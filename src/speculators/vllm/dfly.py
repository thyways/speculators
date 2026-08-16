"""Register Speculators-format DFly checkpoints with vLLM.

DFly keeps DFlash's parallel backbone, but its hidden correction samples the
block from left to right.  vLLM's DSpark runtime already owns that sequential
sampling layout, so this plugin routes DFly through it while registering a
DFly-specific model implementation.

The sequential sampler, the config-layer patches, and the speculator ``__init__``
hook all live in :mod:`speculators.vllm._dflash_family`, which owns a single
patch of each vLLM slot; see that module for why sharing is mandatory rather
than tidy.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from speculators.vllm._dflash_family import (
    DSPARK_COMPAT_ARCH,
    install_config_patches,
    install_speculator_patches,
    normalize_rope,
    propagate_intra_block_causality,
    register_config_updater,
    register_init_hook,
    register_speculative_method_alias,
)

logger = logging.getLogger(__name__)

_DFLY_ARCH = "Qwen3DFlyModel"


def _is_dfly_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", ()) or ()
    return _DFLY_ARCH in architectures or getattr(config, "model_arch", None) == "dfly"


def _update_dfly(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators DFly config into vLLM draft config fields."""
    normalize_rope(pre_trained_config)

    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", False))

    # Keep the compatibility architecture second; registry resolution selects
    # DFly first, while the sentinel prevents vLLM's DeepSeek-v4 rewrite.
    pre_trained_config["architectures"] = [
        _DFLY_ARCH,
        DSPARK_COMPAT_ARCH,
    ]
    pre_trained_config["model_arch"] = "dfly"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["num_target_layers"] = len(aux_layer_ids)
    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor
    pre_trained_config["block_size"] = config_dict.get("block_size", 8)
    pre_trained_config["markov_rank"] = 0
    pre_trained_config["enable_hidden_correction"] = config_dict.get(
        "enable_hidden_correction", True
    )
    pre_trained_config["hidden_correction_intermediate_size"] = config_dict.get(
        "hidden_correction_intermediate_size"
    )
    pre_trained_config["hidden_correction_type"] = "swiglu"
    # sample_from_anchor stays top-level only: DFlashSpeculator.__init__, which
    # DSparkSpeculator inherits, raises when it finds sample_from_anchor=True
    # inside dflash_config, and DSparkSpeculator reads the top-level field to
    # size its query layout. DFly defaults to False so the duplicate was
    # harmless in practice, but it would break an anchor-sampling DFly draft.
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "linear_position_ids": True,
    }
    propagate_intra_block_causality(
        config_dict,
        pre_trained_config,
        default=False,
    )


def _dfly_context_width(config: Any) -> int:
    dflash_config = getattr(config, "dflash_config", None) or {}
    target_layer_ids = (
        dflash_config.get("target_layer_ids")
        or getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        or getattr(config, "target_layer_ids", None)
    )
    num_target_layers = getattr(config, "num_target_layers", None)
    if target_layer_ids:
        num_target_layers = len(target_layer_ids)
    if not num_target_layers:
        raise ValueError("DFly requires at least one target hidden-state layer.")

    hidden_size = int(config.hidden_size)
    target_hidden_size = int(getattr(config, "target_hidden_size", None) or hidden_size)
    return int(num_target_layers) * target_hidden_size


def _resize_dfly_context_buffer(speculator: Any, device: torch.device) -> None:
    """DFly consumes the uncollapsed T*D auxiliary states, so widen the buffer."""
    config = speculator.draft_model_config.hf_config
    if not _is_dfly_config(config):
        return
    speculator.hidden_states = torch.zeros(
        speculator.max_num_tokens,
        _dfly_context_width(config),
        dtype=speculator.dtype,
        device=device,
    )
    speculator._speculator_name = "DFly"  # noqa: SLF001


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater("dfly", _update_dfly)
    register_speculative_method_alias("dfly", "dspark")
    register_init_hook(_resize_dfly_context_buffer)
    install_config_patches()
    if _DFLY_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _DFLY_ARCH,
            "speculators.vllm.dfly_model:Qwen3DFlyForCausalLM",
        )
    install_speculator_patches()
    logger.info("Registered Speculators DFly support for vLLM.")
