"""Register Speculators-format Domino checkpoints with vLLM.

Domino is DFlash's backbone plus a GRU that carries state across the block, so
its logits at position ``k`` depend on the tokens sampled before ``k``. That
requires vLLM's DSpark runtime (the sequential in-block sampler); this plugin
routes Domino there and registers a Domino-specific model implementation.

The sampler itself, the config-layer patches, and the speculator ``__init__``
hook dispatch live in :mod:`speculators.vllm._dflash_family` so DFly and Domino
cannot clobber each other's patches. Unlike DFly, Domino needs no ``__init__``
hook: its context projection is plain DFlash, so the speculator's default
hidden-state buffer already has the right width.
"""

from __future__ import annotations

import logging
from typing import Any

from speculators.models.domino.config import (
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_LOGITS_CORRECTION_EMB_DIM,
    DEFAULT_PURE_DRAFT_PREFIX_LEN,
    resolve_suffix_start,
)
from speculators.vllm._dflash_family import (
    DSPARK_COMPAT_ARCH,
    install_config_patches,
    install_speculator_patches,
    normalize_rope,
    propagate_intra_block_causality,
    register_config_updater,
    register_speculative_method_alias,
)

logger = logging.getLogger(__name__)

_DOMINO_ARCH = "Qwen3DominoModel"


def _update_domino(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators Domino config into vLLM draft config fields."""
    normalize_rope(pre_trained_config)

    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", True))

    # Keep the compatibility architecture second; registry resolution selects
    # Domino first, while the sentinel prevents vLLM's DeepSeek-v4 rewrite.
    pre_trained_config["architectures"] = [
        _DOMINO_ARCH,
        DSPARK_COMPAT_ARCH,
    ]
    pre_trained_config["model_arch"] = "domino"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["num_target_layers"] = len(aux_layer_ids)
    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor
    pre_trained_config["block_size"] = config_dict.get("block_size", 8)
    # Domino's correction replaces the Markov bias rather than composing with it.
    pre_trained_config["markov_rank"] = 0
    # Every fallback here must be the training-side default: a checkpoint config
    # that omits a field would otherwise be served with a different head shape
    # or a different corrected-slot range than it was trained with.
    pre_trained_config["gru_hidden_dim"] = config_dict.get(
        "gru_hidden_dim", DEFAULT_GRU_HIDDEN_DIM
    )
    pre_trained_config["logits_correction_emb_dim"] = config_dict.get(
        "logits_correction_emb_dim", DEFAULT_LOGITS_CORRECTION_EMB_DIM
    )
    pure_draft_prefix_len = config_dict.get(
        "pure_draft_prefix_len", DEFAULT_PURE_DRAFT_PREFIX_LEN
    )
    pre_trained_config["pure_draft_prefix_len"] = pure_draft_prefix_len

    block_size = pre_trained_config["block_size"]
    suffix_start = resolve_suffix_start(
        sample_from_anchor=sample_from_anchor,
        pure_draft_prefix_len=pure_draft_prefix_len,
    )
    if not 0 <= suffix_start < block_size:
        raise ValueError(
            "Domino has no corrected slots to serve: pure_draft_prefix_len="
            f"{pure_draft_prefix_len} with sample_from_anchor={sample_from_anchor} "
            f"gives suffix_start={suffix_start} for block_size={block_size}."
        )

    # sample_from_anchor is deliberately *not* mirrored into dflash_config:
    # DFlashSpeculator.__init__ -- which DSparkSpeculator inherits -- raises when
    # it finds sample_from_anchor=True there, while DSparkSpeculator reads the
    # top-level field to size its query layout. Domino defaults to True, so
    # duplicating the key would make every Domino draft fail to start.
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "linear_position_ids": True,
    }
    # Resolve the attention mask exactly as training does. In particular, a
    # false sliding-window flag must not become a global causal override: full
    # attention layers are always bidirectional during training.
    propagate_intra_block_causality(
        config_dict,
        pre_trained_config,
        default=False,
    )


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater("domino", _update_domino)
    register_speculative_method_alias("domino", "dspark")
    install_config_patches()
    if _DOMINO_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _DOMINO_ARCH,
            "speculators.vllm.domino_model:Qwen3DominoForCausalLM",
        )
    install_speculator_patches()
    logger.info("Registered Speculators Domino support for vLLM.")
