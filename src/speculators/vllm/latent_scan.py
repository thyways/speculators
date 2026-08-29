"""Register Speculators-format LatentScan checkpoints with vLLM."""

from __future__ import annotations

import logging
from typing import Any

from speculators.vllm._dflash_family import (
    install_config_patches,
    normalize_rope,
    propagate_intra_block_causality,
    register_config_updater,
    register_speculative_method_alias,
)

logger = logging.getLogger(__name__)

_LATENT_SCAN_ARCH = "Qwen3LatentScanModel"


def _update_latent_scan(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators LatentScan config into vLLM draft fields."""
    normalize_rope(pre_trained_config)

    if bool(config_dict.get("sample_from_anchor", False)):
        raise ValueError("latent_scan serving requires sample_from_anchor=False.")

    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    pre_trained_config["architectures"] = [_LATENT_SCAN_ARCH]
    pre_trained_config["model_arch"] = "latent_scan"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["block_size"] = int(config_dict.get("block_size", 8))
    pre_trained_config["latent_dim"] = int(config_dict.get("latent_dim", 256))
    pre_trained_config["latent_layer_scale_init"] = float(
        config_dict.get("latent_layer_scale_init", 1e-3)
    )
    pre_trained_config["strict_causal_slots"] = bool(
        config_dict.get("strict_causal_slots", True)
    )
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "sample_from_anchor": False,
        "linear_position_ids": True,
    }

    if pre_trained_config["strict_causal_slots"]:
        pre_trained_config["dflash_config"]["causal"] = True
    else:
        propagate_intra_block_causality(
            config_dict,
            pre_trained_config,
            default=False,
        )


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater("latent_scan", _update_latent_scan)
    register_speculative_method_alias("latent_scan", "dflash")
    install_config_patches()
    if _LATENT_SCAN_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _LATENT_SCAN_ARCH,
            "speculators.vllm.latent_scan_model:Qwen3LatentScanForCausalLM",
        )
    logger.info("Registered Speculators LatentScan support for vLLM.")


__all__ = ["register"]
