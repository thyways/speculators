"""Register the parallel token-latent feedback drafter with vLLM."""

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

_FEEDBACK_ARCH = "Qwen3TokenLatentFeedbackModel"


def _first_set(config_dict: dict[str, Any], *keys: str, default: Any) -> Any:
    """Return the first alias that carries a value, else ``default``.

    Checkpoints serialize every unused legacy alias as ``null``, so a plain
    ``config_dict.get(legacy, config_dict.get(current, default))`` resolves to
    ``None`` instead of falling through to the current key.
    """
    for key in keys:
        value = config_dict.get(key)
        if value is not None:
            return value
    return default


def _update_token_latent_feedback(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators v1.2 config into vLLM draft fields."""
    if bool(config_dict.get("sample_from_anchor", False)):
        raise ValueError(
            "token_latent_feedback serving requires sample_from_anchor=False."
        )

    normalize_rope(pre_trained_config)
    aux_layer_ids = list(config_dict.get("aux_hidden_state_layer_ids") or [])
    if not aux_layer_ids:
        raise ValueError("token_latent_feedback requires aux_hidden_state_layer_ids.")

    latent_dim = int(
        _first_set(config_dict, "token_latent_dim", "latent_dim", default=128)
    )
    mixer_mode = str(
        _first_set(
            config_dict,
            "prefix_mixer",
            "feedback_mode",
            "prefix_mixer_mode",
            default="full",
        )
    )
    pre_trained_config["architectures"] = [_FEEDBACK_ARCH]
    pre_trained_config["model_arch"] = "token_latent_feedback"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["block_size"] = int(config_dict.get("block_size", 8))
    pre_trained_config["sample_from_anchor"] = False
    pre_trained_config["latent_dim"] = latent_dim
    pre_trained_config["feedback_stages"] = int(
        _first_set(
            config_dict,
            "latent_feedback_stages",
            "feedback_stages",
            default=1,
        )
    )
    pre_trained_config["prefix_mixer_mode"] = mixer_mode
    pre_trained_config["prefix_mixer_parameterization"] = "toeplitz"
    pre_trained_config["use_reliability_gate"] = bool(
        _first_set(
            config_dict,
            "reliability_gate",
            "use_reliability_gate",
            default=True,
        )
    )
    pre_trained_config["strict_causal_prefix"] = bool(
        config_dict.get("strict_causal_prefix", True)
    )
    pre_trained_config["position_scale_init"] = float(
        config_dict.get("position_scale_init", 1.0)
    )
    pre_trained_config["position_scale_parameterization"] = str(
        config_dict.get("position_scale_parameterization", "direct")
    )
    pre_trained_config["position_scale_min"] = float(
        config_dict.get("position_scale_min", 0.0)
    )
    pre_trained_config["feedback_output_projection_init"] = float(
        config_dict.get("feedback_output_projection_init", 0.0)
    )
    pre_trained_config["feedback_output_projection_init_mode"] = str(
        config_dict.get("feedback_output_projection_init_mode", "constant")
    )
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "sample_from_anchor": False,
        "linear_position_ids": True,
    }
    # v1.2 follows ordinary DFlash attention defaults; propagate an explicitly
    # serialized sliding-window choice without forcing a global causal override.
    propagate_intra_block_causality(
        config_dict,
        pre_trained_config,
        default=False,
    )


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    for algorithm in (
        "token_latent_feedback",
        "parallel_token_latent",
        "latent_feedback",
        "parallel_token_latent_feedback",
    ):
        register_config_updater(algorithm, _update_token_latent_feedback)
        register_speculative_method_alias(algorithm, "dflash")
    install_config_patches()
    if _FEEDBACK_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _FEEDBACK_ARCH,
            "speculators.vllm.token_latent_feedback_model:"
            "Qwen3TokenLatentFeedbackForCausalLM",
        )
    logger.info("Registered Speculators token-latent feedback support for vLLM.")


__all__ = ["register"]
