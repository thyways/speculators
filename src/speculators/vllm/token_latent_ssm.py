"""Register token-latent SSM checkpoints with the sequential vLLM runtime."""

from __future__ import annotations

import logging
from typing import Any

from speculators.models.token_latent_ssm.config import (
    DEFAULT_HIDDEN_CANDIDATES,
    DEFAULT_LOGIT_SCALE_INIT,
    DEFAULT_SSM_STATE_DIM,
    DEFAULT_TOKEN_CODE_DIM,
    DEFAULT_TRANSITION_CANDIDATES,
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

_V11_ARCH = "Qwen3TokenLatentSSMModel"


def _update_token_latent_ssm(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators v1.1 checkpoint into vLLM draft fields."""
    if bool(config_dict.get("sample_from_anchor", False)):
        raise ValueError("token_latent_ssm requires sample_from_anchor=False.")

    normalize_rope(pre_trained_config)
    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    pre_trained_config["architectures"] = [
        _V11_ARCH,
        DSPARK_COMPAT_ARCH,
    ]
    pre_trained_config["model_arch"] = "token_latent_ssm"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["num_target_layers"] = len(aux_layer_ids)
    pre_trained_config["sample_from_anchor"] = False
    pre_trained_config["dspark_bonus_anchor"] = True
    pre_trained_config["block_size"] = config_dict.get("block_size", 8)
    pre_trained_config["markov_rank"] = 0
    pre_trained_config["enable_confidence_head"] = False
    pre_trained_config["token_code_dim"] = config_dict.get(
        "token_code_dim",
        DEFAULT_TOKEN_CODE_DIM,
    )
    pre_trained_config["ssm_state_dim"] = config_dict.get(
        "ssm_state_dim",
        DEFAULT_SSM_STATE_DIM,
    )
    pre_trained_config["hidden_candidate_count"] = config_dict.get(
        "hidden_candidate_count",
        DEFAULT_HIDDEN_CANDIDATES,
    )
    pre_trained_config["transition_candidate_count"] = config_dict.get(
        "transition_candidate_count",
        DEFAULT_TRANSITION_CANDIDATES,
    )
    pre_trained_config["token_latent_logit_scale_init"] = config_dict.get(
        "token_latent_logit_scale_init",
        DEFAULT_LOGIT_SCALE_INIT,
    )
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "linear_position_ids": True,
    }
    propagate_intra_block_causality(
        config_dict,
        pre_trained_config,
        default=True,
    )


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater("token_latent_ssm", _update_token_latent_ssm)
    register_speculative_method_alias("token_latent_ssm", "dspark")
    install_config_patches()
    if _V11_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _V11_ARCH,
            "speculators.vllm.token_latent_ssm_model:Qwen3TokenLatentSSMForCausalLM",
        )
    install_speculator_patches()
    logger.info("Registered token-latent SSM support for vLLM.")


__all__ = ["register"]
