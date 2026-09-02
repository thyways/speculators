"""Register Speculators-format HashGram checkpoints with vLLM.

HashGram uses the DFlash Qwen3 backbone and DSpark's left-to-right block
runtime, then replaces DSpark's dense Markov-only proposal with a top-k
HashGram reranker.  This plugin translates the training checkpoint config,
registers the draft model, and selects the HashGram speculator at runtime.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vllm.plugins.speculators_hashgram")

HASHGRAM_ARCH = "HashGramDraftModel"
_DSPARK_COMPAT_ARCH = "Qwen3DSparkModel"
_MODEL_IMPL = "speculators.vllm.hashgram_model:Qwen3HashGramForCausalLM"
_CONFIG_PATCH_MARKER = "_speculators_hashgram_config_patch"
_FACTORY_PATCH_MARKER = "_speculators_hashgram_factory_patch"

_HASHGRAM_DEFAULTS: dict[str, Any] = {
    "hashgram_rank": 128,
    "hashgram_top_k": 16,
    "hashgram_bigram_buckets": 1_048_576,
    "hashgram_trigram_buckets": 1_048_576,
    "hashgram_num_hashes": 1,
    "hashgram_markov_rank": 256,
    "hashgram_use_markov_recall": True,
    "hashgram_hidden_refine": False,
    "hashgram_use_bigram": True,
    "hashgram_use_trigram": True,
}


def _require_bool(config_dict: dict[str, Any], key: str, default: bool) -> bool:
    value = config_dict.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean.")
    return value


def _update_hashgram(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a HashGram training config into a vLLM draft config."""
    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    if not aux_layer_ids:
        raise ValueError("HashGram requires at least one auxiliary hidden-state layer.")

    sample_from_anchor = _require_bool(config_dict, "sample_from_anchor", True)
    sliding_non_causal = _require_bool(
        config_dict,
        "sliding_window_non_causal",
        True,
    )
    use_bigram = _require_bool(config_dict, "hashgram_use_bigram", True)
    use_trigram = _require_bool(config_dict, "hashgram_use_trigram", True)
    if not use_bigram and not use_trigram:
        raise ValueError(
            "At least one of hashgram_use_bigram/hashgram_use_trigram must be enabled."
        )

    # The second architecture prevents vLLM's method=dspark config path from
    # rewriting a self-contained Qwen3 draft into the embedded DeepSeek-V4 head.
    # ModelRegistry selects the first supported architecture, i.e. HashGram.
    pre_trained_config["architectures"] = [
        HASHGRAM_ARCH,
        _DSPARK_COMPAT_ARCH,
    ]
    pre_trained_config["model_arch"] = "hashgram"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["num_target_layers"] = len(aux_layer_ids)
    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor
    pre_trained_config["block_size"] = int(config_dict.get("block_size", 8))
    pre_trained_config["mask_token_id"] = config_dict["mask_token_id"]

    # Keep sample_from_anchor at the top level. DSparkSpeculator reads it there,
    # while its DFlash base class rejects sample_from_anchor=True inside
    # dflash_config.
    dflash_config: dict[str, Any] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
    }
    if sliding_non_causal:
        dflash_config["causal"] = False
    pre_trained_config["dflash_config"] = dflash_config

    for key, default in _HASHGRAM_DEFAULTS.items():
        pre_trained_config[key] = config_dict.get(key, default)
    # Native DSpark fields used by shared config/runtime checks.
    pre_trained_config["markov_rank"] = (
        int(pre_trained_config["hashgram_markov_rank"])
        if pre_trained_config["hashgram_use_markov_recall"]
        else 0
    )
    pre_trained_config["enable_confidence_head"] = False


def _finalize_speculative_config(
    config_dict: dict[str, Any],
    speculative_config: dict[str, Any],
) -> dict[str, Any]:
    """Route HashGram through the DSpark block runtime and validate its width."""
    if speculative_config.get("method") != "hashgram":
        return speculative_config

    sample_from_anchor = _require_bool(config_dict, "sample_from_anchor", True)
    block_size = int(config_dict.get("block_size", 8))
    expected_tokens = block_size if sample_from_anchor else block_size - 1
    actual_tokens = int(speculative_config["num_speculative_tokens"])
    if actual_tokens != expected_tokens:
        raise ValueError(
            "HashGram serving must use the block width it was trained with: "
            f"block_size={block_size}, sample_from_anchor={sample_from_anchor} "
            f"requires num_speculative_tokens={expected_tokens}, got "
            f"{actual_tokens}."
        )

    result = dict(speculative_config)
    result["method"] = "dspark"
    return result


def _install_config_support() -> None:
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )
    from vllm.transformers_utils.configs.speculators.base import (  # noqa: PLC0415
        SpeculatorsConfig,
    )

    SUPPORTED_SPECULATORS_TYPES["hashgram"] = _update_hashgram
    if getattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, False):
        return

    original = SpeculatorsConfig.build_vllm_speculative_config.__func__

    @classmethod  # type: ignore[misc]
    def build_vllm_speculative_config(
        cls: type,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return _finalize_speculative_config(
            config_dict,
            original(cls, config_dict),
        )

    SpeculatorsConfig.build_vllm_speculative_config = build_vllm_speculative_config
    setattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, True)


def _is_hashgram_draft(speculative_config: Any) -> bool:
    if speculative_config is None or speculative_config.method != "dspark":
        return False
    draft_config = getattr(speculative_config, "draft_model_config", None)
    if draft_config is None:
        return False
    return HASHGRAM_ARCH in (draft_config.architectures or ())


def _install_speculator_factory() -> None:
    from vllm.v1.worker.gpu import model_runner, spec_decode  # noqa: PLC0415

    if getattr(spec_decode.init_speculator, _FACTORY_PATCH_MARKER, False):
        return

    original = spec_decode.init_speculator

    def init_speculator(vllm_config: Any, device: Any) -> Any:
        if _is_hashgram_draft(vllm_config.speculative_config):
            from speculators.vllm.hashgram_speculator import (  # noqa: PLC0415
                HashGramSpeculator,
            )

            return HashGramSpeculator(vllm_config, device)
        return original(vllm_config, device)

    setattr(init_speculator, _FACTORY_PATCH_MARKER, True)
    spec_decode.init_speculator = init_speculator
    # model_runner imports the factory by name, so update that bound reference too.
    model_runner.init_speculator = init_speculator


def register() -> None:
    """vLLM ``general_plugins`` entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    _install_config_support()
    if HASHGRAM_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(HASHGRAM_ARCH, _MODEL_IMPL)
    _install_speculator_factory()
    logger.info("Registered Speculators HashGram support for vLLM.")


__all__ = ["HASHGRAM_ARCH", "register"]
