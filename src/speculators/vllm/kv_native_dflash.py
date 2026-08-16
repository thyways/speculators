"""Register dual-stream raw-KV DFlash checkpoints with vLLM V1 and V2."""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any

import torch
from hs_connectors.verifier_kv import discover_selected_verifier_kv

from speculators.vllm._dflash_family import (
    install_config_patches,
    register_config_updater,
    register_speculative_config_updater,
    register_speculative_method_alias,
)

logger = logging.getLogger(__name__)

_ALGORITHM = "kv_native_dflash"
_KV_NATIVE_ARCH = "Qwen3KVNativeDFlashForCausalLM"
_V1_PROPOSER_PATCH_MARKER = "_speculators_kv_native_dflash_v1_proposer_patched"
_V2_SPECULATOR_PATCH_MARKER = "_speculators_kv_native_dflash_v2_speculator_patched"
_V1_RUNNER_PATCH_MARKER = "_speculators_kv_native_dflash_v1_runner_patched"
_V2_RUNNER_PATCH_MARKER = "_speculators_kv_native_dflash_v2_runner_patched"


def _is_kv_native_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", ()) or ()
    return (
        _KV_NATIVE_ARCH in architectures
        or getattr(config, "model_arch", None) == _ALGORITHM
    )


def _draft_hf_config(vllm_config: Any) -> Any | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return getattr(draft_model_config, "hf_config", None)


def _update_kv_native_dflash(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate the final training checkpoint into vLLM's DFlash config."""

    if config_dict.get("kv_native_architecture") != "dual_stream_raw_kv":
        raise ValueError(
            "This checkpoint predates the final dual_stream_raw_kv architecture. "
            "Train a new checkpoint from scratch."
        )
    if config_dict.get("sample_from_anchor", False):
        raise ValueError(
            "KV-native DFlash requires sample_from_anchor=false so the anchor "
            "remains the verifier bonus token."
        )
    if config_dict.get("anchor_hidden_injection", False):
        raise ValueError(
            "This checkpoint was trained with anchor_hidden_injection=true, which "
            "this runtime cannot reproduce: the draft model is called with only "
            "(input_ids, positions) and never receives the verifier hidden state "
            "at the last verified position, so serving would silently drop an "
            "input the draft was trained on. Retrain with the flag disabled, or "
            "plumb the context state through to the draft forward first."
        )
    layer_types = list(pre_trained_config.get("layer_types") or [])
    if not layer_types or any(value != "full_attention" for value in layer_types):
        raise NotImplementedError(
            "Dual-stream raw-KV serving currently requires every draft layer "
            "to use full_attention."
        )

    verifier_layer_ids = [
        int(layer_id) for layer_id in config_dict["verifier_kv_layer_ids"]
    ]
    verifier_layer_mapping = [
        int(layer_id) for layer_id in config_dict["verifier_kv_layer_mapping"]
    ]
    if not verifier_layer_ids or len(verifier_layer_ids) != len(
        set(verifier_layer_ids)
    ):
        raise ValueError("verifier_kv_layer_ids must be non-empty and unique")
    if len(verifier_layer_mapping) != len(layer_types):
        raise ValueError(
            "verifier_kv_layer_mapping must contain one source per draft layer"
        )
    unknown_mapping = sorted(set(verifier_layer_mapping) - set(verifier_layer_ids))
    if unknown_mapping:
        raise ValueError(
            "verifier_kv_layer_mapping references non-exported layers: "
            f"{unknown_mapping}"
        )
    mask_token_id = config_dict.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("KV-native DFlash requires mask_token_id in the checkpoint")

    partial_rotary_factor = float(config_dict["verifier_partial_rotary_factor"])
    rope_theta = float(config_dict["verifier_rope_theta"])
    pre_trained_config["architectures"] = [_KV_NATIVE_ARCH]
    pre_trained_config["model_arch"] = _ALGORITHM
    pre_trained_config["kv_native_architecture"] = "dual_stream_raw_kv"
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    pre_trained_config["target_hidden_size"] = config_dict.get(
        "target_hidden_size"
    ) or pre_trained_config.get("hidden_size")
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = []
    pre_trained_config["num_target_layers"] = 0
    pre_trained_config["sample_from_anchor"] = False
    pre_trained_config["block_size"] = int(config_dict.get("block_size", 16))
    pre_trained_config["verifier_num_key_value_heads"] = int(
        config_dict["verifier_num_key_value_heads"]
    )
    pre_trained_config["verifier_head_dim"] = int(config_dict["verifier_head_dim"])
    pre_trained_config["verifier_kv_layer_ids"] = verifier_layer_ids
    pre_trained_config["verifier_kv_layer_mapping"] = verifier_layer_mapping
    pre_trained_config["verifier_partial_rotary_factor"] = partial_rotary_factor
    pre_trained_config["verifier_rope_theta"] = rope_theta
    pre_trained_config["partial_rotary_factor"] = partial_rotary_factor
    pre_trained_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary_factor,
    }
    pre_trained_config["dflash_config"] = {
        "mask_token_id": int(mask_token_id),
        "target_layer_ids": [],
        "use_aux_hidden_state": False,
        "sample_from_anchor": False,
        "linear_position_ids": True,
        "causal": False,
    }


def _finalize_kv_native_speculative_config(
    config_dict: dict[str, Any],
    speculative_config: dict[str, Any],
) -> None:
    block_size = int(config_dict.get("block_size", 16))
    trained_tokens = block_size - 1
    num_tokens = int(
        config_dict.get(
            "num_speculative_tokens",
            speculative_config["num_speculative_tokens"],
        )
    )
    if num_tokens != trained_tokens:
        raise ValueError(
            "KV-native DFlash inference requires num_speculative_tokens to equal "
            f"block_size-1: got {num_tokens}, expected {trained_tokens}."
        )
    speculative_config["num_speculative_tokens"] = num_tokens


def _unwrap_model(model: Any) -> Any:
    unwrap = getattr(model, "unwrap", None)
    return unwrap() if callable(unwrap) else model


def _bind_verifier_kv_runtime(speculator: Any, kv_cache_config: Any) -> None:
    config = speculator.draft_model_config.hf_config
    if not _is_kv_native_config(config):
        return
    parallel = speculator.vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError(
            "KV-native DFlash inference requires tensor_parallel_size=1 and "
            "pipeline_parallel_size=1."
        )
    if getattr(parallel, "use_ubatching", False):
        raise ValueError("KV-native DFlash does not support ubatching.")

    metadata = discover_selected_verifier_kv(
        kv_cache_config,
        list(config.verifier_kv_layer_ids),
    )
    from vllm.config import get_layers_from_vllm_config  # noqa: PLC0415
    from vllm.model_executor.layers.attention_layer_base import (  # noqa: PLC0415
        AttentionLayerBase,
    )

    all_attention = get_layers_from_vllm_config(
        speculator.vllm_config,
        AttentionLayerBase,
    )
    missing = [name for name in metadata.layer_names if name not in all_attention]
    if missing:
        raise KeyError(f"Selected verifier attention layers are missing: {missing}")
    model = _unwrap_model(speculator.model)
    model.bind_verifier_kv(
        verifier_attention={name: all_attention[name] for name in metadata.layer_names},
        verifier_metadata=metadata,
        slot_mapping_capacity=int(speculator.max_num_tokens),
        slot_mapping_device=speculator.device,
    )
    logger.info(
        "Bound raw verifier KV layer/group pairs: %s.",
        list(zip(metadata.layer_ids, metadata.cache_group_ids, strict=True)),
    )


def _snapshot_verifier_slot_mappings(
    speculator: Any,
    num_tokens: int,
    slot_mappings: Any,
) -> Any | None:
    config = speculator.draft_model_config.hf_config
    if not _is_kv_native_config(config):
        return None
    model = _unwrap_model(speculator.model)
    model.clear_verifier_slot_mappings()
    if slot_mappings is None:
        raise RuntimeError(
            "KV-native DFlash requires target slot mappings for live verifier K/V."
        )
    if isinstance(slot_mappings, list):
        raise ValueError("KV-native DFlash does not support ubatched slot mappings.")
    model.snapshot_verifier_slot_mappings(slot_mappings, int(num_tokens))
    return model


def _install_v1_dflash_proposer_patch() -> None:
    from vllm.v1.spec_decode.dflash import DFlashProposer  # noqa: PLC0415

    if getattr(DFlashProposer, _V1_PROPOSER_PATCH_MARKER, False):
        return
    original_initialize_attn_backend = DFlashProposer.initialize_attn_backend
    original_propose = DFlashProposer.propose
    original_determine = (
        DFlashProposer._determine_batch_execution_and_padding  # noqa: SLF001
    )
    propose_signature = inspect.signature(original_propose)

    @wraps(original_initialize_attn_backend)
    def initialize_attn_backend(
        self: Any,
        kv_cache_config: Any,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        original_initialize_attn_backend(self, kv_cache_config, kernel_block_sizes)
        _bind_verifier_kv_runtime(self, kv_cache_config)

    @wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = propose_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        model = _snapshot_verifier_slot_mappings(
            self,
            int(bound.arguments["target_token_ids"].shape[0]),
            bound.arguments["slot_mappings"],
        )
        try:
            return original_propose(self, *args, **kwargs)
        finally:
            if model is not None:
                model.clear_verifier_slot_mappings()

    @wraps(original_determine)
    def determine(self: Any, num_tokens: int, use_cudagraphs: bool = True):
        if _is_kv_native_config(self.draft_model_config.hf_config):
            use_cudagraphs = False
        return original_determine(self, num_tokens, use_cudagraphs)

    DFlashProposer.initialize_attn_backend = initialize_attn_backend
    DFlashProposer.propose = propose
    DFlashProposer._determine_batch_execution_and_padding = determine  # noqa: SLF001
    setattr(DFlashProposer, _V1_PROPOSER_PATCH_MARKER, True)


def _install_v2_dflash_speculator_patch() -> None:
    from vllm.config.compilation import CUDAGraphMode  # noqa: PLC0415
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (  # noqa: PLC0415
        DFlashSpeculator,
    )

    if getattr(DFlashSpeculator, _V2_SPECULATOR_PATCH_MARKER, False):
        return
    original_set_attn = DFlashSpeculator.set_attn
    original_propose = DFlashSpeculator.propose
    original_init_cudagraph = DFlashSpeculator.init_cudagraph_manager
    set_attn_signature = inspect.signature(original_set_attn)
    propose_signature = inspect.signature(original_propose)

    @wraps(original_set_attn)
    def set_attn(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = set_attn_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_set_attn(self, *args, **kwargs)
        _bind_verifier_kv_runtime(self, bound.arguments["kv_cache_config"])

    @wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = propose_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        model = None
        if not bool(bound.arguments["dummy_run"]):
            model = _snapshot_verifier_slot_mappings(
                self,
                int(bound.arguments["input_batch"].num_tokens),
                bound.arguments["slot_mappings"],
            )
        try:
            return original_propose(self, *args, **kwargs)
        finally:
            if model is not None:
                model.clear_verifier_slot_mappings()

    @wraps(original_init_cudagraph)
    def init_cudagraph_manager(self: Any, cudagraph_mode: Any) -> None:
        if _is_kv_native_config(self.draft_model_config.hf_config):
            cudagraph_mode = CUDAGraphMode.NONE
        original_init_cudagraph(self, cudagraph_mode)

    DFlashSpeculator.set_attn = set_attn
    DFlashSpeculator.propose = propose
    DFlashSpeculator.init_cudagraph_manager = init_cudagraph_manager
    setattr(DFlashSpeculator, _V2_SPECULATOR_PATCH_MARKER, True)


def _disable_kv_native_aux_hidden_states(model_runner: Any) -> None:
    config = _draft_hf_config(model_runner.vllm_config)
    if config is None or not _is_kv_native_config(config):
        return
    model_runner.use_aux_hidden_state_outputs = False
    logger.info("Disabled verifier auxiliary hidden states for KV-native DFlash.")


def _patch_runner(runner_class: type, marker: str) -> None:
    if getattr(runner_class, marker, False):
        return
    original_init = runner_class.__init__

    @wraps(original_init)
    def init(
        self: Any, vllm_config: Any, device: torch.device, *args: Any, **kwargs: Any
    ) -> None:
        original_init(self, vllm_config, device, *args, **kwargs)
        _disable_kv_native_aux_hidden_states(self)

    runner_class.__init__ = init
    setattr(runner_class, marker, True)


def _install_model_runner_patches() -> None:
    from vllm.v1.worker.gpu.model_runner import (  # noqa: PLC0415
        GPUModelRunner as V2GPUModelRunner,
    )
    from vllm.v1.worker.gpu_model_runner import (  # noqa: PLC0415
        GPUModelRunner as V1GPUModelRunner,
    )

    _patch_runner(V1GPUModelRunner, _V1_RUNNER_PATCH_MARKER)
    _patch_runner(V2GPUModelRunner, _V2_RUNNER_PATCH_MARKER)


def register() -> None:
    """vLLM general-plugin entry point."""

    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater(_ALGORITHM, _update_kv_native_dflash)
    register_speculative_method_alias(_ALGORITHM, "dflash")
    register_speculative_config_updater(
        _ALGORITHM,
        _finalize_kv_native_speculative_config,
    )
    install_config_patches()
    _install_v1_dflash_proposer_patch()
    _install_v2_dflash_speculator_patch()
    _install_model_runner_patches()
    if _KV_NATIVE_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _KV_NATIVE_ARCH,
            "speculators.vllm.kv_native_dflash_model:Qwen3KVNativeDFlashForCausalLM",
        )
    logger.info("Registered dual-stream raw-KV DFlash support for vLLM V1 and V2.")


__all__ = ["register"]
