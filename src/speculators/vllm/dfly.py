"""Register Speculators-format DFly checkpoints with vLLM.

DFly keeps DFlash's parallel backbone, but its hidden correction samples the
block from left to right.  vLLM's DSpark runtime already owns that sequential
sampling layout, so this plugin routes DFly through it while registering a
DFly-specific model implementation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DFLY_ARCH = "Qwen3DFlyModel"
_DSPARK_COMPAT_ARCH = "Qwen3DSparkModel"
_PATCH_MARKER = "_speculators_dfly_patched"


def _is_dfly_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", ()) or ()
    return _DFLY_ARCH in architectures or getattr(config, "model_arch", None) == "dfly"


def _normalize_rope(pre_trained_config: dict[str, Any]) -> None:
    """Convert target M-RoPE metadata to the drafter's linear positions."""
    rope_parameters = pre_trained_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        return
    if "mrope_section" not in rope_parameters:
        return

    rope_parameters = dict(rope_parameters)
    rope_parameters.pop("mrope_section", None)
    rope_parameters.pop("mrope_interleaved", None)
    pre_trained_config["rope_parameters"] = rope_parameters


def _update_dfly(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators DFly config into vLLM draft config fields."""
    _normalize_rope(pre_trained_config)

    aux_layer_ids = list(config_dict["aux_hidden_state_layer_ids"])
    sample_from_anchor = bool(config_dict.get("sample_from_anchor", False))

    # vLLM 0.26 treats every method=dspark architecture that does not also
    # advertise Qwen3DSparkModel as a DeepSeek-v4 embedded head.  Keep the
    # compatibility architecture second; registry resolution selects DFly
    # first, while the sentinel prevents that incorrect rewrite.
    pre_trained_config["architectures"] = [
        _DFLY_ARCH,
        _DSPARK_COMPAT_ARCH,
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
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [layer_id - 1 for layer_id in aux_layer_ids],
        "linear_position_ids": True,
        "sample_from_anchor": sample_from_anchor,
        "causal": not config_dict.get("sliding_window_non_causal", True),
    }


def _map_dfly_speculative_method(config: dict[str, Any]) -> dict[str, Any]:
    """Route DFly through vLLM's sequential block proposer."""
    if config.get("method") == "dfly":
        config = dict(config)
        config["method"] = "dspark"
    return config


def _preserve_dspark_anchor_mode(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Propagate the checkpoint's DSpark anchor sampling convention."""
    if "sample_from_anchor" not in config_dict:
        return

    sample_from_anchor = config_dict["sample_from_anchor"]
    if not isinstance(sample_from_anchor, bool):
        raise TypeError("DSpark sample_from_anchor must be a boolean.")

    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor


def _resolve_dspark_target_hidden_size(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Fill the target width omitted by same-width DSpark checkpoints."""
    target_hidden_size = config_dict.get("target_hidden_size")
    if target_hidden_size is None:
        target_hidden_size = pre_trained_config.get("hidden_size")
    if not isinstance(target_hidden_size, int) or target_hidden_size <= 0:
        raise ValueError("DSpark target_hidden_size must be a positive integer.")
    pre_trained_config["target_hidden_size"] = target_hidden_size


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


def _sample_sequential_with_hidden_correction(
    self: Any,
    num_reqs: int,
    head_hidden: torch.Tensor,
) -> None:
    """Sample a parallel block left-to-right for DSpark and DFly."""
    num_steps = self.num_speculative_steps
    num_samples = num_reqs * num_steps
    sample_hidden = head_hidden[self.sample_indices[:num_samples]]
    hidden_per_step = sample_hidden.view(num_reqs, num_steps, -1)

    has_hidden_correction = bool(
        getattr(self.model, "has_hidden_correction", lambda: False)()
    )
    base_logits = None
    if not has_hidden_correction:
        logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = logits.view(num_reqs, num_steps, -1)

    has_markov = bool(getattr(self.model, "has_markov", lambda: True)())
    idx_map = self.sample_idx_mapping[:num_samples].view(num_reqs, num_steps)
    sample_pos = self.sample_pos[:num_samples].view(num_reqs, num_steps)
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].long()

    for step in range(num_steps):
        if has_hidden_correction:
            corrected = self.model.apply_hidden_correction(
                hidden_per_step[:, step], prev
            )
            logits_step = self.model.compute_draft_logits(corrected)
        else:
            if base_logits is None:
                raise RuntimeError("Missing precomputed DSpark draft logits.")
            logits_step = base_logits[:, step]

        if has_markov:
            markov_embed = self.model.markov_embed(prev)
            logits_step = logits_step + self.model.markov_bias(markov_embed)

        if self.draft_logits is not None:
            if self._d2t_scatter_index is not None:
                if self._draft_scatter_buf is None:
                    raise RuntimeError("Missing reduced-vocabulary scatter buffer.")
                scatter = self._draft_scatter_buf[:num_reqs]
                scatter.index_copy_(
                    1,
                    self._d2t_scatter_index,
                    logits_step.to(scatter.dtype),
                )
                logits_step = scatter
            from vllm.v1.worker.gpu.sample.gumbel import (  # noqa: PLC0415
                gumbel_sample,
            )

            sampled = gumbel_sample(
                logits_step,
                idx_map[:, step],
                self.temperature,
                self.seeds,
                sample_pos[:, step] - 1,
                apply_temperature=True,
                output_processed_logits=self.draft_logits,
                output_processed_logits_col=self._step_cols[step],
                use_fp64=self.use_fp64_gumbel,
            )
        else:
            sampled = self.model.map_draft_to_target(logits_step.argmax(dim=-1))

        self.draft_tokens[:num_reqs, step] = sampled
        prev = sampled


def _patch_speculators_config() -> None:
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )
    from vllm.transformers_utils.configs.speculators.base import (  # noqa: PLC0415
        SpeculatorsConfig,
    )

    SUPPORTED_SPECULATORS_TYPES.setdefault("dfly", _update_dfly)
    if getattr(SpeculatorsConfig, _PATCH_MARKER, False):
        return

    original_dspark_updater = SUPPORTED_SPECULATORS_TYPES["dspark"]

    def update_dspark(
        config_dict: dict[str, Any],
        pre_trained_config: dict[str, Any],
    ) -> None:
        original_dspark_updater(config_dict, pre_trained_config)
        _preserve_dspark_anchor_mode(config_dict, pre_trained_config)
        _resolve_dspark_target_hidden_size(config_dict, pre_trained_config)

    SUPPORTED_SPECULATORS_TYPES["dspark"] = update_dspark

    original = SpeculatorsConfig.build_vllm_speculative_config.__func__

    @classmethod
    def build_vllm_speculative_config(
        cls: type,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        config = original(cls, config_dict)
        return _map_dfly_speculative_method(config)

    SpeculatorsConfig.build_vllm_speculative_config = build_vllm_speculative_config
    setattr(SpeculatorsConfig, _PATCH_MARKER, True)


def _patch_dspark_speculator() -> None:
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (  # noqa: PLC0415
        DSparkSpeculator,
    )

    if getattr(DSparkSpeculator, _PATCH_MARKER, False):
        return

    original_init = DSparkSpeculator.__init__

    def init(self: Any, vllm_config: Any, device: torch.device) -> None:
        original_init(self, vllm_config, device)
        config = self.draft_model_config.hf_config
        if not _is_dfly_config(config):
            return
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            _dfly_context_width(config),
            dtype=self.dtype,
            device=device,
        )
        self._speculator_name = "DFly"

    DSparkSpeculator.__init__ = init
    DSparkSpeculator._sample_sequential = (  # noqa: SLF001
        _sample_sequential_with_hidden_correction
    )
    setattr(DSparkSpeculator, _PATCH_MARKER, True)


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    _patch_speculators_config()
    if _DFLY_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            _DFLY_ARCH,
            "speculators.vllm.dfly_model:Qwen3DFlyForCausalLM",
        )
    _patch_dspark_speculator()
    logger.info("Registered Speculators DFly support for vLLM.")
