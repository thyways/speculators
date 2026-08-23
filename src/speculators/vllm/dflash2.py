"""Register Speculators-format DFlash2 checkpoints with vLLM.

DFlash2 is DFlash's backbone plus a grouped dynamic convolution inside each block
and a candidate selector over adjacent draft slots, both from
vllm-project/vllm#52816. That PR targets a native z-lab-format checkpoint, so it
carries the model and the speculator but not the Speculators config translation.
This plugin supplies the translation and routes a ``dflash2`` checkpoint onto the
ported runtime:

* ``vllm/transformers_utils/configs/speculators/algos.py`` dispatches on
  ``speculators_model_type`` and ``base.py`` rejects a value it has no updater
  for, so a ``dflash2`` checkpoint cannot be loaded at all without an entry here.
* ``base.py`` also derives ``speculative_config.method`` from that same value,
  while #52816 selects DFlash2 on ``method == "dflash"`` plus the
  ``DFlash2DraftModel`` architecture. The method is aliased back onto ``dflash``,
  the way vLLM already aliases ``peagle`` onto ``eagle3``.
* the V1 ``DFlashProposer`` has no candidate selector, so a DFlash2 checkpoint
  reaching it would draft as DFlash1 without raising. Serve with
  ``VLLM_USE_V2_MODEL_RUNNER=1``; :func:`_check_v2_model_runner` refuses to start
  otherwise rather than report acceptance for a drafter that never ran.

Modeled on ``speculators.vllm.domino`` from the main branch. The config-patch
helpers here duplicate a subset of that branch's ``_dflash_family`` module; fold
them together when the two branches meet, so the algorithms cannot clobber each
other's single patch slot.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

# Named into vLLM's logger hierarchy rather than this module's. vLLM configures a
# handler on the `vllm` logger and leaves the root one bare, so a
# `speculators.vllm.dflash2` logger writes nowhere -- and this module's one log
# line is the evidence that a DFlash2 checkpoint really drafted as DFlash2 rather
# than silently as DFlash1, which a caller reads out of the server log.
logger = init_logger("vllm.plugins.speculators_dflash2")

_DFLASH2_ARCH = "DFlash2DraftModel"
_MODEL_IMPL = "speculators.vllm.dflash2_model:DFlash2Qwen3ForCausalLM"
_CONFIG_PATCH_MARKER = "_speculators_dflash2_config_patched"
_SPECULATOR_PATCH_MARKER = "_speculators_dflash2_speculator_patched"

# Draft-config keys DFlash2 adds on top of DFlash. Every one of them sizes a
# module or a decision the checkpoint was trained with, so a missing key is an
# error rather than a default: serving a different shape than training used is
# exactly what this plugin exists to prevent.
_REQUIRED_KEYS = (
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)
_OPTIONAL_KEYS = (
    "input_embedding_scale",
    "output_multiplier",
    "final_logit_softcapping",
)


def _propagate_intra_block_causality(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Match vLLM's per-layer draft mask to the training-time mask.

    Training always makes full-attention draft layers non-causal;
    ``sliding_window_non_causal`` only controls sliding-window layers. vLLM's
    ``dflash_config.causal`` is a *global* override, so translating a false flag
    into ``causal=True`` would turn full-attention layers causal too -- a
    train/serve mismatch that costs acceptance and raises nothing. Set the
    override only to force non-causal; otherwise drop it and let vLLM's per-layer
    fallback decide.
    """
    non_causal = config_dict.get("sliding_window_non_causal")
    if non_causal is None:
        return
    if not isinstance(non_causal, bool):
        raise TypeError("sliding_window_non_causal must be a boolean.")

    dflash_config = dict(pre_trained_config.get("dflash_config") or {})
    if non_causal:
        dflash_config["causal"] = False
    else:
        dflash_config.pop("causal", None)
    pre_trained_config["dflash_config"] = dflash_config


def _update_dflash2(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    """Translate a Speculators DFlash2 config into vLLM draft config fields."""
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )

    SUPPORTED_SPECULATORS_TYPES["dflash"](config_dict, pre_trained_config)
    pre_trained_config["architectures"] = [_DFLASH2_ARCH]

    draft_config = pre_trained_config["dflash_config"]
    missing = [key for key in _REQUIRED_KEYS if config_dict.get(key) is None]
    if missing:
        raise ValueError(
            f"DFlash2 checkpoint is missing required config fields: {missing}. "
            "A checkpoint saved by speculators.models.dflash2 carries all of them."
        )
    for key in _REQUIRED_KEYS:
        draft_config[key] = config_dict[key]
    for key in _OPTIONAL_KEYS:
        if config_dict.get(key) is not None:
            draft_config[key] = config_dict[key]
    # Carried so the runtime can check the served block against the trained one.
    # Upstream's dflash updater drops block_size, which leaves vLLM unable to
    # tell that the convolution's block boundary moved.
    if config_dict.get("block_size") is not None:
        draft_config["block_size"] = config_dict["block_size"]

    if config_dict.get("sample_from_anchor"):
        raise ValueError(
            "DFlash2 requires sample_from_anchor=False: the convolution's block "
            "boundary at inference is the query block, 1 + num_speculative_tokens, "
            "which equals block_size only when the anchor is the bonus token."
        )
    _propagate_intra_block_causality(config_dict, pre_trained_config)


def _map_method(speculative_config: dict[str, Any]) -> dict[str, Any]:
    if speculative_config.get("method") != "dflash2":
        return speculative_config
    speculative_config = dict(speculative_config)
    speculative_config["method"] = "dflash"
    return speculative_config


def _install_config_patches() -> None:
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )
    from vllm.transformers_utils.configs.speculators.base import (  # noqa: PLC0415
        SpeculatorsConfig,
    )

    SUPPORTED_SPECULATORS_TYPES.setdefault("dflash2", _update_dflash2)

    if getattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, False):
        return

    original = SpeculatorsConfig.build_vllm_speculative_config.__func__

    @classmethod  # type: ignore[misc]
    def build_vllm_speculative_config(
        cls: type,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return _map_method(original(cls, config_dict))

    SpeculatorsConfig.build_vllm_speculative_config = build_vllm_speculative_config
    setattr(SpeculatorsConfig, _CONFIG_PATCH_MARKER, True)


def _is_dflash2_draft(speculative_config: Any) -> bool:
    """Whether this draft is a DFlash2 one, by the architecture #52816 keys on."""
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft_config = getattr(speculative_config, "draft_model_config", None)
    if draft_config is None:
        return False
    return _DFLASH2_ARCH in (draft_config.architectures or [])


def _check_block_size(speculative_config: Any) -> None:
    """Refuse a served block that is not the trained one.

    The convolution's taps and the selector's per-slot statistics are trained at
    ``block_size``; at inference the block is ``1 + num_speculative_tokens``, read
    from the serving config rather than the checkpoint. DFlash can legitimately
    draft fewer tokens than it was trained for, DFlash2 cannot.
    """
    draft_config = speculative_config.draft_model_config.hf_config.dflash_config
    block_size = draft_config.get("block_size")
    if block_size is None:
        return
    served = 1 + speculative_config.num_speculative_tokens
    if served != int(block_size):
        raise ValueError(
            f"DFlash2 was trained with block_size={block_size}, so it must be "
            f"served with num_speculative_tokens={int(block_size) - 1}; got "
            f"{speculative_config.num_speculative_tokens}, which convolves over "
            f"a block of {served}."
        )


def _check_v2_model_runner(vllm_config: Any) -> None:
    if not getattr(vllm_config, "use_v2_model_runner", True):
        raise ValueError(
            "DFlash2 requires the V2 model runner, the only one that runs its "
            "candidate selector; on V1 the same checkpoint drafts as DFlash1 "
            "without raising. Relaunch with VLLM_USE_V2_MODEL_RUNNER=1."
        )


def _install_speculator_patches() -> None:
    """Route a DFlash2 draft to the ported speculator."""
    from vllm.v1.worker.gpu import (
        model_runner,  # noqa: PLC0415
        spec_decode,  # noqa: PLC0415
    )

    if getattr(spec_decode.init_speculator, _SPECULATOR_PATCH_MARKER, False):
        return

    original = spec_decode.init_speculator

    def init_speculator(vllm_config: Any, device: Any) -> Any:
        speculative_config = vllm_config.speculative_config
        if _is_dflash2_draft(speculative_config):
            from speculators.vllm.dflash2_speculator import (  # noqa: PLC0415
                DFlash2Speculator,
            )

            _check_v2_model_runner(vllm_config)
            _check_block_size(speculative_config)
            return DFlash2Speculator(vllm_config, device)
        return original(vllm_config, device)

    setattr(init_speculator, _SPECULATOR_PATCH_MARKER, True)
    spec_decode.init_speculator = init_speculator
    # model_runner binds the name at import time, so the module attribute there
    # is a second, independent reference that has to be replaced as well.
    model_runner.init_speculator = init_speculator


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    _install_config_patches()
    if _DFLASH2_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(_DFLASH2_ARCH, _MODEL_IMPL)
    _install_speculator_patches()
    logger.info("Registered Speculators DFlash2 support for vLLM.")
