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

Like its siblings, this plugin registers into
:mod:`speculators.vllm._dflash_family` and lets that module patch each vLLM slot
once; see it for why sharing is mandatory rather than tidy. DFlash2 is the one
family member whose drafting loop is not DSpark's, so instead of an init hook it
registers a speculator factory.
"""

from __future__ import annotations

import logging
from typing import Any

from speculators.vllm._dflash_family import (
    install_config_patches,
    install_speculator_factory_patches,
    propagate_intra_block_causality,
    register_config_updater,
    register_speculative_method_alias,
    register_speculator_factory,
)

# Named into vLLM's logger hierarchy rather than this module's. vLLM configures a
# handler on the `vllm` logger and leaves the root one bare, so a
# `speculators.vllm.dflash2` logger writes nowhere -- and this module's one log
# line is the evidence that a DFlash2 checkpoint really drafted as DFlash2 rather
# than silently as DFlash1, which a caller reads out of the server log.
# ``logging.getLogger`` rather than vLLM's ``init_logger``: the name is what puts
# the record under vLLM's handler, and keeping vLLM out of this module's imports
# lets the config translation below be unit-tested without vLLM installed.
logger = logging.getLogger("vllm.plugins.speculators_dflash2")

_DFLASH2_ARCH = "DFlash2DraftModel"
_MODEL_IMPL = "speculators.vllm.dflash2_model:DFlash2Qwen3ForCausalLM"

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
    propagate_intra_block_causality(config_dict, pre_trained_config)


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


def _make_dflash2_speculator(vllm_config: Any, device: Any) -> Any | None:
    """Build the ported DFlash2 speculator, or decline a draft we do not own."""
    speculative_config = vllm_config.speculative_config
    if not _is_dflash2_draft(speculative_config):
        return None

    from speculators.vllm.dflash2_speculator import (  # noqa: PLC0415
        DFlash2Speculator,
    )

    _check_v2_model_runner(vllm_config)
    _check_block_size(speculative_config)
    return DFlash2Speculator(vllm_config, device)


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm import ModelRegistry  # noqa: PLC0415

    register_config_updater("dflash2", _update_dflash2)
    # base.py derives speculative_config.method from speculators_model_type,
    # while #52816 selects DFlash2 on method == "dflash" plus the architecture.
    register_speculative_method_alias("dflash2", "dflash")
    register_speculator_factory(_make_dflash2_speculator)
    install_config_patches()
    if _DFLASH2_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(_DFLASH2_ARCH, _MODEL_IMPL)
    install_speculator_factory_patches()
    logger.info("Registered Speculators DFlash2 support for vLLM.")
