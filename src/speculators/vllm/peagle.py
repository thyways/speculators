"""Normalize Speculators-format P-EAGLE checkpoints for vLLM inference.

vLLM already knows P-EAGLE: ``update_peagle`` translates the checkpoint onto
the EAGLE-3 architectures, and ``build_vllm_speculative_config`` maps
``speculators_model_type=peagle`` to ``method=eagle3`` with
``parallel_drafting=True``. No new model implementation is needed.

What is missing is the rotary normalization the DFlash family performs.
Parallel drafting on the EAGLE-3 runtime is only implemented by vLLM's V1
``EagleProposer``, and that proposer refuses any draft whose config still
advertises ``mrope_section`` (``_raise_if_mrope``). A drafter trained against an
M-RoPE verifier -- Qwen3.6-35B-A3B, for instance -- inherits the verifier's
``rope_parameters`` verbatim, so P-EAGLE checkpoints for those verifiers cannot
be served at all without this plugin.

Dropping ``mrope_section`` is semantics-preserving here. Training sets
``partial_rotary_factor=1.0`` and rescales ``mrope_section`` to cover the whole
head (``draft_mrope_full_head_hack``), and the drafter only ever sees text, so
the three M-RoPE position components are identical and M-RoPE reduces exactly to
1-D RoPE. This is the same normalization :mod:`speculators.vllm.dfly` and
:mod:`speculators.vllm.domino` apply to their own drafts.
"""

from __future__ import annotations

import logging
from typing import Any

from speculators.vllm._dflash_family import normalize_rope

logger = logging.getLogger(__name__)

_ALGORITHM = "peagle"
# Marks an updater this module already wrapped, so repeated register() calls
# (vLLM loads general plugins once per process, but processes fork) do not stack
# wrappers on top of each other.
_PATCH_MARKER = "_speculators_peagle_rope_normalized"


def _wrap_updater(
    native_updater: Any,
) -> Any:
    """Return the native P-EAGLE updater plus linear-rope normalization."""

    def _update_peagle(
        config_dict: dict[str, Any],
        pre_trained_config: dict[str, Any],
    ) -> None:
        native_updater(
            config_dict=config_dict,
            pre_trained_config=pre_trained_config,
        )
        normalize_rope(pre_trained_config)

    setattr(_update_peagle, _PATCH_MARKER, True)
    return _update_peagle


def register() -> None:
    """vLLM general-plugin entry point."""
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )

    native_updater = SUPPORTED_SPECULATORS_TYPES.get(_ALGORITHM)
    if native_updater is None:
        raise RuntimeError(
            "vLLM does not register a P-EAGLE speculators updater; this "
            "plugin expects vllm.transformers_utils.configs.speculators."
            "algos.update_peagle to exist."
        )
    if getattr(native_updater, _PATCH_MARKER, False):
        return

    SUPPORTED_SPECULATORS_TYPES[_ALGORITHM] = _wrap_updater(native_updater)
    logger.debug("Normalized P-EAGLE draft rope to linear positions.")
