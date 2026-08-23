from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlash2SpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash2")
class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus the local convolution and the candidate selector.

    Both additions come from vllm-project/vllm#52816, which carries them on a
    separate ``DFlash2DraftModel`` architecture so existing DFlash checkpoints are
    untouched. The convolution lets a proposal position see the ones before it
    without another backbone pass; the selector conditions each slot's
    distribution on the token that actually landed in the previous slot. All
    DFlash fields are inherited unchanged.

    The field names below are the ``dflash_config`` keys the inference side reads,
    so they are the contract between the two halves.
    """

    speculators_model_type: Literal["dflash2"] = "dflash2"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlash2Speculator"],
        description="Model architectures that can load these weights",
    )

    # Grouped dynamic depthwise convolution.
    conv_kernel_size: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of convolution taps per sublayer. Tap t reaches back t draft "
            "positions; taps are zero across the block boundary, so this must not "
            "exceed block_size."
        ),
    )
    conv_group_size: int = Field(
        default=64,
        ge=1,
        description=(
            "Channels sharing one dynamic convolution coefficient. Must divide "
            "hidden_size. Smaller means more input-dependent taps and a wider "
            "kernel projection."
        ),
    )

    # Candidate selector.
    selector_rank: int = Field(
        default=256,
        ge=1,
        description=(
            "Rank of the predecessor/successor codebooks scoring adjacent draft "
            "transitions."
        ),
    )
    selector_top_k: int = Field(
        default=16,
        ge=2,
        description=(
            "Candidates kept per slot from the target head at inference. The "
            "selector loss and the path-walk diagnostics score the same K, so "
            "changing it changes what training optimizes."
        ),
    )

    # Optional scalars the inference side applies; the defaults are no-ops.
    input_embedding_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Multiplier applied to the draft's input embeddings.",
    )
    output_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Multiplier applied to the draft logits before the selector bias is added."
        ),
    )
    final_logit_softcapping: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "When set, softcap the multiplied draft logits as tanh(x / cap) * cap "
            "before the selector bias is added. None disables it; a non-positive "
            "cap is rejected rather than silently treated as disabled, which is "
            "what the inference side would do with it."
        ),
    )
