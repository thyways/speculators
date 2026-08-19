from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlySpeculatorConfig",
]


@SpeculatorModelConfig.register("dfly")
class DFlySpeculatorConfig(DFlashSpeculatorConfig):
    """DFly configuration built on the DFlash block-parallel backbone."""

    speculators_model_type: Literal["dfly"] = "dfly"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlySpeculator"],
        description="Model architectures that can load these weights",
    )

    enable_hidden_correction: bool = Field(
        default=True,
        description=(
            "Apply the previous-token-conditioned hidden-state correction before "
            "the draft LM head."
        ),
    )
    hidden_correction_intermediate_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "SwiGLU width of the hidden-state correction. Defaults to the draft "
            "hidden size."
        ),
    )
