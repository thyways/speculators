from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "LatentScanSpeculatorConfig",
]


@SpeculatorModelConfig.register("latent_scan")
class LatentScanSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash backbone with a token-latent scan after every draft layer."""

    speculators_model_type: Literal["latent_scan"] = "latent_scan"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["LatentScanSpeculator"],
        description="Model architectures that can load these weights",
    )
    latent_dim: int = Field(
        default=256,
        gt=0,
        description="Width of the shared token-latent stream.",
    )
    latent_layer_scale_init: float = Field(
        default=1e-3,
        ge=0.0,
        description="Initial residual scale for every per-layer latent scan.",
    )
    strict_causal_slots: bool = Field(
        default=True,
        description="Force every draft layer's intra-block attention to be causal.",
    )

    @model_validator(mode="after")
    def _validate_latent_scan_layout(self) -> "LatentScanSpeculatorConfig":
        if self.sample_from_anchor:
            raise ValueError(
                "latent_scan requires sample_from_anchor=False: slot 0 carries the "
                "verified anchor and seeds the shifted token-latent stream."
            )
        if self.block_size < 2:  # noqa: PLR2004
            raise ValueError("latent_scan requires block_size >= 2.")
        return self
