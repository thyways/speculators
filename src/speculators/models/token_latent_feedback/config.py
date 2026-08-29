"""Configuration for the block-parallel token-latent feedback drafter."""

from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DEFAULT_LATENT_DIM",
    "DEFAULT_LATENT_LOSS_ALPHA",
    "DEFAULT_POSITION_SCALE_INIT",
    "LatentFeedbackSpeculatorConfig",
    "ParallelTokenLatentFeedbackSpeculatorConfig",
    "ParallelTokenLatentSpeculatorConfig",
    "TokenLatentFeedbackSpeculatorConfig",
]

DEFAULT_LATENT_DIM = 128
DEFAULT_LATENT_LOSS_ALPHA = 0.1
DEFAULT_POSITION_SCALE_INIT = 1.0


class _TokenLatentFeedbackConfig(DFlashSpeculatorConfig):
    """Shared fields for the v1.2 naming aliases."""

    architectures: list[str] = Field(
        default_factory=lambda: ["TokenLatentFeedbackDraftModel"],
        description="Model architectures that can load these weights.",
    )
    sample_from_anchor: bool = Field(
        default=False,
        description="Keep slot zero as the verified anchor; draft slots start at one.",
    )
    latent_dim: int = Field(
        default=DEFAULT_LATENT_DIM,
        gt=0,
        description="Width of the token-intent latent used for prefix feedback.",
    )
    # ``token_latent_dim`` is accepted as a readable alias for experiment files.
    token_latent_dim: int | None = Field(
        default=None,
        gt=0,
        description="Optional alias for latent_dim.",
    )
    feedback_stages: int = Field(
        default=1,
        ge=1,
        description="Number of parallel latent-feedback stages (v1.2 uses one).",
    )
    latent_feedback_stages: int | None = Field(
        default=None,
        ge=1,
        description="Optional alias for feedback_stages.",
    )
    prefix_mixer_mode: Literal["full", "shifted", "none"] = Field(
        default="full",
        description=(
            "Causal mixer ablation: full prefix, one-position shifted prefix, "
            "or no feedback."
        ),
    )
    # Friendly alias used by early experiment scripts.
    feedback_mode: Literal["full", "shifted", "none"] | None = Field(
        default=None,
        description="Optional alias for prefix_mixer_mode.",
    )
    prefix_mixer: Literal["full", "shifted", "none"] | None = Field(
        default=None,
        description="Optional alias for prefix_mixer_mode.",
    )
    prefix_mixer_parameterization: Literal["toeplitz"] = Field(
        default="toeplitz",
        description="Parameterization of the causal mixer matrix.",
    )
    use_reliability_gate: bool = Field(
        default=True,
        description="Gate each source latent with a learned scalar reliability.",
    )
    reliability_gate: bool | None = Field(
        default=None,
        description="Optional alias for use_reliability_gate.",
    )
    strict_causal_prefix: bool = Field(
        default=True,
        description="Use only strictly earlier slots in the prefix mixer.",
    )
    feedback_output_projection_init: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute initialization value for the hidden feedback projection.",
    )
    position_scale_init: float = Field(
        default=DEFAULT_POSITION_SCALE_INIT,
        ge=0.0,
        description="Initial learned per-slot feedback scale.",
    )
    latent_loss_alpha: float = Field(
        default=DEFAULT_LATENT_LOSS_ALPHA,
        ge=0.0,
        description="Weight of the token-latent cosine auxiliary loss.",
    )
    latent_loss_weight: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional alias for latent_loss_alpha.",
    )

    @property
    def resolved_latent_dim(self) -> int:
        """Return the dimension after applying the optional alias."""
        return int(self.token_latent_dim or self.latent_dim)

    @property
    def resolved_feedback_stages(self) -> int:
        return int(self.latent_feedback_stages or self.feedback_stages)

    @property
    def resolved_prefix_mixer_mode(self) -> str:
        """Return the canonical mixer mode after applying the optional alias."""
        return self.prefix_mixer or self.feedback_mode or self.prefix_mixer_mode

    @property
    def resolved_reliability_gate(self) -> bool:
        return bool(
            self.reliability_gate
            if self.reliability_gate is not None
            else self.use_reliability_gate
        )

    @property
    def resolved_latent_loss_alpha(self) -> float:
        """Return the canonical auxiliary-loss weight."""
        return float(
            self.latent_loss_weight
            if self.latent_loss_weight is not None
            else self.latent_loss_alpha
        )

    @model_validator(mode="after")
    def _validate_layout(self):
        if self.sample_from_anchor:
            raise ValueError(
                "token_latent_feedback requires sample_from_anchor=False: "
                "slot zero is the verified anchor."
            )
        if self.block_size < 2:  # noqa: PLR2004
            raise ValueError("token_latent_feedback requires block_size >= 2.")
        if self.resolved_feedback_stages != 1:
            raise ValueError(
                "token_latent_feedback currently implements exactly one "
                f"feedback stage, got {self.resolved_feedback_stages}."
            )
        if self.token_latent_dim is not None and self.latent_dim not in {
            DEFAULT_LATENT_DIM,
            self.token_latent_dim,
        }:
            raise ValueError(
                "latent_dim and token_latent_dim disagree: "
                f"{self.latent_dim} != {self.token_latent_dim}."
            )
        if self.feedback_mode is not None and self.prefix_mixer_mode not in {
            "full",
            self.feedback_mode,
        }:
            raise ValueError(
                "prefix_mixer_mode and feedback_mode disagree: "
                f"{self.prefix_mixer_mode!r} != {self.feedback_mode!r}."
            )
        return self


@SpeculatorModelConfig.register("token_latent_feedback")
class TokenLatentFeedbackSpeculatorConfig(_TokenLatentFeedbackConfig):
    """DFlash plus one block-parallel causal token-latent feedback stage."""

    speculators_model_type: Literal["token_latent_feedback"] = "token_latent_feedback"  # type: ignore[assignment]


@SpeculatorModelConfig.register("parallel_token_latent")
class ParallelTokenLatentSpeculatorConfig(_TokenLatentFeedbackConfig):
    """Compatibility alias for the ParallelTokenLatent experiment name."""

    speculators_model_type: Literal["parallel_token_latent"] = "parallel_token_latent"  # type: ignore[assignment]


@SpeculatorModelConfig.register("latent_feedback")
class LatentFeedbackSpeculatorConfig(_TokenLatentFeedbackConfig):
    """Short compatibility alias for the feedback experiment."""

    speculators_model_type: Literal["latent_feedback"] = "latent_feedback"  # type: ignore[assignment]


@SpeculatorModelConfig.register("parallel_token_latent_feedback")
class ParallelTokenLatentFeedbackSpeculatorConfig(_TokenLatentFeedbackConfig):
    """Explicit long-form alias for the Parallel Token-Latent method."""

    speculators_model_type: Literal["parallel_token_latent_feedback"] = (
        "parallel_token_latent_feedback"  # type: ignore[assignment]
    )
