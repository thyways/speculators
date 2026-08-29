from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DEFAULT_CONDITIONAL_LOSS_WEIGHT",
    "DEFAULT_HIDDEN_CANDIDATES",
    "DEFAULT_LOGIT_SCALE_INIT",
    "DEFAULT_RETRIEVAL_LOSS_WEIGHT",
    "DEFAULT_SSM_STATE_DIM",
    "DEFAULT_TOKEN_CODE_DIM",
    "DEFAULT_TRAINING_NEGATIVES",
    "DEFAULT_TRANSITION_CANDIDATES",
    "TokenLatentSSMSpeculatorConfig",
]

DEFAULT_TOKEN_CODE_DIM = 128
DEFAULT_SSM_STATE_DIM = 256
DEFAULT_HIDDEN_CANDIDATES = 32
DEFAULT_TRANSITION_CANDIDATES = 32
DEFAULT_TRAINING_NEGATIVES = 128
DEFAULT_RETRIEVAL_LOSS_WEIGHT = 0.5
DEFAULT_CONDITIONAL_LOSS_WEIGHT = 1.0
DEFAULT_LOGIT_SCALE_INIT = 1.0 / 0.07


@SpeculatorModelConfig.register("token_latent_ssm")
class TokenLatentSSMSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash feature extractor with a token-conditioned candidate decoder."""

    speculators_model_type: Literal["token_latent_ssm"] = "token_latent_ssm"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["TokenLatentSSMSpeculator"],
        description="Model architectures that can load these weights",
    )
    sample_from_anchor: bool = Field(
        default=False,
        description="Slot zero is a verified anchor and is not sampled.",
    )
    sliding_window_non_causal: bool = Field(
        default=True,
        description="Use the DFlash stack as a bidirectional block feature extractor.",
    )
    token_code_dim: int = Field(
        default=DEFAULT_TOKEN_CODE_DIM,
        gt=0,
        description="Width of the trainable target-vocabulary token codebook.",
    )
    ssm_state_dim: int = Field(
        default=DEFAULT_SSM_STATE_DIM,
        gt=0,
        description="Width of the diagonal token-conditioned SSM state.",
    )
    hidden_candidate_count: int = Field(
        default=DEFAULT_HIDDEN_CANDIDATES,
        gt=0,
        description="Candidates retrieved from each parallel DFlash hidden state.",
    )
    transition_candidate_count: int = Field(
        default=DEFAULT_TRANSITION_CANDIDATES,
        gt=0,
        description="Candidates retrieved from the actual previous token.",
    )
    training_negative_count: int = Field(
        default=DEFAULT_TRAINING_NEGATIVES,
        gt=0,
        description="Sampled negatives per position during training.",
    )
    retrieval_loss_weight: float = Field(
        default=DEFAULT_RETRIEVAL_LOSS_WEIGHT,
        ge=0.0,
        description="Weight of the hidden and transition retrieval objective.",
    )
    conditional_loss_weight: float = Field(
        default=DEFAULT_CONDITIONAL_LOSS_WEIGHT,
        ge=0.0,
        description="Weight of the token-conditioned candidate CE objective.",
    )
    token_latent_logit_scale_init: float = Field(
        default=DEFAULT_LOGIT_SCALE_INIT,
        gt=0.0,
        description="Initial cosine-logit scale for retrieval and candidate scoring.",
    )

    @model_validator(mode="after")
    def _validate_layout(self) -> "TokenLatentSSMSpeculatorConfig":
        if self.sample_from_anchor:
            raise ValueError("token_latent_ssm requires sample_from_anchor=False.")
        if self.block_size < 2:  # noqa: PLR2004
            raise ValueError("token_latent_ssm requires block_size >= 2.")
        hidden_size = int(self.transformer_layer_config.hidden_size)
        if self.token_code_dim > hidden_size:
            raise ValueError(
                "token_code_dim must not exceed the draft hidden size: "
                f"{self.token_code_dim} > {hidden_size}."
            )
        return self
