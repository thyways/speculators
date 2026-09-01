"""Configuration for the HashGram candidate selector."""

from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = ["HashGramSpeculatorConfig"]


@SpeculatorModelConfig.register("hashgram")
class HashGramSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash plus low-rank recall and hashed vector n-gram reranking."""

    speculators_model_type: Literal["hashgram"] = "hashgram"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["HashGramDraftModel"],
        description="Model architectures that can load these weights",
    )

    hashgram_rank: int = Field(
        default=128,
        ge=1,
        description="Dimension of each hashed bigram/trigram vector.",
    )
    hashgram_top_k: int = Field(
        default=16,
        ge=1,
        description="Number of unary candidates reranked by HashGram.",
    )
    hashgram_bigram_buckets: int = Field(
        default=1_048_576,
        ge=1,
        description="Number of buckets in the hashed bigram table.",
    )
    hashgram_trigram_buckets: int = Field(
        default=1_048_576,
        ge=1,
        description="Number of buckets in the hashed trigram table.",
    )
    hashgram_num_hashes: int = Field(
        default=1,
        ge=1,
        description="Independent hash probes averaged for each n-gram.",
    )
    hashgram_loss_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the HashGram candidate-selector loss.",
    )
    hashgram_markov_rank: int = Field(
        default=256,
        ge=0,
        description=(
            "Rank of the DSpark-style full-vocabulary recall bias; 0 disables it."
        ),
    )
    hashgram_use_markov_recall: bool = Field(
        default=True,
        description="Apply the low-rank previous-token bias before Top-K recall.",
    )
    hashgram_hidden_refine: bool = Field(
        default=False,
        description=(
            "Use candidate-specific vector-to-hidden refinement before gathering "
            "candidate LM-head rows. Disabled by default to keep training memory low."
        ),
    )
    hashgram_use_bigram: bool = Field(
        default=True,
        description="Enable the hashed bigram table.",
    )
    hashgram_use_trigram: bool = Field(
        default=True,
        description="Enable the hashed trigram table and hidden-state gate.",
    )

    @model_validator(mode="after")
    def _at_least_one_ngram_order(self) -> "HashGramSpeculatorConfig":
        if not self.hashgram_use_bigram and not self.hashgram_use_trigram:
            raise ValueError(
                "At least one of hashgram_use_bigram/hashgram_use_trigram "
                "must be enabled."
            )
        return self
