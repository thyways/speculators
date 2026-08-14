from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig

__all__ = ["KVNativeDSparkSpeculatorConfig"]

_MROPE_COORDINATES = 3
_ModelType = Literal["kv_native_dspark"]


@SpeculatorModelConfig.register("kv_native_dspark")
class KVNativeDSparkSpeculatorConfig(DSparkSpeculatorConfig):
    """DSpark trained to query real verifier K/V as external prefix memory."""

    speculators_model_type: _ModelType = "kv_native_dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["KVNativeDSparkDraftModel"],
        description="Model architectures that can load these weights",
    )
    aux_hidden_state_layer_ids: list[int] = Field(
        default_factory=list,
        description=(
            "KV-native training does not consume auxiliary verifier hidden states; "
            "only the verifier's final hidden state is exported as the logit target."
        ),
    )
    verifier_kv_layer_ids: list[int] = Field(
        default_factory=lambda: [3, 11, 19, 27, 35, 39],
        description="Verifier full-attention layers exported by the vLLM connector.",
    )
    verifier_kv_layer_mapping: list[int] = Field(
        default_factory=lambda: [3, 11, 19, 27, 35, 39],
        description=(
            "One verifier K/V layer selected for each draft layer. Must be a "
            "subset of verifier_kv_layer_ids."
        ),
    )
    verifier_num_key_value_heads: int = Field(default=2, gt=0)
    verifier_head_dim: int = Field(default=256, gt=0)
    verifier_partial_rotary_factor: float = Field(default=0.25, gt=0.0, le=1.0)
    verifier_rope_theta: float = Field(default=10_000_000.0, gt=0.0)
    verifier_mrope_section: list[int] = Field(default_factory=lambda: [11, 11, 10])
    num_speculative_tokens: int = Field(
        default=7,
        gt=0,
        description=(
            "Number of proposal slots used for speculative decoding and formal "
            "evaluation. This is independent of the trained block_size."
        ),
    )

    @model_validator(mode="after")
    def _validate_verifier_kv(self) -> "KVNativeDSparkSpeculatorConfig":
        # This validator must tolerate ``KVNativeDSparkSpeculatorConfig()`` with
        # no arguments: ``save_pretrained`` default-constructs the config class to
        # diff generation parameters, so anything that cross-checks two fields
        # whose defaults are independent would make every checkpoint save fail.
        # The mapping-length invariant is therefore enforced where both sides are
        # known to be real: ``TrainConfig`` for the CLI and
        # ``KVNativeDSparkDraftModel`` for the model.
        if not self.verifier_kv_layer_ids:
            raise ValueError("verifier_kv_layer_ids must not be empty")
        if len(self.verifier_kv_layer_ids) != len(set(self.verifier_kv_layer_ids)):
            raise ValueError("verifier_kv_layer_ids must not contain duplicates")
        unknown = sorted(
            set(self.verifier_kv_layer_mapping) - set(self.verifier_kv_layer_ids)
        )
        if unknown:
            raise ValueError(
                f"verifier_kv_layer_mapping references non-exported layers: {unknown}"
            )
        rotary_dim = int(self.verifier_head_dim * self.verifier_partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                "verifier partial rotary dimension must be positive and even"
            )
        if len(self.verifier_mrope_section) != _MROPE_COORDINATES:
            raise ValueError("verifier_mrope_section must contain exactly 3 values")
        if sum(self.verifier_mrope_section) != rotary_dim // 2:
            raise ValueError(
                "verifier_mrope_section must sum to half the partial rotary dimension"
            )
        trained_tokens = (
            self.block_size if self.sample_from_anchor else self.block_size - 1
        )
        if self.num_speculative_tokens > trained_tokens:
            raise ValueError(
                "num_speculative_tokens exceeds the proposal slots represented by "
                f"this checkpoint: {self.num_speculative_tokens} > {trained_tokens}"
            )
        return self
