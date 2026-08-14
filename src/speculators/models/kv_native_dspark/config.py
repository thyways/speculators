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
        default_factory=lambda: [3, 11, 19, 27, 31, 39],
        description="Verifier full-attention layers exported by the vLLM connector.",
    )
    verifier_kv_layer_mapping: list[int] = Field(
        default_factory=lambda: [3, 11, 19, 27, 31, 39],
        description=(
            "Direct-read mode only: one verifier K/V layer selected for each "
            "draft layer. The learned bridge consumes every exported layer."
        ),
    )
    verifier_num_key_value_heads: int = Field(default=2, gt=0)
    verifier_head_dim: int = Field(default=256, gt=0)
    verifier_partial_rotary_factor: float = Field(default=0.25, gt=0.0, le=1.0)
    verifier_rope_theta: float = Field(default=10_000_000.0, gt=0.0)
    verifier_mrope_section: list[int] = Field(default_factory=lambda: [11, 11, 10])
    kv_bridge_enabled: bool = Field(
        default=False,
        description=(
            "Fuse every exported verifier K/V layer into a draft-consumable cache "
            "instead of moving draft Q/local K/V into verifier space."
        ),
    )
    kv_bridge_rank: int = Field(
        default=32,
        gt=0,
        description=(
            "Width of each per-source low-rank projection in the learned all-layer "
            "KV fusion bridge."
        ),
    )
    kv_bridge_residual_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Fixed multiplier on the learned low-rank correction added to the "
            "softmax-fused verifier K/V base."
        ),
    )
    kv_bridge_max_correction_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional per-token RMS cap on the learned correction relative to "
            "the softmax-fused base. Unset preserves the original unbounded bridge."
        ),
    )
    kv_bridge_normalize_keys: bool = Field(
        default=False,
        description=(
            "Apply parameter-free per-token RMS normalization to mapped content "
            "Keys before verifier MRoPE is restored."
        ),
    )
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
        # The direct-read mapping-length invariant is therefore enforced where both
        # sides are known to be real: ``TrainConfig`` for the CLI and
        # ``KVNativeDSparkDraftModel`` for the model. The learned bridge has no
        # manual source mapping; it always consumes every exported verifier layer.
        if not self.verifier_kv_layer_ids:
            raise ValueError("verifier_kv_layer_ids must not be empty")
        if len(self.verifier_kv_layer_ids) != len(set(self.verifier_kv_layer_ids)):
            raise ValueError("verifier_kv_layer_ids must not contain duplicates")
        if not self.kv_bridge_enabled:
            unknown = sorted(
                set(self.verifier_kv_layer_mapping) - set(self.verifier_kv_layer_ids)
            )
            if unknown:
                raise ValueError(
                    "verifier_kv_layer_mapping references non-exported layers: "
                    f"{unknown}"
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
