from typing import Literal

from pydantic import Field, model_validator

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = ["KVNativeDFlashSpeculatorConfig"]

_ModelType = Literal["kv_native_dflash"]


@SpeculatorModelConfig.register("kv_native_dflash")
class KVNativeDFlashSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash trained to query real verifier K/V as external prefix memory."""

    speculators_model_type: _ModelType = "kv_native_dflash"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["KVNativeDFlashDraftModel"],
        description="Model architectures that can load these weights",
    )
    aux_hidden_state_layer_ids: list[int] = Field(
        default_factory=list,
        description=(
            "KV-native training does not consume auxiliary verifier hidden states; "
            "only the verifier's final hidden state is exported as the logit target."
        ),
    )
    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "DFlash treats the anchor as the bonus token and trains only the "
            "remaining block_size-1 proposal positions."
        ),
    )
    block_size: int = Field(
        default=16,
        gt=1,
        description=(
            "Size of each anchored DFlash block; one anchor plus block_size-1 "
            "trained proposal positions."
        ),
    )
    verifier_kv_layer_ids: list[int] = Field(
        default_factory=lambda: [3, 11, 19, 27, 35],
        description="Verifier full-attention layers exported by the vLLM connector.",
    )
    verifier_kv_layer_mapping: list[int] = Field(
        default_factory=lambda: [3, 11, 19, 27, 35],
        description=("One depth-matched raw verifier K/V source per draft layer."),
    )
    verifier_num_key_value_heads: int = Field(default=2, gt=0)
    verifier_head_dim: int = Field(default=256, gt=0)
    verifier_partial_rotary_factor: float = Field(default=0.25, gt=0.0, le=1.0)
    verifier_rope_theta: float = Field(default=10_000_000.0, gt=0.0)
    kv_native_architecture: Literal["dual_stream_raw_kv"] = Field(
        default="dual_stream_raw_kv",
        description=(
            "Final KV-native architecture: independent local/raw-prefix softmaxes "
            "with identity-initialized query/output adapters."
        ),
    )
    anchor_hidden_injection: bool = Field(
        default=False,
        description=(
            "Add the verifier's final hidden state at the last verified position "
            "into the anchor slot's draft input. Training-only: the vLLM runtime "
            "does not pass that state to the draft, so a checkpoint trained with "
            "this enabled cannot be served."
        ),
    )
    num_speculative_tokens: int = Field(
        default=15,
        gt=0,
        description=(
            "Number of proposal slots used for speculative decoding and formal "
            "evaluation. This must fit within block_size-1 for DFlash."
        ),
    )

    @model_validator(mode="after")
    def _validate_verifier_kv(self) -> "KVNativeDFlashSpeculatorConfig":
        if self.sample_from_anchor:
            raise ValueError(
                "KV-native DFlash requires sample_from_anchor=false so the anchor "
                "remains the verifier bonus token"
            )
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
        trained_tokens = self.block_size - 1
        if self.num_speculative_tokens != trained_tokens:
            raise ValueError(
                "num_speculative_tokens must equal the complete proposal block "
                f"represented by this checkpoint: {self.num_speculative_tokens} "
                f"!= {trained_tokens}"
            )
        return self
