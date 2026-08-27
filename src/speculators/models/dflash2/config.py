from copy import deepcopy
from typing import Any, Literal

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize one checkpoint for both training and native vLLM serving.

        vLLM 0.28 ships the DFlash2 model and proposal runtime, but its
        Speculators-format config translator does not recognize
        ``speculators_model_type=dflash2``.  Saving the draft transformer's fields
        at the top level makes the checkpoint a normal Qwen3 config to vLLM, while
        retaining the nested Speculators fields keeps ``from_pretrained`` and
        checkpoint resumption lossless on the training side.

        The serving command must pass this checkpoint explicitly as
        ``--spec-model`` with ``--spec-method dflash``.  The top-level
        ``model_type`` deliberately prevents vLLM from routing draft-config
        loading through its incomplete Speculators translator.
        """
        config_dict = super().to_dict()
        transformer_config = deepcopy(config_dict["transformer_layer_config"])

        # Start from the draft transformer's native HF config, then keep the
        # Speculators fields as extra metadata for training-side round trips.
        native_config = {**transformer_config, **config_dict}
        native_config["model_type"] = transformer_config.get(
            "model_type", self.transformer_layer_config.model_type
        )
        native_config["architectures"] = ["DFlash2DraftModel"]
        native_config.pop("auto_map", None)

        aux_layer_ids = self.aux_hidden_state_layer_ids
        if aux_layer_ids is not None:
            native_config["eagle_aux_hidden_state_layer_ids"] = list(aux_layer_ids)

        dflash_config: dict[str, Any] = {
            "sample_from_anchor": self.sample_from_anchor,
            "block_size": self.block_size,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_group_size": self.conv_group_size,
            "selector_rank": self.selector_rank,
            "selector_top_k": self.selector_top_k,
            "input_embedding_scale": self.input_embedding_scale,
            "output_multiplier": self.output_multiplier,
        }
        if self.mask_token_id is not None:
            dflash_config["mask_token_id"] = self.mask_token_id
        if aux_layer_ids is not None:
            # vLLM's DFlash family indexes target outputs one layer earlier than
            # Speculators' extraction-side IDs.
            dflash_config["target_layer_ids"] = [i - 1 for i in aux_layer_ids]
        if self.final_logit_softcapping is not None:
            dflash_config["final_logit_softcapping"] = self.final_logit_softcapping
        if self.sliding_window_non_causal:
            # A global False matches training for both sliding and full-attention
            # draft layers.  When False, omit the override so vLLM applies its
            # per-layer default instead of forcing full-attention layers causal.
            dflash_config["causal"] = False
        native_config["dflash_config"] = dflash_config

        # The verifier may be multimodal and carry M-RoPE metadata, whereas this
        # draft is a text-only Qwen3 stack with linear position IDs.
        rope_parameters = native_config.get("rope_parameters")
        if isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)
            native_config["rope_parameters"] = rope_parameters

        return native_config
