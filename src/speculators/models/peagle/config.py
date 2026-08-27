from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.eagle3.config import Eagle3SpeculatorConfig

__all__ = [
    "PEagleSpeculatorConfig",
]


@SpeculatorModelConfig.register("peagle")
class PEagleSpeculatorConfig(Eagle3SpeculatorConfig):
    """
    Configuration for P-EAGLE (Parallel EAGLE) speculator.

    P-EAGLE extends EAGLE-3 with parallel multi-token prediction using
    Conditional Drop Token (COD) sampling for memory-efficient training.

    :param mask_token_id: Token ID used for masking
    """

    speculators_model_type: Literal["peagle"] = "peagle"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["PEagleSpeculator"],
        description="Model architectures that can load these weights",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for padding unused positions in parallel groups",
    )

    # Override Eagle3 default: P-EAGLE requires trainable embeddings
    # (matches p-eagle-train)
    embed_requires_grad: bool = Field(
        default=True,
        description=(
            "Whether embedding layer weights require gradients during "
            "training (True for P-EAGLE)"
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a P-EAGLE checkpoint for native vLLM serving.

        P-EAGLE drafts text with one-dimensional position IDs.  A draft built
        from a multimodal verifier can nevertheless inherit ``mrope_section``
        metadata, which makes vLLM's V1 proposer reject parallel drafting even
        though the training-side Llama/Qwen3 rotary layer uses ordinary 1-D
        RoPE.  Remove only that stale metadata from the serialized nested draft
        config so vLLM 0.28's native Speculators translator can load it without
        a plugin.

        All Speculators fields remain intact, so checkpoint resumption still
        resolves :class:`PEagleSpeculatorConfig` normally.
        """
        config_dict = super().to_dict()
        transformer_config = deepcopy(config_dict["transformer_layer_config"])

        for rope_key in ("rope_parameters", "rope_scaling"):
            rope_config = transformer_config.get(rope_key)
            if not isinstance(rope_config, dict):
                continue
            rope_config = dict(rope_config)
            rope_config.pop("mrope_section", None)
            rope_config.pop("mrope_interleaved", None)
            transformer_config[rope_key] = rope_config

        config_dict["transformer_layer_config"] = transformer_config
        return config_dict
