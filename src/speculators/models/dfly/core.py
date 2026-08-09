from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dfly.config import DFlySpeculatorConfig
from speculators.models.dfly.model_definitions import build_hidden_correction

__all__ = [
    "DFlyDraftModel",
]

_CONTEXT_TENSOR_NDIM = 3


@SpeculatorModel.register("dfly")
class DFlyDraftModel(DFlashDraftModel):
    """DFly: DFlash context plus per-layer fusion and hidden correction.

    Each draft layer consumes a normalized sum of the shared DFlash FC context
    and a learnable weighted mixture of the captured verifier hidden states. An
    optional previous-token-conditioned SwiGLU residual is applied immediately
    before the frozen draft LM head.
    """

    config_class: ClassVar[type[DFlySpeculatorConfig]] = DFlySpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DFlySpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = int(config.transformer_layer_config.hidden_size)
        target_hidden_size = config.target_hidden_size or hidden_size
        if target_hidden_size != hidden_size:
            raise ValueError(
                "DFly residual fusion requires target_hidden_size to match the "
                f"draft hidden size, got {target_hidden_size} and {hidden_size}."
            )

        self.num_target_layers = len(self.target_layer_ids)
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(len(self.layers), self.num_target_layers)
        )
        self._init_fusion_weights()
        self.hidden_correction = build_hidden_correction(config)

    def _init_fusion_weights(self) -> None:
        """Initialize a shallow-to-deep target-layer preference per draft layer."""
        num_draft_layers, num_target_layers = self.layer_fusion_weights.shape
        with torch.no_grad():
            target_depths = torch.arange(
                num_target_layers,
                dtype=self.layer_fusion_weights.dtype,
                device=self.layer_fusion_weights.device,
            )
            for layer_idx in range(num_draft_layers):
                preferred_depth = round(
                    layer_idx / max(num_draft_layers - 1, 1) * (num_target_layers - 1)
                )
                logits = -2.0 * (target_depths - preferred_depth).abs()
                self.layer_fusion_weights[layer_idx].copy_(logits)

    def _project_base_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep the shared DFlash FC output unnormalized until fusion is added."""
        return self.fc(hidden_states)

    def _build_layer_context(
        self,
        hidden_states: torch.Tensor,
        base_context: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        expected_size = self.num_target_layers * self.hidden_size
        if (
            hidden_states.ndim != _CONTEXT_TENSOR_NDIM
            or hidden_states.shape[-1] != expected_size
        ):
            raise ValueError(
                "DFly hidden_states must have shape [batch, sequence, "
                f"{expected_size}], got {tuple(hidden_states.shape)}."
            )

        context_features = hidden_states.reshape(
            *hidden_states.shape[:-1],
            self.num_target_layers,
            self.hidden_size,
        )
        fusion_probs = functional.softmax(
            self.layer_fusion_weights[layer_idx],
            dim=-1,
        ).to(base_context.dtype)
        residual_context = torch.einsum(
            "t,bstd->bsd",
            fusion_probs,
            context_features.to(base_context.dtype),
        )
        return self.hidden_norm(base_context + residual_context)

    def _previous_token_ids(
        self,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> torch.Tensor:
        block_tokens = input_ids[:, anchored_block_indices].reshape(
            input_ids.shape[0],
            -1,
            self.block_size,
        )
        if self.config.sample_from_anchor:
            previous = block_tokens
        else:
            previous = torch.cat(
                [block_tokens[:, :, :1], block_tokens[:, :, :-1]],
                dim=-1,
            )
        return previous.reshape(input_ids.shape[0], -1)

    def _compute_draft_logits(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        anchored_block_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_correction is not None:
            previous_ids = self._previous_token_ids(
                input_ids,
                anchored_block_indices,
            )
            previous_embeds = self.embed_tokens(previous_ids)
            hidden = self.hidden_correction(hidden, previous_embeds)
        return self.lm_head(hidden)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlyDraftModel":
        """Create a DFly model from the shared training arguments."""
        config = DFlySpeculatorConfig(
            **cls._build_base_config_kwargs(
                "dfly",
                verifier_config,
                **kwargs,
            ),
            enable_hidden_correction=kwargs.get(
                "enable_hidden_correction",
                True,
            ),
            hidden_correction_intermediate_size=kwargs.get(
                "hidden_correction_intermediate_size"
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model
