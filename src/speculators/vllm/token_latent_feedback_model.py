# SPDX-License-Identifier: Apache-2.0
"""vLLM model for block-parallel token-latent feedback checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from speculators.models.token_latent_feedback.model_definitions import (
    TokenLatentFeedbackHead,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_HIDDEN_RANK = 2


class Qwen3TokenLatentFeedbackModel(DFlashQwen3Model):
    """DFlash backbone followed by one parallel latent-feedback head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        config = self.config
        self.block_size = int(getattr(config, "block_size", 8))
        latent_dim = getattr(config, "token_latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(config, "latent_dim", None)
        self.latent_dim = 128 if latent_dim is None else int(latent_dim)
        self.token_latent_head = TokenLatentFeedbackHead(
            hidden_size=int(config.hidden_size),
            latent_dim=self.latent_dim,
            block_size=self.block_size,
            rms_norm_eps=float(config.rms_norm_eps),
            initializer_range=float(getattr(config, "initializer_range", None) or 0.02),
            prefix_mixer_mode=str(
                getattr(config, "prefix_mixer", None)
                or getattr(config, "feedback_mode", None)
                or getattr(config, "prefix_mixer_mode", "full")
            ),
            use_reliability_gate=bool(
                getattr(config, "reliability_gate", None)
                if getattr(config, "reliability_gate", None) is not None
                else getattr(config, "use_reliability_gate", True)
            ),
            strict_causal_prefix=bool(getattr(config, "strict_causal_prefix", True)),
            position_scale_init=float(getattr(config, "position_scale_init", 1.0)),
            position_scale_parameterization=str(
                getattr(config, "position_scale_parameterization", "direct")
            ),
            position_scale_min=float(getattr(config, "position_scale_min", 0.0)),
            feedback_output_projection_init=float(
                getattr(config, "feedback_output_projection_init", 0.0)
            ),
            feedback_output_projection_init_mode=str(
                getattr(
                    config,
                    "feedback_output_projection_init_mode",
                    "constant",
                )
            ),
        ).to(dtype=vllm_config.model_config.dtype)
        # Training-only target projection is not part of this module; the
        # checkpoint loader below drops it before vLLM sees the weight stream.
        self.token_latent_head.requires_grad_(False)

    def _reshape_blocks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != _HIDDEN_RANK:
            raise ValueError(
                "token_latent_feedback expects [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] % self.block_size:
            raise ValueError(
                "token_latent_feedback token count must be divisible by block_size: "
                f"{hidden_states.shape[0]} % {self.block_size} != 0"
            )
        return hidden_states.reshape(-1, self.block_size, hidden_states.shape[-1])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        # The vLLM DFlash layer keeps the residual deferred until the final norm.
        # Materialize the pre-norm tensor so the feedback stage sits at the same
        # boundary as the training model; apply the final norm only afterward.
        if residual is not None:
            hidden_states = hidden_states + residual
        blocks = self._reshape_blocks(hidden_states)
        corrected = self.token_latent_head(blocks).hidden_states
        return self.norm(corrected.reshape_as(hidden_states))


class Qwen3TokenLatentFeedbackForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level vLLM draft with a single final full-vocabulary LM head."""

    model_cls = Qwen3TokenLatentFeedbackModel

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if "_target_code_projection" not in name
        )


__all__ = [
    "Qwen3TokenLatentFeedbackForCausalLM",
    "Qwen3TokenLatentFeedbackModel",
]
