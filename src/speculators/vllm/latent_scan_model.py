# SPDX-License-Identifier: Apache-2.0
"""vLLM model implementation for all-layer LatentScan checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from speculators.models.latent_scan.model_definitions import (
    LatentCausalScan,
    LatentRMSNorm,
)

_VLLM_HIDDEN_RANK = 2


class Qwen3LatentScanDecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer followed by the checkpoint-compatible latent scan."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.latent_scan = LatentCausalScan(
            hidden_size=int(config.hidden_size),
            latent_dim=int(getattr(config, "latent_dim", 256)),
            block_size=int(getattr(config, "block_size", 8)),
            rms_norm_eps=float(config.rms_norm_eps),
            layer_scale_init=float(getattr(config, "latent_layer_scale_init", 1e-3)),
            initializer_range=float(getattr(config, "initializer_range", None) or 0.02),
        ).to(dtype=vllm_config.model_config.dtype)


class Qwen3LatentScanModel(DFlashQwen3Model):
    """DFlash backbone carrying one shared token-latent stream through all layers."""

    decoder_layer_cls = Qwen3LatentScanDecoderLayer

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
        self.block_size = int(getattr(self.config, "block_size", 8))
        self.latent_dim = int(getattr(self.config, "latent_dim", 256))
        dtype = vllm_config.model_config.dtype
        self.token_latent_norm = LatentRMSNorm(
            int(self.config.hidden_size),
            float(self.config.rms_norm_eps),
        ).to(dtype=dtype)
        self.token_latent_projection = nn.Linear(
            int(self.config.hidden_size),
            self.latent_dim,
            bias=False,
            dtype=dtype,
        )
        nn.init.normal_(
            self.token_latent_projection.weight,
            mean=0.0,
            std=float(getattr(self.config, "initializer_range", None) or 0.02),
        )

    def _reshape_blocks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != _VLLM_HIDDEN_RANK:
            raise ValueError(
                "LatentScan vLLM forward expects [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[0] % self.block_size:
            raise ValueError(
                "LatentScan query token count must be divisible by block_size: "
                f"{hidden_states.shape[0]} % {self.block_size} != 0."
            )
        return hidden_states.reshape(-1, self.block_size, hidden_states.shape[-1])

    def _predict_token_latents(self, hidden_states: torch.Tensor) -> torch.Tensor:
        blocks = self._reshape_blocks(hidden_states)
        return self.token_latent_projection(self.token_latent_norm(blocks))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds
        latent_stream = self._predict_token_latents(hidden_states)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            # vLLM defers the decoder residual add to the next layer. LatentScan
            # lives at the true layer boundary, so materialize that add first and
            # restart the deferred-residual chain after applying the correction.
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_blocks = self._reshape_blocks(hidden_states)
            current_latents = self._predict_token_latents(hidden_states)
            hidden_blocks, latent_stream = layer.latent_scan(
                hidden_blocks,
                latent_stream,
                current_latents,
            )
            hidden_states = hidden_blocks.reshape_as(hidden_states)
            residual = None

        return self.norm(hidden_states)


class Qwen3LatentScanForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level LatentScan draft with DFlash's single final LM head."""

    model_cls = Qwen3LatentScanModel

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Drop the training-only latent target before vLLM weight loading.

        The target projection is persisted so resumed training sees the same
        global-seed initialization.  It is never used by inference, and keeping
        it out of vLLM avoids treating the extra training buffer as a draft
        parameter.
        """
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if "_token_target_projection" not in name
        )


__all__ = [
    "Qwen3LatentScanDecoderLayer",
    "Qwen3LatentScanForCausalLM",
    "Qwen3LatentScanModel",
]
