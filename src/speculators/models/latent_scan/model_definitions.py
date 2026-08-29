"""Per-layer token-latent associative scan for LatentScan drafts."""

import math

import torch
from torch import nn
from torch.nn import functional
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer

__all__ = [
    "LatentCausalScan",
    "LatentRMSNorm",
    "Qwen3LatentScanDecoderLayer",
    "associative_affine_scan",
]

_SCAN_TENSOR_RANK = 3


def _keep_initialization(module: nn.Module) -> None:
    """Prevent HF ``post_init`` from replacing explicit scan initialization."""
    for submodule in module.modules():
        submodule._is_hf_initialized = True  # type: ignore[assignment]  # noqa: SLF001


class LatentRMSNorm(nn.Module):
    """Small RMSNorm shared by the training and vLLM implementations."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * values.to(input_dtype)


def associative_affine_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Inclusive scan of ``state[k] = decay[k] * state[k-1] + update[k]``.

    The affine maps are composed with Hillis--Steele steps. Every step operates
    on all block positions in parallel, so the dependency depth is ``log2(B)``
    rather than a Python loop over ``B`` positions. The initial state is zero.
    """
    if decay.shape != update.shape:
        raise ValueError(
            "decay and update must have the same shape, got "
            f"{tuple(decay.shape)} and {tuple(update.shape)}"
        )
    if decay.ndim != _SCAN_TENSOR_RANK:
        raise ValueError(
            "associative_affine_scan expects [blocks, positions, channels], got "
            f"{tuple(decay.shape)}"
        )

    prefix_decay = decay
    prefix_update = update
    offset = 1
    block_size = decay.shape[1]
    while offset < block_size:
        shifted_decay = torch.cat(
            [
                torch.ones_like(prefix_decay[:, :offset]),
                prefix_decay[:, :-offset],
            ],
            dim=1,
        )
        shifted_update = torch.cat(
            [
                torch.zeros_like(prefix_update[:, :offset]),
                prefix_update[:, :-offset],
            ],
            dim=1,
        )
        next_update = prefix_update + prefix_decay * shifted_update
        next_decay = prefix_decay * shifted_decay
        prefix_decay = next_decay
        prefix_update = next_update
        offset *= 2
    return prefix_update


class LatentCausalScan(nn.Module):
    """One layer of full-prefix causal mixing in a token-aligned latent space."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_dim: int,
        block_size: int,
        rms_norm_eps: float,
        layer_scale_init: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        if block_size < 2:  # noqa: PLR2004
            raise ValueError(f"block_size must be >= 2, got {block_size}")

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.initializer_range = initializer_range

        self.hidden_norm = LatentRMSNorm(hidden_size, rms_norm_eps)
        self.latent_norm = LatentRMSNorm(latent_dim, rms_norm_eps)
        self.hidden_proj = nn.Linear(hidden_size, latent_dim, bias=False)
        self.prev_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decay_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.gate_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.output_proj = nn.Linear(latent_dim, hidden_size, bias=False)

        self.slot_embedding = nn.Parameter(torch.empty(block_size, latent_dim))
        self.decay_bias = nn.Parameter(torch.empty(latent_dim))
        self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

        self.reset_parameters()
        _keep_initialization(self)

    def reset_parameters(self) -> None:
        """Initialize stable log-spaced decay times and a small residual branch."""
        for projection in (
            self.hidden_proj,
            self.prev_proj,
            self.value_proj,
            self.gate_proj,
            self.output_proj,
        ):
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                std=self.initializer_range,
            )
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.slot_embedding)

        # a = exp(-rate), with timescales log-spaced from 1 to block_size.
        positions = torch.linspace(0.0, 1.0, self.latent_dim)
        timescales = torch.exp(positions * math.log(float(self.block_size)))
        rates = timescales.reciprocal()
        inverse_softplus = torch.log(torch.expm1(rates))
        with torch.no_grad():
            self.decay_bias.copy_(inverse_softplus)

    @staticmethod
    def shift_latents(latents: torch.Tensor) -> torch.Tensor:
        """Slot 0 is the anchor; slot k reads the latent from slot k-1."""
        return torch.cat([latents[:, :1], latents[:, :-1]], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [N, B, hidden]
        latent_stream: torch.Tensor,  # [N, B, latent]
        current_latents: torch.Tensor,  # [N, B, latent]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got "
                f"{hidden_states.shape[1]} positions."
            )

        base_latents = self.latent_norm(latent_stream + current_latents)
        previous_latents = self.shift_latents(base_latents)
        scan_input = (
            self.hidden_proj(self.hidden_norm(hidden_states))
            + self.prev_proj(previous_latents)
            + self.slot_embedding.unsqueeze(0).to(hidden_states.dtype)
        )

        # The coefficients depend only on parallel-computable inputs, not on the
        # previous state. This is what preserves associative scan semantics.
        decay_rate = functional.softplus(
            self.decay_proj(scan_input).float() + self.decay_bias.float()
        )
        decay = torch.exp(-decay_rate)
        values = functional.silu(self.value_proj(scan_input)).float()
        updates = (1.0 - decay) * values
        scan_state = associative_affine_scan(decay, updates).to(scan_input.dtype)

        gated_state = torch.sigmoid(self.gate_proj(scan_input)) * scan_state
        next_latents = self.latent_norm(base_latents + gated_state)
        correction = self.output_proj(next_latents)
        hidden_states = hidden_states + self.layer_scale.to(hidden_states.dtype) * (
            correction.to(hidden_states.dtype)
        )
        return hidden_states, next_latents


class Qwen3LatentScanDecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer carrying one post-layer token-latent scan."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        latent_dim: int,
        block_size: int,
        layer_scale_init: float,
    ) -> None:
        super().__init__(config, layer_idx)
        self.latent_scan = LatentCausalScan(
            hidden_size=int(config.hidden_size),
            latent_dim=latent_dim,
            block_size=block_size,
            rms_norm_eps=float(config.rms_norm_eps),
            layer_scale_init=layer_scale_init,
            initializer_range=float(config.initializer_range or 0.02),
        )
