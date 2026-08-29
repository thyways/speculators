"""Block-parallel token-latent feedback modules."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional

from speculators.models.latent_scan.model_definitions import LatentRMSNorm

__all__ = [
    "TokenLatentFeedbackHead",
    "TokenLatentFeedbackOutput",
    "build_causal_toeplitz_matrix",
]

_TENSOR_RANK = 3


def build_causal_toeplitz_matrix(
    coefficients: torch.Tensor,
    *,
    mode: str = "full",
    strict: bool = True,
) -> torch.Tensor:
    """Build the fixed-shape causal Toeplitz matrix used by the mixer."""
    if coefficients.ndim != 1:
        raise ValueError(
            f"coefficients must be one-dimensional, got {tuple(coefficients.shape)}"
        )
    block_size = coefficients.shape[0] + 1
    positions = torch.arange(block_size, device=coefficients.device)
    distance = positions[:, None] - positions[None, :]
    valid = distance > 0 if strict else distance >= 0
    if mode == "shifted":
        valid = valid & (distance == 1)
    elif mode == "none":
        valid = torch.zeros_like(valid)
    elif mode != "full":
        raise ValueError(f"Unsupported prefix mixer mode: {mode!r}")
    indices = (distance - 1).clamp_min(0)
    return coefficients[indices] * valid.to(coefficients.dtype)


class TokenLatentFeedbackOutput(NamedTuple):
    """Outputs of the parallel feedback head."""

    hidden_states: torch.Tensor
    latents: torch.Tensor
    gated_latents: torch.Tensor
    prefix_latents: torch.Tensor
    reliability: torch.Tensor


class TokenLatentFeedbackHead(nn.Module):
    """Predict, mix, and write back token-intent latents in one block.

    The only position-to-position operation is a dense multiplication by a
    fixed-shape strictly lower-triangular Toeplitz matrix.  Consequently its
    graph has constant depth with respect to the number of draft positions and
    does not expose a Python/CUDA-graph loop over those positions.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_dim: int,
        block_size: int,
        rms_norm_eps: float,
        initializer_range: float,
        prefix_mixer_mode: str = "full",
        use_reliability_gate: bool = True,
        strict_causal_prefix: bool = True,
        position_scale_init: float = 1.0,
        feedback_output_projection_init: float = 0.0,
    ) -> None:
        super().__init__()
        if block_size < 2:  # noqa: PLR2004
            raise ValueError(f"block_size must be >= 2, got {block_size}")
        if prefix_mixer_mode not in {"full", "shifted", "none"}:
            raise ValueError(
                "prefix_mixer_mode must be one of 'full', 'shifted', or 'none', "
                f"got {prefix_mixer_mode!r}"
            )

        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.block_size = int(block_size)
        self.prefix_mixer_mode = prefix_mixer_mode
        self.use_reliability_gate = bool(use_reliability_gate)
        self.strict_causal_prefix = bool(strict_causal_prefix)
        self.initializer_range = float(initializer_range)

        self.input_norm = LatentRMSNorm(self.hidden_size, rms_norm_eps)
        # The latent projection and scalar gate share one packed GEMM.
        self.intent_gate_proj = nn.Linear(
            self.hidden_size,
            self.latent_dim + 1,
            bias=True,
        )
        self.prefix_norm = LatentRMSNorm(self.latent_dim, rms_norm_eps)
        self.feedback_up_proj = nn.Linear(
            self.latent_dim,
            self.hidden_size,
            bias=False,
        )
        self.toeplitz_coeff = nn.Parameter(torch.empty(self.block_size - 1))
        self.position_scale = nn.Parameter(
            torch.full((self.block_size,), float(position_scale_init))
        )

        self.reset_parameters(feedback_output_projection_init)

    # Read-only compatibility names used by experiment notebooks.  Properties
    # avoid registering the same module twice (which would duplicate checkpoint
    # keys) while keeping the canonical packed-GEMM names above.
    @property
    def token_latent_projection(self) -> nn.Linear:
        return self.intent_gate_proj

    @property
    def feedback_output_projection(self) -> nn.Linear:
        return self.feedback_up_proj

    @property
    def prefix_mixer_coefficients(self) -> nn.Parameter:
        return self.toeplitz_coeff

    def reset_parameters(self, feedback_output_projection_init: float = 0.0) -> None:
        """Use a stable small initialization and an exact zero feedback path."""
        nn.init.normal_(
            self.intent_gate_proj.weight,
            mean=0.0,
            std=self.initializer_range,
        )
        nn.init.zeros_(self.intent_gate_proj.bias)
        # A short-distance prior gives the first trained update a useful signal;
        # the output projection remains zero so the whole model starts as DFlash.
        distances = torch.arange(
            1,
            self.block_size,
            dtype=self.toeplitz_coeff.dtype,
        )
        with torch.no_grad():
            self.toeplitz_coeff.copy_(distances.reciprocal().sqrt())
            self.position_scale[0] = 0.0
        if feedback_output_projection_init == 0.0:
            nn.init.zeros_(self.feedback_up_proj.weight)
        else:
            nn.init.constant_(
                self.feedback_up_proj.weight,
                float(feedback_output_projection_init),
            )

    def _mixer_matrix(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Materialize the small strict lower-triangular Toeplitz matrix."""
        coefficients = self.toeplitz_coeff.to(device=device, dtype=dtype)
        return build_causal_toeplitz_matrix(
            coefficients,
            mode=self.prefix_mixer_mode,
            strict=self.strict_causal_prefix,
        )

    def mix_prefix(self, gated_latents: torch.Tensor) -> torch.Tensor:
        """Mix ``[batch, block, latent]`` source latents causally in parallel."""
        if gated_latents.ndim != _TENSOR_RANK:
            raise ValueError(
                "gated_latents must have shape [batch, block, latent], got "
                f"{tuple(gated_latents.shape)}"
            )
        if gated_latents.shape[1] != self.block_size:
            raise ValueError(
                f"Expected {self.block_size} block positions, got "
                f"{gated_latents.shape[1]}"
            )
        mixer = self._mixer_matrix(gated_latents.dtype, gated_latents.device)
        return torch.matmul(mixer, gated_latents)

    def forward(self, hidden_states: torch.Tensor) -> TokenLatentFeedbackOutput:
        """Apply the packed latent projection, prefix mixer, and write-back."""
        if hidden_states.ndim != _TENSOR_RANK:
            raise ValueError(
                "hidden_states must have shape [batch, block, hidden], got "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] != self.block_size:
            raise ValueError(
                f"Expected {self.block_size} block positions, got "
                f"{hidden_states.shape[1]}"
            )

        packed = self.intent_gate_proj(self.input_norm(hidden_states))
        latent_logits, gate_logits = packed.split([self.latent_dim, 1], dim=-1)
        latents = functional.normalize(latent_logits.float(), dim=-1).to(
            hidden_states.dtype
        )
        reliability = (
            torch.sigmoid(gate_logits.float()).to(hidden_states.dtype)
            if self.use_reliability_gate
            else torch.ones_like(gate_logits, dtype=hidden_states.dtype)
        )
        gated_latents = latents * reliability
        prefix_latents = self.mix_prefix(gated_latents)
        prefix_latents = self.prefix_norm(prefix_latents)
        correction = self.feedback_up_proj(prefix_latents)
        scales = self.position_scale.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).view(1, self.block_size, 1)
        corrected_hidden = hidden_states + correction * scales
        return TokenLatentFeedbackOutput(
            corrected_hidden,
            latents,
            gated_latents,
            prefix_latents,
            reliability.squeeze(-1),
        )
