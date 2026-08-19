"""DFly-specific hidden-state correction components."""

import torch
from torch import nn
from torch.nn import functional
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.models.dfly.config import DFlySpeculatorConfig

__all__ = [
    "HiddenStatesCorrection",
    "build_hidden_correction",
]


class HiddenStatesCorrection(nn.Module):
    """Previous-token-conditioned residual correction used by DFly.

    The correction follows the TreeFlash form used by AngelSpec::

        h' = h + down(silu(gate([norm(h), norm(e_prev)])) * up(...))

    The output projection is zero-initialized, so a newly initialized DFly
    model starts with an identity correction.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = Qwen3RMSNorm(embed_size, eps=rms_norm_eps)

        input_size = hidden_size + embed_size
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_norm = self.hidden_norm(hidden_states)
        embed_norm = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        correction_input = torch.cat([hidden_norm, embed_norm], dim=-1)
        correction = self.down_proj(
            functional.silu(self.gate_proj(correction_input))
            * self.up_proj(correction_input)
        )
        return hidden_states + correction.to(hidden_states.dtype)


def build_hidden_correction(
    config: DFlySpeculatorConfig,
) -> HiddenStatesCorrection | None:
    """Build the optional DFly hidden-state correction module."""
    if not config.enable_hidden_correction:
        return None

    transformer_config = config.transformer_layer_config
    hidden_size = int(transformer_config.hidden_size)
    intermediate_size = config.hidden_correction_intermediate_size or hidden_size
    return HiddenStatesCorrection(
        hidden_size=hidden_size,
        embed_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=transformer_config.rms_norm_eps,
    )
