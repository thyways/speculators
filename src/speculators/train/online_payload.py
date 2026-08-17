from __future__ import annotations

import torch

__all__ = ["check_online_payload"]

_HIDDEN_STATES_NDIM = 3


def check_online_payload(
    data: dict[str, torch.Tensor],
    tokens: list[int],
) -> None:
    """Validate one ephemeral payload returned by the online vLLM service."""

    token_ids = data["token_ids"].tolist()
    if token_ids != tokens:
        raise ValueError(f"Token ids don't match expected token ids {tokens}")

    hidden_states = data["hidden_states"]
    if hidden_states.ndim != _HIDDEN_STATES_NDIM:
        raise ValueError(
            "Hidden states must have shape [tokens, layers, hidden_size], got "
            f"{tuple(hidden_states.shape)}"
        )
    if not torch.isfinite(hidden_states).all():
        raise ValueError("Hidden states contain NaN or Inf values")
    if hidden_states.shape[0] != len(tokens):
        raise ValueError(
            f"Sequence length of hidden states {hidden_states.shape[0]} doesn't "
            f"match num tokens {len(tokens)}"
        )
