from __future__ import annotations

import torch

__all__ = ["check_online_payload"]

_HIDDEN_STATES_NDIM = 3
_VERIFIER_KV_NDIM = 4


def check_online_payload(  # noqa: C901
    data: dict[str, torch.Tensor],
    tokens: list[int],
    *,
    require_verifier_kv: bool = False,
    expected_verifier_kv_shape: tuple[int, int, int] | None = None,
    expected_verifier_kv_layer_ids: list[int] | None = None,
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

    kv_names = {
        "verifier_keys",
        "verifier_values",
        "verifier_kv_layer_ids",
    }
    present = kv_names.intersection(data)
    if not present:
        if require_verifier_kv:
            raise ValueError(
                "Verifier K/V is required, but the online vLLM payload contains "
                "only auxiliary hidden states. Launch vLLM with "
                "--verifier-kv-layer-ids."
            )
        return
    if present != kv_names:
        missing = sorted(kv_names - present)
        raise ValueError(f"Incomplete verifier K/V payload; missing tensors: {missing}")

    if "position_ids" not in data:
        raise ValueError(
            "Verifier K/V payload is missing text position_ids. Relaunch vLLM "
            "with the KV-native connector from this checkout."
        )
    position_ids = data["position_ids"]
    if position_ids.ndim != 1 or position_ids.shape[0] != len(tokens):
        raise ValueError(
            "Current text-only verifier position_ids must have shape [tokens], "
            f"got {tuple(position_ids.shape)}"
        )
    if position_ids.dtype == torch.bool or torch.is_floating_point(position_ids):
        raise ValueError("Verifier position_ids must use an integer dtype")
    expected_positions = torch.arange(
        len(tokens), dtype=position_ids.dtype, device=position_ids.device
    )
    if not torch.equal(position_ids, expected_positions):
        raise ValueError(
            "KV-native training currently supports text-only contiguous position_ids "
            f"[0, ..., T-1], got {position_ids.tolist()}"
        )

    keys = data["verifier_keys"]
    values = data["verifier_values"]
    layer_ids = data["verifier_kv_layer_ids"]
    if keys.shape != values.shape:
        raise ValueError(
            "Verifier K/V shapes differ: "
            f"keys={tuple(keys.shape)}, values={tuple(values.shape)}"
        )
    if keys.ndim != _VERIFIER_KV_NDIM:
        raise ValueError(
            "Verifier K/V must have shape [tokens, layers, kv_heads, head_dim], "
            f"got {tuple(keys.shape)}"
        )
    if layer_ids.ndim != 1 or layer_ids.shape[0] != keys.shape[1]:
        raise ValueError(
            "verifier_kv_layer_ids must match the K/V layer axis, got IDs "
            f"{tuple(layer_ids.shape)} and K/V {tuple(keys.shape)}"
        )
    if keys.shape[0] != len(tokens):
        raise ValueError(
            f"Sequence length of verifier K/V {keys.shape[0]} doesn't match "
            f"num tokens {len(tokens)}"
        )
    if not torch.is_floating_point(keys) or not torch.is_floating_point(values):
        raise ValueError("Verifier K/V tensors must use a floating-point dtype")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("Verifier K/V contains NaN or Inf values")
    layer_id_values = [int(item) for item in layer_ids.tolist()]
    if len(layer_id_values) != len(set(layer_id_values)):
        raise ValueError(
            f"verifier_kv_layer_ids contains duplicates: {layer_id_values}"
        )
    if expected_verifier_kv_shape is not None:
        expected = (len(tokens), *expected_verifier_kv_shape)
        if tuple(keys.shape) != expected:
            raise ValueError(
                f"Verifier K/V shape mismatch: expected {expected}, got "
                f"{tuple(keys.shape)}"
            )
    if (
        expected_verifier_kv_layer_ids is not None
        and layer_id_values != expected_verifier_kv_layer_ids
    ):
        raise ValueError(
            "Verifier K/V layer IDs mismatch: expected "
            f"{expected_verifier_kv_layer_ids}, got {layer_id_values}"
        )
