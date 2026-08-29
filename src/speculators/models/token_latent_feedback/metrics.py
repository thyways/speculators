"""Loss helpers for Parallel Token-Latent Feedback."""

from __future__ import annotations

import torch

__all__ = ["compute_latent_cosine_loss"]

_TENSOR_RANK = 3


def compute_latent_cosine_loss(
    predicted_latents: torch.Tensor,
    target_codes: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked ``1 - cosine`` loss and the mean cosine similarity."""
    if predicted_latents.shape != target_codes.shape:
        raise ValueError(
            "predicted_latents and target_codes must have the same shape, got "
            f"{tuple(predicted_latents.shape)} and {tuple(target_codes.shape)}"
        )
    if predicted_latents.ndim != _TENSOR_RANK:
        raise ValueError(
            "latent tensors must have shape [batch, positions, dim], got "
            f"{tuple(predicted_latents.shape)}"
        )
    if loss_mask.shape != predicted_latents.shape[:2]:
        raise ValueError(
            "loss_mask must match the first two latent dimensions, got "
            f"{tuple(loss_mask.shape)} and {tuple(predicted_latents.shape[:2])}"
        )
    predicted = torch.nn.functional.normalize(predicted_latents.float(), dim=-1)
    target = torch.nn.functional.normalize(target_codes.float(), dim=-1)
    cosine = (predicted * target).sum(dim=-1)
    mask = loss_mask.to(cosine.dtype)
    count = mask.sum()
    mean_cosine = (cosine * mask).sum() / count.clamp_min(1.0)
    loss = (1.0 - mean_cosine) * (count > 0).to(mean_cosine.dtype)
    return loss, mean_cosine
