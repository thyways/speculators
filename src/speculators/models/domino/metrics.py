"""Loss and metrics for the Domino draft model.

``loss = (1 - lambda_base) * L(final_logits) + lambda_base * L(base_logits)``

``L`` is the shared DFlash objective, so Domino inherits every configured loss
term, the per-position decay, and the accuracy / EAL telemetry unchanged. The
base term anchors early training on the backbone alone (its gradient cannot
reach the correction head, and the draft LM head is frozen); ``lambda_base``
decays to zero so the deployed -- corrected -- logits are what ends up being
optimized.

When ``lambda_base`` is zero the base term is skipped entirely rather than
multiplied by zero: with no chunked reduction in this repo, a second pass over
``[1, T, draft_vocab_size]`` is the dominant memory cost of the step. The
``base_*`` metrics are therefore only emitted while the base term is live;
``lambda_base`` is always logged so their absence is unambiguous.
"""

from typing import Any

import torch

from speculators.models.dflash.metrics import (
    compute_metrics as dflash_compute_metrics,
)
from speculators.models.metrics import LossConfig

__all__ = [
    "compute_metrics",
]

# Base-side keys worth carrying: the backbone-only loss and the two headline
# quality numbers. Everything else would collide with the final-side keys.
_BASE_METRIC_KEYS = ("loss", "full_acc", "eal")


def compute_metrics(
    final_logits: torch.Tensor,  # [1, T, draft_vocab_size]
    base_logits: torch.Tensor,  # [1, T, draft_vocab_size]
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    *,
    lambda_base: torch.Tensor,  # scalar
    include_base: bool,
    loss_config: LossConfig | None = None,
    gamma: float = 4.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Blend the final and base DFlash objectives and merge their metrics."""
    shared: dict[str, Any] = {
        "targets": targets,
        "loss_mask": loss_mask,
        "block_size": block_size,
        "gamma": gamma,
        "loss_config": loss_config,
        "per_position_loss_weight": per_position_loss_weight,
        "dpace_alpha": dpace_alpha,
        "sample_from_anchor": sample_from_anchor,
    }

    final_loss, metrics = dflash_compute_metrics(final_logits, **shared)
    ones = torch.ones((), device=final_logits.device)
    metrics["final_loss_sum"] = final_loss.detach().clone()
    metrics["final_loss_total"] = ones.clone()

    if include_base:
        base_loss, base_metrics = dflash_compute_metrics(base_logits, **shared)
        # Keep the final term even at lambda_base == 1: `0 * final_loss` is what
        # routes a zero gradient to the correction head, without which DDP's
        # reducer reports it as an unused parameter.
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        for key in _BASE_METRIC_KEYS:
            metrics[f"base_{key}_sum"] = (
                base_metrics[f"{key}_sum"].detach().clone()
            )
            metrics[f"base_{key}_total"] = (
                base_metrics[f"{key}_total"].detach().clone()
            )
    else:
        loss = final_loss

    # The headline `loss` must be the quantity actually optimized: the trainer
    # logs it and `maybe_update_best` selects checkpoints on its epoch mean.
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones.clone()
    # Detached copy, never the live buffer: the trainer all-reduces metric
    # tensors in place, which would otherwise scale lambda_base by world_size.
    metrics["lambda_base"] = lambda_base.detach().clone().float()
    return loss, metrics
