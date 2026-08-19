"""Loss and metrics for the DFlash2 draft model.

The loss is DFlash's, computed on the selector-corrected logits -- the correction
is an additive low-rank term over the whole vocabulary, so every existing loss
(``ce``, ``kl_div``, D-PACE weighting) applies unchanged.

On top of that this module reports what the *inference* path will do, which the
full-vocabulary loss cannot show on its own: at serving time the selector only
ever ranks the target head's top-K per slot, and it walks the block using the
token it picked itself rather than the ground-truth one. The diagnostics below
replay exactly that walk, and replay it a second time with the correction
switched off, so a single run reports the selector's contribution.

They run every step, with no way to turn them off: a DFlash2 run whose selector is
not earning anything should be visible in the metrics rather than at eval time. The
cost is the vocabulary top-K -- the same operation the inference side calls the
selector's largest single cost -- and it lands in the single-digit percent of a
training step at production shapes. The walk itself is negligible.
"""

from typing import Any

import torch

from speculators.losses import LossConfig
from speculators.models.dflash.metrics import compute_metrics as dflash_compute_metrics
from speculators.models.dflash2.model_definitions import CandidateSelector

__all__ = [
    "compute_metrics",
    "compute_selector_diagnostics",
]


def _walk_metrics(
    prefix: str,
    correct: torch.Tensor,  # [num_blocks, block_size], already masked
    valid: torch.Tensor,  # [num_blocks, block_size], 1 where the slot is trained
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Per-position accuracy, DFlash-style EAL, and mean accepted run length.

    ``{prefix}_eal`` uses DFlash's estimator (the product of independent
    per-position accuracies) so it lines up with the ``eal`` the DFlash metrics
    already report. ``{prefix}_accept_len`` is the directly measured mean leading
    run of correct tokens in a block; add 1 for the bonus token to compare it with
    the acceptance length vLLM reports.
    """
    metrics: dict[str, torch.Tensor] = {}
    ones = torch.tensor(1.0, device=correct.device)

    eal = torch.zeros((), device=correct.device)
    cumulative = torch.ones((), device=correct.device)
    for pos in range(1, block_size):
        pos_correct = correct[:, pos].sum()
        pos_total = valid[:, pos].sum()
        metrics[f"{prefix}_position_{pos}_acc_sum"] = pos_correct.float()
        metrics[f"{prefix}_position_{pos}_acc_total"] = pos_total.float()
        cumulative = cumulative * (pos_correct / pos_total.clamp(min=1.0))
        eal = eal + cumulative
    metrics[f"{prefix}_eal_sum"] = eal
    metrics[f"{prefix}_eal_total"] = ones.clone()

    # A masked-out slot terminates the run: cumprod over the drafted slots.
    run_length = correct[:, 1:].to(torch.int64).cumprod(dim=1).sum(dim=1)
    counted = valid[:, 1] if block_size > 1 else torch.zeros_like(valid[:, 0])
    metrics[f"{prefix}_accept_len_sum"] = (run_length * counted).sum().float()
    metrics[f"{prefix}_accept_len_total"] = counted.sum().float()
    return metrics


@torch.no_grad()
def compute_selector_diagnostics(
    unary_logits: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, num_blocks*block_size]
    hidden_states: torch.Tensor,  # [1, num_blocks*block_size, hidden_size]
    anchor_token_ids: torch.Tensor,  # [num_blocks]
    selector: CandidateSelector,
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Replay the inference-time candidate walk, with and without the selector.

    Args:
        unary_logits: Draft logits before the selector correction, already scaled
            by ``output_multiplier`` and softcapped, as the inference side sees
            them.
        targets: Verifier logits; their argmax is the label, matching DFlash's
            accuracy metrics.
        loss_mask: The aligned block mask; slot 0 (the anchor) is already zeroed.
        hidden_states: Post-norm draft hidden states.
        anchor_token_ids: The verified token in slot 0 of each block, i.e. the
            first step's predecessor.
        selector: The trained selector, read only.
        block_size: Draft block size; slots ``1..block_size-1`` are drafted.

    Returns:
        ``_sum``/``_total`` metric pairs for the selector walk, the top-1 (plain
        DFlash) walk, and the top-K candidate recall that bounds both.
    """
    num_blocks = unary_logits.shape[1] // block_size
    top_k = min(selector.top_k, unary_logits.shape[-1])

    unary_blocks = unary_logits.view(num_blocks, block_size, -1)
    label_ids = targets.argmax(dim=-1).view(num_blocks, block_size)
    valid = loss_mask.view(num_blocks, block_size).bool()

    candidate_values, candidate_ids = torch.topk(unary_blocks, top_k, dim=-1)
    gate = selector.project(hidden_states).view(num_blocks, block_size, -1)
    predecessors = selector.predecessor_codebook
    successors = selector.successor_codebook

    metrics: dict[str, torch.Tensor] = {}

    in_candidates = (candidate_ids == label_ids.unsqueeze(-1)).any(dim=-1) & valid
    metrics["candidate_recall_sum"] = in_candidates.sum().float()
    metrics["candidate_recall_total"] = valid.sum().float()

    selector_tokens = torch.empty_like(label_ids)
    selector_tokens[:, 0] = anchor_token_ids
    previous = anchor_token_ids
    for pos in range(1, block_size):
        slot_candidates = candidate_ids[:, pos]  # [num_blocks, top_k]
        query = (predecessors[previous] * gate[:, pos]).unsqueeze(1)
        edge = (query * successors[slot_candidates]).sum(dim=-1)
        chosen = (candidate_values[:, pos] + edge).argmax(dim=-1)
        previous = slot_candidates.gather(1, chosen.unsqueeze(1)).squeeze(1)
        selector_tokens[:, pos] = previous

    metrics.update(
        _walk_metrics(
            "selector",
            (selector_tokens == label_ids) & valid,
            valid,
            block_size,
        )
    )
    # Selector off: the walk degenerates to the per-slot top-1, i.e. DFlash.
    metrics.update(
        _walk_metrics(
            "unary",
            (candidate_ids[..., 0] == label_ids) & valid,
            valid,
            block_size,
        )
    )
    return metrics


def compute_metrics(
    logits: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, num_blocks*block_size]
    block_size: int,
    *,
    diagnostics: dict[str, torch.Tensor] | None = None,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """DFlash's loss on the corrected logits, merged with the selector diagnostics."""
    loss, metrics = dflash_compute_metrics(
        logits,
        targets,
        loss_mask,
        block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=False,
    )
    if diagnostics:
        metrics.update(diagnostics)
    return loss, metrics
