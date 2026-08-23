"""Loss and metrics for the DFlash2 draft model.

The loss has two terms, because inference makes two separate decisions::

    loss = dflash_loss(unary, targets) + w * CE(unary + bias over top-K, target)

The draft head picks the candidate set -- ``topK(unary)``, fixed before the selector
contributes anything -- and the selector picks one candidate out of that set. The
first term is DFlash's own full-vocabulary loss on the unary logits, and it is what
puts the target token in the set at all (``candidate_recall``). The second is the
cross-entropy of the selector's actual decision: a categorical over the ``top_k``
kept candidates, scored exactly as ``vllm/v1/worker/gpu/spec_decode/dflash2`` scores
them, so training optimizes the distribution serving samples from rather than a
full-vocabulary stand-in for it.

Slots whose target the draft head left out of the candidate set carry no selector
loss. The selector cannot recover a token that is not in the set -- widening the set
is the unary term's job -- so including them would only ask the bias to move
probability onto a candidate that is wrong whatever it does. ``candidate_recall`` is
the fraction of slots that clears the bar, and the ceiling on the selector.

On top of the loss this module reports what the inference path will do: the walk
conditions on the token *it* picked in slot ``k-1``, not on the ground-truth one the
loss feeds in, so the walk is replayed here in full, and replayed a second time with
the correction switched off. A single run therefore reports the selector's
contribution (``selector_accept_len`` vs ``unary_accept_len``).

The diagnostics run every step, with no way to turn them off: a DFlash2 run whose
selector is not earning anything should be visible in the metrics rather than at
eval time. They share the candidate top-K with the loss, so they cost the walk and
nothing else.
"""

from functools import partial
from typing import Any, NamedTuple

import torch

from speculators.losses import (
    LossConfig,
    dflash_loss_decay,
    dpace_loss_decay,
    masked_decayed_mean,
)
from speculators.models.dflash.metrics import compute_metrics as dflash_compute_metrics
from speculators.models.dflash2.model_definitions import CandidateSelector

__all__ = [
    "SelectorCandidates",
    "compute_metrics",
    "compute_selector_diagnostics",
    "select_candidates",
    "selector_cross_entropy",
]

_IGNORE_INDEX = -100


class SelectorCandidates(NamedTuple):
    """The per-slot candidate set the selector chooses from.

    ``values`` are the unary logits of the kept tokens and carry gradient: they are
    the first half of the walk's score, so the selector loss trains the draft head
    through them as well as through the codebooks.
    """

    values: torch.Tensor  # [num_blocks, block_size, top_k]
    ids: torch.Tensor  # [num_blocks, block_size, top_k]


def select_candidates(
    unary_logits: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    selector: CandidateSelector,
    block_size: int,
) -> SelectorCandidates:
    """The candidate set ``compute_candidates`` builds at inference.

    The top-K spans the vocabulary and is taken on the draft head's own logits,
    before the selector contributes anything -- which is why a correction that would
    only pay off outside the top-K is wasted, and why widening the set is the unary
    term's job.

    ``top_k`` is clamped to the vocabulary so a small-vocab test configuration keeps
    working; production never hits that.
    """
    num_blocks = unary_logits.shape[1] // block_size
    top_k = min(selector.top_k, unary_logits.shape[-1])
    values, ids = torch.topk(
        unary_logits.view(num_blocks, block_size, -1), top_k, dim=-1
    )
    return SelectorCandidates(values=values, ids=ids)


def selector_cross_entropy(
    candidates: SelectorCandidates,
    label_ids: torch.Tensor,  # [num_blocks, block_size]
    prev_token_ids: torch.Tensor,  # [num_blocks, block_size]
    hidden_states: torch.Tensor,  # [1, num_blocks*block_size, hidden_size]
    selector: CandidateSelector,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the inference-time decision, and the slots it covers.

    The decision is a categorical over the slot's candidates with logits
    ``unary[c] + bias[prev, c]`` -- the scores the walk takes its argmax over at
    ``temperature=0``, and, above it, samples from after renormalizing over the same
    K. Training conditions on the ground-truth predecessor; the walk conditions on
    its own previous choice, which is what the diagnostics measure.

    Returns:
        ``(elementwise, in_candidates)``: per-slot cross-entropy shaped ``[1, T]``,
        zero where the target is not among the candidates, and the boolean mask of
        the slots where it is.
    """
    num_blocks, block_size, top_k = candidates.ids.shape
    scores = candidates.values + selector.candidate_bias(
        prev_token_ids,
        hidden_states.view(num_blocks, block_size, -1),
        candidates.ids,
    )

    label_slot = candidates.ids == label_ids.unsqueeze(-1)
    in_candidates = label_slot.any(dim=-1)
    # argmax picks the one True column; rows without one are ignored below.
    label_index = torch.where(
        in_candidates,
        label_slot.to(torch.int64).argmax(dim=-1),
        torch.full_like(label_ids, _IGNORE_INDEX),
    )
    elementwise = torch.nn.functional.cross_entropy(
        scores.reshape(-1, top_k).float(),
        label_index.reshape(-1),
        reduction="none",
        ignore_index=_IGNORE_INDEX,
    )
    return elementwise.view(1, -1), in_candidates.view(1, -1)


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
    candidates: SelectorCandidates | None = None,
    label_ids: torch.Tensor | None = None,  # [num_blocks, block_size]
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
        candidates: The candidate set from :func:`select_candidates`; recomputed
            when absent, so the loss path hands over the one it already built.
        label_ids: ``targets.argmax(-1)``, likewise recomputed when absent.

    Returns:
        ``_sum``/``_total`` metric pairs for the selector walk, the top-1 (plain
        DFlash) walk, and the top-K candidate recall that bounds both.
    """
    num_blocks = unary_logits.shape[1] // block_size
    if candidates is None:
        candidates = select_candidates(unary_logits, selector, block_size)
    if label_ids is None:
        label_ids = targets.argmax(dim=-1).view(num_blocks, block_size)
    candidate_values, candidate_ids = candidates
    valid = loss_mask.view(num_blocks, block_size).bool()

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
    unary_logits: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_blocks*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # [1, num_blocks*block_size]
    block_size: int,
    *,
    selector: CandidateSelector,
    hidden_states: torch.Tensor,  # [1, num_blocks*block_size, hidden_size]
    prev_token_ids: torch.Tensor,  # [num_blocks, block_size]
    anchor_token_ids: torch.Tensor,  # [num_blocks]
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    selector_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """DFlash's loss on the unary logits, plus the selector's top-K decision loss.

    ``metrics["loss"]`` is the total; ``unary_loss`` and ``selector_loss`` are the
    two terms before ``selector_loss_weight``, so a run shows which one is moving.
    """
    device = unary_logits.device
    num_blocks = unary_logits.shape[1] // block_size
    candidates = select_candidates(unary_logits, selector, block_size)
    with torch.no_grad():
        label_ids = targets.argmax(dim=-1).view(num_blocks, block_size)

    # The candidate set is topK(unary), so this term is what puts the target in it.
    loss, metrics = dflash_compute_metrics(
        unary_logits,
        targets,
        loss_mask,
        block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=False,
    )
    ones = torch.ones((), device=device)
    metrics["unary_loss_sum"] = loss.detach().clone()
    metrics["unary_loss_total"] = ones

    elementwise, in_candidates = selector_cross_entropy(
        candidates, label_ids, prev_token_ids, hidden_states, selector
    )
    # Only slots whose target is inside the candidate set are decisions the selector
    # can win, and the denominator counts exactly those, so the term's scale does not
    # move with candidate_recall.
    selector_mask = loss_mask.bool() & in_candidates
    pos_idx = (torch.arange(loss_mask.shape[1], device=device) % block_size).unsqueeze(
        0
    )
    if per_position_loss_weight == "dpace":
        # Weighted like the unary term: D-PACE closes over the full block mask, and
        # so treats the slots this term drops as neutral rather than as rejections.
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(dflash_loss_decay, gamma=gamma, sample_from_anchor=False)
    selector_loss = masked_decayed_mean(elementwise, selector_mask, pos_idx, decay_fn)
    loss = loss + selector_loss_weight * selector_loss

    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = ones.clone()
    metrics["loss_sum"] = loss.detach().clone()  # overwrite DFlash's: report the total
    metrics.update(
        compute_selector_diagnostics(
            unary_logits,
            targets,
            loss_mask,
            hidden_states,
            anchor_token_ids,
            selector,
            block_size,
            candidates=candidates,
            label_ids=label_ids,
        )
    )
    return loss, metrics
