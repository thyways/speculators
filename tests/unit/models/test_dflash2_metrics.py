"""Unit tests for the DFlash2 selector diagnostics.

The diagnostics replay the inference-time walk, so they are the only place in
training where the top-K restriction and the selector's own predecessor choice are
exercised. These tests construct cases where the per-slot top-1 and the corrected
walk disagree, so a walk that silently ignored the bias -- or propagated the wrong
predecessor -- would fail.
"""

import torch

from speculators.models.dflash2.metrics import compute_selector_diagnostics
from speculators.models.dflash2.model_definitions import CandidateSelector

VOCAB = 8
HIDDEN = 4
RANK = 2
BLOCK_SIZE = 3
NUM_BLOCKS = 2


def make_selector(top_k: int = 3) -> CandidateSelector:
    selector = CandidateSelector(
        verifier_vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        hidden_size=HIDDEN,
        rank=RANK,
        top_k=top_k,
    )
    with torch.no_grad():
        # project(h) == h[:2]; the predecessor codebook is a per-token one-hot on
        # rank 0, so the bias reduces to successor_codebook[:, 0] * h[0] and is
        # fully controllable from the test.
        selector.hidden_projection.weight.zero_()
        selector.hidden_projection.weight[0, 0] = 1.0
        selector.hidden_projection.weight[1, 1] = 1.0
        selector.predecessor_codebook.zero_()
        selector.successor_codebook.zero_()
    return selector


def one_hot_logits(token_ids: torch.Tensor, high: float = 10.0) -> torch.Tensor:
    """Unary logits whose per-slot ranking is ``token_ids`` in order."""
    logits = torch.zeros(1, token_ids.shape[0] * token_ids.shape[1], VOCAB)
    flat = token_ids.reshape(-1)
    for slot in range(flat.shape[0]):
        logits[0, slot, flat[slot]] = high
    return logits


def targets_for(label_ids: torch.Tensor) -> torch.Tensor:
    return one_hot_logits(label_ids, high=5.0)


def test_walk_falls_back_to_the_top_1_when_the_bias_is_zero():
    selector = make_selector()
    top1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    diagnostics = compute_selector_diagnostics(
        one_hot_logits(top1),
        targets_for(top1),
        torch.tensor([[0, 1, 1, 0, 1, 1]]),
        torch.randn(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN),
        top1[:, 0],
        selector,
        BLOCK_SIZE,
    )
    for pos in range(1, BLOCK_SIZE):
        assert diagnostics[f"selector_position_{pos}_acc_sum"] == NUM_BLOCKS
        assert diagnostics[f"unary_position_{pos}_acc_sum"] == NUM_BLOCKS
    assert diagnostics["selector_accept_len_sum"] == NUM_BLOCKS * (BLOCK_SIZE - 1)
    assert diagnostics["candidate_recall_sum"] == NUM_BLOCKS * (BLOCK_SIZE - 1)


def test_a_trained_bias_overrides_the_top_1():
    """The selector must be able to pick a candidate the unary head ranked lower.

    Slot 1's unary top-1 is token 1 but the label is token 2, so the unary walk is
    wrong. A bias that favours token 2 by more than the unary gap must flip it.
    """
    selector = make_selector()
    unary = torch.zeros(1, NUM_BLOCKS * BLOCK_SIZE, VOCAB)
    unary[0, :, 1] = 2.0  # top-1 everywhere
    unary[0, :, 2] = 1.0  # runner-up everywhere
    labels = torch.tensor([[7, 2, 2], [7, 2, 2]])

    hidden = torch.zeros(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN)
    hidden[0, :, 0] = 1.0  # project(h) = [1, 0]
    anchors = torch.tensor([7, 7])
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1]])

    args = (
        unary,
        targets_for(labels),
        loss_mask,
        hidden,
        anchors,
        selector,
        BLOCK_SIZE,
    )

    off = compute_selector_diagnostics(*args)
    for pos in range(1, BLOCK_SIZE):
        assert off[f"selector_position_{pos}_acc_sum"] == 0
        assert off[f"unary_position_{pos}_acc_sum"] == 0

    with torch.no_grad():
        # bias[v] = <A[prev] * [1, 0], B[v]> = A[prev, 0] * B[v, 0]
        selector.predecessor_codebook[:, 0] = 1.0
        selector.successor_codebook[2, 0] = 5.0
    on = compute_selector_diagnostics(*args)
    for pos in range(1, BLOCK_SIZE):
        assert on[f"selector_position_{pos}_acc_sum"] == NUM_BLOCKS
        assert on[f"unary_position_{pos}_acc_sum"] == 0
    assert on["selector_accept_len_sum"] == NUM_BLOCKS * (BLOCK_SIZE - 1)
    assert on["unary_accept_len_sum"] == 0


def test_the_walk_conditions_on_its_own_previous_choice():
    """Only the token the walk picked at slot k-1 may drive slot k.

    The bias is keyed on the predecessor, and only predecessor 3 favours token 6.
    Slot 1's walk is steered to token 3, so slot 2 must then pick 6 -- which it can
    only do if ``previous`` carried the walk's own choice rather than the anchor.
    """
    selector = make_selector()
    unary = torch.zeros(1, NUM_BLOCKS * BLOCK_SIZE, VOCAB)
    unary[0, :, 1] = 2.0
    unary[0, :, 3] = 1.0
    unary[0, :, 6] = 0.5
    labels = torch.tensor([[7, 3, 6], [7, 3, 6]])

    hidden = torch.zeros(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN)
    hidden[0, :, 0] = 1.0
    anchors = torch.tensor([7, 7])
    loss_mask = torch.tensor([[0, 1, 1, 0, 1, 1]])

    with torch.no_grad():
        # From the anchor (7): favour token 3. From token 3: favour token 6.
        selector.predecessor_codebook[7, 0] = 1.0
        selector.predecessor_codebook[3, 1] = 1.0
        selector.successor_codebook[3, 0] = 5.0
        selector.successor_codebook[6, 1] = 5.0
        # project(h) = [h[0], h[1]] = [1, 1] so both rank slots are live.
        hidden[0, :, 1] = 1.0

    diagnostics = compute_selector_diagnostics(
        unary, targets_for(labels), loss_mask, hidden, anchors, selector, BLOCK_SIZE
    )
    # Slot 1 -> token 3 (label 3), slot 2 -> token 6 (label 6): both correct.
    assert diagnostics["selector_position_1_acc_sum"] == NUM_BLOCKS
    assert diagnostics["selector_position_2_acc_sum"] == NUM_BLOCKS
    assert diagnostics["unary_position_1_acc_sum"] == 0
    assert diagnostics["selector_accept_len_sum"] == NUM_BLOCKS * (BLOCK_SIZE - 1)


def test_masked_slots_terminate_the_accepted_run():
    """A padded block contributes nothing; a masked slot ends the run."""
    selector = make_selector()
    top1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # Block 0 is valid; block 1 is entirely padded.
    loss_mask = torch.tensor([[0, 1, 1, 0, 0, 0]])
    diagnostics = compute_selector_diagnostics(
        one_hot_logits(top1),
        targets_for(top1),
        loss_mask,
        torch.randn(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN),
        top1[:, 0],
        selector,
        BLOCK_SIZE,
    )
    assert diagnostics["selector_accept_len_total"] == 1
    assert diagnostics["selector_accept_len_sum"] == BLOCK_SIZE - 1
    assert diagnostics["candidate_recall_total"] == BLOCK_SIZE - 1

    # Now truncate block 0 at slot 2: the run stops after slot 1.
    diagnostics = compute_selector_diagnostics(
        one_hot_logits(top1),
        targets_for(top1),
        torch.tensor([[0, 1, 0, 0, 0, 0]]),
        torch.randn(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN),
        top1[:, 0],
        selector,
        BLOCK_SIZE,
    )
    assert diagnostics["selector_accept_len_total"] == 1
    assert diagnostics["selector_accept_len_sum"] == 1


def test_metrics_are_scalar_tensors():
    """The trainer all-reduces then ``.item()``s every value."""
    selector = make_selector()
    top1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    diagnostics = compute_selector_diagnostics(
        one_hot_logits(top1),
        targets_for(top1),
        torch.ones(1, NUM_BLOCKS * BLOCK_SIZE, dtype=torch.long),
        torch.randn(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN),
        top1[:, 0],
        selector,
        BLOCK_SIZE,
    )
    for key, value in diagnostics.items():
        assert isinstance(value, torch.Tensor), key
        assert value.ndim == 0, key
        assert value.dtype == torch.float32, key
        assert not value.requires_grad, key
    # Every key is half of a _sum/_total pair, so normalize_counted_metrics
    # reduces it to a ratio instead of dividing it by the world size.
    sums = {key.removesuffix("_sum") for key in diagnostics if key.endswith("_sum")}
    totals = {
        key.removesuffix("_total") for key in diagnostics if key.endswith("_total")
    }
    assert sums == totals
    assert len(sums) * 2 == len(diagnostics)


def test_top_k_larger_than_the_vocabulary_is_clamped():
    selector = make_selector(top_k=VOCAB + 4)
    top1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    diagnostics = compute_selector_diagnostics(
        one_hot_logits(top1),
        targets_for(top1),
        torch.ones(1, NUM_BLOCKS * BLOCK_SIZE, dtype=torch.long),
        torch.randn(1, NUM_BLOCKS * BLOCK_SIZE, HIDDEN),
        top1[:, 0],
        selector,
        BLOCK_SIZE,
    )
    # Every token is a candidate, so recall is perfect.
    assert diagnostics["candidate_recall_sum"] == diagnostics["candidate_recall_total"]
