"""Unit tests for the DFlash2 convolution and candidate selector.

The first two tests are ports of the reference tests shipped with
vllm-project/vllm#52816 (``tests/v1/spec_decode/test_dflash2.py``), run against
this repo's implementations. They are the contract: if they pass, a checkpoint
trained here means the same thing to the inference kernels.
"""

import pytest
import torch

from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    DFlashGroupedConv,
    grouped_conv,
    score_edges,
)


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    """Port of ``test_grouped_conv_matches_reference`` from the vLLM PR."""
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    """Port of ``test_selector_edges_match_sequential_reference`` from the vLLM PR."""
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = score_edges(
        predecessors, successors, candidate_ids, unary, hidden, anchors, top_k
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


class TestGroupedConv:
    def _conv(self, hidden_size=16, taps=3, group_size=4, block_size=8):
        torch.manual_seed(0)
        return DFlashGroupedConv(
            hidden_size=hidden_size,
            taps=taps,
            group_size=group_size,
            block_size=block_size,
        )

    def test_identity_at_initialization(self):
        conv = self._conv()
        hidden = torch.randn(3 * 8, 16)
        convolved, coefficients = conv.prepare(hidden)
        torch.testing.assert_close(convolved, hidden)
        torch.testing.assert_close(conv.finish(hidden, coefficients), hidden)

    def test_accepts_a_leading_batch_dimension(self):
        """Training feeds ``[1, T, H]``; inference feeds ``[T, H]``."""
        conv = self._conv()
        with torch.no_grad():
            conv.kernel_projection.weight.normal_(std=0.1)
            conv.base_kernel.normal_(std=0.1)
        hidden = torch.randn(3 * 8, 16)
        flat, flat_coefficients = conv.prepare(hidden)
        batched, batched_coefficients = conv.prepare(hidden.unsqueeze(0))
        assert batched.shape == (1, 24, 16)
        torch.testing.assert_close(flat, batched.squeeze(0))
        torch.testing.assert_close(
            conv.finish(hidden, flat_coefficients),
            conv.finish(hidden.unsqueeze(0), batched_coefficients).squeeze(0),
        )

    def test_taps_do_not_cross_the_block_boundary(self):
        conv = self._conv(block_size=4)
        with torch.no_grad():
            conv.kernel_projection.weight.normal_(std=0.1)
            conv.base_kernel.normal_(std=0.1)
        hidden = torch.randn(3 * 4, 16)
        baseline, _ = conv.prepare(hidden)

        perturbed = hidden.clone()
        perturbed[3] += 10.0  # last slot of block 0
        actual, _ = conv.prepare(perturbed)

        # Block 0 changes; blocks 1 and 2 must not see it at all.
        assert not torch.allclose(actual[:4], baseline[:4])
        torch.testing.assert_close(actual[4:], baseline[4:])

    def test_first_positions_only_see_available_taps(self):
        """Position 0 of a block has no history, so only tap 0 contributes."""
        conv = self._conv(block_size=4, taps=3)
        with torch.no_grad():
            conv.base_kernel.normal_(std=0.1)
        hidden = torch.zeros(2 * 4, 16)
        hidden[0] = 1.0
        convolved, _ = conv.prepare(hidden)
        expected = conv.base_kernel[0, 0] * hidden[0]
        torch.testing.assert_close(convolved[0], expected)

    def test_group_size_must_divide_hidden_size(self):
        with pytest.raises(ValueError, match="must divide hidden_size"):
            DFlashGroupedConv(hidden_size=16, taps=2, group_size=5, block_size=8)

    def test_taps_must_fit_in_the_block(self):
        with pytest.raises(ValueError, match="must not exceed block_size"):
            DFlashGroupedConv(hidden_size=16, taps=9, group_size=4, block_size=8)

    def test_taps_must_be_positive(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            DFlashGroupedConv(hidden_size=16, taps=0, group_size=4, block_size=8)


class TestCandidateSelector:
    def _selector(self, vocab=17, hidden_size=16, rank=5, top_k=3):
        torch.manual_seed(0)
        return CandidateSelector(
            verifier_vocab_size=vocab,
            draft_vocab_size=vocab,
            hidden_size=hidden_size,
            rank=rank,
            top_k=top_k,
        )

    def test_bias_is_exactly_zero_at_initialization(self):
        selector = self._selector()
        bias = selector.block_bias(
            torch.randint(17, (4, 6)),
            torch.randn(4, 6, 16),
        )
        assert bias.shape == (4, 6, 17)
        assert bias.abs().max() == 0.0

    def test_block_bias_matches_the_inference_edge_scores(self):
        """The two forms of the bias score the same function.

        ``block_bias`` scores the whole vocabulary against the ground-truth
        predecessor; ``edge_scores`` scores the kept top-K against the predecessor
        the walk actually chose. Where the predecessor agrees, the two must return
        the same number.
        """
        selector = self._selector(vocab=17, hidden_size=16, rank=5, top_k=3)
        with torch.no_grad():
            selector.successor_codebook.normal_(std=0.5)

        num_blocks, block_size, top_k = 4, 6, 3
        torch.manual_seed(2)
        block_tokens = torch.randint(17, (num_blocks, block_size))
        hidden = torch.randn(num_blocks, block_size, 16)
        # Slot k's predecessor is slot k-1; slot 0's entry is unused.
        prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
        bias = selector.block_bias(prev_token_ids, hidden)

        # Inference step l corresponds to slot l+1, so the walk sees slots 1..B-1.
        steps = block_size - 1
        candidate_ids = torch.randint(17, (num_blocks, steps, top_k))
        # Put the ground-truth path at candidate index 0 so the predecessor the
        # edge scorer uses at step l equals prev_token_ids[:, l + 1].
        candidate_ids[:, :, 0] = block_tokens[:, 1:]
        unary = torch.zeros(num_blocks, steps, top_k)
        edges = selector.edge_scores(
            candidate_ids, unary, hidden[:, 1:], block_tokens[:, 0]
        )

        for step in range(steps):
            torch.testing.assert_close(
                edges[:, step, 0],
                bias[:, step + 1].gather(1, candidate_ids[:, step]),
            )

    def test_candidate_bias_matches_the_inference_edge_scores(self):
        """The loss and the serving path score the same function.

        ``candidate_bias`` is what the selector loss adds to the unary logits, and
        ``edge_scores`` is what the walk scores. For the ground-truth predecessor the
        two must agree column by column -- otherwise training would optimize
        something the drafter does not compute.
        """
        selector = self._selector(vocab=17, hidden_size=16, rank=5, top_k=3)
        with torch.no_grad():
            selector.successor_codebook.normal_(std=0.5)

        num_blocks, block_size, top_k = 4, 6, 3
        torch.manual_seed(2)
        block_tokens = torch.randint(17, (num_blocks, block_size))
        hidden = torch.randn(num_blocks, block_size, 16)
        prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

        # Inference step l corresponds to slot l+1, so the walk sees slots 1..B-1.
        steps = block_size - 1
        candidate_ids = torch.randint(17, (num_blocks, steps, top_k))
        # Put the ground-truth path at candidate index 0 so the predecessor the edge
        # scorer uses at step l is prev_token_ids[:, l + 1], i.e. row p=0 is the row
        # the loss trains.
        candidate_ids[:, :, 0] = block_tokens[:, 1:]
        unary = torch.zeros(num_blocks, steps, top_k)
        edges = selector.edge_scores(
            candidate_ids, unary, hidden[:, 1:], block_tokens[:, 0]
        )

        # Slot 0 carries no loss; its candidates are never read.
        block_candidates = torch.cat([candidate_ids[:, :1], candidate_ids], dim=1)
        bias = selector.candidate_bias(prev_token_ids, hidden, block_candidates)
        assert bias.shape == (num_blocks, block_size, top_k)
        for step in range(steps):
            torch.testing.assert_close(edges[:, step, 0], bias[:, step + 1])

    def test_candidate_bias_is_block_bias_on_the_kept_candidates(self):
        """The top-K restriction is a gather, not a different computation."""
        selector = self._selector(vocab=17, hidden_size=16, rank=5, top_k=3)
        with torch.no_grad():
            selector.successor_codebook.normal_(std=0.5)

        torch.manual_seed(3)
        prev_token_ids = torch.randint(17, (4, 6))
        hidden = torch.randn(4, 6, 16)
        candidate_ids = torch.randint(17, (4, 6, 3))

        torch.testing.assert_close(
            selector.candidate_bias(prev_token_ids, hidden, candidate_ids),
            selector.block_bias(prev_token_ids, hidden).gather(2, candidate_ids),
        )

    def test_bias_depends_on_the_predecessor(self):
        selector = self._selector()
        with torch.no_grad():
            selector.successor_codebook.normal_(std=0.5)
        hidden = torch.randn(1, 1, 16)
        first = selector.block_bias(torch.tensor([[1]]), hidden)
        second = selector.block_bias(torch.tensor([[2]]), hidden)
        assert not torch.allclose(first, second)

    def test_rejects_a_degenerate_rank(self):
        with pytest.raises(ValueError, match="selector_rank"):
            CandidateSelector(
                verifier_vocab_size=17,
                draft_vocab_size=17,
                hidden_size=16,
                rank=0,
                top_k=3,
            )

    def test_rejects_a_top_k_with_no_transition(self):
        with pytest.raises(ValueError, match="selector_top_k"):
            CandidateSelector(
                verifier_vocab_size=17,
                draft_vocab_size=17,
                hidden_size=16,
                rank=4,
                top_k=1,
            )
