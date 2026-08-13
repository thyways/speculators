"""Tests for the shared sequential in-block sampler used by DFlash-family models.

The sampler is a single patched slot on vLLM's DSparkSpeculator, so all three
correction styles (memoryless Markov bias, hidden correction, recurrent logit
correction) must coexist in one implementation. These tests drive it with fake
speculator state, so they need neither vLLM nor a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from speculators.models.domino.model_definitions import DominoLogitsCorrection
from speculators.vllm._dflash_family import sample_sequential_block

HIDDEN_SIZE = 8
VOCAB_SIZE = 16
GRU_HIDDEN = 6
EMB_DIM = 5
BLOCK_SIZE = 4


def _speculator_state(
    model, num_steps: int, head_hidden: torch.Tensor, anchor: int
):
    return SimpleNamespace(
        num_speculative_steps=num_steps,
        sample_indices=torch.arange(num_steps),
        sample_idx_mapping=torch.zeros(num_steps, dtype=torch.int32),
        sample_pos=torch.arange(1, num_steps + 1),
        input_buffers=SimpleNamespace(
            input_ids=torch.tensor([anchor] + [0] * num_steps)
        ),
        _anchor_idx=torch.tensor([0]),
        model=model,
        draft_logits=None,
        draft_tokens=torch.zeros((1, num_steps), dtype=torch.long),
        _d2t_scatter_index=None,
        _draft_scatter_buf=None,
    )


class _FakeDFlyModel:
    def __init__(self):
        self.previous_tokens: list[torch.Tensor] = []

    def has_hidden_correction(self):
        return True

    def has_markov(self):
        return False

    def apply_hidden_correction(self, hidden, previous):
        self.previous_tokens.append(previous.clone())
        return previous.to(hidden.dtype).unsqueeze(-1)

    def compute_draft_logits(self, hidden):
        next_ids = hidden[:, 0].long() + 1
        logits = torch.full((hidden.shape[0], 16), -1000.0)
        logits.scatter_(1, next_ids.unsqueeze(1), 0.0)
        return logits

    def map_draft_to_target(self, token_ids):
        return token_ids


def test_dfly_hidden_correction_uses_previous_sampled_token():
    model = _FakeDFlyModel()
    state = _speculator_state(model, 3, torch.zeros((3, 1)), anchor=3)

    sample_sequential_block(state, num_reqs=1, head_hidden=torch.zeros((3, 1)))

    assert state.draft_tokens.tolist() == [[4, 5, 6]]
    assert [tokens.item() for tokens in model.previous_tokens] == [3, 4, 5]


class _FakeDSparkModel:
    def __init__(self):
        self.previous_tokens: list[torch.Tensor] = []

    def compute_draft_logits(self, hidden):
        return torch.zeros((hidden.shape[0], 16))

    def markov_embed(self, previous):
        self.previous_tokens.append(previous.clone())
        return previous

    def markov_bias(self, previous):
        next_ids = previous.long() + 1
        logits = torch.full((previous.shape[0], 16), -1000.0)
        logits.scatter_(1, next_ids.unsqueeze(1), 0.0)
        return logits

    def map_draft_to_target(self, token_ids):
        return token_ids


def test_sampler_preserves_dspark_markov_path():
    model = _FakeDSparkModel()
    state = _speculator_state(model, 3, torch.zeros((3, 1)), anchor=3)

    sample_sequential_block(state, num_reqs=1, head_hidden=torch.zeros((3, 1)))

    assert state.draft_tokens.tolist() == [[4, 5, 6]]
    assert [tokens.item() for tokens in model.previous_tokens] == [3, 4, 5]


class _FakeDominoModel:
    """Serving-side Domino hooks backed by the real training head module."""

    def __init__(
        self, *, sample_from_anchor: bool, pure_draft_prefix_len: int
    ):
        torch.manual_seed(0)
        self.correction = DominoLogitsCorrection(
            hidden_size=HIDDEN_SIZE,
            gru_hidden_dim=GRU_HIDDEN,
            emb_dim=EMB_DIM,
            draft_vocab_size=VOCAB_SIZE,
        )
        # A zero-initialized output layer would make the correction a no-op and
        # the test vacuous.
        torch.nn.init.normal_(self.correction.output_projection.weight)
        self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.sample_from_anchor = sample_from_anchor
        self.suffix_start = pure_draft_prefix_len + (
            0 if sample_from_anchor else 1
        )

    def has_markov(self):
        return False

    def has_hidden_correction(self):
        return False

    def has_recurrent_logits_correction(self):
        return True

    def compute_draft_logits(self, hidden):
        return self.head(hidden)

    def embed_input_ids(self, token_ids):
        return self.embed(token_ids)

    def init_recurrent_state(self, num_reqs, reference):
        return self.correction.prefix_gru.initial_state(reference, num_reqs)

    def advance_recurrent_state(self, prev_token_ids, state):
        gru = self.correction.prefix_gru
        embeds = self.embed_input_ids(prev_token_ids)
        return gru.step(gru.project_inputs(embeds.to(state.dtype)), state)

    def recurrent_logit_correction(self, step, hidden_states, state):
        slot = step + (0 if self.sample_from_anchor else 1)
        if slot < self.suffix_start:
            return None
        return self.correction(hidden_states, state)

    def map_draft_to_target(self, token_ids):
        return token_ids

    def training_side_block_logits(self, hidden, block_tokens):
        """Recompute the block's final logits the way training does."""
        base = self.head(hidden).view(1, BLOCK_SIZE, VOCAB_SIZE)
        states = self.correction.block_states(self.embed(block_tokens))
        low = self.suffix_start - (0 if self.sample_from_anchor else 1)
        states_suffix = states[:, low : low + BLOCK_SIZE - self.suffix_start]
        correction = self.correction(
            hidden.view(1, BLOCK_SIZE, HIDDEN_SIZE)[:, self.suffix_start :],
            states_suffix,
        )
        final = base.clone()
        final[:, self.suffix_start :] += correction
        return final


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_domino_sequential_sampling_matches_the_training_scan(
    sample_from_anchor,
):
    """Greedy serving output must equal the training-time block computation.

    The sampler advances the GRU one token at a time while training scans the
    whole block at once; both consume the same shared module, so the realized
    block must score identically under either route.
    """
    model = _FakeDominoModel(
        sample_from_anchor=sample_from_anchor,
        pure_draft_prefix_len=1,
    )
    anchor = 3
    num_steps = BLOCK_SIZE if sample_from_anchor else BLOCK_SIZE - 1
    torch.manual_seed(1)
    # Serving only scores the drafted slots; with a bonus anchor the anchor slot
    # has no hidden state of its own, so pad it to keep block-shaped indexing.
    drafted_hidden = torch.randn(num_steps, HIDDEN_SIZE)
    block_hidden = (
        drafted_hidden
        if sample_from_anchor
        else torch.cat([torch.zeros(1, HIDDEN_SIZE), drafted_hidden])
    )

    state = _speculator_state(model, num_steps, drafted_hidden, anchor=anchor)
    with torch.no_grad():
        sample_sequential_block(state, num_reqs=1, head_hidden=drafted_hidden)
    sampled = state.draft_tokens[0]

    # The realized block: the anchor followed by every token sampled before the
    # last slot.
    block_tokens = torch.cat(
        [torch.tensor([anchor]), sampled[: BLOCK_SIZE - 1]]
    ).view(1, BLOCK_SIZE)
    with torch.no_grad():
        final = model.training_side_block_logits(block_hidden, block_tokens)
    expected = final[0].argmax(dim=-1)

    slot_offset = 0 if sample_from_anchor else 1
    torch.testing.assert_close(sampled, expected[slot_offset:])


def test_domino_leading_slots_keep_uncorrected_logits():
    model = _FakeDominoModel(sample_from_anchor=True, pure_draft_prefix_len=2)
    hidden = torch.randn(1, HIDDEN_SIZE)
    state = model.init_recurrent_state(1, hidden)

    assert model.recurrent_logit_correction(0, hidden, state) is None
    assert model.recurrent_logit_correction(1, hidden, state) is None
    assert model.recurrent_logit_correction(2, hidden, state) is not None
