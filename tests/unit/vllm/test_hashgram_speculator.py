from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from speculators.vllm.hashgram_speculator import (
    HashGramSpeculator,
    sample_hashgram_block,
)


class _HistoryModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            hashgram_selector=SimpleNamespace(top_k=2),
        )
        self.history: list[tuple[torch.Tensor, torch.Tensor]] = []

    def compute_draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = hidden.new_full((hidden.shape[0], 8), -10.0)
        logits[:, 1] = 2.0
        logits[:, 2] = 1.0
        return logits

    def has_markov(self) -> bool:
        return False

    def score_hashgram_candidates(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        del unary_logits, hidden_states
        self.history.append((previous2_ids.clone(), previous_ids.clone()))
        # Alternate selected candidates through the recorded token history.
        choose_second = ((previous_ids + previous2_ids) % 2).bool()
        scores = torch.zeros_like(candidate_ids, dtype=torch.float32)
        scores[:, 0] = (~choose_second).float()
        scores[:, 1] = choose_second.float()
        return scores

    def map_draft_to_target(self, ids: torch.Tensor) -> torch.Tensor:
        return ids


def _greedy_sample(logits, idx_map, sample_pos, step):
    del idx_map, sample_pos, step
    return logits.argmax(dim=-1)


def test_hashgram_sampler_rolls_two_token_history_left_to_right():
    model = _HistoryModel()
    state = SimpleNamespace(
        num_speculative_steps=3,
        sample_indices=torch.arange(3),
        sample_idx_mapping=torch.zeros(3, dtype=torch.int32),
        sample_pos=torch.arange(1, 4),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([5])),
        _anchor_idx=torch.tensor([0]),
        previous2_ids=torch.tensor([4]),
        model=model,
        hashgram_logits=torch.empty(1, 8),
        draft_tokens=torch.zeros(1, 3, dtype=torch.long),
        _sample_logits=_greedy_sample,
    )

    sample_hashgram_block(state, 1, torch.zeros(3, 4))

    assert len(model.history) == 3
    assert [(int(p2), int(p1)) for p2, p1 in model.history] == [
        (4, 5),
        (5, 2),
        (2, 2),
    ]
    assert state.draft_tokens.tolist() == [[2, 2, 1]]


def test_hashgram_captures_the_last_non_rejected_context_token():
    state = SimpleNamespace(
        previous2_ids=torch.full((4,), 127, dtype=torch.long),
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(mask_token_id=127),
        ),
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 4, 9], dtype=torch.int32),
        input_ids=torch.tensor([10, 11, 12, 13, 20, 21, 22, 23, 24]),
    )

    HashGramSpeculator._capture_previous2(
        state,
        input_batch,
        num_rejected=torch.tensor([1, 2], dtype=torch.int32),
    )

    # Request 0 keeps [10, 11, 12]; request 1 keeps [20, 21, 22].
    assert state.previous2_ids.tolist() == [12, 22, 127, 127]
