from types import SimpleNamespace

import pytest
import torch

from speculators.vllm.dfly import (
    _dfly_context_width,
    _map_dfly_speculative_method,
    _preserve_dspark_anchor_mode,
    _resolve_dspark_target_hidden_size,
    _sample_sequential_with_hidden_correction,
    _update_dfly,
)


def test_dfly_config_is_mapped_to_sequential_runtime():
    mapped = _map_dfly_speculative_method(
        {"method": "dfly", "num_speculative_tokens": 7}
    )
    assert mapped == {
        "method": "dspark",
        "num_speculative_tokens": 7,
    }


@pytest.mark.parametrize(
    ("sample_from_anchor", "dspark_bonus_anchor"),
    [(False, True), (True, False)],
)
def test_dfly_config_translation_preserves_training_semantics(
    sample_from_anchor,
    dspark_bonus_anchor,
):
    source = {
        "aux_hidden_state_layer_ids": [2, 7, 12],
        "draft_vocab_size": 128,
        "target_hidden_size": 16,
        "mask_token_id": 127,
        "block_size": 8,
        "sample_from_anchor": sample_from_anchor,
        "sliding_window_non_causal": True,
        "enable_hidden_correction": True,
        "hidden_correction_intermediate_size": 24,
    }
    translated = {
        "hidden_size": 16,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
    }

    _update_dfly(source, translated)

    assert translated["architectures"][0] == "Qwen3DFlyModel"
    assert translated["sample_from_anchor"] is sample_from_anchor
    assert translated["dspark_bonus_anchor"] is dspark_bonus_anchor
    assert translated["markov_rank"] == 0
    assert translated["enable_hidden_correction"] is True
    assert translated["hidden_correction_intermediate_size"] == 24
    assert translated["eagle_aux_hidden_state_layer_ids"] == [2, 7, 12]
    assert translated["dflash_config"]["target_layer_ids"] == [1, 6, 11]
    assert translated["dflash_config"]["causal"] is False
    assert "mrope_section" not in translated["rope_parameters"]
    assert _dfly_context_width(SimpleNamespace(**translated)) == 48


@pytest.mark.parametrize(
    ("sample_from_anchor", "dspark_bonus_anchor"),
    [(False, True), (True, False)],
)
def test_dspark_config_translation_preserves_training_semantics(
    sample_from_anchor,
    dspark_bonus_anchor,
):
    translated = {"dspark_bonus_anchor": True}

    _preserve_dspark_anchor_mode(
        {"sample_from_anchor": sample_from_anchor},
        translated,
    )

    assert translated["sample_from_anchor"] is sample_from_anchor
    assert translated["dspark_bonus_anchor"] is dspark_bonus_anchor


def test_dspark_config_translation_keeps_legacy_default_when_field_is_missing():
    translated = {"dspark_bonus_anchor": True}

    _preserve_dspark_anchor_mode({}, translated)

    assert translated == {"dspark_bonus_anchor": True}


def test_dspark_config_translation_rejects_non_boolean_anchor_mode():
    with pytest.raises(TypeError, match="must be a boolean"):
        _preserve_dspark_anchor_mode(
            {"sample_from_anchor": "true"},
            {},
        )


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [(None, 16), (32, 32)],
)
def test_dspark_config_translation_resolves_target_hidden_size(
    source_size,
    expected_size,
):
    translated = {"hidden_size": 16}

    _resolve_dspark_target_hidden_size(
        {"target_hidden_size": source_size},
        translated,
    )

    assert translated["target_hidden_size"] == expected_size


def test_dspark_config_translation_requires_a_target_hidden_size():
    with pytest.raises(ValueError, match="must be a positive integer"):
        _resolve_dspark_target_hidden_size({}, {})


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
    state = SimpleNamespace(
        num_speculative_steps=3,
        sample_indices=torch.tensor([0, 1, 2]),
        sample_idx_mapping=torch.zeros(3, dtype=torch.int32),
        sample_pos=torch.tensor([1, 2, 3]),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([3, 0, 0, 0])),
        _anchor_idx=torch.tensor([0]),
        model=model,
        draft_logits=None,
        draft_tokens=torch.zeros((1, 3), dtype=torch.long),
        _d2t_scatter_index=None,
        _draft_scatter_buf=None,
    )

    _sample_sequential_with_hidden_correction(
        state,
        num_reqs=1,
        head_hidden=torch.zeros((3, 1)),
    )

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


def test_patched_sequential_sampler_preserves_dspark_markov_path():
    model = _FakeDSparkModel()
    state = SimpleNamespace(
        num_speculative_steps=3,
        sample_indices=torch.tensor([0, 1, 2]),
        sample_idx_mapping=torch.zeros(3, dtype=torch.int32),
        sample_pos=torch.tensor([1, 2, 3]),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([3, 0, 0, 0])),
        _anchor_idx=torch.tensor([0]),
        model=model,
        draft_logits=None,
        draft_tokens=torch.zeros((1, 3), dtype=torch.long),
        _d2t_scatter_index=None,
        _draft_scatter_buf=None,
    )

    _sample_sequential_with_hidden_correction(
        state,
        num_reqs=1,
        head_hidden=torch.zeros((3, 1)),
    )

    assert state.draft_tokens.tolist() == [[4, 5, 6]]
    assert [tokens.item() for tokens in model.previous_tokens] == [3, 4, 5]
