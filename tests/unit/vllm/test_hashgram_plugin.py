from typing import Any

import pytest

pytest.importorskip("vllm")

from speculators.vllm.hashgram import (
    _DSPARK_COMPAT_ARCH,
    HASHGRAM_ARCH,
    _finalize_speculative_config,
    _update_hashgram,
)


def _source(**overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {
        "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
        "draft_vocab_size": 128,
        "target_hidden_size": None,
        "mask_token_id": 127,
        "block_size": 8,
        "sample_from_anchor": True,
        "sliding_window_non_causal": True,
        "hashgram_rank": 16,
        "hashgram_top_k": 8,
        "hashgram_bigram_buckets": 64,
        "hashgram_trigram_buckets": 128,
        "hashgram_num_hashes": 2,
        "hashgram_markov_rank": 12,
        "hashgram_use_markov_recall": True,
        "hashgram_hidden_refine": False,
        "hashgram_use_bigram": True,
        "hashgram_use_trigram": True,
    }
    source.update(overrides)
    return source


def test_hashgram_config_translation_preserves_training_semantics():
    translated: dict[str, Any] = {"hidden_size": 32}

    _update_hashgram(_source(), translated)

    assert translated["architectures"] == [HASHGRAM_ARCH, _DSPARK_COMPAT_ARCH]
    assert translated["model_arch"] == "hashgram"
    assert translated["sample_from_anchor"] is True
    assert translated["dspark_bonus_anchor"] is False
    assert translated["target_hidden_size"] == 32
    assert translated["eagle_aux_hidden_state_layer_ids"] == [1, 9, 17, 25, 33]
    assert translated["dflash_config"] == {
        "mask_token_id": 127,
        "target_layer_ids": [0, 8, 16, 24, 32],
        "causal": False,
    }
    assert "sample_from_anchor" not in translated["dflash_config"]
    assert translated["hashgram_top_k"] == 8
    assert translated["hashgram_num_hashes"] == 2
    assert translated["markov_rank"] == 12


@pytest.mark.parametrize(
    ("sample_from_anchor", "speculative_tokens"),
    [(True, 8), (False, 7)],
)
def test_hashgram_config_routes_to_dspark_with_matching_block(
    sample_from_anchor: bool,
    speculative_tokens: int,
):
    source = _source(sample_from_anchor=sample_from_anchor)
    result = _finalize_speculative_config(
        source,
        {"method": "hashgram", "num_speculative_tokens": speculative_tokens},
    )

    assert result == {
        "method": "dspark",
        "num_speculative_tokens": speculative_tokens,
    }


def test_hashgram_config_rejects_a_different_serving_block():
    with pytest.raises(ValueError, match="block width"):
        _finalize_speculative_config(
            _source(),
            {"method": "hashgram", "num_speculative_tokens": 7},
        )
