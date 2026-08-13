"""Tests for the Domino vLLM plugin's config translation.

The sequential sampler shared with DFly and DSpark is covered by
``test_dflash_family_sampler.py``.
"""

from typing import Any

import pytest

from speculators.models.domino.config import (
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_LOGITS_CORRECTION_EMB_DIM,
    DEFAULT_PURE_DRAFT_PREFIX_LEN,
    DominoSpeculatorConfig,
    resolve_suffix_start,
)
from speculators.train.config import TrainConfig
from speculators.vllm._dflash_family import (
    map_speculative_method,
    register_speculative_method_alias,
)
from speculators.vllm.domino import _update_domino


def _source(**overrides):
    source = {
        "aux_hidden_state_layer_ids": [2, 7, 12],
        "draft_vocab_size": 128,
        "mask_token_id": 127,
        "block_size": 16,
        "sample_from_anchor": True,
        "sliding_window_non_causal": True,
        "gru_hidden_dim": 32,
        "logits_correction_emb_dim": 8,
        "pure_draft_prefix_len": 1,
    }
    source.update(overrides)
    return source


def test_domino_config_is_mapped_to_sequential_runtime():
    register_speculative_method_alias("domino", "dspark")
    mapped = map_speculative_method(
        {"method": "domino", "num_speculative_tokens": 16}
    )

    assert mapped == {"method": "dspark", "num_speculative_tokens": 16}


def test_domino_and_dfly_aliases_coexist():
    """Both plugins share one method-alias table, so neither erases the other."""
    register_speculative_method_alias("dfly", "dspark")
    register_speculative_method_alias("domino", "dspark")

    assert map_speculative_method({"method": "dfly"})["method"] == "dspark"
    assert map_speculative_method({"method": "domino"})["method"] == "dspark"
    assert map_speculative_method({"method": "eagle3"})["method"] == "eagle3"


@pytest.mark.parametrize(
    ("sample_from_anchor", "dspark_bonus_anchor"),
    [(False, True), (True, False)],
)
def test_domino_config_translation_preserves_training_semantics(
    sample_from_anchor,
    dspark_bonus_anchor,
):
    translated: dict[str, Any] = {
        "hidden_size": 16,
        "rope_parameters": {
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
    }

    _update_domino(_source(sample_from_anchor=sample_from_anchor), translated)

    assert translated["architectures"][0] == "Qwen3DominoModel"
    # The compatibility sentinel keeps vLLM from rewriting a method=dspark
    # architecture into a DeepSeek-v4 embedded head.
    assert translated["architectures"][1] == "Qwen3DSparkModel"
    assert translated["model_arch"] == "domino"
    assert translated["sample_from_anchor"] is sample_from_anchor
    assert translated["dspark_bonus_anchor"] is dspark_bonus_anchor
    # Domino's recurrent correction replaces the Markov bias.
    assert translated["markov_rank"] == 0
    assert translated["gru_hidden_dim"] == 32
    assert translated["logits_correction_emb_dim"] == 8
    assert translated["pure_draft_prefix_len"] == 1
    assert translated["block_size"] == 16
    assert translated["eagle_aux_hidden_state_layer_ids"] == [2, 7, 12]
    assert translated["num_target_layers"] == 3
    # Upstream indexes hidden_states[layer_id + 1]; speculators uses layer ids.
    assert translated["dflash_config"]["target_layer_ids"] == [1, 6, 11]
    assert translated["dflash_config"]["causal"] is False
    assert "mrope_section" not in translated["rope_parameters"]


def test_domino_target_hidden_size_defaults_to_the_draft_width():
    translated: dict[str, Any] = {"hidden_size": 16}

    _update_domino(_source(), translated)

    assert translated["target_hidden_size"] == 16


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_a_minimal_config_serves_the_slots_it_was_trained_with(
    sample_from_anchor,
):
    """Omitted fields must fall back to the *training* defaults.

    A serving fallback that disagreed with the training default would apply the
    correction to a slot that was trained to keep the base logits -- correct
    output shapes, silently worse acceptance.
    """
    minimal = {
        "aux_hidden_state_layer_ids": [2, 7, 12],
        "draft_vocab_size": 128,
        "mask_token_id": 127,
        "block_size": 16,
        "sample_from_anchor": sample_from_anchor,
    }
    translated: dict[str, Any] = {"hidden_size": 16}

    _update_domino(minimal, translated)

    trained = DominoSpeculatorConfig(
        draft_vocab_size=128,
        block_size=16,
        sample_from_anchor=sample_from_anchor,
    )
    assert translated["gru_hidden_dim"] == trained.gru_hidden_dim
    assert (
        translated["logits_correction_emb_dim"]
        == trained.logits_correction_emb_dim
    )
    assert translated["pure_draft_prefix_len"] == trained.pure_draft_prefix_len
    served_suffix_start = resolve_suffix_start(
        sample_from_anchor=translated["sample_from_anchor"],
        pure_draft_prefix_len=translated["pure_draft_prefix_len"],
    )
    assert served_suffix_start == trained.suffix_start
    # DFlashSpeculatorConfig defaults sliding_window_non_causal to False.
    assert translated["dflash_config"]["causal"] is True


def test_cli_config_and_serving_defaults_all_agree():
    """Three layers declare these defaults; they must not drift apart."""
    cli = TrainConfig(speculator_type="domino").flatten()
    config = DominoSpeculatorConfig(draft_vocab_size=128, block_size=16)

    assert (
        cli["gru_hidden_dim"]
        == config.gru_hidden_dim
        == DEFAULT_GRU_HIDDEN_DIM
    )
    assert (
        cli["logits_correction_emb_dim"]
        == config.logits_correction_emb_dim
        == DEFAULT_LOGITS_CORRECTION_EMB_DIM
    )
    assert (
        cli["pure_draft_prefix_len"]
        == config.pure_draft_prefix_len
        == DEFAULT_PURE_DRAFT_PREFIX_LEN
    )
    assert cli["lambda_base_start"] == config.lambda_base_start
    assert cli["lambda_base_decay_ratio"] == config.lambda_base_decay_ratio


def test_translation_rejects_a_block_with_no_corrected_slots():
    source = _source(pure_draft_prefix_len=16)

    with pytest.raises(ValueError, match="no corrected slots"):
        _update_domino(source, {"hidden_size": 16})


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_sample_from_anchor_stays_out_of_dflash_config(sample_from_anchor):
    """vLLM's DFlashSpeculator raises on sample_from_anchor inside dflash_config.

    DSparkSpeculator (which Domino runs on) inherits that __init__ and reads the
    *top-level* field to size its query layout. Duplicating the key into
    dflash_config makes an anchor-sampling draft fail to start outright -- and
    Domino defaults to sample_from_anchor=True.
    """
    translated: dict[str, Any] = {"hidden_size": 16}

    _update_domino(_source(sample_from_anchor=sample_from_anchor), translated)

    assert translated["sample_from_anchor"] is sample_from_anchor
    assert "sample_from_anchor" not in translated["dflash_config"]
