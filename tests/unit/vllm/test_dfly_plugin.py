from types import SimpleNamespace
from typing import Any

import pytest

from speculators.vllm._dflash_family import (
    install_config_patches,
    map_speculative_method,
    preserve_dspark_anchor_mode,
    propagate_intra_block_causality,
    register_speculative_method_alias,
    resolve_dspark_target_hidden_size,
)
from speculators.vllm.dfly import (
    _dfly_context_width,
    _update_dfly,
)


def test_dfly_config_is_mapped_to_sequential_runtime():
    register_speculative_method_alias("dfly", "dspark")
    mapped = map_speculative_method({"method": "dfly", "num_speculative_tokens": 7})
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
    translated: dict[str, Any] = {
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
    translated: dict[str, Any] = {"dspark_bonus_anchor": True}

    preserve_dspark_anchor_mode(
        {"sample_from_anchor": sample_from_anchor},
        translated,
    )

    assert translated["sample_from_anchor"] is sample_from_anchor
    assert translated["dspark_bonus_anchor"] is dspark_bonus_anchor


def test_dspark_config_translation_keeps_legacy_default_when_field_is_missing():
    translated: dict[str, Any] = {"dspark_bonus_anchor": True}

    preserve_dspark_anchor_mode({}, translated)

    assert translated == {"dspark_bonus_anchor": True}


def test_dspark_config_translation_rejects_non_boolean_anchor_mode():
    with pytest.raises(TypeError, match="must be a boolean"):
        preserve_dspark_anchor_mode(
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
    translated: dict[str, Any] = {"hidden_size": 16}

    resolve_dspark_target_hidden_size(
        {"target_hidden_size": source_size},
        translated,
    )

    assert translated["target_hidden_size"] == expected_size


def test_dspark_config_translation_requires_a_target_hidden_size():
    with pytest.raises(ValueError, match="must be a positive integer"):
        resolve_dspark_target_hidden_size({}, {})


def _runtime_causal_by_layer(translated: dict[str, Any]) -> list[bool]:
    from vllm.model_executor.models.qwen3_dflash import (  # noqa: PLC0415
        _dflash_layer_causal,
    )

    config = SimpleNamespace(**translated)
    return [
        _dflash_layer_causal(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ]


@pytest.mark.parametrize(
    ("non_causal", "expected_override", "expected_causal_by_layer"),
    [
        (True, False, [False, False]),
        (False, None, [True, False]),
    ],
)
def test_dspark_intra_block_causality_follows_the_checkpoint(
    non_causal,
    expected_override,
    expected_causal_by_layer,
):
    """The serving mask must match training for both layer types."""
    translated: dict[str, Any] = {
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        # Simulate the incorrect global override written by upstream DFlash.
        "dflash_config": {"causal": True},
    }

    propagate_intra_block_causality(
        {"sliding_window_non_causal": non_causal},
        translated,
    )

    assert translated["dflash_config"].get("causal") is expected_override
    assert _runtime_causal_by_layer(translated) == expected_causal_by_layer


def test_intra_block_causality_preserves_existing_dflash_config_keys():
    translated: dict[str, Any] = {"dflash_config": {"mask_token_id": 7}}

    propagate_intra_block_causality(
        {"sliding_window_non_causal": True},
        translated,
    )

    assert translated["dflash_config"] == {"mask_token_id": 7, "causal": False}


def test_intra_block_causality_leaves_external_checkpoints_alone():
    """External DSpark checkpoints never had the field; keep vLLM's fallback."""
    translated: dict[str, Any] = {"hidden_size": 16}

    propagate_intra_block_causality({}, translated)

    assert "dflash_config" not in translated


def test_intra_block_causality_honours_an_explicit_default():
    translated: dict[str, Any] = {}

    propagate_intra_block_causality({}, translated, default=False)

    assert "dflash_config" not in translated


def test_config_patch_repairs_upstream_dflash_all_full_attention():
    from vllm.transformers_utils.configs.speculators.algos import (  # noqa: PLC0415
        SUPPORTED_SPECULATORS_TYPES,
    )

    install_config_patches()
    translated: dict[str, Any] = {
        "num_hidden_layers": 5,
        "layer_types": ["full_attention"] * 5,
    }
    SUPPORTED_SPECULATORS_TYPES["dflash"](
        {
            "aux_hidden_state_layer_ids": [2, 7, 12],
            "mask_token_id": 7,
            "sliding_window_non_causal": False,
        },
        translated,
    )

    assert "causal" not in translated["dflash_config"]
    assert _runtime_causal_by_layer(translated) == [False] * 5


def test_intra_block_causality_rejects_a_non_boolean():
    with pytest.raises(TypeError, match="must be a boolean"):
        propagate_intra_block_causality(
            {"sliding_window_non_causal": "true"},
            {},
        )


def test_dfly_keeps_sample_from_anchor_out_of_dflash_config():
    """Same constraint as Domino: the key belongs at the top level only."""
    translated: dict[str, Any] = {"hidden_size": 16}

    _update_dfly(
        {
            "aux_hidden_state_layer_ids": [2, 7, 12],
            "draft_vocab_size": 128,
            "mask_token_id": 127,
            "block_size": 8,
            "sample_from_anchor": True,
        },
        translated,
    )

    assert translated["sample_from_anchor"] is True
    assert "sample_from_anchor" not in translated["dflash_config"]
    assert "causal" not in translated["dflash_config"]
