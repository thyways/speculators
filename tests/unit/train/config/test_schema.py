"""Data-model seam tests for the train config schema.

These exercise the schema at its two pure seams -- ``flatten()`` and
``from_flat()`` -- not CLI parsing or YAML. Backward compatibility against the
real parser is proven separately by the example-recipe tests, not a golden
``vars(args)`` snapshot here.
"""

import pytest

from speculators.train.config import TrainConfig
from speculators.train.config.schema import (
    CONFIG_DESTS,
    DataArgs,
    DFlashArgs,
    DraftArgs,
    GenerationArgs,
    KVNativeDFlashArgs,
    LossArgs,
    OptimizerArgs,
)


def test_constructs_from_defaults():
    # The whole point of the schema seam: a config exists with no inputs.
    TrainConfig()


def test_flatten_covers_exactly_the_schema_fields():
    # flatten() emits every schema dest and nothing else; consumers bind the flat
    # dict by name (**kwargs / args.<field>), so the key set is the contract.
    flat = TrainConfig().flatten()
    assert set(flat) == CONFIG_DESTS
    # Order is deterministic (declaration order) so the run.yaml dump stays stable.
    assert list(flat) == list(TrainConfig(speculator_type="dflash").flatten())


def test_flatten_resolves_eagle3_derived_defaults():
    # Mirrors the tail of the pre-refactor parse_args for the default (eagle3) run.
    flat = TrainConfig().flatten()
    assert flat["draft_arch"] == "llama"
    assert flat["norm_before_fc"] is True
    assert flat["norm_output"] is True
    assert flat["muon_lr"] == pytest.approx(10 * flat["lr"])


def test_flatten_resolves_non_eagle3_derived_defaults():
    flat = TrainConfig(speculator_type="dflash").flatten()
    assert flat["draft_arch"] == "qwen3"
    assert flat["norm_before_fc"] is False
    assert flat["norm_output"] is False


def test_flatten_resolves_dflash_derived_defaults():
    # Best-practices recipe from https://github.com/vllm-project/speculators/issues/979:
    # dflash gets 5 draft layers, D-PACE weighting, CE loss, and block_size=16
    # out of the box.
    flat = TrainConfig(speculator_type="dflash").flatten()
    assert flat["num_layers"] == 5
    assert flat["per_position_loss_weight"] == "dpace"
    assert flat["loss_fn"] == "ce"
    assert flat["block_size"] == 16


def test_flatten_resolves_kv_native_dflash_derived_defaults():
    flat = TrainConfig(speculator_type="kv_native_dflash").flatten()
    assert flat["num_layers"] == 5
    assert flat["per_position_loss_weight"] == "dpace"
    assert flat["loss_fn"] == "ce"
    assert flat["block_size"] == 16
    assert flat["num_speculative_tokens"] == 15
    assert flat["verifier_kv_layer_mapping"] == [3, 11, 19, 27, 35]
    assert "verifier_partial_rotary_factor" not in flat
    assert "verifier_rope_theta" not in flat
    assert "verifier_" + "mrope_section" not in flat


def test_flatten_leaves_non_dflash_derived_defaults_unchanged():
    # Only dflash gets the new defaults; every other speculator type keeps the
    # pre-existing behavior.
    for speculator_type in ("eagle3", "dspark", "peagle", "mtp"):
        flat = TrainConfig(speculator_type=speculator_type).flatten()
        assert flat["num_layers"] == 1
        assert flat["per_position_loss_weight"] == "fixed-exp-decay"
        assert flat["loss_fn"] == "kl_div"
        assert flat["block_size"] == 8


def test_dflash_derived_defaults_do_not_override_explicit_values():
    cfg = TrainConfig(
        speculator_type="dflash",
        draft=DraftArgs(num_layers=3),
        loss=LossArgs(loss_fn="kl_div"),
        dflash=DFlashArgs(per_position_loss_weight="fixed-exp-decay", block_size=8),
    )
    assert cfg.draft.num_layers == 3
    assert cfg.loss.loss_fn == "kl_div"
    assert cfg.dflash.per_position_loss_weight == "fixed-exp-decay"
    assert cfg.dflash.block_size == 8


def test_from_flat_inverts_flatten():
    cfg = TrainConfig(
        speculator_type="dspark",
        draft=DraftArgs(num_layers=4, full_attention_indices=[2, 18, 33]),
        optimizer=OptimizerArgs(lr=3e-4, kv_bridge_lr=3e-5),
    )
    assert TrainConfig.from_flat(cfg.flatten()) == cfg


def test_from_flat_default_round_trip():
    cfg = TrainConfig()
    assert TrainConfig.from_flat(cfg.flatten()) == cfg


def test_from_flat_ignores_non_config_keys():
    flat = TrainConfig().flatten()
    flat["config"] = "run.yaml"
    flat["dump_config"] = True
    recovered = TrainConfig.from_flat(flat)
    assert recovered == TrainConfig()


def test_from_flat_accepts_partial_working_dict():
    recovered = TrainConfig.from_flat({"lr": 5e-4, "num_layers": 6})
    assert recovered.optimizer.lr == pytest.approx(5e-4)
    assert recovered.draft.num_layers == 6
    # Untouched fields fall back to their schema defaults.
    assert recovered.trainer.epochs == 20


def _kv_native_dflash_config(**kv_kwargs):
    return TrainConfig(
        speculator_type="kv_native_dflash",
        draft=DraftArgs(num_layers=5),
        data=DataArgs(hidden_states_backend="file"),
        generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        kv_native_dflash=KVNativeDFlashArgs(**kv_kwargs),
    )


def test_kv_native_config_is_online_only():
    cfg = _kv_native_dflash_config()
    assert cfg.generation.on_missing == "generate"
    assert cfg.generation.on_generate == "delete"


@pytest.mark.parametrize(
    ("generation", "match"),
    [
        (GenerationArgs(on_missing="raise"), "on-missing=generate"),
        (GenerationArgs(on_generate="cache"), "on-generate=delete"),
    ],
)
def test_kv_native_rejects_non_online_generation(generation, match):
    with pytest.raises(ValueError, match=match):
        TrainConfig(
            speculator_type="kv_native_dflash",
            draft=DraftArgs(num_layers=5),
            data=DataArgs(hidden_states_backend="file"),
            generation=generation,
        )


def test_kv_native_rejects_pretrained_initialization():
    with pytest.raises(ValueError, match="from-scratch training only"):
        TrainConfig(
            speculator_type="kv_native_dflash",
            draft=DraftArgs(num_layers=5, from_pretrained="checkpoint"),
            data=DataArgs(hidden_states_backend="file"),
            generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        )


def test_kv_native_rejects_auxiliary_hidden_layers():
    with pytest.raises(ValueError, match="omit --target-layer-ids"):
        TrainConfig(
            speculator_type="kv_native_dflash",
            draft=DraftArgs(num_layers=5, target_layer_ids=[2, 20, 37]),
            data=DataArgs(hidden_states_backend="file"),
            generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        )


def test_kv_native_dflash_defaults_to_fifteen_proposal_slots():
    cfg = _kv_native_dflash_config()
    assert cfg.kv_native_dflash.num_speculative_tokens == 15
    assert cfg.kv_native_dflash.verifier_kv_layer_ids == [3, 11, 19, 27, 35]
    assert cfg.dflash.sample_from_anchor is None
    assert cfg.draft.num_layers == 5


@pytest.mark.parametrize("num_speculative_tokens", [14, 16])
def test_kv_native_dflash_requires_the_complete_proposal_block(
    num_speculative_tokens,
):
    with pytest.raises(ValueError, match="must equal the complete proposal block"):
        _kv_native_dflash_config(num_speculative_tokens=num_speculative_tokens)


def test_kv_native_dflash_rejects_sampling_from_anchor():
    with pytest.raises(ValueError, match="no-sample-from-anchor"):
        TrainConfig(
            speculator_type="kv_native_dflash",
            draft=DraftArgs(num_layers=5),
            data=DataArgs(hidden_states_backend="file"),
            generation=GenerationArgs(on_missing="generate", on_generate="delete"),
            dflash=DFlashArgs(sample_from_anchor=True),
            kv_native_dflash=KVNativeDFlashArgs(),
        )


def test_final_raw_kv_uses_depth_matched_sources():
    cfg = _kv_native_dflash_config()
    assert cfg.kv_native_dflash.verifier_kv_layer_mapping == [3, 11, 19, 27, 35]


def test_raw_kv_layer_fields_round_trip_through_flat_schema():
    cfg = _kv_native_dflash_config(
        verifier_kv_layer_ids=[3, 11, 19, 27, 35],
        verifier_kv_layer_mapping=[3, 3, 19, 27, 35],
    )
    recovered = TrainConfig.from_flat(cfg.flatten())
    args = recovered.kv_native_dflash
    assert args.verifier_kv_layer_ids == [3, 11, 19, 27, 35]
    assert args.verifier_kv_layer_mapping == [3, 3, 19, 27, 35]


def test_raw_kv_rejects_non_exported_anchor_mapping():
    with pytest.raises(ValueError, match="non-exported layers"):
        _kv_native_dflash_config(
            verifier_kv_layer_mapping=[7, 15, 23, 31, 999],
        )


def test_kv_direct_read_rejects_wrong_mapping_length():
    with pytest.raises(ValueError, match="verifier-kv-layer-mapping"):
        _kv_native_dflash_config(
            verifier_kv_layer_mapping=[7, 15],
        )


def test_kv_direct_read_rejects_non_exported_layer():
    with pytest.raises(ValueError, match="non-exported layers"):
        _kv_native_dflash_config(
            verifier_kv_layer_mapping=[7, 15, 23, 31, 35],
        )
