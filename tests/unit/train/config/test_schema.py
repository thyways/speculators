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
    DraftArgs,
    GenerationArgs,
    KVNativeDSparkArgs,
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


def _kv_native_config(**kv_kwargs):
    return TrainConfig(
        speculator_type="kv_native_dspark",
        draft=DraftArgs(num_layers=6),
        data=DataArgs(hidden_states_backend="file"),
        generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        kv_native_dspark=KVNativeDSparkArgs(**kv_kwargs),
    )


def test_kv_native_config_is_online_only():
    cfg = _kv_native_config()
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
            speculator_type="kv_native_dspark",
            draft=DraftArgs(num_layers=6),
            data=DataArgs(hidden_states_backend="file"),
            generation=generation,
        )


def test_kv_native_rejects_pretrained_initialization():
    with pytest.raises(ValueError, match="from-scratch training only"):
        TrainConfig(
            speculator_type="kv_native_dspark",
            draft=DraftArgs(num_layers=6, from_pretrained="checkpoint"),
            data=DataArgs(hidden_states_backend="file"),
            generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        )


def test_kv_native_rejects_auxiliary_hidden_layers():
    with pytest.raises(ValueError, match="omit --target-layer-ids"):
        TrainConfig(
            speculator_type="kv_native_dspark",
            draft=DraftArgs(num_layers=6, target_layer_ids=[2, 20, 37]),
            data=DataArgs(hidden_states_backend="file"),
            generation=GenerationArgs(on_missing="generate", on_generate="delete"),
        )


def test_kv_native_defaults_to_spec7():
    cfg = _kv_native_config()
    assert cfg.kv_native_dspark.num_speculative_tokens == 7
    assert cfg.kv_native_dspark.verifier_kv_layer_ids == [3, 11, 19, 27, 31, 39]


def test_kv_native_rejects_speculative_length_beyond_block():
    with pytest.raises(ValueError, match="num-speculative-tokens exceeds"):
        _kv_native_config(num_speculative_tokens=9)


def test_kv_bridge_uses_all_exported_layers_without_manual_mapping():
    cfg = _kv_native_config(
        kv_bridge_enabled=True,
        kv_bridge_rank=16,
        verifier_kv_layer_mapping=[],
    )
    assert cfg.kv_native_dspark.kv_bridge_enabled is True
    assert cfg.kv_native_dspark.kv_bridge_rank == 16
    assert cfg.kv_native_dspark.verifier_kv_layer_mapping == []


def test_kv_bridge_stability_fields_round_trip_through_flat_schema():
    cfg = _kv_native_config(
        kv_bridge_enabled=True,
        kv_bridge_residual_scale=0.1,
        kv_bridge_max_correction_ratio=0.5,
        kv_bridge_normalize_keys=True,
    )
    cfg.optimizer.kv_bridge_lr = 6e-5
    recovered = TrainConfig.from_flat(cfg.flatten())
    assert recovered.kv_native_dspark.kv_bridge_residual_scale == pytest.approx(0.1)
    assert recovered.kv_native_dspark.kv_bridge_max_correction_ratio == pytest.approx(
        0.5
    )
    assert recovered.kv_native_dspark.kv_bridge_normalize_keys is True
    assert recovered.optimizer.kv_bridge_lr == pytest.approx(6e-5)


def test_kv_bridge_ignores_direct_read_mapping():
    cfg = _kv_native_config(
        kv_bridge_enabled=True,
        verifier_kv_layer_mapping=[999],
    )
    assert cfg.kv_native_dspark.kv_bridge_enabled is True


def test_kv_direct_read_rejects_wrong_mapping_length():
    with pytest.raises(ValueError, match="verifier-kv-layer-mapping"):
        _kv_native_config(verifier_kv_layer_mapping=[3, 11])


def test_kv_direct_read_rejects_non_exported_layer():
    with pytest.raises(ValueError, match="non-exported layers"):
        _kv_native_config(
            verifier_kv_layer_mapping=[3, 11, 19, 27, 31, 35],
        )
