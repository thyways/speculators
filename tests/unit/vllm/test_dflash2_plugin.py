import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from speculators.vllm._dflash_family import (
    _SPECULATOR_FACTORIES,
    map_speculative_method,
    register_speculative_method_alias,
    register_speculator_factory,
)
from speculators.vllm.dflash2 import (
    _check_block_size,
    _check_v2_model_runner,
    _is_dflash2_draft,
    _make_dflash2_speculator,
    _update_dflash2,
)


@pytest.fixture
def upstream_dflash_updater(monkeypatch):
    """Stand in for the vLLM dflash updater that ``_update_dflash2`` builds on.

    DFlash2 translates a config by delegating to whatever is registered for
    ``dflash`` and then adding its own keys, so the delegation target is the one
    piece of vLLM this module still needs. Stubbing it keeps the DFlash2-specific
    half of the translation testable without vLLM installed, the same way the
    DFly and Domino plugin tests run.
    """

    def update_dflash(
        config_dict: dict[str, Any],
        pre_trained_config: dict[str, Any],
    ) -> None:
        pre_trained_config["architectures"] = ["DFlashDraftModel"]
        pre_trained_config["dflash_config"] = {
            "mask_token_id": config_dict["mask_token_id"],
            "target_layer_ids": [1, 6, 11],
        }

    supported = {"dflash": update_dflash}
    leaf_name = "vllm.transformers_utils.configs.speculators.algos"
    parents = [
        "vllm",
        "vllm.transformers_utils",
        "vllm.transformers_utils.configs",
        "vllm.transformers_utils.configs.speculators",
    ]
    modules = {name: ModuleType(name) for name in parents}
    leaf = ModuleType(leaf_name)
    leaf.SUPPORTED_SPECULATORS_TYPES = supported  # type: ignore[attr-defined]
    modules[leaf_name] = leaf
    # Bind each child onto its parent so the import machinery can walk the chain.
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return supported


def _dflash2_source() -> dict[str, Any]:
    return {
        "mask_token_id": 127,
        "conv_kernel_size": 3,
        "conv_group_size": 64,
        "selector_rank": 256,
        "selector_top_k": 16,
        "block_size": 16,
    }


def test_dflash2_config_is_mapped_to_the_dflash_runtime():
    register_speculative_method_alias("dflash2", "dflash")
    mapped = map_speculative_method({"method": "dflash2", "num_speculative_tokens": 15})
    assert mapped == {
        "method": "dflash",
        "num_speculative_tokens": 15,
    }


def test_dflash2_alias_leaves_the_other_family_methods_alone():
    register_speculative_method_alias("dflash2", "dflash")
    register_speculative_method_alias("dfly", "dspark")
    assert map_speculative_method({"method": "dfly"})["method"] == "dspark"
    assert map_speculative_method({"method": "dflash"})["method"] == "dflash"


def test_dflash2_translation_carries_the_trained_module_shapes(
    upstream_dflash_updater,
):
    source = _dflash2_source()
    source["input_embedding_scale"] = 2.0
    translated: dict[str, Any] = {}

    _update_dflash2(source, translated)

    assert translated["architectures"] == ["DFlash2DraftModel"]
    draft_config = translated["dflash_config"]
    # The upstream updater's own keys survive alongside the DFlash2 additions.
    assert draft_config["mask_token_id"] == 127
    assert draft_config["target_layer_ids"] == [1, 6, 11]
    assert draft_config["conv_kernel_size"] == 3
    assert draft_config["conv_group_size"] == 64
    assert draft_config["selector_rank"] == 256
    assert draft_config["selector_top_k"] == 16
    assert draft_config["input_embedding_scale"] == 2.0
    # Carried so the runtime can compare the served block against the trained one.
    assert draft_config["block_size"] == 16


def test_dflash2_translation_omits_unset_optional_keys(upstream_dflash_updater):
    translated: dict[str, Any] = {}
    _update_dflash2(_dflash2_source(), translated)
    assert "final_logit_softcapping" not in translated["dflash_config"]
    assert "output_multiplier" not in translated["dflash_config"]


@pytest.mark.parametrize(
    "missing",
    ["conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"],
)
def test_dflash2_translation_rejects_a_missing_module_shape(
    missing,
    upstream_dflash_updater,
):
    source = _dflash2_source()
    del source[missing]
    with pytest.raises(ValueError, match=missing):
        _update_dflash2(source, {})


def test_dflash2_translation_rejects_anchor_sampling(upstream_dflash_updater):
    source = _dflash2_source()
    source["sample_from_anchor"] = True
    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        _update_dflash2(source, {})


def test_dflash2_translation_forces_non_causal_sliding_windows(
    upstream_dflash_updater,
):
    source = _dflash2_source()
    source["sliding_window_non_causal"] = True
    translated: dict[str, Any] = {}
    _update_dflash2(source, translated)
    assert translated["dflash_config"]["causal"] is False


def test_dflash2_translation_leaves_the_causal_override_off_by_default(
    upstream_dflash_updater,
):
    source = _dflash2_source()
    source["sliding_window_non_causal"] = False
    translated: dict[str, Any] = {}
    _update_dflash2(source, translated)
    assert "causal" not in translated["dflash_config"]


def _speculative_config(
    method: str = "dflash",
    architectures: list[str] | None = None,
    num_speculative_tokens: int = 15,
    block_size: int | None = 16,
) -> Any:
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config=SimpleNamespace(
            architectures=(
                ["DFlash2DraftModel"] if architectures is None else architectures
            ),
            hf_config=SimpleNamespace(dflash_config={"block_size": block_size}),
        ),
    )


def test_a_dflash2_draft_is_recognized_by_its_architecture():
    assert _is_dflash2_draft(_speculative_config()) is True


@pytest.mark.parametrize(
    "config",
    [
        None,
        _speculative_config(method="eagle3"),
        _speculative_config(architectures=["DFlashDraftModel"]),
    ],
)
def test_other_drafts_are_not_mistaken_for_dflash2(config):
    assert _is_dflash2_draft(config) is False


def test_the_served_block_must_be_the_trained_one():
    # A DFlash2 checkpoint trained at block_size=16 must serve 15 draft tokens.
    _check_block_size(_speculative_config(num_speculative_tokens=15))
    with pytest.raises(ValueError, match="block_size=16"):
        _check_block_size(_speculative_config(num_speculative_tokens=7))


def test_a_checkpoint_without_a_block_size_is_left_to_vllm():
    _check_block_size(_speculative_config(num_speculative_tokens=7, block_size=None))


def test_the_v1_model_runner_is_refused():
    _check_v2_model_runner(SimpleNamespace(use_v2_model_runner=True))
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        _check_v2_model_runner(SimpleNamespace(use_v2_model_runner=False))


def test_the_factory_declines_a_draft_it_does_not_own():
    # Returning None rather than raising is what lets every other algorithm's
    # draft fall through to vLLM's own speculator.
    vllm_config = SimpleNamespace(
        speculative_config=_speculative_config(method="eagle3")
    )
    assert _make_dflash2_speculator(vllm_config, "cpu") is None


def test_registering_the_factory_twice_installs_it_once():
    before = list(_SPECULATOR_FACTORIES)
    register_speculator_factory(_make_dflash2_speculator)
    register_speculator_factory(_make_dflash2_speculator)
    try:
        assert _SPECULATOR_FACTORIES.count(_make_dflash2_speculator) == 1
    finally:
        _SPECULATOR_FACTORIES[:] = before
