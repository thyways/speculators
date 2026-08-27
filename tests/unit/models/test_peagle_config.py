"""P-EAGLE checkpoint configuration tests."""

import json

import pytest
from transformers import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def make_config() -> PEagleSpeculatorConfig:
    transformer_config = LlamaConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
            "partial_rotary_factor": 1.0,
        },
    )
    return PEagleSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        eagle_aux_hidden_state_layer_ids=[1, 2, 3],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="peagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3_5MoeForCausalLM"],
            ),
        ),
    )


def test_serialized_draft_uses_linear_rope_without_mutating_training_config():
    config = make_config()

    serialized = config.to_dict()
    runtime_rope = serialized["transformer_layer_config"]["rope_parameters"]

    assert "mrope_section" not in runtime_rope
    assert "mrope_interleaved" not in runtime_rope
    assert runtime_rope["rope_theta"] == 1_000_000.0
    assert runtime_rope["partial_rotary_factor"] == 1.0
    assert "mrope_section" in config.transformer_layer_config.rope_parameters


def test_saved_config_is_resumable_and_keeps_speculators_metadata(tmp_path):
    config = make_config()
    config.save_pretrained(tmp_path)

    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["speculators_model_type"] == "peagle"
    assert raw["speculators_config"]["algorithm"] == "peagle"
    assert raw["transformer_layer_config"]["model_type"] == "llama"
    assert "mrope_section" not in raw["transformer_layer_config"]["rope_parameters"]

    restored = PEagleSpeculatorConfig.from_pretrained(tmp_path)
    assert restored.speculators_model_type == "peagle"
    assert restored.mask_token_id == 0
    assert restored.eagle_aux_hidden_state_layer_ids == [1, 2, 3]
    assert "mrope_section" not in restored.transformer_layer_config.rope_parameters


def test_vllm_reads_saved_config_without_a_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "")
    config_module = pytest.importorskip("vllm.transformers_utils.config")
    algos = pytest.importorskip("vllm.transformers_utils.configs.speculators.algos")
    speculators_base = pytest.importorskip(
        "vllm.transformers_utils.configs.speculators.base"
    )

    make_config().save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    runtime = config_module.get_config(str(tmp_path), trust_remote_code=False)
    speculative = speculators_base.SpeculatorsConfig.extract_vllm_speculative_config(
        raw
    )

    assert type(runtime).__name__ == "SpeculatorsConfig"
    assert runtime.architectures == ["PeagleLlamaForCausalLM"]
    assert runtime.pard_token == 0
    assert "mrope_section" not in runtime.rope_parameters
    assert not config_module.uses_mrope(runtime)
    assert speculative["method"] == "eagle3"
    assert speculative["parallel_drafting"] is True
    assert speculative["num_speculative_tokens"] == 7
    assert algos.SUPPORTED_SPECULATORS_TYPES["peagle"] is algos.update_peagle
