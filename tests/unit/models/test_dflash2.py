"""Unit tests for the DFlash2 draft model: config guards, weight contract, forward."""

import copy
import json
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

VOCAB_SIZE = 64
HIDDEN_SIZE = 32
NUM_TARGET_LAYERS = 2
BLOCK_SIZE = 4

TINY_QWEN3 = Qwen3Config(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)


def make_config(**overrides) -> DFlash2SpeculatorConfig:
    transformer_config = copy.deepcopy(TINY_QWEN3)
    transformer_config._attn_implementation = "eager"
    kwargs: dict = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
        "aux_hidden_state_layer_ids": list(range(NUM_TARGET_LAYERS)),
        "mask_token_id": 0,
        "conv_kernel_size": 3,
        "conv_group_size": 8,
        "selector_rank": 6,
        "selector_top_k": 4,
        "speculators_config": SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    }
    kwargs.update(overrides)
    return DFlash2SpeculatorConfig(**kwargs)


def make_model(**overrides) -> DFlash2DraftModel:
    torch.manual_seed(0)
    model = DFlash2DraftModel(config=make_config(**overrides))
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buffer in model.buffers():
            if buffer.is_floating_point() and buffer.isnan().any():
                buffer.zero_()
    return model.float()


def make_batch(total_seq_len: int = 48) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "hidden_states": torch.randn(1, total_seq_len, NUM_TARGET_LAYERS * HIDDEN_SIZE),
        "input_ids": torch.randint(1, VOCAB_SIZE, (1, total_seq_len)),
        "loss_mask": torch.ones(1, total_seq_len, dtype=torch.long),
        "verifier_last_hidden_states": torch.randn(1, total_seq_len, HIDDEN_SIZE),
        "document_ids": torch.zeros(1, total_seq_len, dtype=torch.long),
    }


CALL_KWARGS = {
    "loss_config": resolve_loss_config("ce", "eager"),
    "gamma": 4.0,
    "max_anchors": 6,
    "per_position_loss_weight": "fixed-exp-decay",
    "dpace_alpha": 0.5,
}


class TestConfigGuards:
    def test_registered_under_dflash2(self):
        assert DFlash2SpeculatorConfig().speculators_model_type == "dflash2"
        assert DFlash2DraftModel.config_class is DFlash2SpeculatorConfig

    def test_inherits_dflash_fields(self):
        config = make_config()
        assert not config.sample_from_anchor
        assert config.target_vocab_size == VOCAB_SIZE

    def test_pruned_draft_vocab_is_rejected(self):
        with pytest.raises(ValueError, match="requires the full vocabulary"):
            DFlash2DraftModel(config=make_config(draft_vocab_size=VOCAB_SIZE // 2))

    def test_sample_from_anchor_is_rejected(self):
        with pytest.raises(ValueError, match="sample_from_anchor=False"):
            DFlash2DraftModel(config=make_config(sample_from_anchor=True))

    def test_conv_kernel_larger_than_block_is_rejected(self):
        with pytest.raises(ValueError, match="must not exceed"):
            DFlash2DraftModel(config=make_config(conv_kernel_size=BLOCK_SIZE + 1))

    @pytest.mark.parametrize("cap", [0.0, -1.0])
    def test_non_positive_softcap_is_rejected(self, cap):
        """Inference reads a non-positive cap as "disabled"; training would divide
        by it. Reject rather than train something that serves differently."""
        with pytest.raises(ValidationError):
            make_config(final_logit_softcapping=cap)

    def test_config_round_trips_through_serialization(self):
        config = make_config(
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=3,
            selector_top_k=5,
            input_embedding_scale=2.0,
            output_multiplier=0.5,
            final_logit_softcapping=30.0,
        )
        restored = DFlash2SpeculatorConfig.model_validate(config.model_dump())
        assert restored.conv_kernel_size == 2
        assert restored.conv_group_size == 4
        assert restored.selector_rank == 3
        assert restored.selector_top_k == 5
        assert restored.input_embedding_scale == 2.0
        assert restored.output_multiplier == 0.5
        assert restored.final_logit_softcapping == 30.0


class TestWeightContract:
    """The exported state dict is the interface to vLLM's DFlash2 model.

    ``DFlashQwen3ForCausalLM.load_weights`` prefixes every key except ``lm_head``
    and ``d2t`` with ``model.``, so these names must land on
    ``DFlash2Qwen3Model.layers[i].{attention,mlp}_conv`` and
    ``DFlash2Qwen3Model.candidate_selector`` exactly.
    """

    def test_new_keys_are_exactly_the_vllm_ones(self):
        model = make_model()
        expected = set()
        for layer_idx in range(TINY_QWEN3.num_hidden_layers):
            for conv in ("attention_conv", "mlp_conv"):
                expected.add(f"layers.{layer_idx}.{conv}.base_kernel")
                expected.add(f"layers.{layer_idx}.{conv}.kernel_projection.weight")
        expected |= {
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "candidate_selector.hidden_projection.weight",
        }
        actual = {
            key
            for key in model.state_dict()
            if "_conv." in key or key.startswith("candidate_selector.")
        }
        assert actual == expected

    def test_new_weight_shapes(self):
        config = make_config()
        model = make_model()
        state = model.state_dict()
        num_groups = HIDDEN_SIZE // config.conv_group_size
        assert state["layers.0.attention_conv.base_kernel"].shape == (
            2,
            config.conv_kernel_size,
            HIDDEN_SIZE,
        )
        assert state["layers.0.attention_conv.kernel_projection.weight"].shape == (
            2 * config.conv_kernel_size * num_groups,
            HIDDEN_SIZE,
        )
        assert state["candidate_selector.predecessor_codebook"].shape == (
            VOCAB_SIZE,
            config.selector_rank,
        )
        assert state["candidate_selector.successor_codebook"].shape == (
            VOCAB_SIZE,
            config.selector_rank,
        )
        assert state["candidate_selector.hidden_projection.weight"].shape == (
            config.selector_rank,
            HIDDEN_SIZE,
        )

    def test_everything_else_matches_dflash(self):
        """Only the conv and the selector are new; the backbone keys are DFlash's."""
        dflash_config_dict = make_config().model_dump()
        dflash_config_dict["speculators_model_type"] = "dflash"
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "input_embedding_scale",
            "output_multiplier",
            "final_logit_softcapping",
        ):
            dflash_config_dict.pop(key)
        dflash_config_dict["architectures"] = ["DFlashSpeculator"]
        transformer_config = copy.deepcopy(TINY_QWEN3)
        transformer_config._attn_implementation = "eager"
        dflash_config_dict["transformer_layer_config"] = transformer_config

        dflash_model = DFlashDraftModel(
            config=DFlashSpeculatorConfig(**dflash_config_dict)
        )
        new_keys = set(make_model().state_dict()) - set(dflash_model.state_dict())
        assert all(
            "_conv." in key or key.startswith("candidate_selector.") for key in new_keys
        )
        assert not set(dflash_model.state_dict()) - set(make_model().state_dict())

    def test_selector_and_conv_are_trainable(self):
        model = make_model()
        for name, param in model.named_parameters():
            if "_conv." in name or name.startswith("candidate_selector."):
                assert param.requires_grad, name

    def test_checkpoint_round_trips(self, tmp_path):
        """``--from-pretrained`` must recover the conv and the selector exactly.

        This is what would break silently if the registry entry or the
        ``_is_hf_initialized`` marking regressed: ``from_pretrained`` would resolve
        the wrong class, or ``post_init`` would overwrite loaded weights.
        """
        model = make_model(
            input_embedding_scale=1.5,
            output_multiplier=0.5,
            final_logit_softcapping=20.0,
        )
        with torch.no_grad():
            # Move the new modules off their initialization so a silent re-init shows.
            model.candidate_selector.successor_codebook.normal_(std=0.3)
            for layer in cast("Any", model.layers):
                layer.attention_conv.kernel_projection.weight.normal_(std=0.1)
                layer.mlp_conv.base_kernel.normal_(std=0.1)
        before = {key: value.clone() for key, value in model.state_dict().items()}

        model.save_pretrained(str(tmp_path))
        restored = SpeculatorModel.from_pretrained(str(tmp_path))

        assert isinstance(restored, DFlash2DraftModel)
        assert restored.config.speculators_model_type == "dflash2"
        assert restored.input_embedding_scale == 1.5
        assert restored.output_multiplier == 0.5
        assert restored.final_logit_softcapping == 20.0
        after = restored.state_dict()
        assert set(after) == set(before)
        for key, value in before.items():
            if key in ("verifier_lm_head.weight", "verifier_norm.weight"):
                continue  # reloaded from the verifier, excluded on save
            assert torch.equal(value, after[key]), key

    def test_saved_config_is_native_vllm_dflash2_and_resumable(self, tmp_path):
        model = make_model(
            input_embedding_scale=1.5,
            output_multiplier=0.5,
            final_logit_softcapping=20.0,
            sliding_window_non_causal=True,
        )
        model.save_pretrained(str(tmp_path))

        raw = json.loads((tmp_path / "config.json").read_text())
        assert raw["model_type"] == "qwen3"
        assert raw["architectures"] == ["DFlash2DraftModel"]
        assert raw["eagle_aux_hidden_state_layer_ids"] == list(range(NUM_TARGET_LAYERS))
        assert raw["speculators_model_type"] == "dflash2"
        assert raw["speculators_config"]["algorithm"] == "dflash2"
        assert raw["transformer_layer_config"]["model_type"] == "qwen3"
        assert "auto_map" not in raw

        runtime = raw["dflash_config"]
        assert runtime["mask_token_id"] == 0
        assert runtime["target_layer_ids"] == [-1, 0]
        assert runtime["sample_from_anchor"] is False
        assert runtime["block_size"] == BLOCK_SIZE
        assert runtime["conv_kernel_size"] == 3
        assert runtime["conv_group_size"] == 8
        assert runtime["selector_rank"] == 6
        assert runtime["selector_top_k"] == 4
        assert runtime["input_embedding_scale"] == 1.5
        assert runtime["output_multiplier"] == 0.5
        assert runtime["final_logit_softcapping"] == 20.0
        assert runtime["causal"] is False

        # The same hybrid config must still recover the training-side subclass.
        restored = DFlash2SpeculatorConfig.from_pretrained(str(tmp_path))
        assert restored.speculators_model_type == "dflash2"
        assert restored.speculators_config.algorithm == "dflash2"
        assert restored.transformer_layer_config.model_type == "qwen3"

    def test_vllm_reads_saved_config_without_a_plugin(self, tmp_path, monkeypatch):
        config_module = pytest.importorskip("vllm.transformers_utils.config")

        model = make_model()
        model.save_pretrained(str(tmp_path))
        monkeypatch.setenv("VLLM_PLUGINS", "")

        runtime = config_module.get_config(str(tmp_path), trust_remote_code=False)
        assert runtime.model_type == "qwen3"
        assert runtime.architectures == ["DFlash2DraftModel"]
        assert runtime.dflash_config["selector_top_k"] == 4


class TestForward:
    def test_returns_finite_loss_and_metrics(self):
        model = make_model()
        _draft_tokens, loss, metrics = model(**make_batch(), **CALL_KWARGS)
        assert loss.isfinite()
        assert loss.requires_grad
        for key, value in metrics.items():
            assert isinstance(value, torch.Tensor), key
            assert value.isfinite().all(), key

    def test_reports_the_selector_diagnostics(self):
        model = make_model()
        _draft_tokens, _loss, metrics = model(**make_batch(), **CALL_KWARGS)
        for prefix in ("selector", "unary"):
            assert f"{prefix}_eal_sum" in metrics
            assert f"{prefix}_accept_len_sum" in metrics
            for pos in range(1, BLOCK_SIZE):
                assert f"{prefix}_position_{pos}_acc_sum" in metrics
        assert "candidate_recall_sum" in metrics
        assert metrics["candidate_recall_total"] > 0

    def test_at_initialization_the_selector_walk_equals_the_top1_walk(self):
        """The zero-initialized selector must not change any decision."""
        model = make_model()
        _draft_tokens, _loss, metrics = model(**make_batch(), **CALL_KWARGS)
        for pos in range(1, BLOCK_SIZE):
            assert (
                metrics[f"selector_position_{pos}_acc_sum"]
                == metrics[f"unary_position_{pos}_acc_sum"]
            )

    def test_the_total_is_the_two_reported_terms(self):
        """``loss = unary_loss + selector_loss_weight * selector_loss``.

        Both terms are reported unweighted, so a run can tell which one is moving.
        """
        model = make_model()
        torch.manual_seed(3)  # select_anchors draws the anchor positions
        _draft_tokens, total, metrics = model(**make_batch(), **CALL_KWARGS)
        torch.testing.assert_close(
            total, metrics["unary_loss_sum"] + metrics["selector_loss_sum"]
        )
        assert metrics["selector_loss_sum"] > 0
        torch.testing.assert_close(metrics["loss_sum"], total.detach())

    def test_weight_zero_leaves_dflashs_loss_on_the_unary_logits(self):
        """The selector term is additive, so 0 recovers DFlash's own objective --
        which is also what the candidate set is trained by."""
        model = make_model()
        torch.manual_seed(3)
        _draft_tokens, total, metrics = model(**make_batch(), **CALL_KWARGS)
        torch.manual_seed(3)
        _draft_tokens, unary_only, unweighted = model(
            **make_batch(), **CALL_KWARGS, selector_loss_weight=0.0
        )
        torch.testing.assert_close(unary_only, metrics["unary_loss_sum"])
        assert total > unary_only
        # Still measured and reported, just not optimized.
        assert unweighted["selector_loss_sum"] > 0

    def test_gradients_reach_the_conv_and_the_selector(self):
        model = make_model()
        _draft_tokens, loss, metrics = model(**make_batch(), **CALL_KWARGS)
        # The selector term only covers slots whose target is a candidate, so a
        # fixture with no such slot would starve the codebooks below.
        assert metrics["candidate_recall_sum"] > 0
        loss.backward()
        for name in (
            "layers.0.attention_conv.base_kernel",
            "layers.0.attention_conv.kernel_projection.weight",
            "layers.0.mlp_conv.base_kernel",
            "layers.0.mlp_conv.kernel_projection.weight",
            "candidate_selector.successor_codebook",
        ):
            param = dict(model.named_parameters())[name]
            assert param.grad is not None, name
            assert param.grad.abs().sum() > 0, name

    def test_selector_gradients_unblock_once_the_successor_side_is_nonzero(self):
        """The zero init costs exactly one step, LoRA-style.

        The bias is a product of three factors, so with ``successor_codebook = 0``
        only that factor gets gradient at step 0; the other two start moving as
        soon as it is nonzero.
        """
        model = make_model()
        _draft_tokens, loss, _metrics = model(**make_batch(), **CALL_KWARGS)
        loss.backward()
        blocked = ("predecessor_codebook", "hidden_projection.weight")
        for name in blocked:
            param = dict(model.named_parameters())[f"candidate_selector.{name}"]
            assert param.grad is not None, name
            assert param.grad.abs().sum() == 0, name

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.candidate_selector.successor_codebook.normal_(std=0.02)
        _draft_tokens, loss, _metrics = model(**make_batch(), **CALL_KWARGS)
        loss.backward()
        for name in blocked:
            param = dict(model.named_parameters())[f"candidate_selector.{name}"]
            assert param.grad is not None, name
            assert param.grad.abs().sum() > 0, name

    def test_dpace_weighting_runs(self):
        model = make_model()
        kwargs = {**CALL_KWARGS, "per_position_loss_weight": "dpace"}
        _draft_tokens, loss, _metrics = model(**make_batch(), **kwargs)
        assert loss.isfinite()

    def test_output_multiplier_and_softcap_are_applied(self):
        batch = make_batch()
        plain = make_model()
        _t, plain_loss, _m = plain(**batch, **CALL_KWARGS)
        capped = make_model(output_multiplier=0.5, final_logit_softcapping=5.0)
        _t, capped_loss, _m = capped(**batch, **CALL_KWARGS)
        assert not torch.allclose(plain_loss, capped_loss)

    def test_input_embedding_scale_changes_the_forward(self):
        batch = make_batch()
        _t, plain_loss, _m = make_model()(**batch, **CALL_KWARGS)
        _t, scaled_loss, _m = make_model(input_embedding_scale=4.0)(
            **batch, **CALL_KWARGS
        )
        assert not torch.allclose(plain_loss, scaled_loss)


class TestTrainerIntegration:
    def test_get_trainer_kwargs_mirrors_dflash(self):
        train_kwargs, val_kwargs = DFlash2DraftModel.get_trainer_kwargs(
            loss_fn="ce",
            loss_implementation="eager",
            dflash_decay_gamma=3.0,
            max_anchors=128,
            per_position_loss_weight="dpace",
            dpace_alpha=0.25,
        )
        assert train_kwargs == val_kwargs
        assert train_kwargs["gamma"] == 3.0
        assert train_kwargs["max_anchors"] == 128
        assert train_kwargs["per_position_loss_weight"] == "dpace"
        assert train_kwargs["dpace_alpha"] == 0.25
        assert set(train_kwargs["loss_config"]) == {"ce"}

    def test_verify_training_compatible(self):
        SpeculatorModel.verify_training_compatible(make_model())
