"""Unit tests for the DFly DFlash-family draft model."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dfly import DFlyDraftModel, DFlySpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.optimizers import split_named_params_for_muon

HIDDEN_SIZE = 16
VOCAB_SIZE = 64
BLOCK_SIZE = 4
_EAGER_LOSS_CONFIG = resolve_loss_config("kl_div", "eager")


def _config(
    *,
    enable_hidden_correction: bool = True,
    hidden_correction_intermediate_size: int | None = None,
    sample_from_anchor: bool = False,
    target_hidden_size: int | None = None,
) -> DFlySpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    speculative_tokens = BLOCK_SIZE if sample_from_anchor else BLOCK_SIZE - 1
    return DFlySpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        target_hidden_size=target_hidden_size,
        enable_hidden_correction=enable_hidden_correction,
        hidden_correction_intermediate_size=(hidden_correction_intermediate_size),
        speculators_config=SpeculatorsConfig(
            algorithm="dfly",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=speculative_tokens,
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )


def _initialized_model(**kwargs) -> DFlyDraftModel:
    torch.manual_seed(0)
    model = DFlyDraftModel(_config(**kwargs))
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    return model


def test_dfly_is_registered_and_round_trips_config():
    config = _config()
    restored = SpeculatorModelConfig.from_dict(config.to_dict())

    assert isinstance(restored, DFlySpeculatorConfig)
    assert SpeculatorModelConfig.registry is not None
    assert SpeculatorModelConfig.registry["dfly"] is DFlySpeculatorConfig
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dfly"] is DFlyDraftModel


def test_dfly_structure_and_zero_initialized_correction():
    model = DFlyDraftModel(_config())

    assert model.layer_fusion_weights.shape == (2, 3)
    assert model.layer_fusion_weights[0].argmax().item() == 0
    assert model.layer_fusion_weights[1].argmax().item() == 2
    assert model.hidden_correction is not None
    assert torch.count_nonzero(model.hidden_correction.down_proj.weight) == 0

    keys = set(model.state_dict())
    assert "fc.weight" in keys
    assert "layer_fusion_weights" in keys
    assert "hidden_correction.gate_proj.weight" in keys
    assert "hidden_correction.up_proj.weight" in keys
    assert "hidden_correction.down_proj.weight" in keys
    assert not any("markov_head" in key for key in keys)
    assert not any("confidence_head" in key for key in keys)


def test_hidden_correction_can_be_disabled():
    model = DFlyDraftModel(_config(enable_hidden_correction=False))

    assert model.hidden_correction is None
    assert not any("hidden_correction" in key for key in model.state_dict())


def test_hidden_correction_uses_configured_intermediate_size():
    model = DFlyDraftModel(_config(hidden_correction_intermediate_size=24))

    assert model.hidden_correction is not None
    assert model.hidden_correction.gate_proj.out_features == 24
    assert model.hidden_correction.up_proj.out_features == 24


def test_target_hidden_size_must_match_draft_hidden_size():
    with pytest.raises(ValueError, match="target_hidden_size"):
        DFlyDraftModel(_config(target_hidden_size=HIDDEN_SIZE + 8))


def test_layer_context_adds_per_layer_fusion_residual():
    model = DFlyDraftModel(_config())
    with torch.no_grad():
        model.fc.weight.zero_()

    context_features = torch.zeros(1, 2, 3, HIDDEN_SIZE)
    context_features[:, :, 0, 0] = 1.0
    context_features[:, :, 1, 1] = 1.0
    context_features[:, :, 2, 2] = 1.0
    hidden_states = context_features.flatten(start_dim=2)
    base_context = model._project_base_context(hidden_states)

    shallow_context = model._build_layer_context(
        hidden_states,
        base_context,
        0,
    )
    deep_context = model._build_layer_context(
        hidden_states,
        base_context,
        1,
    )

    assert shallow_context[0, 0].argmax().item() == 0
    assert deep_context[0, 0].argmax().item() == 2
    assert not torch.allclose(shallow_context, deep_context)


@pytest.mark.parametrize(
    ("sample_from_anchor", "expected"),
    [
        (False, [[2, 2, 3, 4, 6, 6, 7, 8]]),
        (True, [[2, 3, 4, 5, 6, 7, 8, 9]]),
    ],
)
def test_previous_token_alignment(sample_from_anchor, expected):
    model = DFlyDraftModel(_config(sample_from_anchor=sample_from_anchor))
    input_ids = torch.arange(12).unsqueeze(0)
    anchored_block_indices = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9])

    previous_ids = model._previous_token_ids(
        input_ids,
        anchored_block_indices,
    )

    assert previous_ids.tolist() == expected


def test_hidden_correction_changes_logits_after_zero_init_is_perturbed():
    model = _initialized_model()
    hidden = torch.randn(1, 2 * BLOCK_SIZE, HIDDEN_SIZE)
    input_ids = torch.arange(12).unsqueeze(0)
    anchored_block_indices = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9])

    base_logits = model.lm_head(hidden)
    initial_logits = model._compute_draft_logits(
        hidden,
        input_ids,
        anchored_block_indices,
    )
    torch.testing.assert_close(initial_logits, base_logits)

    assert model.hidden_correction is not None
    with torch.no_grad():
        model.hidden_correction.down_proj.weight.normal_()
    corrected_logits = model._compute_draft_logits(
        hidden,
        input_ids,
        anchored_block_indices,
    )

    assert not torch.allclose(corrected_logits, base_logits)


def test_dfly_full_forward_backpropagates_through_new_paths():
    model = _initialized_model().train()
    sequence_length = 24
    input_ids = torch.randint(
        1,
        VOCAB_SIZE,
        (1, sequence_length),
    )
    hidden_states = torch.randn(
        1,
        sequence_length,
        3 * HIDDEN_SIZE,
    )
    verifier_last_hidden_states = torch.randn(
        1,
        sequence_length,
        HIDDEN_SIZE,
    )
    loss_mask = torch.ones(1, sequence_length)
    document_ids = torch.zeros(
        1,
        sequence_length,
        dtype=torch.long,
    )

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last_hidden_states,
        document_ids=document_ids,
        loss_config=_EAGER_LOSS_CONFIG,
        max_anchors=3,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "eal_sum" in metrics
    assert model.layer_fusion_weights.grad is not None
    assert model.layer_fusion_weights.grad.abs().sum() > 0
    assert model.hidden_correction is not None
    assert model.hidden_correction.down_proj.weight.grad is not None
    assert model.hidden_correction.down_proj.weight.grad.abs().sum() > 0


def test_dfly_checkpoint_round_trip(tmp_path):
    model = _initialized_model()
    with torch.no_grad():
        model.layer_fusion_weights.add_(0.25)
        assert model.hidden_correction is not None
        model.hidden_correction.down_proj.weight.normal_()

    model.save_pretrained(tmp_path)
    restored = SpeculatorModel.from_pretrained(tmp_path)

    assert isinstance(restored, DFlyDraftModel)
    torch.testing.assert_close(
        restored.layer_fusion_weights,
        model.layer_fusion_weights,
    )
    assert restored.hidden_correction is not None
    torch.testing.assert_close(
        restored.hidden_correction.down_proj.weight,
        model.hidden_correction.down_proj.weight,
    )


def test_dfly_fusion_logits_use_adamw_with_muon_optimizer():
    model = DFlyDraftModel(_config())
    muon_params, adamw_params = split_named_params_for_muon(model)
    muon_names = {name for name, _ in muon_params}
    adamw_names = {name for name, _ in adamw_params}

    assert "layer_fusion_weights" in adamw_names
    assert "layer_fusion_weights" not in muon_names
