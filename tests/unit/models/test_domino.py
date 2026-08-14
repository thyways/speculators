"""Unit tests for the Domino DFlash-family draft model."""

import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.domino import (
    DominoDraftModel,
    DominoSpeculatorConfig,
    linear_lambda_base,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.optimizers import split_named_params_for_muon

HIDDEN_SIZE = 16
VOCAB_SIZE = 64
BLOCK_SIZE = 6
GRU_HIDDEN = 8
EMB_DIM = 5
_EAGER_LOSS_CONFIG = resolve_loss_config("kl_div", "eager")


def _config(
    *,
    sample_from_anchor: bool = True,
    pure_draft_prefix_len: int = 1,
    block_size: int = BLOCK_SIZE,
    lambda_base_start: float = 1.0,
    lambda_base_decay_ratio: float = 0.5,
) -> DominoSpeculatorConfig:
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
    speculative_tokens = block_size if sample_from_anchor else block_size - 1
    return DominoSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=VOCAB_SIZE,
        block_size=block_size,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        gru_hidden_dim=GRU_HIDDEN,
        logits_correction_emb_dim=EMB_DIM,
        pure_draft_prefix_len=pure_draft_prefix_len,
        lambda_base_start=lambda_base_start,
        lambda_base_decay_ratio=lambda_base_decay_ratio,
        speculators_config=SpeculatorsConfig(
            algorithm="domino",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )


def _initialized_model(**kwargs) -> DominoDraftModel:
    torch.manual_seed(0)
    model = DominoDraftModel(_config(**kwargs))
    nn.init.normal_(model.embed_tokens.weight)
    nn.init.normal_(model.lm_head.weight)
    nn.init.normal_(model.verifier_lm_head.weight)
    return model


def _batch(sequence_length: int = 24, *, loss_config=None) -> dict:
    return {
        "hidden_states": torch.randn(1, sequence_length, 3 * HIDDEN_SIZE),
        "input_ids": torch.randint(1, VOCAB_SIZE, (1, sequence_length)),
        "loss_mask": torch.ones(1, sequence_length),
        "verifier_last_hidden_states": torch.randn(1, sequence_length, HIDDEN_SIZE),
        "document_ids": torch.zeros(1, sequence_length, dtype=torch.long),
        "loss_config": loss_config or _EAGER_LOSS_CONFIG,
    }


def test_domino_is_registered_and_round_trips_config():
    config = _config()
    restored = SpeculatorModelConfig.from_dict(config.to_dict())

    assert isinstance(restored, DominoSpeculatorConfig)
    assert SpeculatorModelConfig.registry is not None
    assert SpeculatorModelConfig.registry["domino"] is DominoSpeculatorConfig
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["domino"] is DominoDraftModel


def test_domino_structure_and_upstream_compatible_weight_names():
    model = DominoDraftModel(_config())
    keys = set(model.state_dict())

    # These names are what the converter renames upstream checkpoints onto.
    assert "logits_correction.prefix_gru.weight_ih_l0" in keys
    assert "logits_correction.prefix_gru.weight_hh_l0" in keys
    assert "logits_correction.embed_proj.0.weight" in keys
    assert "logits_correction.embed_proj.2.weight" in keys
    assert not any("markov_head" in key for key in keys)
    assert not any("confidence_head" in key for key in keys)
    # Schedule state must not leak into checkpoints.
    assert "lambda_base" not in keys

    assert model.logits_correction.embed_proj[0].out_features == EMB_DIM  # type: ignore[index]
    assert model.logits_correction.output_projection.out_features == VOCAB_SIZE
    # SpecForge uses the standard nn.Linear initialization for both projections.
    assert torch.count_nonzero(model.logits_correction.output_projection.weight) > 0


@pytest.mark.parametrize(
    ("sample_from_anchor", "pure_draft_prefix_len", "expected"),
    [(True, 0, 0), (True, 1, 1), (False, 0, 1), (False, 2, 3)],
)
def test_suffix_start_maps_the_anchor_convention(
    sample_from_anchor,
    pure_draft_prefix_len,
    expected,
):
    config = _config(
        sample_from_anchor=sample_from_anchor,
        pure_draft_prefix_len=pure_draft_prefix_len,
    )

    assert config.suffix_start == expected
    assert DominoDraftModel(config).suffix_start == expected


def test_correction_needs_at_least_one_slot():
    with pytest.raises(ValueError, match="at least one corrected slot"):
        DominoDraftModel(_config(pure_draft_prefix_len=BLOCK_SIZE))


def test_fresh_head_changes_the_corrected_suffix_only():
    model = _initialized_model()
    hidden = torch.randn(1, 2 * BLOCK_SIZE, HIDDEN_SIZE)
    base_logits = model.lm_head(hidden)
    block_tokens = torch.randint(1, VOCAB_SIZE, (2, BLOCK_SIZE))

    final_logits = model._correct_suffix_logits(hidden, base_logits, block_tokens, 2)
    torch.testing.assert_close(
        final_logits.view(2, BLOCK_SIZE, -1)[:, : model.suffix_start],
        base_logits.view(2, BLOCK_SIZE, -1)[:, : model.suffix_start],
    )
    assert not torch.allclose(
        final_logits.view(2, BLOCK_SIZE, -1)[:, model.suffix_start :],
        base_logits.view(2, BLOCK_SIZE, -1)[:, model.suffix_start :],
    )


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_correction_is_causal_within_the_block(sample_from_anchor):
    """Slot k may depend on the tokens before its label, and nothing later.

    With ``sample_from_anchor`` the label of slot k is token p+k+1 and its state
    may cover p+0..p+k; without it the label is p+k and the state may cover
    p+0..p+k-1. Anything beyond that is label leakage.
    """
    model = _initialized_model(sample_from_anchor=sample_from_anchor)
    with torch.no_grad():
        model.logits_correction.output_projection.weight.normal_()
    anchor_offset = 0 if sample_from_anchor else 1

    hidden = torch.randn(1, BLOCK_SIZE, HIDDEN_SIZE)
    base_logits = model.lm_head(hidden)
    block_tokens = torch.arange(1, BLOCK_SIZE + 1).view(1, BLOCK_SIZE)
    with torch.no_grad():
        reference = model._correct_suffix_logits(hidden, base_logits, block_tokens, 1)

    # Uncorrected prefix stays exactly at the DFlash logits.
    torch.testing.assert_close(
        reference[0, : model.suffix_start],
        base_logits[0, : model.suffix_start],
    )

    for position in range(BLOCK_SIZE):
        perturbed = block_tokens.clone()
        perturbed[0, position] = VOCAB_SIZE - 1
        with torch.no_grad():
            actual = model._correct_suffix_logits(hidden, base_logits, perturbed, 1)
        changed = [
            slot
            for slot in range(BLOCK_SIZE)
            if not torch.allclose(reference[0, slot], actual[0, slot], atol=1e-7)
        ]
        expected = [
            slot
            for slot in range(model.suffix_start, BLOCK_SIZE)
            if slot - anchor_offset >= position
        ]
        assert changed == expected, f"token {position} affected slots {changed}"


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_correction_matches_the_upstream_logit_head(sample_from_anchor):
    """Reproduce upstream Domino's ``apply_logits_head`` and compare in float64.

    Upstream runs the block through ``nn.GRU`` on 4-D tensors and slices the
    states differently per anchor mode; this port folds both into one path, so
    the equivalence is asserted rather than assumed.
    """
    model = _initialized_model(sample_from_anchor=sample_from_anchor).double()
    with torch.no_grad():
        model.logits_correction.output_projection.weight.normal_()

    upstream_gru = nn.GRU(
        HIDDEN_SIZE,
        GRU_HIDDEN,
        num_layers=1,
        batch_first=True,
        bias=False,
    ).double()
    # Loading straight across also re-asserts that the key names match.
    upstream_gru.load_state_dict(model.logits_correction.prefix_gru.state_dict())
    embed_proj = model.logits_correction.embed_proj
    suffix_start = model.suffix_start

    def upstream_apply_logits_head(base_logits, prev_embeds, hidden):
        # Verbatim port of DominoDraftModel.apply_logits_head (4-D path).
        num_batch, num_blocks, block_size = base_logits.shape[:3]
        if sample_from_anchor:  # upstream: shift_label
            flat = prev_embeds.reshape(num_batch * num_blocks, block_size, -1)
            states = upstream_gru(flat)[0].reshape(
                num_batch, num_blocks, block_size, -1
            )
            prefix_states = states[:, :, suffix_start:, :]
        else:
            flat = prev_embeds[:, :, : block_size - 1, :].reshape(
                num_batch * num_blocks, block_size - 1, -1
            )
            states = upstream_gru(flat)[0].reshape(
                num_batch, num_blocks, block_size - 1, -1
            )
            prefix_states = states[:, :, suffix_start - 1 :, :]
        correction = embed_proj(
            torch.cat([hidden[:, :, suffix_start:, :], prefix_states], dim=-1)
        )
        return torch.cat(
            [
                base_logits[:, :, :suffix_start, :],
                base_logits[:, :, suffix_start:, :] + correction,
            ],
            dim=2,
        )

    num_blocks = 3
    hidden = torch.randn(1, num_blocks * BLOCK_SIZE, HIDDEN_SIZE, dtype=torch.float64)
    base_logits = model.lm_head(hidden)
    block_tokens = torch.randint(1, VOCAB_SIZE, (num_blocks, BLOCK_SIZE))

    with torch.no_grad():
        actual = model._correct_suffix_logits(
            hidden, base_logits, block_tokens, num_blocks
        ).view(num_blocks, BLOCK_SIZE, VOCAB_SIZE)
        expected = upstream_apply_logits_head(
            base_logits.view(1, num_blocks, BLOCK_SIZE, VOCAB_SIZE),
            model.embed_tokens(block_tokens).unsqueeze(0),
            hidden.view(1, num_blocks, BLOCK_SIZE, HIDDEN_SIZE),
        )[0]

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_forward_without_a_schedule_optimizes_the_corrected_objective_only():
    model = _initialized_model().train()
    with torch.no_grad():
        model.logits_correction.output_projection.weight.normal_()

    _, loss, metrics = model(**_batch(), max_anchors=3)

    assert torch.isfinite(loss)
    # No on_training_step call yet, so lambda_base is 0 and the base term --
    # a second pass over the full vocab -- is skipped entirely.
    assert float(metrics["lambda_base"]) == 0.0
    assert not any(key.startswith("base_") for key in metrics)
    torch.testing.assert_close(metrics["loss_sum"], metrics["final_loss_sum"])
    assert "eal_sum" in metrics


def test_forward_blends_the_base_objective_while_lambda_is_live():
    model = _initialized_model().train()
    with torch.no_grad():
        model.logits_correction.output_projection.weight.normal_()
    model.on_training_step(25, 100)  # lambda_base_start=1.0, decay_ratio=0.5

    assert float(model.lambda_base) == pytest.approx(0.5)

    _, loss, metrics = model(**_batch(), max_anchors=3)

    assert float(metrics["lambda_base"]) == pytest.approx(0.5)
    assert {"base_loss_sum", "base_full_acc_sum", "base_eal_sum"} <= set(metrics)
    expected = 0.5 * metrics["final_loss_sum"] + 0.5 * metrics["base_loss_sum"]
    torch.testing.assert_close(metrics["loss_sum"], expected)
    # The reported headline loss must be the blend that is actually optimized.
    torch.testing.assert_close(metrics["loss_sum"], loss.detach())


def test_multi_term_losses_do_not_collide_with_the_base_metrics():
    """A compound loss makes compound_loss emit per-term keys for both passes."""
    model = _initialized_model().train()
    model.on_training_step(25, 100)  # lambda_base = 0.5, both terms live

    _, _, metrics = model(
        **_batch(loss_config=resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")),
        max_anchors=3,
    )

    # The per-term keys belong to the final pass; the base pass contributes only
    # its three renamed keys, so nothing is silently overwritten.
    assert {"ce_loss_sum", "tv_loss_sum"} <= set(metrics)
    assert {"base_loss_sum", "base_full_acc_sum", "base_eal_sum"} <= set(metrics)
    assert not any(key.startswith(("base_ce", "base_tv")) for key in metrics)
    expected = 0.5 * metrics["final_loss_sum"] + 0.5 * metrics["base_loss_sum"]
    torch.testing.assert_close(metrics["loss_sum"], expected)


def test_metrics_never_alias_the_lambda_buffer():
    """The trainer all-reduces metric tensors in place."""
    model = _initialized_model().train()
    model.on_training_step(10, 100)

    _, _, metrics = model(**_batch(), max_anchors=3)
    metrics["lambda_base"].mul_(8.0)

    assert float(model.lambda_base) == pytest.approx(0.8)


def test_gradients_reach_the_correction_head():
    model = _initialized_model().train()
    model.on_training_step(40, 100)  # lambda_base = 0.2, both terms live

    _, loss, _ = model(**_batch(), max_anchors=3)
    loss.backward()

    gru = model.logits_correction.prefix_gru
    assert gru.weight_ih_l0.grad is not None
    assert gru.weight_ih_l0.grad.abs().sum() > 0
    assert gru.weight_hh_l0.grad.abs().sum() > 0
    input_grad = model.logits_correction.embed_proj[0].weight.grad  # type: ignore[index]
    assert input_grad is not None
    assert input_grad.abs().sum() > 0
    output_grad = model.logits_correction.output_projection.weight.grad
    assert output_grad is not None
    assert output_grad.abs().sum() > 0


def test_correction_head_still_receives_a_gradient_at_lambda_one():
    """DDP's reducer errors out on a parameter with no gradient at all."""
    model = _initialized_model().train()
    model.on_training_step(0, 100)  # lambda_base = 1.0

    _, loss, _ = model(**_batch(), max_anchors=3)
    loss.backward()

    gru = model.logits_correction.prefix_gru
    assert gru.weight_ih_l0.grad is not None
    assert gru.weight_hh_l0.grad is not None


def test_validation_scores_the_corrected_objective_only():
    """Otherwise val loss drifts with lambda and --save-best is meaningless."""
    model = _initialized_model()
    model.on_training_step(0, 100)  # lambda_base = 1.0
    model.eval()

    with torch.no_grad():
        _, _, metrics = model(**_batch(), max_anchors=3)

    assert float(metrics["lambda_base"]) == 0.0
    assert not any(key.startswith("base_") for key in metrics)


def test_dpace_weighting_is_rejected():
    model = _initialized_model().train()

    with pytest.raises(ValueError, match="dpace"):
        model(**_batch(), max_anchors=3, per_position_loss_weight="dpace")


@pytest.mark.parametrize(
    ("global_step", "total_steps", "expected"),
    [
        (0, 100, 1.0),
        (25, 100, 0.5),
        (50, 100, 0.0),
        (500, 100, 0.0),
        (0, None, 0.0),
        (0, 0, 0.0),
    ],
)
def test_linear_lambda_base_schedule(global_step, total_steps, expected):
    assert linear_lambda_base(global_step, total_steps) == pytest.approx(expected)


def test_linear_lambda_base_honours_start_and_ratio():
    assert linear_lambda_base(0, 100, 0.4, 0.25) == pytest.approx(0.4)
    assert linear_lambda_base(12, 100, 0.4, 0.25) == pytest.approx(0.4 * 0.52)
    assert linear_lambda_base(25, 100, 0.4, 0.25) == pytest.approx(0.0)


def test_domino_checkpoint_round_trip(tmp_path):
    model = _initialized_model()
    with torch.no_grad():
        model.logits_correction.prefix_gru.weight_hh_l0.normal_()
        model.logits_correction.output_projection.weight.normal_()

    model.save_pretrained(tmp_path)
    restored = SpeculatorModel.from_pretrained(tmp_path)

    assert isinstance(restored, DominoDraftModel)
    torch.testing.assert_close(
        restored.logits_correction.prefix_gru.weight_hh_l0,
        model.logits_correction.prefix_gru.weight_hh_l0,
    )
    torch.testing.assert_close(
        restored.logits_correction.embed_proj[2].weight,
        model.logits_correction.output_projection.weight,
    )
    assert restored.config.gru_hidden_dim == GRU_HIDDEN
    assert restored.config.pure_draft_prefix_len == 1
    assert restored.config.lambda_base_start == 1.0


def test_correction_head_uses_adamw_with_muon_optimizer():
    model = DominoDraftModel(_config())
    muon_params, adamw_params = split_named_params_for_muon(model)
    muon_names = {name for name, _ in muon_params}
    adamw_names = {name for name, _ in adamw_params}

    for name in (
        "logits_correction.prefix_gru.weight_ih_l0",
        "logits_correction.prefix_gru.weight_hh_l0",
        "logits_correction.embed_proj.2.weight",
    ):
        assert name in adamw_names
        assert name not in muon_names
