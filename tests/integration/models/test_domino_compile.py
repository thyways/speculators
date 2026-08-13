"""Domino's lambda schedule must not fight ``torch.compile``.

``DFlashDraftModel.forward`` is wrapped in ``conditional_torch_compile``, which
only compiles when CUDA is present -- so the CPU unit tests never exercise the
compiled path. Domino changes a loss weight on every optimizer step, and Dynamo
specializes on Python scalars by value: holding lambda as a float instead of a
tensor buffer would recompile the whole forward every single step. That failure
is silent (correct results, an order-of-magnitude slower, eventually a cache
blowout), so it is pinned here.
"""

import pytest
import torch
from torch._dynamo.utils import counters
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.domino import DominoDraftModel, DominoSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="conditional_torch_compile only compiles the forward on CUDA",
)

HIDDEN_SIZE = 128
VOCAB_SIZE = 512
BLOCK_SIZE = 8
SEQUENCE_LENGTH = 256
MAX_ANCHORS = 8
TOTAL_STEPS = 40


def _model() -> DominoDraftModel:
    transformer_config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        _attn_implementation="simple_flex_attention",  # type: ignore[call-arg]
    )
    config = DominoSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        sample_from_anchor=True,
        gru_hidden_dim=64,
        logits_correction_emb_dim=32,
        pure_draft_prefix_len=1,
        speculators_config=SpeculatorsConfig(
            algorithm="domino",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    torch.manual_seed(0)
    model = DominoDraftModel(config).cuda()  # type: ignore[call-arg]
    for weight in (
        model.embed_tokens.weight,
        model.lm_head.weight,
        model.verifier_lm_head.weight,
        model.verifier_norm.weight,
    ):
        torch.nn.init.normal_(weight)
    with torch.no_grad():
        model.logits_correction.output_projection.weight.normal_(std=0.02)
    return model.train()


def _batch() -> dict:
    return {
        "hidden_states": torch.randn(
            1, SEQUENCE_LENGTH, 3 * HIDDEN_SIZE, device="cuda"
        ),
        "input_ids": torch.randint(
            1, VOCAB_SIZE, (1, SEQUENCE_LENGTH), device="cuda"
        ),
        "loss_mask": torch.ones(1, SEQUENCE_LENGTH, device="cuda"),
        "verifier_last_hidden_states": torch.randn(
            1, SEQUENCE_LENGTH, HIDDEN_SIZE, device="cuda"
        ),
        "document_ids": torch.zeros(
            1, SEQUENCE_LENGTH, dtype=torch.long, device="cuda"
        ),
    }


def _unique_graphs() -> int:
    return counters["stats"]["unique_graphs"]


def _step(model: DominoDraftModel, batch: dict, global_step: int):
    model.on_training_step(global_step, TOTAL_STEPS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss, metrics = model(**batch, max_anchors=MAX_ANCHORS)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return loss, metrics


@pytest.mark.sanity
def test_changing_lambda_does_not_recompile_the_forward():
    model = _model()
    batch = _batch()

    _step(model, batch, 0)
    baseline = _unique_graphs()
    assert (
        baseline > 0
    ), "the forward was never traced; compilation is not active"

    # Five more steps, each with a different live lambda value.
    for global_step in range(1, 6):
        loss, _ = _step(model, batch, global_step)
        assert torch.isfinite(loss)
        assert float(model.lambda_base) > 0.0

    assert _unique_graphs() == baseline


@pytest.mark.sanity
def test_dropping_the_base_term_settles_after_one_recompile():
    """The base term is gated by a Python bool, which flips at most once."""
    model = _model()
    batch = _batch()

    _step(model, batch, 0)
    assert model._base_loss_active is True

    # Past the decay horizon the base term is skipped entirely.
    _step(model, batch, TOTAL_STEPS // 2)
    assert model._base_loss_active is False
    after_flip = _unique_graphs()

    for global_step in (TOTAL_STEPS // 2 + 4, TOTAL_STEPS // 2 + 8):
        loss, metrics = _step(model, batch, global_step)
        assert torch.isfinite(loss)
        assert not any(key.startswith("base_") for key in metrics)

    assert _unique_graphs() == after_flip


@pytest.mark.sanity
def test_correction_head_receives_a_gradient_at_every_lambda():
    """At lambda 1 the gradient is zero but must still exist for DDP."""
    model = _model()
    batch = _batch()
    gru = model.logits_correction.prefix_gru

    model.on_training_step(0, TOTAL_STEPS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss, _ = model(**batch, max_anchors=MAX_ANCHORS)
    loss.backward()
    assert gru.weight_ih_l0.grad is not None
    assert float(gru.weight_ih_l0.grad.abs().sum()) == 0.0
    model.zero_grad(set_to_none=True)

    model.on_training_step(TOTAL_STEPS // 4, TOTAL_STEPS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss, _ = model(**batch, max_anchors=MAX_ANCHORS)
    loss.backward()
    assert float(gru.weight_ih_l0.grad.abs().sum()) > 0.0
