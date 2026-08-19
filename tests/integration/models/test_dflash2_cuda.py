"""DFlash2 on the real training path: flex attention, bf16 autocast, fused loss.

The unit tests run on CPU, which forces eager attention, the eager losses, and no
``torch.compile``. The default training configuration uses none of those: the draft
attention is ``simple_flex_attention``, the losses are the fused Triton kernels, and
``conditional_torch_compile`` compiles ``forward`` whenever CUDA is available. These
tests cover that path, and pin the property that makes a DFlash checkpoint a valid
warm start -- a freshly initialized DFlash2 computes exactly what DFlash computes.
"""

import copy

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from tests.conftest import requires_cuda

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_TARGET_LAYERS = 2
BLOCK_SIZE = 8
MAX_ANCHORS = 16
TOTAL_SEQ_LEN = 512

TINY_QWEN3 = Qwen3Config(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=1024,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=64,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)

DEVICE = "cuda:0"


def _speculators_config(algorithm: str) -> SpeculatorsConfig:
    return SpeculatorsConfig(
        algorithm=algorithm,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE - 1)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path=None, architectures=["Qwen3ForCausalLM"]),
    )


def _base_kwargs(attn_impl: str) -> dict:
    transformer_config = copy.deepcopy(TINY_QWEN3)
    transformer_config._attn_implementation = attn_impl
    return {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
        "aux_hidden_state_layer_ids": list(range(NUM_TARGET_LAYERS)),
        "mask_token_id": 0,
    }


def _fill(model, seed: int = 0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buffer in model.buffers():
            if buffer.is_floating_point() and buffer.isnan().any():
                buffer.zero_()
    return model


def make_dflash2(attn_impl: str = "simple_flex_attention", **overrides):
    config = DFlash2SpeculatorConfig(
        **_base_kwargs(attn_impl),
        conv_kernel_size=3,
        conv_group_size=64,
        selector_rank=32,
        selector_top_k=8,
        speculators_config=_speculators_config("dflash2"),
        **overrides,
    )
    return _fill(DFlash2DraftModel(config=config)).to(DEVICE)


def make_dflash(attn_impl: str = "simple_flex_attention"):
    config = DFlashSpeculatorConfig(
        **_base_kwargs(attn_impl),
        speculators_config=_speculators_config("dflash"),
    )
    return _fill(DFlashDraftModel(config=config)).to(DEVICE)


def make_batch(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "hidden_states": torch.randn(
            1,
            TOTAL_SEQ_LEN,
            NUM_TARGET_LAYERS * HIDDEN_SIZE,
            generator=generator,
        ).to(DEVICE),
        "input_ids": torch.randint(
            1, VOCAB_SIZE, (1, TOTAL_SEQ_LEN), generator=generator
        ).to(DEVICE),
        "loss_mask": torch.ones(1, TOTAL_SEQ_LEN, dtype=torch.long, device=DEVICE),
        "verifier_last_hidden_states": torch.randn(
            1, TOTAL_SEQ_LEN, HIDDEN_SIZE, generator=generator
        ).to(DEVICE),
        "document_ids": torch.zeros(1, TOTAL_SEQ_LEN, dtype=torch.long, device=DEVICE),
    }


def call_kwargs(loss_fn: str = "ce", per_position: str = "dpace") -> dict:
    return {
        "loss_config": resolve_loss_config(loss_fn, "fused"),
        "gamma": 4.0,
        "max_anchors": MAX_ANCHORS,
        "per_position_loss_weight": per_position,
        "dpace_alpha": 0.5,
    }


@requires_cuda
@pytest.mark.parametrize("loss_fn", ["ce", "kl_div"])
def test_forward_backward_on_the_default_training_path(loss_fn):
    """Flex attention + bf16 autocast + fused loss + compiled forward."""
    model = make_dflash2()
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _draft_tokens, loss, metrics = model(**make_batch(), **call_kwargs(loss_fn))
    assert loss.isfinite()
    loss.backward()

    named = dict(model.named_parameters())
    for name in (
        "layers.0.attention_conv.base_kernel",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.1.mlp_conv.base_kernel",
        "layers.1.mlp_conv.kernel_projection.weight",
        "candidate_selector.successor_codebook",
    ):
        assert named[name].grad is not None, name
        assert named[name].grad.isfinite().all(), name
        assert named[name].grad.abs().sum() > 0, name

    for key, value in metrics.items():
        assert isinstance(value, torch.Tensor), key
        assert value.device.type == "cuda", key
        assert value.isfinite().all(), key


@requires_cuda
def test_a_fresh_dflash2_matches_dflash_exactly():
    """The identity conv and the zero selector make DFlash2 a superset at step 0.

    This is what lets a DFlash checkpoint warm-start a DFlash2 run: with the shared
    backbone weights copied across, the two models must compute the same thing.

    Asserted under ``force_eager``. ``conditional_torch_compile`` wraps each model's
    ``forward`` separately, and Inductor is free to fuse two different graphs
    differently, which moves the loss by ~1e-5 relative -- rounding, not a
    difference in the arithmetic. Eager pins the arithmetic itself, bit for bit;
    the compiled paths are then compared at a tolerance that admits that rounding.
    """
    dflash = make_dflash()
    dflash2 = make_dflash2()
    shared = {
        key: value
        for key, value in dflash.state_dict().items()
        if key in dflash2.state_dict()
    }
    missing, unexpected = dflash2.load_state_dict(shared, strict=False)
    assert not unexpected
    assert all(
        "_conv." in key or key.startswith("candidate_selector.") for key in missing
    ), missing

    batch = make_batch()

    def both():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7)  # select_anchors draws the anchor positions
            _t, dflash_loss, dflash_metrics = dflash(**batch, **call_kwargs())
            torch.manual_seed(7)
            _t, dflash2_loss, dflash2_metrics = dflash2(**batch, **call_kwargs())
        return dflash_loss, dflash_metrics, dflash2_loss, dflash2_metrics

    with torch.compiler.set_stance("force_eager"):
        base_loss, base_metrics, port_loss, port_metrics = both()
    torch.testing.assert_close(port_loss, base_loss, rtol=0, atol=0)
    for key, value in base_metrics.items():
        torch.testing.assert_close(port_metrics[key], value, rtol=0, atol=0)

    base_loss, base_metrics, port_loss, port_metrics = both()
    torch.testing.assert_close(port_loss, base_loss, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_the_conv_and_the_selector_change_the_output_once_trained():
    """Guard against the modules being silently inert (e.g. a dropped call)."""
    model = make_dflash2()
    batch = make_batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(7)
        _t, before, _m = model(**batch, **call_kwargs())

    with torch.no_grad():
        for layer in model.layers:
            layer.attention_conv.kernel_projection.weight.normal_(std=0.05)
            layer.mlp_conv.base_kernel.normal_(std=0.05)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(7)
        _t, conv_changed, _m = model(**batch, **call_kwargs())
    assert not torch.allclose(conv_changed, before)

    with torch.no_grad():
        model.candidate_selector.successor_codebook.normal_(std=0.3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(7)
        _t, selector_changed, _m = model(**batch, **call_kwargs())
    assert not torch.allclose(selector_changed, conv_changed)


@requires_cuda
def test_flex_and_eager_attention_agree():
    """The conv must not depend on the attention backend's masking details."""
    flex = make_dflash2("simple_flex_attention")
    eager = make_dflash2("eager")
    eager.load_state_dict(flex.state_dict())
    with torch.no_grad():
        for model in (flex, eager):
            for layer in model.layers:
                layer.attention_conv.kernel_projection.weight.normal_(std=0.05)
                layer.mlp_conv.base_kernel.normal_(std=0.05)
        eager.load_state_dict(flex.state_dict())

    batch = make_batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(7)
        _t, flex_loss, _m = flex(**batch, **call_kwargs())
        torch.manual_seed(7)
        _t, eager_loss, _m = eager(**batch, **call_kwargs())
    torch.testing.assert_close(flex_loss, eager_loss, rtol=2e-2, atol=2e-2)
