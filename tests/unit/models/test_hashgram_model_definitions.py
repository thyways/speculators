"""Unit tests for the HashGram vector n-gram selector."""

import copy

import pytest
import torch

from speculators.models.hashgram.model_definitions import HashGramSelector
from speculators.train.optimizers import split_named_params_for_muon


def _selector(**overrides) -> HashGramSelector:
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "rank": 8,
        "top_k": 4,
        "bigram_buckets": 17,
        "trigram_buckets": 19,
        "num_hashes": 2,
    }
    values.update(overrides)
    torch.manual_seed(0)
    return HashGramSelector(**values)


def test_hash_ids_are_deterministic_and_in_range():
    selector = _selector()
    previous = torch.tensor([[1, 2], [3, 4]])
    previous2 = torch.tensor([[5, 6], [7, 8]])
    candidates = torch.tensor([[[0, 4, 9], [2, 3, 7]], [[1, 5, 6], [8, 9, 10]]])

    bigram_a = selector.hash_bigram(previous, candidates)
    bigram_b = selector.hash_bigram(previous, candidates)
    trigram = selector.hash_trigram(previous2, previous, candidates)

    torch.testing.assert_close(bigram_a, bigram_b)
    assert bigram_a.shape == candidates.shape
    assert trigram.shape == candidates.shape
    assert int(bigram_a.min()) >= 0
    assert int(bigram_a.max()) < selector.bigram_buckets
    assert int(trigram.min()) >= 0
    assert int(trigram.max()) < selector.trigram_buckets


def test_independent_probe_can_separate_a_first_probe_collision():
    selector = _selector(vocab_size=32, bigram_buckets=17, num_hashes=2)
    previous = torch.arange(32).repeat_interleave(32)
    candidates = torch.arange(32).repeat(32).unsqueeze(-1)
    keys = selector.hash_bigram_probes(previous, candidates).squeeze(-1)

    first_seen: dict[int, int] = {}
    separated_collision = False
    for pair_index, first_key in enumerate(keys[0].tolist()):
        earlier = first_seen.get(first_key)
        if earlier is not None and keys[1, earlier] != keys[1, pair_index]:
            separated_collision = True
            break
        first_seen[first_key] = pair_index

    assert separated_collision


def test_candidate_features_and_scores_have_expected_shapes():
    selector = _selector(num_hashes=1)
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    previous = torch.randint(0, 32, (2, 3))
    previous2 = torch.randint(0, 32, (2, 3))
    candidates = torch.randint(0, 32, (2, 3, 4))
    unary = torch.randn(2, 3, 32)

    features = selector.candidate_features(hidden, previous, previous2, candidates)
    scores = selector.score_candidates(
        unary_logits=unary,
        hidden_states=hidden,
        previous_ids=previous,
        previous2_ids=previous2,
        candidate_ids=candidates,
    )

    assert features.shape == (2, 3, 4, 8)
    assert scores.shape == (2, 3, 4)
    assert torch.isfinite(features).all()
    assert torch.isfinite(scores).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_gpu_hash_and_score_parity():
    cpu_selector = _selector().eval()
    gpu_selector = copy.deepcopy(cpu_selector).cuda().eval()
    hidden = torch.randn(2, 3, 16)
    previous = torch.randint(0, 32, (2, 3))
    previous2 = torch.randint(0, 32, (2, 3))
    candidates = torch.randint(0, 32, (2, 3, 4))
    unary = torch.randn(2, 3, 32)

    with torch.no_grad():
        cpu_bigram = cpu_selector.hash_bigram_probes(previous, candidates)
        gpu_bigram = gpu_selector.hash_bigram_probes(
            previous.cuda(), candidates.cuda()
        ).cpu()
        cpu_trigram = cpu_selector.hash_trigram_probes(previous2, previous, candidates)
        gpu_trigram = gpu_selector.hash_trigram_probes(
            previous2.cuda(), previous.cuda(), candidates.cuda()
        ).cpu()
        cpu_scores = cpu_selector.score_candidates(
            unary_logits=unary,
            hidden_states=hidden,
            previous_ids=previous,
            previous2_ids=previous2,
            candidate_ids=candidates,
        )
        gpu_scores = gpu_selector.score_candidates(
            unary_logits=unary.cuda(),
            hidden_states=hidden.cuda(),
            previous_ids=previous.cuda(),
            previous2_ids=previous2.cuda(),
            candidate_ids=candidates.cuda(),
        ).cpu()
        gpu_diagnostics = gpu_selector.diagnostics(
            unary_logits=unary.cuda(),
            hidden_states=hidden.cuda(),
            previous_ids=previous.cuda(),
            previous2_ids=previous2.cuda(),
            candidate_ids=candidates.cuda(),
            warmup=0,
            repeats=2,
        )

    torch.testing.assert_close(cpu_bigram, gpu_bigram, rtol=0, atol=0)
    torch.testing.assert_close(cpu_trigram, gpu_trigram, rtol=0, atol=0)
    torch.testing.assert_close(cpu_scores, gpu_scores, rtol=1e-4, atol=1e-5)
    assert gpu_diagnostics["hash_lookup_ms"] >= 0
    assert gpu_diagnostics["candidate_refinement_ms"] >= 0
    assert gpu_diagnostics["selector_total_ms"] >= 0


def test_diagnostics_report_latency_collisions_and_table_memory():
    selector = _selector(
        bigram_buckets=1,
        trigram_buckets=1,
        num_hashes=2,
    ).eval()
    hidden = torch.randn(2, 3, 16)
    previous = torch.randint(0, 32, (2, 3))
    previous2 = torch.randint(0, 32, (2, 3))
    candidates = torch.randint(0, 32, (2, 3, 4))
    unary = torch.randn(2, 3, 32)

    report = selector.diagnostics(
        unary_logits=unary,
        hidden_states=hidden,
        previous_ids=previous,
        previous2_ids=previous2,
        candidate_ids=candidates,
        warmup=0,
        repeats=2,
    )

    expected_table_bytes = 2 * 2 * 1 * 8 * 4
    assert report["hash_table_bytes"] == expected_table_bytes
    assert report["bigram_lookups"] == candidates.numel() * 2
    assert report["trigram_lookups"] == candidates.numel() * 2
    assert report["bigram_collision_rate"] > 0
    assert report["trigram_collision_rate"] > 0
    assert report["bigram_bucket_load"] == pytest.approx(1.0)
    assert report["trigram_bucket_load"] == pytest.approx(1.0)
    assert report["hash_lookup_ms"] >= 0
    assert report["candidate_refinement_ms"] >= 0
    assert report["selector_total_ms"] >= 0


def test_table_and_projection_receive_gradients():
    selector = _selector()
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    previous = torch.randint(0, 32, (2, 3))
    previous2 = torch.randint(0, 32, (2, 3))
    candidates = torch.randint(0, 32, (2, 3, 4))
    unary = torch.randn(2, 3, 32)
    scores = selector.score_candidates(
        unary_logits=unary,
        hidden_states=hidden,
        previous_ids=previous,
        previous2_ids=previous2,
        candidate_ids=candidates,
    )
    scores.sum().backward()

    parameter_names = (
        "bigram_table.weight",
        "trigram_table.weight",
        "hidden_projection.weight",
    )
    for name in parameter_names:
        parameter = dict(selector.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_hidden_refinement_uses_candidate_lm_rows():
    selector = _selector(hidden_refine=True)
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    previous = torch.randint(0, 32, (2, 3))
    previous2 = torch.randint(0, 32, (2, 3))
    candidates = torch.randint(0, 32, (2, 3, 4))
    unary = torch.randn(2, 3, 32)
    lm_head = torch.randn(32, 16)
    scores = selector.score_candidates(
        unary_logits=unary,
        hidden_states=hidden,
        previous_ids=previous,
        previous2_ids=previous2,
        candidate_ids=candidates,
        lm_head_weight=lm_head,
    )
    assert scores.shape == (2, 3, 4)
    scores.sum().backward()
    assert selector.residual_projection is not None
    assert selector.residual_projection.weight.grad is not None


def test_hidden_refinement_requires_lm_head_weight():
    selector = _selector(hidden_refine=True)
    hidden = torch.randn(1, 1, 16)
    previous = torch.tensor([[1]])
    previous2 = torch.tensor([[0]])
    candidates = torch.tensor([[[2, 3, 4, 5]]])
    unary = torch.randn(1, 1, 32)
    with pytest.raises(ValueError, match="lm_head_weight"):
        selector.score_candidates(
            unary_logits=unary,
            hidden_states=hidden,
            previous_ids=previous,
            previous2_ids=previous2,
            candidate_ids=candidates,
        )


@pytest.mark.parametrize(
    ("use_bigram", "use_trigram", "missing_table"),
    [(True, False, "trigram_table"), (False, True, "bigram_table")],
)
def test_single_ngram_order_ablation_runs(
    use_bigram: bool,
    use_trigram: bool,
    missing_table: str,
):
    selector = _selector(use_bigram=use_bigram, use_trigram=use_trigram)
    hidden = torch.randn(1, 2, 16)
    previous = torch.randint(0, 32, (1, 2))
    previous2 = torch.randint(0, 32, (1, 2))
    candidates = torch.randint(0, 32, (1, 2, 4))
    unary = torch.randn(1, 2, 32)
    scores = selector.score_candidates(
        unary_logits=unary,
        hidden_states=hidden,
        previous_ids=previous,
        previous2_ids=previous2,
        candidate_ids=candidates,
    )
    assert scores.shape == (1, 2, 4)
    assert getattr(selector, missing_table) is None


def test_selector_rejects_disabling_both_ngram_orders():
    with pytest.raises(ValueError, match="At least one"):
        _selector(use_bigram=False, use_trigram=False)


def test_hash_tables_use_adamw_under_muon_optimizer():
    selector = _selector()
    muon, adamw = split_named_params_for_muon(selector)
    muon_names = {name for name, _ in muon}
    adamw_names = {name for name, _ in adamw}
    assert "hidden_projection.weight" in muon_names
    assert "bigram_table.weight" in adamw_names
    assert "trigram_table.weight" in adamw_names
