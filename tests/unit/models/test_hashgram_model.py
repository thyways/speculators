"""Unit tests for HashGram configuration and predecessor alignment."""

from typing import Any

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.hashgram import HashGramDraftModel, HashGramSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


def _tiny_config(**overrides: Any) -> HashGramSpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values: dict[str, Any] = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "hashgram_rank": 8,
        "hashgram_top_k": 3,
        "hashgram_bigram_buckets": 17,
        "hashgram_trigram_buckets": 19,
        "hashgram_markov_rank": 4,
    }
    values.update(overrides)
    return HashGramSpeculatorConfig(**values)


def test_model_rejects_pruned_vocab():
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        HashGramDraftModel(_tiny_config(draft_vocab_size=32))


def test_config_rejects_disabling_both_ngram_orders():
    with pytest.raises(ValueError, match="At least one"):
        _tiny_config(hashgram_use_bigram=False, hashgram_use_trigram=False)


def test_config_round_trip_preserves_hashgram_fields(tmp_path):
    config = _tiny_config(
        hashgram_num_hashes=2,
        hashgram_use_trigram=False,
        speculators_config=SpeculatorsConfig(
            algorithm="hashgram",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B",
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, HashGramSpeculatorConfig)
    assert loaded.speculators_model_type == "hashgram"
    assert loaded.speculators_config.algorithm == "hashgram"
    assert loaded.hashgram_rank == config.hashgram_rank
    assert loaded.hashgram_top_k == config.hashgram_top_k
    assert loaded.hashgram_bigram_buckets == config.hashgram_bigram_buckets
    assert loaded.hashgram_trigram_buckets == config.hashgram_trigram_buckets
    assert loaded.hashgram_num_hashes == 2
    assert loaded.hashgram_use_bigram is True
    assert loaded.hashgram_use_trigram is False
    assert (
        SpeculatorModel.registered_model_class_from_config(loaded) is HashGramDraftModel
    )


def test_predecessor_ids_shift_inside_each_block():
    model = HashGramDraftModel(_tiny_config())
    input_ids = torch.arange(16).unsqueeze(0)
    documents = torch.zeros_like(input_ids)
    block_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])
    prev1, prev2 = model._previous_ids(input_ids, documents, block_indices)
    torch.testing.assert_close(prev1, torch.tensor([[1, 2, 3, 4, 7, 8, 9, 10]]))
    torch.testing.assert_close(prev2, torch.tensor([[0, 1, 2, 3, 6, 7, 8, 9]]))


def test_predecessor_ids_sample_from_anchor():
    model = HashGramDraftModel(_tiny_config(sample_from_anchor=True))
    input_ids = torch.arange(16).unsqueeze(0)
    documents = torch.zeros_like(input_ids)
    block_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])
    prev1, prev2 = model._previous_ids(input_ids, documents, block_indices)
    torch.testing.assert_close(prev1, torch.tensor([[2, 3, 4, 5, 8, 9, 10, 11]]))
    torch.testing.assert_close(prev2, torch.tensor([[1, 2, 3, 4, 7, 8, 9, 10]]))


def test_predecessor_ids_do_not_cross_document_boundary():
    model = HashGramDraftModel(_tiny_config())
    input_ids = torch.arange(16).unsqueeze(0)
    documents = torch.zeros_like(input_ids)
    documents[:, :3] = 1
    block_indices = torch.tensor([3, 4, 5, 6])
    prev1, prev2 = model._previous_ids(input_ids, documents, block_indices)
    assert prev1[0, 0].item() == model.mask_token_id
    assert prev2[0, 0].item() == model.mask_token_id


def test_cross_document_slots_are_masked_but_new_document_history_recovers():
    model = HashGramDraftModel(_tiny_config())
    input_ids = torch.arange(12).unsqueeze(0)
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    block_indices = torch.tensor([1, 2, 3, 4])

    prev1, prev2 = model._previous_ids(input_ids, documents, block_indices)
    expected_prev1 = torch.tensor([[0, 1, model.mask_token_id, 3]])
    expected_prev2 = torch.tensor(
        [[model.mask_token_id, 0, model.mask_token_id, model.mask_token_id]]
    )
    torch.testing.assert_close(prev1, expected_prev1)
    torch.testing.assert_close(prev2, expected_prev2)
    torch.testing.assert_close(
        model._same_document_block_mask(documents, block_indices),
        torch.tensor([[True, True, False, False]]),
    )


def test_sequential_greedy_rollout_uses_selected_predecessors(monkeypatch):
    model = HashGramDraftModel(_tiny_config())
    model.markov_head = None
    hidden = torch.zeros(1, 4, 16)
    unary = torch.full((1, 4, 64), -100.0)
    unary[..., 2] = 3.0
    unary[..., 3] = 2.0
    unary[..., 4] = 1.0
    previous = torch.tensor([[0, 1, 9, 9]])
    previous2 = torch.tensor([[0, 0, 8, 8]])

    def score_from_selected_previous(**kwargs):
        desired = kwargs["previous_ids"].unsqueeze(-1) + 1
        return -(kwargs["candidate_ids"] - desired).abs().float()

    monkeypatch.setattr(
        model.hashgram_selector,
        "score_candidates",
        score_from_selected_previous,
    )
    predictions = model._sequential_greedy_predictions(
        hidden=hidden,
        unary_logits=unary,
        previous_ids=previous,
        previous2_ids=previous2,
    )

    torch.testing.assert_close(predictions, torch.tensor([[0, 2, 3, 4]]))


def test_greedy_rollout_metrics_distinguish_slot_accuracy_from_prefix_acceptance():
    model = HashGramDraftModel(_tiny_config())
    predictions = torch.tensor([[0, 9, 3, 4]])
    targets = torch.tensor([[0, 2, 3, 4]])
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

    metrics = model._greedy_rollout_metrics(
        predictions=predictions,
        target_ids=targets,
        loss_mask=loss_mask,
    )

    assert metrics["hashgram_greedy_position_1_acc_sum"].item() == 0
    assert metrics["hashgram_greedy_position_2_acc_sum"].item() == 1
    assert metrics["hashgram_greedy_position_2_accept_sum"].item() == 0
    assert metrics["hashgram_greedy_eal_sum"].item() == 0
    assert metrics["hashgram_greedy_eal_total"].item() == 1


def test_checkpoint_round_trip_preserves_hashgram_weights(tmp_path, monkeypatch):
    metadata = SpeculatorsConfig(
        algorithm="hashgram",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path="Qwen/Qwen3-4B",
            architectures=["Qwen3ForCausalLM"],
        ),
    )
    model = HashGramDraftModel(
        _tiny_config(hashgram_num_hashes=2, speculators_config=metadata)
    )
    assert model.hashgram_selector.bigram_table is not None
    assert model.hashgram_selector.trigram_table is not None
    with torch.no_grad():
        model.hashgram_selector.bigram_table.weight.fill_(0.125)
        model.hashgram_selector.trigram_table.weight.fill_(-0.25)
        assert model.markov_head is not None
        model.markov_head.markov_w1.weight.fill_(0.375)
    expected_bigram = model.hashgram_selector.bigram_table.weight.detach().clone()
    expected_trigram = model.hashgram_selector.trigram_table.weight.detach().clone()
    expected_markov = model.markov_head.markov_w1.weight.detach().clone()
    model.save_pretrained(tmp_path)

    verifier_weights = {
        "embed_tokens.weight": torch.randn(64, 16),
        "lm_head.weight": torch.randn(64, 16),
        "model.norm.weight": torch.ones(16),
    }

    def fake_loader(weights_to_load, _name_or_path):
        return {
            name: verifier_weights[name]
            for name in weights_to_load
            if name in verifier_weights
        }

    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers",
        fake_loader,
    )
    loaded = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, HashGramDraftModel)
    assert loaded.hashgram_selector.bigram_table is not None
    assert loaded.hashgram_selector.trigram_table is not None
    assert loaded.markov_head is not None
    torch.testing.assert_close(
        loaded.hashgram_selector.bigram_table.weight,
        expected_bigram,
    )
    torch.testing.assert_close(
        loaded.hashgram_selector.trigram_table.weight,
        expected_trigram,
    )
    torch.testing.assert_close(loaded.markov_head.markov_w1.weight, expected_markov)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tiny_forward_produces_selector_loss_and_hash_gradients():
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = HashGramDraftModel(_tiny_config()).to(device).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.embed_tokens.weight, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.02)
        torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.02)
        torch.nn.init.ones_(model.verifier_norm.weight)

    seq_len = 24
    hidden_size = model.config.transformer_layer_config.hidden_size
    hidden_states = torch.randn(1, seq_len, 2 * hidden_size, device=device)
    input_ids = torch.randint(0, model.verifier_vocab_size, (1, seq_len), device=device)
    loss_mask = torch.ones(1, seq_len, device=device)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)

    _, loss, metrics = model(
        hidden_states=hidden_states,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=torch.randn(1, seq_len, hidden_size, device=device),
        document_ids=document_ids,
        loss_config=resolve_loss_config("ce", "eager"),
        max_anchors=2,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "selector_loss_sum" in metrics
    assert "base_candidate_recall_at_3_sum" in metrics
    assert "recall_candidate_recall_at_3_sum" in metrics
    assert "hashgram_greedy_position_1_acc_sum" in metrics
    assert "hashgram_greedy_position_1_accept_sum" in metrics
    assert "hashgram_greedy_eal_sum" in metrics
    loss.backward()
    assert model.hashgram_selector.bigram_table is not None
    assert model.hashgram_selector.trigram_table is not None
    assert model.hashgram_selector.bigram_table.weight.grad is not None
    assert model.hashgram_selector.trigram_table.weight.grad is not None
