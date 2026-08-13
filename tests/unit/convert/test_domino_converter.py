"""Unit tests for DominoConverter config building and head-key remapping."""

from unittest.mock import patch

import pytest
import torch

from speculators.convert.domino.converter import (
    DominoConverter,
    remap_domino_head_keys,
)


def _source_config(**overrides):
    """A config shaped like upstream's configs/qwen3-8b-domino.json."""
    config = {
        "model_type": "qwen3",
        "architectures": ["DominoDraftModel"],
        "auto_map": {"AutoModel": "domino.DominoDraftModel"},
        "block_size": 16,
        "num_target_layers": 36,
        # Upstream duplicates emb_dim at the top level; it must not leak into
        # the transformer config.
        "emb_dim": 256,
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "projector_type": "domino",
            "pure_draft_prefix_len": 1,
            "emb_dim": 256,
            "gru_hidden_dim": 1024,
            "shift_label": True,
        },
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    config.update(overrides)
    return config


class TestBuildConfig:
    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_happy_path(self, mock_get_config):
        mock_get_config.return_value = (
            {
                "hidden_size": 4096,
                "vocab_size": 151936,
                "architectures": ["Qwen3ForCausalLM"],
            },
            None,
        )
        config = DominoConverter()._build_config(
            _source_config(), "Qwen/Qwen3-8B", None
        )

        assert config.speculators_config.algorithm == "domino"
        assert (
            config.speculators_config.verifier.name_or_path == "Qwen/Qwen3-8B"
        )
        assert config.block_size == 16
        assert config.draft_vocab_size == 151936
        assert config.mask_token_id == 151669
        assert config.gru_hidden_dim == 1024
        assert config.logits_correction_emb_dim == 256
        assert config.pure_draft_prefix_len == 1
        # shift_label maps onto sample_from_anchor, which decides the token count
        assert config.sample_from_anchor is True
        assert config.speculators_config.proposal_methods[0].speculative_tokens == 16  # type: ignore[attr-defined]
        assert config.suffix_start == 1
        # upstream target_layer_ids are offset by +1 to speculators layer ids
        assert config.aux_hidden_state_layer_ids == [2, 10, 18, 26, 34]
        assert config.transformer_layer_config.num_hidden_layers == 5
        assert not hasattr(config.transformer_layer_config, "dflash_config")
        assert not hasattr(config.transformer_layer_config, "emb_dim")

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_shift_label_false_drops_the_anchor_from_the_draft(
        self, mock_get_config
    ):
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        source = _source_config()
        source["dflash_config"] = {
            **source["dflash_config"],
            "shift_label": False,
        }

        config = DominoConverter()._build_config(source, "Qwen/Qwen3-8B", None)

        assert config.sample_from_anchor is False
        assert config.speculators_config.proposal_methods[0].speculative_tokens == 15  # type: ignore[attr-defined]
        # slot 0 is the anchor, so the correction starts one slot later
        assert config.suffix_start == 2

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_explicit_aux_layer_ids_override(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        config = DominoConverter()._build_config(
            _source_config(), "Qwen/Qwen3-8B", [3, 11, 19]
        )

        assert config.aux_hidden_state_layer_ids == [3, 11, 19]

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_rejects_a_non_domino_projector(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        source = _source_config()
        source["dflash_config"] = {
            **source["dflash_config"],
            "projector_type": "dspark",
        }

        with pytest.raises(ValueError, match="not a Domino draft model"):
            DominoConverter()._build_config(source, "Qwen/Qwen3-8B", None)

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_requires_the_head_dimensions(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 4096}, None)
        source = _source_config()
        source["dflash_config"] = {
            k: v
            for k, v in source["dflash_config"].items()
            if k != "gru_hidden_dim"
        }

        with pytest.raises(ValueError, match="head dimensions"):
            DominoConverter()._build_config(source, "Qwen/Qwen3-8B", None)

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_rejects_a_hidden_size_mismatch(self, mock_get_config):
        mock_get_config.return_value = ({"hidden_size": 2048}, None)

        with pytest.raises(ValueError, match="Architecture mismatch"):
            DominoConverter()._build_config(
                _source_config(), "Qwen/Qwen3-8B", None
            )

    @patch(
        "speculators.convert.domino.converter.PretrainedConfig.get_config_dict"
    )
    def test_rejects_a_vocab_size_mismatch(self, mock_get_config):
        """Conversion yields a full-vocabulary draft, so the vocabs must agree.

        Without this check the mismatch would only surface later, while copying
        the verifier's embedding and LM head.
        """
        mock_get_config.return_value = (
            {"hidden_size": 4096, "vocab_size": 128256},
            None,
        )

        with pytest.raises(ValueError, match="Vocabulary mismatch"):
            DominoConverter()._build_config(
                _source_config(), "Qwen/Qwen3-8B", None
            )


class TestRemapHeadKeys:
    def test_flat_upstream_layout(self):
        weights = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "prefix_gru.weight_ih_l0": torch.zeros(2),
            "prefix_gru.weight_hh_l0": torch.zeros(3),
            "embed_proj.0.weight": torch.zeros(4),
            "embed_proj.2.weight": torch.zeros(5),
        }

        remapped = remap_domino_head_keys(weights)

        assert set(remapped) == {
            "layers.0.self_attn.q_proj.weight",
            "logits_correction.prefix_gru.weight_ih_l0",
            "logits_correction.prefix_gru.weight_hh_l0",
            "logits_correction.embed_proj.0.weight",
            "logits_correction.embed_proj.2.weight",
        }
        assert (
            remapped["logits_correction.prefix_gru.weight_hh_l0"].numel() == 3
        )

    def test_legacy_logit_head_container_layout(self):
        """Early upstream checkpoints nested the head under ``logit_head``."""
        weights = {
            "logit_head.prefix_gru.weight_ih_l0": torch.zeros(2),
            "logit_head.embed_proj.2.weight": torch.zeros(5),
        }

        remapped = remap_domino_head_keys(weights)

        assert set(remapped) == {
            "logits_correction.prefix_gru.weight_ih_l0",
            "logits_correction.embed_proj.2.weight",
        }

    def test_unrelated_keys_are_untouched(self):
        weights = {"norm.weight": torch.zeros(1), "fc.weight": torch.zeros(1)}

        assert set(remap_domino_head_keys(weights)) == set(weights)
