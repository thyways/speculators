"""CLI/config coverage for token-latent feedback."""

from speculators.train.config.schema import TrainConfig


def test_token_latent_feedback_defaults_are_resolved():
    config = TrainConfig.from_sources(
        cli={
            "speculator_type": "token_latent_feedback",
            "verifier_name_or_path": "dummy-verifier",
        },
        argv=[],
    )
    flat = config.flatten()
    assert flat["latent_dim"] == 128
    assert flat["feedback_stages"] == 1
    assert flat["prefix_mixer_mode"] == "full"
    assert flat["prefix_mixer_parameterization"] == "toeplitz"
    assert flat["latent_loss_alpha"] == 0.1
    assert config.loss.loss_fn == '{"ce": 0.1, "tv": 0.9}'
    assert config.dflash.per_position_loss_weight == "fixed-exp-decay"
    assert "target_projection_seed" not in flat
    assert "latent_projection_seed" not in flat


def test_token_latent_feedback_cli_values_round_trip():
    config = TrainConfig.from_sources(
        cli={
            "speculator_type": "token_latent_feedback",
            "verifier_name_or_path": "dummy-verifier",
            "latent_dim": 64,
            "prefix_mixer_mode": "shifted",
            "use_reliability_gate": False,
            "feedback_output_projection_init": 0.001,
            "latent_loss_alpha": 0.25,
        },
        argv=[
            "--speculator-type",
            "token_latent_feedback",
            "--verifier-name-or-path",
            "dummy-verifier",
        ],
    )
    flat = config.flatten()
    assert flat["latent_dim"] == 64
    assert flat["prefix_mixer_mode"] == "shifted"
    assert flat["use_reliability_gate"] is False
    assert flat["feedback_output_projection_init"] == 0.001
    assert flat["latent_loss_alpha"] == 0.25
