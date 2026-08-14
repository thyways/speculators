import warnings

from speculators.losses import kl_div_loss
from speculators.models.domino import DominoDraftModel
from speculators.train.config import TrainConfig


def test_domino_cli_resolves_algorithm_specific_settings():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            "domino",
            "--gru-hidden-dim",
            "512",
            "--logits-correction-emb-dim",
            "128",
            "--pure-draft-prefix-len",
            "2",
            "--lambda-base-start",
            "0.8",
            "--lambda-base-decay-ratio",
            "0.25",
        ]
    )
    args = cfg.flatten()

    assert args["speculator_type"] == "domino"
    assert args["draft_arch"] == "qwen3"
    assert args["gru_hidden_dim"] == 512
    assert args["logits_correction_emb_dim"] == 128
    assert args["pure_draft_prefix_len"] == 2
    assert args["lambda_base_start"] == 0.8
    assert args["lambda_base_decay_ratio"] == 0.25


def test_domino_reads_the_shared_dflash_group_without_warning():
    """Domino is a DFlash-family model, so --block-size must not warn.

    Resolution warns when an algorithm group is set under a speculator type that
    ignores it; forgetting to add domino to the dflash group's users would make
    every DFlash-family flag emit a spurious warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TrainConfig.resolve(
            [
                "--verifier-name-or-path",
                "dummy",
                "--speculator-type",
                "domino",
                "--block-size",
                "16",
                "--max-anchors",
                "512",
            ]
        )

    group_warnings = [
        str(item.message) for item in caught if "algorithm group" in str(item.message)
    ]
    assert not group_warnings, group_warnings

    args = cfg.flatten()
    assert args["block_size"] == 16
    assert args["max_anchors"] == 512


def test_domino_settings_warn_under_another_speculator_type():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrainConfig.resolve(
            [
                "--verifier-name-or-path",
                "dummy",
                "--speculator-type",
                "dflash",
                "--gru-hidden-dim",
                "512",
            ]
        )

    assert any("'domino'" in str(item.message) for item in caught)


def test_domino_reuses_dflash_training_loss_arguments():
    args = TrainConfig(speculator_type="domino").flatten()
    train_kwargs, validation_kwargs = DominoDraftModel.get_trainer_kwargs(**args)

    assert train_kwargs["loss_config"]["kl_div"][0] is kl_div_loss
    assert validation_kwargs["loss_config"]["kl_div"][0] is kl_div_loss
    assert train_kwargs["gamma"] == 4.0
