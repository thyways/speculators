from speculators.models.dfly import DFlyDraftModel
from speculators.models.metrics import kl_div_loss
from speculators.train.config import TrainConfig


def test_dfly_cli_resolves_algorithm_specific_settings():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            "dfly",
            "--no-enable-hidden-correction",
            "--hidden-correction-intermediate-size",
            "96",
        ]
    )
    args = cfg.flatten()

    assert args["speculator_type"] == "dfly"
    assert args["draft_arch"] == "qwen3"
    assert args["enable_hidden_correction"] is False
    assert args["hidden_correction_intermediate_size"] == 96


def test_dfly_reuses_dflash_training_loss_arguments():
    args = TrainConfig(
        speculator_type="dfly",
    ).flatten()
    train_kwargs, validation_kwargs = DFlyDraftModel.get_trainer_kwargs(**args)

    assert train_kwargs["loss_config"]["kl_div"][0] is kl_div_loss
    assert validation_kwargs["loss_config"]["kl_div"][0] is kl_div_loss
    assert train_kwargs["gamma"] == 4.0
