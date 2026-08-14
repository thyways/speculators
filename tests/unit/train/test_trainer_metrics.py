import torch

from speculators.train.trainer import _optimizer_lrs, _reduce_step_metrics


def test_step_metric_reduction_does_not_mutate_shared_source_storage(monkeypatch):
    shared = torch.tensor([2.0, 3.0])
    metrics = {"first": shared[0], "second": shared[1]}

    def fake_reduce(stacked, **_kwargs):
        stacked.add_(10)

    monkeypatch.setattr("speculators.train.trainer.dist.reduce", fake_reduce)
    reduced = _reduce_step_metrics(metrics, distributed=True)

    torch.testing.assert_close(shared, torch.tensor([2.0, 3.0]))
    assert reduced == {"first": 12.0, "second": 13.0}


def test_optimizer_lr_logging_reports_each_named_parameter_group():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter], "lr": 6e-4, "name": "base"},
            {
                "params": [torch.nn.Parameter(torch.zeros(()))],
                "lr": 6e-5,
                "name": "kv_bridge",
            },
        ]
    )

    assert _optimizer_lrs([optimizer]) == {
        "AdamW/base": 6e-4,
        "AdamW/kv_bridge": 6e-5,
    }
