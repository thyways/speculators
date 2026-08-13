"""Numerical parity between Domino's hand-written GRU and ``nn.GRU``.

Domino's recurrent head is hand-written (see the module docstring in
``models/domino/model_definitions.py`` for why) but must stay bit-compatible
with ``nn.GRU(bias=False)``: upstream Domino checkpoints are trained with the
real thing, and the converter only renames their keys. A drifted gate order or
update rule would load cleanly and silently degrade acceptance, so the parity is
checked in float64 -- cuDNN and the hand-written loop reduce in different
orders, which shows up in bfloat16 but not in the math.
"""

import torch
from torch import nn

from speculators.models.domino.model_definitions import DominoGRU

INPUT_SIZE = 7
HIDDEN_SIZE = 5
NUM_ROWS = 4
NUM_STEPS = 6


def _paired_grus() -> tuple[nn.GRU, DominoGRU]:
    torch.manual_seed(0)
    reference = nn.GRU(
        INPUT_SIZE,
        HIDDEN_SIZE,
        num_layers=1,
        batch_first=True,
        bias=False,
    ).double()
    ported = DominoGRU(INPUT_SIZE, HIDDEN_SIZE).double()
    # Loading nn.GRU's state dict straight into the port only works because the
    # key names are identical -- which is exactly what the converter relies on.
    ported.load_state_dict(reference.state_dict())
    return reference, ported


def test_state_dict_keys_match_nn_gru():
    """The converter relies on identical key names to do a plain rename."""
    reference, ported = _paired_grus()

    assert set(ported.state_dict()) == set(reference.state_dict())
    assert set(ported.state_dict()) == {"weight_ih_l0", "weight_hh_l0"}


def test_scan_matches_nn_gru_in_float64():
    reference, ported = _paired_grus()
    inputs = torch.randn(NUM_ROWS, NUM_STEPS, INPUT_SIZE, dtype=torch.float64)

    with torch.no_grad():
        expected, _ = reference(inputs)
        actual = ported(inputs)

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_stepwise_advance_matches_the_batched_scan():
    """Serving advances one token at a time; training scans the whole block."""
    _, ported = _paired_grus()
    inputs = torch.randn(NUM_ROWS, NUM_STEPS, INPUT_SIZE, dtype=torch.float64)

    with torch.no_grad():
        scanned = ported(inputs)
        state = ported.initial_state(inputs, NUM_ROWS)
        stepped = []
        for position in range(NUM_STEPS):
            state = ported.step(
                ported.project_inputs(inputs[:, position]),
                state,
            )
            stepped.append(state)

    torch.testing.assert_close(torch.stack(stepped, dim=1), scanned)


def test_initial_state_is_zero_per_row():
    _, ported = _paired_grus()
    reference = torch.randn(NUM_ROWS, INPUT_SIZE, dtype=torch.float64)

    state = ported.initial_state(reference, NUM_ROWS)

    assert state.shape == (NUM_ROWS, HIDDEN_SIZE)
    assert state.dtype == reference.dtype
    assert torch.count_nonzero(state) == 0


def test_gradients_reach_both_weight_matrices():
    _, ported = _paired_grus()
    inputs = torch.randn(NUM_ROWS, NUM_STEPS, INPUT_SIZE, dtype=torch.float64)

    ported(inputs).sum().backward()

    assert ported.weight_ih_l0.grad is not None
    assert ported.weight_ih_l0.grad.abs().sum() > 0
    assert ported.weight_hh_l0.grad is not None
    assert ported.weight_hh_l0.grad.abs().sum() > 0
