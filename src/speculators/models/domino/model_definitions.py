"""Recurrent logit-correction head for the Domino draft model.

The head is deliberately built from plain ``nn.Parameter``s rather than
``nn.GRU``, while keeping ``nn.GRU``'s exact parameter names and math:

* ``nn.GRU`` is routed to ``aten::_cudnn_rnn``, whose autocast policy is pinned
  to fp16 -- under the trainer's bf16 autocast its output comes back fp16 and
  concatenating it with bf16 hidden states silently promotes to fp32.
* ``aten::_cudnn_rnn`` cannot be decomposed, so it breaks the ``torch.compile``
  region that wraps the DFlash forward.
* ``nn.RNNBase`` caches ``_flat_weights``, which only stays coherent under FSDP2
  because of an implementation detail of how unsharded parameters are assigned
  back onto the module.
* vLLM needs a single-step cell for sequential in-block sampling anyway, so one
  implementation serves both training and serving and the two cannot drift.

Parameter names match ``nn.GRU(..., bias=False)`` (``weight_ih_l0`` /
``weight_hh_l0``) so upstream Domino checkpoints convert with a plain prefix
rename.
"""

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    "DominoGRU",
    "DominoLogitsCorrection",
]

_NUM_GRU_GATES = 3
# Position of the vocabulary-output layer inside ``embed_proj``.
_OUTPUT_PROJ_INDEX = 2


class DominoGRU(nn.Module):
    """Single-layer bias-free GRU, numerically identical to ``nn.GRU``.

    The input projection does not depend on the recurrent state, so it is
    applied to every block position with one batched GEMM; only the much
    smaller hidden-to-hidden product runs inside the per-position loop.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(
            torch.empty(_NUM_GRU_GATES * hidden_size, input_size)
        )
        self.weight_hh_l0 = nn.Parameter(
            torch.empty(_NUM_GRU_GATES * hidden_size, hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match ``nn.RNNBase.reset_parameters``: U(-1/sqrt(H), 1/sqrt(H))."""
        stdv = (
            1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        )
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def project_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project inputs for every gate: [..., input_size] -> [..., 3*H]."""
        return functional.linear(inputs, self.weight_ih_l0)

    def step(
        self,
        projected_input: torch.Tensor,  # [..., 3*H]
        state: torch.Tensor,  # [..., H]
    ) -> torch.Tensor:
        """Advance one position. Gate order and update rule follow ``nn.GRU``:

        ``r = sigmoid(W_ir x + W_hr h)``, ``z = sigmoid(W_iz x + W_hz h)``,
        ``n = tanh(W_in x + r * (W_hn h))`` (the reset gate multiplies only the
        hidden term), ``h' = (1 - z) * n + z * h`` (``z`` weights the *old*
        state).
        """
        gates_hidden = functional.linear(state, self.weight_hh_l0)
        reset_i, update_i, new_i = projected_input.chunk(
            _NUM_GRU_GATES, dim=-1
        )
        reset_h, update_h, new_h = gates_hidden.chunk(_NUM_GRU_GATES, dim=-1)
        reset = torch.sigmoid(reset_i + reset_h)
        update = torch.sigmoid(update_i + update_h)
        candidate = torch.tanh(new_i + reset * new_h)
        return (1.0 - update) * candidate + update * state

    def initial_state(
        self, reference: torch.Tensor, num_rows: int
    ) -> torch.Tensor:
        """Zero state shaped [num_rows, H], matching ``reference``'s dtype."""
        return reference.new_zeros(num_rows, self.hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Scan a whole block: [N, T, in] -> [N, T, H].

        The state is reset to zero for every row, mirroring upstream Domino's
        per-block ``reshape(bsz * n_blocks, block_size, -1)`` GRU call.

        This is the module's ``forward`` rather than a named method so callers go
        through ``__call__``: FSDP's pre-forward hook is what unshards the
        parameters, and it only runs on a module call. ``step`` /
        ``project_inputs`` stay separate for serving, which advances one position
        at a time outside any FSDP wrapper.
        """
        projected = self.project_inputs(inputs)
        state = self.initial_state(projected, projected.shape[0])
        states = []
        for position in range(projected.shape[1]):
            state = self.step(projected[:, position], state)
            states.append(state)
        return torch.stack(states, dim=1)


class DominoLogitsCorrection(nn.Module):
    """GRU state + draft hidden state -> additive draft-vocabulary logits.

    All layers use PyTorch's standard initialization, matching SpecForge's
    ``DominoDraftModel``. In particular, the output projection must not start at
    zero: otherwise the first backward pass with a non-zero corrected-objective
    weight cannot reach the GRU or the input projection, because both sit behind
    a zero matrix.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        gru_hidden_dim: int,
        emb_dim: int,
        draft_vocab_size: int,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.prefix_gru = DominoGRU(hidden_size, gru_hidden_dim)
        up_proj = nn.Linear(hidden_size + gru_hidden_dim, emb_dim, bias=False)
        out_proj = nn.Linear(emb_dim, draft_vocab_size, bias=False)
        nn.init.normal_(up_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(out_proj.weight, mean=0.0, std=initializer_range)
        # nn.Sequential keeps the upstream ``embed_proj.0`` / ``embed_proj.2``
        # state-dict names.
        self.embed_proj = nn.Sequential(up_proj, nn.SiLU(), out_proj)

    @property
    def output_projection(self) -> nn.Linear:
        """Typed accessor for ``embed_proj``'s vocabulary-output layer.

        ``nn.Sequential.__getitem__`` is typed as returning ``Module``, so index
        it here once instead of at every call site.
        """
        return cast("nn.Linear", self.embed_proj[_OUTPUT_PROJ_INDEX])

    def block_states(self, prev_token_embeds: torch.Tensor) -> torch.Tensor:
        """Recurrent state after consuming each block position's token."""
        return self.prefix_gru(prev_token_embeds)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [N, S, hidden_size]
        states: torch.Tensor,  # [N, S, gru_hidden_dim]
    ) -> torch.Tensor:
        """Additive logit correction for the slots the caller sliced out."""
        features = torch.cat(
            [hidden_states, states.to(hidden_states.dtype)],
            dim=-1,
        )
        return self.embed_proj(features)
