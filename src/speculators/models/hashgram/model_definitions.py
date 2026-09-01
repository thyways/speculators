"""Hashed vector n-gram candidate selector components."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["HashGramSelector"]


class HashGramSelector(nn.Module):
    """Vector-valued hashed bigram/trigram selector.

    The tables are deliberately factorized by hash bucket rather than by the full
    vocabulary cross product.  ``candidate_features`` returns one vector per
    ``(previous, candidate)`` pair, while ``score_candidates`` contracts those
    vectors with the current draft hidden state (or, optionally, uses them as a
    candidate-specific hidden residual).
    """

    _BIGRAM_PREVIOUS_BASE = 1_000_003
    _BIGRAM_PREVIOUS_STEP = 209_458
    _BIGRAM_CANDIDATE_BASE = 97_003
    _BIGRAM_CANDIDATE_STEP = 260_726
    _TRIGRAM_PREVIOUS2_BASE = 1_000_003
    _TRIGRAM_PREVIOUS2_STEP = 311_842
    _TRIGRAM_PREVIOUS_BASE = 97_003
    _TRIGRAM_PREVIOUS_STEP = 393_226
    _TRIGRAM_CANDIDATE_BASE = 9_973
    _TRIGRAM_CANDIDATE_STEP = 473_794
    _PROBE_SALT_STEP = 1_000_033

    bigram_previous_coefficients: torch.Tensor
    bigram_candidate_coefficients: torch.Tensor
    trigram_previous2_coefficients: torch.Tensor
    trigram_previous_coefficients: torch.Tensor
    trigram_candidate_coefficients: torch.Tensor
    probe_salts: torch.Tensor

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
        bigram_buckets: int,
        trigram_buckets: int,
        num_hashes: int = 1,
        initializer_range: float = 0.02,
        hidden_refine: bool = False,
        use_bigram: bool = True,
        use_trigram: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if hidden_size <= 0 or rank <= 0 or top_k <= 0:
            raise ValueError(
                "hidden_size, rank, and top_k must be positive; got "
                f"hidden_size={hidden_size}, rank={rank}, top_k={top_k}"
            )
        if top_k > vocab_size:
            raise ValueError(f"top_k ({top_k}) cannot exceed vocab_size ({vocab_size})")
        if bigram_buckets <= 0 or trigram_buckets <= 0 or num_hashes <= 0:
            raise ValueError(
                "bigram_buckets, trigram_buckets, and num_hashes must be positive"
            )
        if not use_bigram and not use_trigram:
            raise ValueError("At least one of use_bigram/use_trigram must be enabled")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.top_k = top_k
        self.bigram_buckets = bigram_buckets
        self.trigram_buckets = trigram_buckets
        self.num_hashes = num_hashes
        self.hidden_refine = hidden_refine
        self.use_bigram = use_bigram
        self.use_trigram = use_trigram

        # Flatten the probe dimension into one embedding. The table names route
        # these embedding-like matrices to AdamW rather than Muon.
        self.bigram_table: nn.Embedding | None = (
            nn.Embedding(num_hashes * bigram_buckets, rank) if use_bigram else None
        )
        self.trigram_table: nn.Embedding | None = (
            nn.Embedding(num_hashes * trigram_buckets, rank) if use_trigram else None
        )
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.trigram_gate: nn.Linear | None = (
            nn.Linear(hidden_size, 1) if use_trigram else None
        )

        self.residual_projection: nn.Linear | None
        self.refine_gate: nn.Linear | None
        if hidden_refine:
            self.residual_projection = nn.Linear(rank, hidden_size, bias=False)
            self.refine_gate = nn.Linear(hidden_size, 1)
        else:
            self.residual_projection = None
            self.refine_gate = None

        probe_ids = torch.arange(num_hashes, dtype=torch.long)
        self.register_buffer(
            "bigram_previous_coefficients",
            self._BIGRAM_PREVIOUS_BASE + probe_ids * self._BIGRAM_PREVIOUS_STEP,
            persistent=False,
        )
        self.register_buffer(
            "bigram_candidate_coefficients",
            self._BIGRAM_CANDIDATE_BASE + probe_ids * self._BIGRAM_CANDIDATE_STEP,
            persistent=False,
        )
        self.register_buffer(
            "trigram_previous2_coefficients",
            self._TRIGRAM_PREVIOUS2_BASE + probe_ids * self._TRIGRAM_PREVIOUS2_STEP,
            persistent=False,
        )
        self.register_buffer(
            "trigram_previous_coefficients",
            self._TRIGRAM_PREVIOUS_BASE + probe_ids * self._TRIGRAM_PREVIOUS_STEP,
            persistent=False,
        )
        self.register_buffer(
            "trigram_candidate_coefficients",
            self._TRIGRAM_CANDIDATE_BASE + probe_ids * self._TRIGRAM_CANDIDATE_STEP,
            persistent=False,
        )
        self.register_buffer(
            "probe_salts",
            probe_ids * self._PROBE_SALT_STEP,
            persistent=False,
        )
        self.reset_parameters(initializer_range)

    def reset_parameters(self, initializer_range: float = 0.02) -> None:
        """Initialize tables and projections with non-zero gradients from step 0."""
        if self.bigram_table is not None:
            nn.init.normal_(self.bigram_table.weight, mean=0.0, std=initializer_range)
        if self.trigram_table is not None:
            nn.init.normal_(self.trigram_table.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.hidden_projection.weight, mean=0.0, std=initializer_range)
        if self.trigram_gate is not None:
            nn.init.normal_(self.trigram_gate.weight, mean=0.0, std=initializer_range)
            nn.init.zeros_(self.trigram_gate.bias)
        if self.residual_projection is not None:
            nn.init.normal_(
                self.residual_projection.weight,
                mean=0.0,
                std=initializer_range,
            )
        if self.refine_gate is not None:
            nn.init.normal_(self.refine_gate.weight, mean=0.0, std=initializer_range)
            nn.init.zeros_(self.refine_gate.bias)

    @staticmethod
    def _as_long(value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=torch.long)

    def _lookup(
        self,
        table: nn.Embedding,
        probe_keys: torch.Tensor,
        buckets: int,
    ) -> torch.Tensor:
        """Lookup and average independent probes for an arbitrary key shape."""
        probe_keys = self._as_long(probe_keys)
        if probe_keys.shape[0] != self.num_hashes:
            raise ValueError(
                "probe_keys must have num_hashes as its leading dimension; got "
                f"shape={tuple(probe_keys.shape)}, num_hashes={self.num_hashes}"
            )
        probe_shape = (self.num_hashes,) + (1,) * (probe_keys.ndim - 1)
        probe_offsets = (
            torch.arange(self.num_hashes, device=probe_keys.device, dtype=torch.long)
            * buckets
        ).view(probe_shape)
        return table(probe_keys + probe_offsets).mean(dim=0)

    def _probe_view(self, values: torch.Tensor, candidate_ndim: int) -> torch.Tensor:
        return values.view((self.num_hashes,) + (1,) * candidate_ndim)

    def hash_bigram_probes(
        self, previous_ids: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return independent probe bucket IDs for ``(previous, candidate)``."""
        previous_ids, candidate_ids = (
            self._as_long(previous_ids),
            self._as_long(candidate_ids),
        )
        candidate_ndim = candidate_ids.ndim
        previous = previous_ids.unsqueeze(-1).unsqueeze(0)
        candidates = candidate_ids.unsqueeze(0)
        previous_coefficients = self._probe_view(
            self.bigram_previous_coefficients, candidate_ndim
        )
        candidate_coefficients = self._probe_view(
            self.bigram_candidate_coefficients, candidate_ndim
        )
        salts = self._probe_view(self.probe_salts, candidate_ndim)
        return torch.remainder(
            previous * previous_coefficients
            + candidates * candidate_coefficients
            + salts,
            self.bigram_buckets,
        )

    def hash_bigram(
        self, previous_ids: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return first-probe bucket IDs for ``(previous, candidate)`` pairs."""
        return self.hash_bigram_probes(previous_ids, candidate_ids)[0]

    def hash_trigram_probes(
        self,
        previous2_ids: torch.Tensor,
        previous_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return independent probe bucket IDs for trigram candidate keys."""
        previous2_ids, previous_ids, candidate_ids = (
            self._as_long(previous2_ids),
            self._as_long(previous_ids),
            self._as_long(candidate_ids),
        )
        candidate_ndim = candidate_ids.ndim
        previous2 = previous2_ids.unsqueeze(-1).unsqueeze(0)
        previous = previous_ids.unsqueeze(-1).unsqueeze(0)
        candidates = candidate_ids.unsqueeze(0)
        previous2_coefficients = self._probe_view(
            self.trigram_previous2_coefficients, candidate_ndim
        )
        previous_coefficients = self._probe_view(
            self.trigram_previous_coefficients, candidate_ndim
        )
        candidate_coefficients = self._probe_view(
            self.trigram_candidate_coefficients, candidate_ndim
        )
        salts = self._probe_view(self.probe_salts, candidate_ndim)
        return torch.remainder(
            previous2 * previous2_coefficients
            + previous * previous_coefficients
            + candidates * candidate_coefficients
            + salts,
            self.trigram_buckets,
        )

    def hash_trigram(
        self,
        previous2_ids: torch.Tensor,
        previous_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return first-probe bucket IDs for trigram candidate keys."""
        return self.hash_trigram_probes(previous2_ids, previous_ids, candidate_ids)[0]

    def candidate_features(
        self,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return composed vector features with shape ``[..., top_k, rank]``."""
        features: torch.Tensor | None = None
        if self.bigram_table is not None:
            bigram_keys = self.hash_bigram_probes(previous_ids, candidate_ids)
            features = self._lookup(
                self.bigram_table,
                bigram_keys,
                self.bigram_buckets,
            )
        if self.trigram_table is not None:
            if self.trigram_gate is None:
                raise RuntimeError("trigram gate was not initialized")
            trigram_keys = self.hash_trigram_probes(
                previous2_ids, previous_ids, candidate_ids
            )
            trigram = self._lookup(
                self.trigram_table,
                trigram_keys,
                self.trigram_buckets,
            )
            gate = torch.sigmoid(self.trigram_gate(hidden_states)).unsqueeze(-1)
            # ``gate`` is [.., 1, 1] and broadcasts over candidates and rank.
            trigram = gate * trigram
            features = trigram if features is None else features + trigram
        if features is None:
            raise RuntimeError("HashGram has no enabled n-gram table")
        return features

    def _score_from_features(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        candidate_ids: torch.Tensor,
        features: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates from already-looked-up n-gram features."""
        if self.hidden_refine:
            if self.residual_projection is None or self.refine_gate is None:
                raise RuntimeError("hidden refinement layers were not initialized")
            if lm_head_weight is None:
                raise ValueError("lm_head_weight is required when hidden_refine=True")
            residual = self.residual_projection(features)
            gate = torch.sigmoid(self.refine_gate(hidden_states)).unsqueeze(-1)
            residual = residual * gate
            candidate_rows = torch.nn.functional.embedding(
                candidate_ids,
                lm_head_weight,
            )
            delta = (candidate_rows * residual).sum(dim=-1)
        else:
            query = self.hidden_projection(hidden_states)
            delta = torch.einsum("...r,...kr->...k", query, features)
            delta = delta / math.sqrt(self.rank)
        return unary_logits.gather(dim=-1, index=candidate_ids) + delta

    def score_candidates(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add HashGram scores to unary logits for a candidate set.

        ``lm_head_weight`` enables the optional candidate-specific hidden residual
        path.  Without it, the default low-rank contraction is used and avoids
        materializing ``[..., top_k, hidden_size]`` activations.
        """
        features = self.candidate_features(
            hidden_states,
            previous_ids,
            previous2_ids,
            candidate_ids,
        )
        return self._score_from_features(
            unary_logits=unary_logits,
            hidden_states=hidden_states,
            candidate_ids=candidate_ids,
            features=features,
            lm_head_weight=lm_head_weight,
        )

    def table_memory_bytes(self) -> dict[str, int]:
        """Return the resident parameter bytes for each enabled hash table."""
        result: dict[str, int] = {}
        if self.bigram_table is not None:
            result["bigram_table_bytes"] = (
                self.bigram_table.weight.numel()
                * self.bigram_table.weight.element_size()
            )
        if self.trigram_table is not None:
            result["trigram_table_bytes"] = (
                self.trigram_table.weight.numel()
                * self.trigram_table.weight.element_size()
            )
        result["hash_table_bytes"] = sum(result.values())
        return result

    @staticmethod
    def _probe_collision_stats(
        probe_keys: torch.Tensor,
        *,
        buckets: int,
        lookups: int,
        prefix: str,
    ) -> dict[str, int | float]:
        """Summarize collisions over distinct n-grams, excluding repeated inputs."""
        unique_ngrams = probe_keys.shape[1]
        if unique_ngrams == 0:
            return {
                f"{prefix}_lookups": lookups,
                f"{prefix}_unique_ngrams": 0,
                f"{prefix}_collision_rate": 0.0,
                f"{prefix}_joint_collision_rate": 0.0,
                f"{prefix}_bucket_load": 0.0,
            }

        unique_buckets = [
            int(torch.unique(probe_keys[probe]).numel())
            for probe in range(probe_keys.shape[0])
        ]
        mean_unique_buckets = sum(unique_buckets) / len(unique_buckets)
        joint_unique = int(torch.unique(probe_keys.mT, dim=0).shape[0])
        return {
            f"{prefix}_lookups": lookups,
            f"{prefix}_unique_ngrams": unique_ngrams,
            f"{prefix}_collision_rate": 1.0 - mean_unique_buckets / unique_ngrams,
            f"{prefix}_joint_collision_rate": 1.0 - joint_unique / unique_ngrams,
            f"{prefix}_bucket_load": mean_unique_buckets / buckets,
        }

    def collision_stats(
        self,
        *,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> dict[str, int | float]:
        """Return observed bucket load and collision rates for a candidate batch.

        Exact duplicate n-grams are deduplicated before collision rates are computed,
        so repeated data is not mislabeled as a hash collision. ``joint`` means that
        all probe bucket IDs collide simultaneously.
        """
        candidates = self._as_long(candidate_ids)
        expanded_previous = (
            self._as_long(previous_ids).unsqueeze(-1).expand_as(candidates)
        )
        result: dict[str, int | float] = {}

        if self.bigram_table is not None:
            bigrams = torch.stack(
                (expanded_previous.reshape(-1), candidates.reshape(-1)),
                dim=-1,
            )
            unique_bigrams = torch.unique(bigrams, dim=0)
            bigram_keys = self.hash_bigram_probes(
                unique_bigrams[:, 0], unique_bigrams[:, 1].unsqueeze(-1)
            ).squeeze(-1)
            result.update(
                self._probe_collision_stats(
                    bigram_keys,
                    buckets=self.bigram_buckets,
                    lookups=candidates.numel() * self.num_hashes,
                    prefix="bigram",
                )
            )

        if self.trigram_table is not None:
            expanded_previous2 = (
                self._as_long(previous2_ids).unsqueeze(-1).expand_as(candidates)
            )
            trigrams = torch.stack(
                (
                    expanded_previous2.reshape(-1),
                    expanded_previous.reshape(-1),
                    candidates.reshape(-1),
                ),
                dim=-1,
            )
            unique_trigrams = torch.unique(trigrams, dim=0)
            trigram_keys = self.hash_trigram_probes(
                unique_trigrams[:, 0],
                unique_trigrams[:, 1],
                unique_trigrams[:, 2].unsqueeze(-1),
            ).squeeze(-1)
            result.update(
                self._probe_collision_stats(
                    trigram_keys,
                    buckets=self.trigram_buckets,
                    lookups=candidates.numel() * self.num_hashes,
                    prefix="trigram",
                )
            )
        return result

    @staticmethod
    def _benchmark_call(
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup: int,
        repeats: int,
    ) -> float:
        for _ in range(warmup):
            function()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                function()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end) / repeats)

        start_time = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - start_time) * 1_000.0 / repeats

    @torch.inference_mode()
    def benchmark_candidates(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
        warmup: int = 5,
        repeats: int = 20,
    ) -> dict[str, float]:
        """Measure lookup, refinement, and end-to-end selector latency in ms.

        This explicit diagnostic synchronizes CUDA and must not be called from the
        training forward path.
        """
        if warmup < 0 or repeats <= 0:
            raise ValueError("warmup must be >= 0 and repeats must be > 0")
        device = hidden_states.device
        features = self.candidate_features(
            hidden_states,
            previous_ids,
            previous2_ids,
            candidate_ids,
        )

        def lookup() -> torch.Tensor:
            return self.candidate_features(
                hidden_states,
                previous_ids,
                previous2_ids,
                candidate_ids,
            )

        def refine() -> torch.Tensor:
            return self._score_from_features(
                unary_logits=unary_logits,
                hidden_states=hidden_states,
                candidate_ids=candidate_ids,
                features=features,
                lm_head_weight=lm_head_weight,
            )

        def total() -> torch.Tensor:
            return self.score_candidates(
                unary_logits=unary_logits,
                hidden_states=hidden_states,
                previous_ids=previous_ids,
                previous2_ids=previous2_ids,
                candidate_ids=candidate_ids,
                lm_head_weight=lm_head_weight,
            )

        return {
            "hash_lookup_ms": self._benchmark_call(
                lookup, device=device, warmup=warmup, repeats=repeats
            ),
            "candidate_refinement_ms": self._benchmark_call(
                refine, device=device, warmup=warmup, repeats=repeats
            ),
            "selector_total_ms": self._benchmark_call(
                total, device=device, warmup=warmup, repeats=repeats
            ),
        }

    def diagnostics(
        self,
        *,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_ids: torch.Tensor,
        previous2_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
        warmup: int = 5,
        repeats: int = 20,
    ) -> dict[str, int | float]:
        """Collect the v2.0 selector latency, collision, and table-memory report."""
        report: dict[str, int | float] = {}
        report.update(self.table_memory_bytes())
        report.update(
            self.collision_stats(
                previous_ids=previous_ids,
                previous2_ids=previous2_ids,
                candidate_ids=candidate_ids,
            )
        )
        report.update(
            self.benchmark_candidates(
                unary_logits=unary_logits,
                hidden_states=hidden_states,
                previous_ids=previous_ids,
                previous2_ids=previous2_ids,
                candidate_ids=candidate_ids,
                lm_head_weight=lm_head_weight,
                warmup=warmup,
                repeats=repeats,
            )
        )
        return report
