"""vLLM connector that exports the final hidden state and selected verifier K/V.

This extends vLLM's file-based ``ExampleHiddenStatesConnector`` so existing
online transfer plumbing can carry the logits teacher state while KV-native
training consumes real, already-rotated verifier cache entries from selected
full-attention layers.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, TypeAlias

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as hidden_states_connector,
)

from hs_connectors.verifier_kv import (
    SelectedVerifierKV,
    discover_selected_verifier_kv,
    extract_selected_verifier_kv,
)

ExampleHiddenStatesConnector: TypeAlias = (
    hidden_states_connector.ExampleHiddenStatesConnector
)
PendingSave: TypeAlias = hidden_states_connector.PendingSave

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

__all__ = ["VerifierKVConnector"]


@dataclass
class PendingVerifierKVSave(PendingSave):
    verifier_kv_block_ids: list[int]


class VerifierKVConnector(ExampleHiddenStatesConnector):
    """Store the final hidden state plus selected full-attention K/V."""

    def __init__(self, vllm_config, role, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)
        layer_ids = self._kv_transfer_config.get_from_extra_config(
            "verifier_kv_layer_ids", []
        )
        if not self._kv_transfer_config.get_from_extra_config("online_only", False):
            raise ValueError(
                "VerifierKVConnector is online-only; set online_only=true in "
                "kv_connector_extra_config"
            )
        self._selected_verifier_kv = discover_selected_verifier_kv(
            kv_cache_config, layer_ids
        )
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError(
                "VerifierKVConnector currently requires tensor_parallel_size=1; "
                "otherwise TP rank 0 would export only a KV-head shard."
            )
        self._all_kv_caches: dict[str, torch.Tensor] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        self._all_kv_caches = kv_caches
        # Validate selected names and physical shapes before serving requests.
        extract_selected_verifier_kv(
            kv_caches,
            self._selected_verifier_kv,
            block_ids=[],
            num_tokens=0,
        )

    def _submit_async_write(self, pending: PendingSave) -> None:
        if not isinstance(pending, PendingVerifierKVSave):
            raise TypeError(
                "VerifierKVConnector received metadata without verifier KV block IDs"
            )
        if not self._is_tp_rank_zero:
            return
        if self._kv_cache is None:
            raise RuntimeError("hidden-state KV cache was not registered")

        num_tokens = pending.token_ids.shape[0]
        block_ids_t = torch.tensor(pending.block_ids, dtype=torch.long)
        block_offsets = torch.arange(self._block_size, dtype=torch.long)
        hidden_slots = (
            block_ids_t[:, None] * self._block_size + block_offsets[None, :]
        ).reshape(-1)[:num_tokens]

        copy_stream = self._get_copy_stream()
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        with torch.cuda.stream(copy_stream):
            hidden_slots = hidden_slots.to(
                device=self._kv_cache.device, non_blocking=True
            )
            hidden_states_gpu = hidden_states_connector.extract_from_kv_cache(
                self._kv_cache, hidden_slots, num_tokens
            )
            verifier_keys_gpu, verifier_values_gpu = extract_selected_verifier_kv(
                self._all_kv_caches,
                self._selected_verifier_kv,
                pending.verifier_kv_block_ids,
                num_tokens,
            )

            pinned_hidden = torch.empty_like(
                hidden_states_gpu, device="cpu", pin_memory=True
            )
            pinned_keys = torch.empty_like(
                verifier_keys_gpu, device="cpu", pin_memory=True
            )
            pinned_values = torch.empty_like(
                verifier_values_gpu, device="cpu", pin_memory=True
            )
            pinned_hidden.copy_(hidden_states_gpu, non_blocking=True)
            pinned_keys.copy_(verifier_keys_gpu, non_blocking=True)
            pinned_values.copy_(verifier_values_gpu, non_blocking=True)

        copy_done = torch.cuda.Event()
        copy_done.record(copy_stream)

        tensors = {
            "hidden_states": pinned_hidden,
            "token_ids": pending.token_ids.clone(),
            "position_ids": torch.arange(num_tokens, dtype=torch.long),
            "verifier_keys": pinned_keys,
            "verifier_values": pinned_values,
            "verifier_kv_layer_ids": torch.tensor(
                self._selected_verifier_kv.layer_ids, dtype=torch.long
            ),
        }

        prior = self._req_futures.get(pending.req_id)
        if prior is not None:
            raise RuntimeError(
                f"another KV transfer request uses req_id={pending.req_id!r}"
            )
        os.makedirs(os.path.dirname(pending.filename), exist_ok=True)

        lock_fd = self._lock_fds.pop(pending.req_id, None)
        if lock_fd is None and self.use_lock:
            lock_path = pending.filename + ".lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        future = self._executor.submit(
            self._write_tensors,
            tensors,
            copy_done,
            pending.filename,
            lock_fd,
        )
        self._req_copy_events[pending.req_id] = copy_done
        self._req_futures[pending.req_id] = future
        future.add_done_callback(partial(self._on_write_done, pending.req_id))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict | None]:
        if request.has_encoder_inputs or request.prompt_token_ids is None:
            raise ValueError(
                "VerifierKVConnector currently supports text token-ID prompts only; "
                "multimodal inputs and prompt embeddings require exported MRoPE IDs."
            )
        required_group = max(
            self._cache_kv_group_id,
            self._selected_verifier_kv.cache_group_id,
        )
        if len(block_ids) <= required_group:
            raise ValueError(
                f"vLLM supplied {len(block_ids)} KV block tables, but verifier K/V "
                f"export requires group {required_group}"
            )
        delayed, params = super().request_finished(
            request, block_ids[self._cache_kv_group_id]
        )
        pending = self._pending_saves.get(request.request_id)
        if pending is None:
            raise RuntimeError(
                f"hidden-state connector did not stage request {request.request_id!r}"
            )
        self._pending_saves[request.request_id] = PendingVerifierKVSave(
            req_id=pending.req_id,
            filename=pending.filename,
            token_ids=pending.token_ids,
            block_ids=pending.block_ids,
            verifier_kv_block_ids=list(
                block_ids[self._selected_verifier_kv.cache_group_id]
            ),
        )
        return delayed, params

    @property
    def selected_verifier_kv(self) -> SelectedVerifierKV:
        """Expose resolved metadata for diagnostics and tests."""

        return self._selected_verifier_kv
