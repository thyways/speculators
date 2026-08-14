from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm")

from hs_connectors.verifier_kv_connector import (
    PendingVerifierKVSave,
    VerifierKVConnector,
)
from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


def _cache(offset: int) -> torch.Tensor:
    return torch.arange(4 * 2 * 3 * 6, dtype=torch.float32).reshape(4, 2, 3, 6) + offset


@pytest.fixture
def connector(tmp_path):
    hidden_spec = HiddenStateCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    verifier_spec = FullAttentionSpec(
        block_size=3,
        num_kv_heads=2,
        head_size=3,
        dtype=torch.float32,
    )
    cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["hidden_states_cache"], hidden_spec),
            KVCacheGroupSpec(
                [
                    "model.layers.3.self_attn",
                    "model.layers.11.self_attn",
                ],
                verifier_spec,
            ),
        ],
    )
    transfer_config = KVTransferConfig(
        kv_connector="VerifierKVConnector",
        kv_role="kv_producer",
        kv_connector_module_path="hs_connectors.verifier_kv_connector",
        kv_connector_extra_config={
            "shared_storage_path": str(tmp_path),
            "verifier_kv_layer_ids": [3, 11],
            "online_only": True,
            "use_synchronization_lock": False,
        },
    )
    vllm_config = VllmConfig(
        kv_transfer_config=transfer_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
    )
    vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1])
        )
    )
    instance = VerifierKVConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        cache_config,
    )
    yield instance
    instance._executor.shutdown(wait=True)


def test_resolves_independent_hidden_and_verifier_block_sizes(connector):
    assert connector._cache_kv_group_id == 0
    assert connector._block_size == 2
    assert connector.selected_verifier_kv.cache_group_id == 1
    assert connector.selected_verifier_kv.block_size == 3


def test_request_finished_all_groups_stages_verifier_blocks(connector, tmp_path):
    filename = str(tmp_path / "payload.safetensors")
    connector._request_filenames["request"] = filename
    request = Request("request", [7, 8, 9], SamplingParams(), None)

    delayed, params = connector.request_finished_all_groups(
        request,
        ([4, 5], [8, 9]),
    )

    assert delayed
    assert params == {"hidden_states_path": filename}
    pending = connector._pending_saves["request"]
    assert isinstance(pending, PendingVerifierKVSave)
    assert pending.block_ids == [4, 5]
    assert pending.verifier_kv_block_ids == [8, 9]


@pytest.mark.parametrize(
    "finished_request",
    [
        SimpleNamespace(has_encoder_inputs=True, prompt_token_ids=[1]),
        Request(
            "embeddings",
            None,
            SamplingParams(),
            None,
            prompt_embeds=torch.randn(2, 4),
        ),
    ],
)
def test_request_finished_all_groups_rejects_non_text_token_prompts(
    connector, finished_request
):
    with pytest.raises(ValueError, match="text token-ID prompts only"):
        connector.request_finished_all_groups(finished_request, ([0], [0]))


def test_submit_async_write_uses_each_cache_block_size_and_writes_positions(
    connector, tmp_path, monkeypatch
):
    connector._is_tp_rank_zero = True
    connector._kv_cache = torch.zeros(4, 2, 4)
    connector._all_kv_caches = {
        "model.layers.3.self_attn": _cache(0),
        "model.layers.11.self_attn": _cache(1000),
    }

    class _Event:
        def record(self, *_args):
            return None

    class _Stream:
        def wait_event(self, _event):
            return None

    class _StreamContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class _Future:
        def add_done_callback(self, callback):
            self.callback = callback

    captured: dict[str, Any] = {}
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: _StreamContext())
    copy_stream = _Stream()
    monkeypatch.setattr(connector, "_get_copy_stream", lambda: copy_stream)
    monkeypatch.setattr(
        "hs_connectors.verifier_kv_connector.hidden_states_connector."
        "extract_from_kv_cache",
        lambda _cache_tensor, slots, num_tokens: (
            captured.setdefault("hidden_slots", slots.cpu().clone())
            .to(torch.float32)
            .view(num_tokens, 1)
        ),
    )

    real_empty_like = torch.empty_like

    def _empty_like(tensor, *, device=None, pin_memory=False):
        del pin_memory
        return real_empty_like(tensor, device=device)

    monkeypatch.setattr(torch, "empty_like", _empty_like)

    future = _Future()

    def _submit(write_fn, tensors, event, filename, lock_fd):
        captured.update(
            tensors=tensors,
            event=event,
            filename=filename,
            lock_fd=lock_fd,
            write_fn=write_fn,
        )
        return future

    monkeypatch.setattr(connector._executor, "submit", _submit)
    pending = PendingVerifierKVSave(
        req_id="request",
        filename=str(tmp_path / "payload.safetensors"),
        token_ids=torch.tensor([10, 11, 12]),
        block_ids=[1, 0],
        verifier_kv_block_ids=[2],
    )

    connector._submit_async_write(pending)

    assert captured["hidden_slots"].tolist() == [2, 3, 0]
    tensors = captured["tensors"]
    assert torch.equal(tensors["position_ids"], torch.arange(3))
    assert torch.equal(tensors["verifier_kv_layer_ids"], torch.tensor([3, 11]))
    expected_key = _cache(0)[2, :, 0, :3]
    assert torch.equal(tensors["verifier_keys"][0, 0], expected_key)
    assert "request" in connector._req_copy_events
    assert connector._req_futures["request"] is future
