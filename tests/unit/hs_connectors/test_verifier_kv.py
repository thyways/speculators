from types import SimpleNamespace

import pytest
import torch
from hs_connectors.transfer import FileBackend, MooncakeBackend
from hs_connectors.verifier_kv import (
    SelectedVerifierKV,
    build_slot_mapping,
    discover_selected_verifier_kv,
    extract_selected_verifier_kv,
)


def _metadata() -> SelectedVerifierKV:
    return SelectedVerifierKV(
        layer_ids=(3, 11),
        layer_names=("model.layers.3.self_attn", "model.layers.11.self_attn"),
        cache_group_id=1,
        block_size=3,
        num_kv_heads=2,
        head_dim=3,
    )


def _cache(offset: int, *, nhd_strides: bool = False) -> torch.Tensor:
    logical = torch.arange(4 * 2 * 3 * 6, dtype=torch.float32).reshape(4, 2, 3, 6)
    logical = logical + offset
    if not nhd_strides:
        return logical
    # Preserve vLLM's logical [B, H, N, 2D] axes while giving the tensor an
    # NHD physical layout/stride order.
    return logical.transpose(1, 2).contiguous().transpose(1, 2)


@pytest.mark.parametrize("nhd_strides", [False, True])
def test_extract_non_contiguous_blocks_and_layouts(nhd_strides):
    metadata = _metadata()
    caches = {
        metadata.layer_names[0]: _cache(0, nhd_strides=nhd_strides),
        metadata.layer_names[1]: _cache(1000, nhd_strides=nhd_strides),
    }
    keys, values = extract_selected_verifier_kv(
        caches, metadata, block_ids=[2, 0], num_tokens=3
    )
    assert keys.shape == (3, 2, 2, 3)
    assert values.shape == keys.shape
    normalized = _cache(0)
    expected_first = normalized[2, :, 0, :3]
    assert torch.equal(keys[0, 0], expected_first)
    assert torch.equal(keys[:, 1], keys[:, 0] + 1000)


def test_slot_mapping_capacity_and_shape_checks():
    assert build_slot_mapping([3, 1], 2, 3).tolist() == [6, 7, 2]
    with pytest.raises(ValueError, match="only hold"):
        build_slot_mapping([0], 2, 3)
    with pytest.raises(ValueError, match="one-dimensional"):
        build_slot_mapping([[0]], 2, 1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-negative"):
        build_slot_mapping([-1], 2, 1)


def test_discover_layers_and_missing_layer():
    spec = SimpleNamespace(block_size=16, num_kv_heads=2, head_size=256)
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.linear_attn"], kv_cache_spec=object()
            ),
            SimpleNamespace(
                layer_names=[
                    "model.layers.3.self_attn",
                    "model.layers.11.self_attn",
                ],
                kv_cache_spec=spec,
            ),
        ]
    )
    selected = discover_selected_verifier_kv(config, [3, 11])
    assert selected.cache_group_id == 1
    assert selected.block_size == 16
    with pytest.raises(ValueError, match="were not found"):
        discover_selected_verifier_kv(config, [7])


def test_file_backend_builds_out_of_tree_connector_config():
    args = SimpleNamespace(
        hidden_states_path="/tmp/online-kv",
        verifier_kv_layer_ids=[3, 11],
    )
    config = FileBackend.build_kv_transfer_config(args)  # type: ignore[arg-type]
    assert config["kv_connector"] == "VerifierKVConnector"
    assert config["kv_connector_module_path"] == "hs_connectors.verifier_kv_connector"
    assert config["kv_connector_extra_config"]["verifier_kv_layer_ids"] == [3, 11]
    assert config["kv_connector_extra_config"]["online_only"] is True


def test_mooncake_explicitly_rejects_verifier_kv():
    args = SimpleNamespace(
        verifier_kv_layer_ids=[3],
        mooncake_metadata_server="P2PHANDSHAKE",
        mooncake_master="127.0.0.1:50051",
        mooncake_protocol="tcp",
    )
    with pytest.raises(NotImplementedError, match="file"):
        MooncakeBackend.build_kv_transfer_config(args)  # type: ignore[arg-type]
