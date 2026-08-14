import pytest
import torch

from speculators.train.online_payload import check_online_payload


def _payload():
    return {
        "token_ids": torch.tensor([1, 2, 3]),
        "hidden_states": torch.zeros(3, 2, 4),
        "position_ids": torch.arange(3),
        "verifier_keys": torch.zeros(3, 2, 2, 4),
        "verifier_values": torch.ones(3, 2, 2, 4),
        "verifier_kv_layer_ids": torch.tensor([3, 11]),
    }


def test_online_payload_checks_hidden_states_and_kv():
    check_online_payload(
        _payload(),
        [1, 2, 3],
        require_verifier_kv=True,
        expected_verifier_kv_shape=(2, 2, 4),
        expected_verifier_kv_layer_ids=[3, 11],
    )


def test_online_payload_requires_complete_kv():
    payload = _payload()
    del payload["verifier_values"]
    with pytest.raises(ValueError, match="Incomplete"):
        check_online_payload(payload, [1, 2, 3], require_verifier_kv=True)


def test_online_payload_rejects_nan_kv():
    payload = _payload()
    payload["verifier_keys"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        check_online_payload(payload, [1, 2, 3], require_verifier_kv=True)


def test_online_payload_rejects_wrong_kv_shape_and_layer_ids():
    with pytest.raises(ValueError, match="shape mismatch"):
        check_online_payload(
            _payload(),
            [1, 2, 3],
            require_verifier_kv=True,
            expected_verifier_kv_shape=(2, 1, 2),
        )
    with pytest.raises(ValueError, match="layer IDs mismatch"):
        check_online_payload(
            _payload(),
            [1, 2, 3],
            require_verifier_kv=True,
            expected_verifier_kv_layer_ids=[3, 15],
        )


def test_online_payload_requires_text_position_ids():
    payload = _payload()
    del payload["position_ids"]
    with pytest.raises(ValueError, match="missing text position_ids"):
        check_online_payload(payload, [1, 2, 3], require_verifier_kv=True)

    payload = _payload()
    payload["position_ids"] = torch.tensor([0, 1, 3])
    with pytest.raises(ValueError, match="text-only contiguous"):
        check_online_payload(payload, [1, 2, 3], require_verifier_kv=True)
