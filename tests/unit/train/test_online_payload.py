import pytest
import torch

from speculators.train.online_payload import check_online_payload


def _payload():
    return {
        "token_ids": torch.tensor([1, 2, 3]),
        "hidden_states": torch.zeros(3, 2, 4),
    }


def test_online_payload_accepts_matching_hidden_states():
    check_online_payload(_payload(), [1, 2, 3])


def test_online_payload_rejects_mismatched_token_ids():
    with pytest.raises(ValueError, match="Token ids don't match"):
        check_online_payload(_payload(), [1, 2, 4])


def test_online_payload_rejects_wrong_hidden_state_rank():
    payload = _payload()
    payload["hidden_states"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match=r"\[tokens, layers, hidden_size\]"):
        check_online_payload(payload, [1, 2, 3])


def test_online_payload_rejects_nan_hidden_states():
    payload = _payload()
    payload["hidden_states"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        check_online_payload(payload, [1, 2, 3])


def test_online_payload_rejects_hidden_state_length_mismatch():
    payload = _payload()
    payload["hidden_states"] = torch.zeros(2, 2, 4)
    with pytest.raises(ValueError, match="Sequence length of hidden states"):
        check_online_payload(payload, [1, 2, 3])
