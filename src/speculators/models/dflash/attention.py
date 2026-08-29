import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_dual_stream_anchor_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    force_causal_local: bool = False,
):
    """Build the independent prefix and local masks for an anchored block.

    The prefix stream reads only verifier tokens before an anchor.  The local
    stream reads only the synthetic tokens in the same anchored block.  Returned
    separately so a caller can either combine them into one mask (see
    :func:`create_anchor_block_mask_mod`) or drive two attention calls with
    independent softmax normalizers.
    """
    non_causal = not force_causal_local and (
        sliding_window is None or sliding_window_non_causal
    )
    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()
    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    num_anchors = anchor_positions.numel()
    query_length = num_anchors * block_size
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def prefix_mask_mod(_b, _h, query_index, key_index):
        query_anchor = query_anchor_positions[query_index]
        query_document = document_ids[query_anchor]
        key_document = document_ids[key_index]
        same_document = (query_document == key_document) & (query_document != -1)
        before_anchor = key_index < query_anchor
        in_window = (
            key_index >= query_anchor - sliding_window
            if sliding_window is not None
            else True
        )
        return same_document & before_anchor & in_window

    def local_mask_mod(_b, _h, query_index, key_index):
        same_block = query_index // block_size == key_index // block_size
        if not non_causal:
            same_block = same_block & (key_index <= query_index)
        return same_block

    return (
        prefix_mask_mod,
        local_mask_mod,
        query_length,
        total_seq_len,
    )


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    force_causal_local: bool = False,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn

    Returns:
        mask_mod, q_len, kv_len
    """
    prefix_mask_mod, local_mask_mod, q_len, _ = create_dual_stream_anchor_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
        force_causal_local=force_causal_local,
    )

    def shifted_local_mask_mod(b, h, q_idx, kv_idx):
        kv_is_local = kv_idx >= total_seq_len
        local_index = kv_idx - total_seq_len
        return kv_is_local & local_mask_mod(b, h, q_idx, local_index)

    def bounded_prefix_mask_mod(b, h, q_idx, kv_idx):
        kv_is_prefix = kv_idx < total_seq_len
        safe_index = torch.remainder(kv_idx, total_seq_len)
        return kv_is_prefix & prefix_mask_mod(b, h, q_idx, safe_index)

    return (
        or_masks(bounded_prefix_mask_mod, shifted_local_mask_mod),
        q_len,
        total_seq_len + q_len,
    )
