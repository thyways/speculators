"""COD sampling and dependency-aware sequence partitioning for P-EAGLE."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CODPartition:
    """One sequence partition and the positions that contribute supervision."""

    anchor_pos: torch.Tensor
    depth: torch.Tensor
    supervision_mask: torch.Tensor


def _validate_sequence_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    seq_length: int,
) -> torch.Tensor:
    if tensor.ndim > 1:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1, got {tensor.shape}")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1 or tensor.shape[0] != seq_length:
        raise ValueError(
            f"{name} must have shape [{seq_length}] or [1, {seq_length}], "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _valid_successors(
    positions: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return valid ``p + 1`` positions without crossing sequence/doc boundaries."""
    successors = positions + 1
    in_bounds = successors < valid_mask.shape[0]
    positions = positions[in_bounds]
    successors = successors[in_bounds]
    keep = valid_mask[successors]
    if document_ids is not None:
        keep = keep & (document_ids[positions] == document_ids[successors])
        keep = keep & (document_ids[successors] != -1)
    return successors[keep]


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.0,
    document_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample P-EAGLE positions with Conditional Drop-token (COD).

    Depth 0 retains the complete sequence. At depth ``d > 0``, COD targets
    ``floor(n * max(r**d, r_min))`` valid positions, sampled from successors of
    the retained positions at depth ``d - 1``. This conditional construction
    guarantees that every retained position ``p`` at depth ``d`` has its
    dependency ``p - 1`` at depth ``d - 1``.

    ``anchor_pos`` stores the depth-0 start of each sampled rollout; the original
    sequence position represented by an entry is ``anchor_pos + depth``.
    """
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if num_depths <= 0:
        raise ValueError(f"num_depths must be positive, got {num_depths}")
    if not 0 < down_sample_ratio <= 1:
        raise ValueError(
            f"down_sample_ratio must be in (0, 1], got {down_sample_ratio}"
        )
    if not 0 <= down_sample_ratio_min <= 1:
        raise ValueError(
            f"down_sample_ratio_min must be in [0, 1], got {down_sample_ratio_min}"
        )

    loss_mask_1d = _validate_sequence_tensor(
        loss_mask, name="loss_mask", seq_length=seq_length
    )
    valid_mask = loss_mask_1d != 0
    device = loss_mask.device

    document_ids_1d = None
    if document_ids is not None:
        document_ids_1d = _validate_sequence_tensor(
            document_ids.to(device), name="document_ids", seq_length=seq_length
        )

    valid_positions = torch.where(valid_mask)[0]
    base_count = valid_positions.numel()

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]

    # A depth-1 position p depends on depth-0 position p-1. Build this first
    # candidate pool explicitly so position zero and packed-document boundaries
    # can never create negative/cross-document anchors.
    depth0_positions = torch.arange(seq_length, device=device)
    prev_positions = _valid_successors(
        depth0_positions,
        valid_mask=valid_mask,
        document_ids=document_ids_1d,
    )

    for d in range(1, num_depths):
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = min(int(base_count * ratio), prev_positions.numel())
        if sample_size <= 0:
            break

        random_selection = torch.randperm(prev_positions.numel(), device=device)[
            :sample_size
        ]
        sampled_positions = prev_positions[random_selection].sort()[0]

        sample_indices.append(sampled_positions - d)
        n_per_depth.append(sample_size)
        prev_positions = _valid_successors(
            sampled_positions,
            valid_mask=valid_mask,
            document_ids=document_ids_1d,
        )

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), d, device=device, dtype=torch.long)
            for d, n in enumerate(n_per_depth)
        ]
    )
    return anchor_pos, depth


def _assign_cod_segments(
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    *,
    seq_length: int,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = anchor_pos + depth
    if torch.any((positions < 0) | (positions >= seq_length)):
        raise ValueError("COD sample contains positions outside the sequence")

    max_depth = int(depth.max().item())
    assignments_by_depth = torch.full(
        (max_depth + 1, seq_length),
        -1,
        dtype=torch.long,
        device=anchor_pos.device,
    )
    assignments = torch.full_like(depth, -1)

    for d in range(max_depth + 1):
        element_mask = depth == d
        depth_positions = positions[element_mask]
        if d <= 1:
            depth_assignments = torch.div(
                depth_positions * num_segments,
                seq_length,
                rounding_mode="floor",
            ).clamp_max(num_segments - 1)
        else:
            depth_assignments = assignments_by_depth[d - 1, depth_positions - 1]
            if torch.any(depth_assignments < 0):
                raise ValueError(
                    f"depth {d} contains a position without its depth-{d - 1} "
                    "dependency"
                )
        assignments[element_mask] = depth_assignments
        assignments_by_depth[d, depth_positions] = depth_assignments

    return positions, assignments


def partition_cod_sample_indices(
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    *,
    seq_length: int,
    num_segments: int,
) -> list[CODPartition]:
    """Implement Algorithm 1 from the P-EAGLE paper.

    Depths 0 and 1 are assigned by original position. Every position ``p`` at
    depth ``d >= 2`` inherits the segment of dependency ``(d - 1, p - 1)``.
    Segment ``s`` additionally receives the cumulative depth-0 prefix before
    boundary ``B[s + 1]``. Repeated prefix entries are marked as context-only
    through ``supervision_mask`` so every sampled training target contributes
    exactly once to the global objective.
    """
    if anchor_pos.ndim != 1 or depth.ndim != 1 or anchor_pos.shape != depth.shape:
        raise ValueError("anchor_pos and depth must be same-shaped 1D tensors")
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if not 1 <= num_segments <= seq_length:
        raise ValueError(
            f"num_segments must be in [1, {seq_length}], got {num_segments}"
        )
    if anchor_pos.numel() == 0:
        raise ValueError("COD samples must contain the complete depth-0 sequence")

    positions, assignments = _assign_cod_segments(
        anchor_pos,
        depth,
        seq_length=seq_length,
        num_segments=num_segments,
    )

    depth0_indices = torch.where(depth == 0)[0]
    if depth0_indices.numel() != seq_length:
        raise ValueError(
            "depth 0 must contain the complete sequence before partitioning"
        )

    partitions: list[CODPartition] = []
    for segment in range(num_segments):
        # B[s + 1] can be fractional. For integer positions, p < B[s + 1]
        # is equivalent to p < ceil(B[s + 1]).
        prefix_end = ((segment + 1) * seq_length + num_segments - 1) // num_segments
        prefix_mask = positions[depth0_indices] < prefix_end
        prefix_indices = depth0_indices[prefix_mask]

        mtp_indices = torch.where((depth > 0) & (assignments == segment))[0]
        partition_indices = torch.cat([prefix_indices, mtp_indices])

        prefix_supervision = assignments[prefix_indices] == segment
        supervision_mask = torch.cat(
            [
                prefix_supervision,
                torch.ones(
                    mtp_indices.numel(), dtype=torch.bool, device=anchor_pos.device
                ),
            ]
        )
        partitions.append(
            CODPartition(
                anchor_pos=anchor_pos[partition_indices],
                depth=depth[partition_indices],
                supervision_mask=supervision_mask,
            )
        )

    return partitions
