"""Unit tests for paper-faithful P-EAGLE COD and sequence partitioning."""

from collections import Counter

import pytest
import torch

from speculators.losses import LossConfig, eager
from speculators.models.peagle.data import (
    generate_cod_sample_indices,
    partition_cod_sample_indices,
)
from speculators.models.peagle.metrics import compute_metrics


def _loss_mask(seq_length: int) -> torch.Tensor:
    return torch.ones(1, seq_length, dtype=torch.float32)


def _sample(
    seq_length: int = 64,
    *,
    num_depths: int = 6,
    ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    return generate_cod_sample_indices(
        seq_length=seq_length,
        loss_mask=_loss_mask(seq_length),
        num_depths=num_depths,
        down_sample_ratio=ratio,
    )


class TestCODSampling:
    def test_depth0_is_the_complete_sequence(self):
        anchor_pos, depth = _sample()
        torch.testing.assert_close(anchor_pos[depth == 0], torch.arange(64))

    def test_geometric_depth_counts_match_paper_schedule(self):
        seq_length = 4096
        ratio = 0.8
        anchor_pos, depth = _sample(seq_length, num_depths=8, ratio=ratio)
        del anchor_pos
        assert [(depth == d).sum().item() for d in range(8)] == [
            seq_length,
            *[int(seq_length * ratio**d) for d in range(1, 8)],
        ]

    def test_every_sample_has_its_previous_depth_dependency(self):
        anchor_pos, depth = _sample()
        positions = anchor_pos + depth
        positions_by_depth = {
            d: set(positions[depth == d].tolist())
            for d in range(int(depth.max().item()) + 1)
        }
        for d in range(1, len(positions_by_depth)):
            for position in positions_by_depth[d]:
                assert position - 1 in positions_by_depth[d - 1]

    def test_never_creates_negative_anchors(self):
        anchor_pos, _depth = _sample(ratio=1.0)
        assert torch.all(anchor_pos >= 0)

    def test_sampling_does_not_cross_packed_document_boundaries(self):
        document_ids = torch.tensor([[0] * 8 + [1] * 8])
        torch.manual_seed(0)
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=16,
            loss_mask=_loss_mask(16),
            document_ids=document_ids,
            num_depths=5,
            down_sample_ratio=1.0,
        )
        positions = anchor_pos + depth
        assert torch.equal(
            document_ids[0, anchor_pos],
            document_ids[0, positions],
        )

    def test_minimum_ratio_is_an_optional_extension(self):
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=100,
            loss_mask=_loss_mask(100),
            num_depths=5,
            down_sample_ratio=0.2,
            down_sample_ratio_min=0.1,
        )
        del anchor_pos
        assert [(depth == d).sum().item() for d in range(5)] == [100, 20, 10, 10, 10]


class TestSequencePartitioning:
    def test_one_segment_is_identical_to_unpartitioned_cod(self):
        anchor_pos, depth = _sample()
        (partition,) = partition_cod_sample_indices(
            anchor_pos,
            depth,
            seq_length=64,
            num_segments=1,
        )
        torch.testing.assert_close(partition.anchor_pos, anchor_pos)
        torch.testing.assert_close(partition.depth, depth)
        assert torch.all(partition.supervision_mask)

    def test_every_sample_contributes_supervision_exactly_once(self):
        anchor_pos, depth = _sample()
        partitions = partition_cod_sample_indices(
            anchor_pos,
            depth,
            seq_length=64,
            num_segments=4,
        )
        expected = Counter(
            zip(depth.tolist(), (anchor_pos + depth).tolist(), strict=True)
        )
        observed: Counter[tuple[int, int]] = Counter()
        for partition in partitions:
            positions = partition.anchor_pos + partition.depth
            observed.update(
                zip(
                    partition.depth[partition.supervision_mask].tolist(),
                    positions[partition.supervision_mask].tolist(),
                    strict=True,
                )
            )
        assert observed == expected

    def test_cross_depth_dependencies_stay_in_the_same_segment(self):
        anchor_pos, depth = _sample()
        partitions = partition_cod_sample_indices(
            anchor_pos,
            depth,
            seq_length=64,
            num_segments=4,
        )
        assignment: dict[tuple[int, int], int] = {}
        for segment, partition in enumerate(partitions):
            positions = partition.anchor_pos + partition.depth
            for d, position in zip(
                partition.depth[partition.supervision_mask].tolist(),
                positions[partition.supervision_mask].tolist(),
                strict=True,
            ):
                assignment[(d, position)] = segment

        for (d, position), segment in assignment.items():
            if d >= 2:
                assert assignment[(d - 1, position - 1)] == segment

    def test_each_segment_has_the_paper_cumulative_depth0_prefix(self):
        seq_length = 17
        num_segments = 4
        anchor_pos, depth = _sample(seq_length, num_depths=5)
        partitions = partition_cod_sample_indices(
            anchor_pos,
            depth,
            seq_length=seq_length,
            num_segments=num_segments,
        )
        for segment, partition in enumerate(partitions):
            prefix_end = ((segment + 1) * seq_length + num_segments - 1) // num_segments
            depth0_positions = partition.anchor_pos[partition.depth == 0]
            torch.testing.assert_close(depth0_positions, torch.arange(prefix_end))

            depth0_supervision = partition.supervision_mask[partition.depth == 0]
            expected = depth0_positions * num_segments // seq_length == segment
            torch.testing.assert_close(depth0_supervision, expected)

    def test_missing_dependency_is_rejected(self):
        anchor_pos = torch.tensor([0, 1, 2, 3, 0])
        depth = torch.tensor([0, 0, 0, 0, 2])
        with pytest.raises(ValueError, match="without its depth-1 dependency"):
            partition_cod_sample_indices(
                anchor_pos,
                depth,
                seq_length=4,
                num_segments=2,
            )


def test_partitioned_loss_and_metrics_equal_unpartitioned_objective():
    seq_length = 24
    num_depths = 5
    anchor_pos, depth = _sample(seq_length, num_depths=num_depths)
    positions = anchor_pos + depth
    partitions = partition_cod_sample_indices(
        anchor_pos,
        depth,
        seq_length=seq_length,
        num_segments=3,
    )

    torch.manual_seed(7)
    logits = torch.randn(1, anchor_pos.numel(), 19, requires_grad=True)
    targets = torch.randn_like(logits)
    loss_mask = _loss_mask(seq_length)
    loss_config: LossConfig = {"kl_div": (eager.kl_div_loss, 1.0)}
    full_loss, full_metrics = compute_metrics(
        logits,
        targets,
        loss_mask,
        anchor_pos,
        depth,
        num_depths,
        loss_config,
    )

    global_indices = {
        (d, position): index
        for index, (d, position) in enumerate(
            zip(depth.tolist(), positions.tolist(), strict=True)
        )
    }
    partitioned_loss = torch.zeros(())
    partitioned_metrics: dict[str, torch.Tensor] = {}
    global_loss_count = loss_mask[:, positions].sum()
    for partition in partitions:
        partition_positions = partition.anchor_pos + partition.depth
        indices = torch.tensor(
            [
                global_indices[(d, position)]
                for d, position in zip(
                    partition.depth.tolist(),
                    partition_positions.tolist(),
                    strict=True,
                )
            ]
        )
        segment_loss, segment_metrics = compute_metrics(
            logits[:, indices],
            targets[:, indices],
            loss_mask,
            partition.anchor_pos,
            partition.depth,
            num_depths,
            loss_config,
            supervision_mask=partition.supervision_mask,
            global_loss_count=global_loss_count,
        )
        partitioned_loss = partitioned_loss + segment_loss
        for key, value in segment_metrics.items():
            partitioned_metrics[key] = partitioned_metrics.get(key, 0) + value

    torch.testing.assert_close(partitioned_loss, full_loss)
    for key, value in full_metrics.items():
        torch.testing.assert_close(partitioned_metrics[key], value)
