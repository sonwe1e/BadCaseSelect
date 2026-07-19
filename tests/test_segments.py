from __future__ import annotations

import pytest

from vfi_hard_miner.segments import (
    ClassifiedInterval,
    FrameInterval,
    merge_classified_intervals,
    merge_intervals,
)


def test_overlapping_and_adjacent_closed_intervals_merge_per_video() -> None:
    merged = merge_intervals(
        [
            FrameInterval("video-a", 4, 6, ("s3",)),
            FrameInterval("video-b", 1, 3, ("other",)),
            FrameInterval("video-a", 1, 3, ("s1",)),
            FrameInterval("video-a", 2, 4, ("s2",)),
        ]
    )

    assert merged == (
        FrameInterval("video-a", 1, 6, ("s1", "s2", "s3")),
        FrameInterval("video-b", 1, 3, ("other",)),
    )


def test_barriers_are_removed_and_force_a_break() -> None:
    merged = merge_intervals(
        [FrameInterval("video", 1, 8, ("wide",))],
        barriers=[FrameInterval("video", 4, 5)],
    )
    assert merged == (
        FrameInterval("video", 1, 3, ("wide",)),
        FrameInterval("video", 6, 8, ("wide",)),
    )


def test_barrier_only_affects_its_own_video_and_minimum_length_is_applied() -> None:
    merged = merge_intervals(
        [FrameInterval("a", 1, 5), FrameInterval("b", 1, 5)],
        barriers=[FrameInterval("a", 2, 4)],
        min_length=2,
    )
    assert merged == (FrameInterval("b", 1, 5),)


def test_classified_intervals_use_invalid_and_out_of_scope_as_barriers() -> None:
    decisions = [
        ClassifiedInterval("v", 1, 3, "accept", "s1", ("broken_structure",)),
        ClassifiedInterval("v", 3, 5, "hard", "s2", ("flicker",)),
        ClassifiedInterval("v", 4, 4, "invalid"),
        ClassifiedInterval("v", 7, 9, "review", "review-only"),
        ClassifiedInterval("v", 10, 12, "out_of_scope"),
    ]

    merged = merge_classified_intervals(decisions)

    assert merged == (
        FrameInterval("v", 1, 3, ("s1", "s2"), ("broken_structure", "flicker")),
        FrameInterval("v", 5, 5, ("s2",), ("flicker",)),
    )


def test_invalid_intervals_and_min_length_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid closed interval"):
        FrameInterval("v", 5, 4)
    with pytest.raises(ValueError, match="min_length"):
        merge_intervals([], min_length=0)
