"""Closed-interval merging with explicit invalid/out-of-scope barriers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Iterable


BARRIER_STATUSES = frozenset({"invalid", "out_of_scope"})
HARD_STATUSES = frozenset({"hard", "accept", "extremely_hard"})


@dataclass(frozen=True, slots=True)
class FrameInterval:
    video_id: str
    start: int
    end: int
    sample_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if self.start > self.end:
            raise ValueError(f"invalid closed interval [{self.start}, {self.end}]")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class ClassifiedInterval:
    video_id: str
    start: int
    end: int
    status: str
    sample_id: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if self.start > self.end:
            raise ValueError(f"invalid closed interval [{self.start}, {self.end}]")


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _merge_plain(
    intervals: Iterable[FrameInterval], *, merge_adjacent: bool
) -> list[FrameInterval]:
    ordered = sorted(intervals, key=lambda item: (item.video_id, item.start, item.end))
    merged: list[FrameInterval] = []
    for interval in ordered:
        if not merged or merged[-1].video_id != interval.video_id:
            merged.append(interval)
            continue
        previous = merged[-1]
        limit = previous.end + (1 if merge_adjacent else 0)
        if interval.start > limit:
            merged.append(interval)
            continue
        merged[-1] = FrameInterval(
            video_id=previous.video_id,
            start=previous.start,
            end=max(previous.end, interval.end),
            sample_ids=_unique_sorted((*previous.sample_ids, *interval.sample_ids)),
            reasons=_unique_sorted((*previous.reasons, *interval.reasons)),
        )
    return merged


def _subtract_barriers(
    interval: FrameInterval, barriers: Iterable[FrameInterval]
) -> list[FrameInterval]:
    pieces = [interval]
    for barrier in barriers:
        next_pieces: list[FrameInterval] = []
        for piece in pieces:
            if barrier.end < piece.start or barrier.start > piece.end:
                next_pieces.append(piece)
                continue
            if piece.start < barrier.start:
                next_pieces.append(
                    FrameInterval(
                        piece.video_id,
                        piece.start,
                        barrier.start - 1,
                        piece.sample_ids,
                        piece.reasons,
                    )
                )
            if barrier.end < piece.end:
                next_pieces.append(
                    FrameInterval(
                        piece.video_id,
                        barrier.end + 1,
                        piece.end,
                        piece.sample_ids,
                        piece.reasons,
                    )
                )
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def merge_intervals(
    intervals: Iterable[FrameInterval],
    *,
    barriers: Iterable[FrameInterval] = (),
    merge_adjacent: bool = True,
    min_length: int = 1,
) -> tuple[FrameInterval, ...]:
    """Union hard intervals per video, removing every barrier frame.

    Both kinds of interval are closed.  With the default adjacency rule,
    ``[1,3]`` and ``[4,6]`` merge.  A barrier is first unioned per video and
    then subtracted from candidate coverage, so invalid/out-of-scope frames
    cannot be copied into a final segment or bridged by neighbouring samples.
    """

    if isinstance(min_length, bool) or min_length <= 0:
        raise ValueError("min_length must be a positive integer")
    candidates = list(intervals)
    barrier_union = _merge_plain(barriers, merge_adjacent=True)
    barriers_by_video = {
        video_id: tuple(group)
        for video_id, group in groupby(barrier_union, key=lambda item: item.video_id)
    }
    pieces_with_provenance: list[FrameInterval] = []
    for candidate in candidates:
        pieces_with_provenance.extend(
            _subtract_barriers(candidate, barriers_by_video.get(candidate.video_id, ()))
        )
    # Merge only after subtraction.  This retains contributor metadata on the
    # side(s) actually covered by each source sample, while a removed barrier
    # leaves at least one frame of separation and therefore cannot be bridged.
    merged = _merge_plain(pieces_with_provenance, merge_adjacent=merge_adjacent)
    return tuple(piece for piece in merged if piece.length >= min_length)


def merge_classified_intervals(
    decisions: Iterable[ClassifiedInterval],
    *,
    hard_statuses: Iterable[str] = HARD_STATUSES,
    barrier_statuses: Iterable[str] = BARRIER_STATUSES,
    min_length: int = 1,
) -> tuple[FrameInterval, ...]:
    """Convenience adapter from per-triplet decisions to final frame segments."""

    accepted = frozenset(hard_statuses)
    blocked = frozenset(barrier_statuses)
    candidates: list[FrameInterval] = []
    barriers: list[FrameInterval] = []
    for decision in decisions:
        sample_ids = () if decision.sample_id is None else (decision.sample_id,)
        interval = FrameInterval(
            video_id=decision.video_id,
            start=decision.start,
            end=decision.end,
            sample_ids=sample_ids,
            reasons=decision.reasons,
        )
        if decision.status in accepted:
            candidates.append(interval)
        elif decision.status in blocked:
            barriers.append(interval)
    return merge_intervals(candidates, barriers=barriers, min_length=min_length)


# Short public alias used by finalization code.
merge_segments = merge_intervals
