"""Stable data contracts shared by the mining stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


DecisionLabel = Literal["accept", "review", "reject"]


@dataclass(frozen=True, slots=True)
class FrameRef:
    path: Path
    game_id: str
    scene_id: str
    video_id: str
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass(frozen=True, slots=True)
class TripletRef:
    sample_id: str
    video_key: str
    img0: FrameRef
    gt: FrameRef
    img1: FrameRef
    stride: int

    @property
    def frame_interval(self) -> tuple[int, int]:
        return self.img0.frame_index, self.img1.frame_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_key": self.video_key,
            "img0": self.img0.to_dict(),
            "gt": self.gt.to_dict(),
            "img1": self.img1.to_dict(),
            "stride": self.stride,
        }


@dataclass(frozen=True, slots=True)
class RegionBox:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("region must have positive area")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class SampleDecision:
    sample_id: str
    label: DecisionLabel
    valid: bool
    in_scope: bool
    p_wrong: float
    p_solvable: float
    reasons: list[str] = field(default_factory=list)
    regions: list[RegionBox] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = [region.to_dict() for region in self.regions]
        return payload


@dataclass(frozen=True, slots=True)
class MergedSegment:
    video_key: str
    start_frame: int
    end_frame: int
    sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
