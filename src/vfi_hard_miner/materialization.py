"""Crash-safe per-video materialization for completed main-stage results."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from .config import AppConfig
from .manifest import read_jsonl
from .outputs import materialize_mapped_frames
from .segments import HARD_STATUSES, ClassifiedInterval, FrameInterval, merge_classified_intervals
from .state import TaskRecord


STAGING_DIRECTORY = ".vfi_hard_miner_staging"
_OUTPUT_MARKER = ".vfi_hard_miner_output.json"
_VIDEO_MARKER = ".vfi_hard_miner_video.json"


@dataclass(frozen=True, slots=True)
class MaterializationSummary:
    strategy: str
    staging_path: Path | None
    videos: int
    segments: int
    frames: int
    link_counts: dict[str, int]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _video_digest(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:20]


def segment_id(run_hash: str, segment: FrameInterval) -> str:
    payload = f"{run_hash}\0{segment.video_id}\0{segment.start}\0{segment.end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def safe_video_relative(video_id: str) -> Path:
    parent, separator, video_key = video_id.partition("::")
    if not separator:
        parent, video_key = "", video_id
    candidate = Path(parent) / (video_key or "video")
    if candidate.is_absolute() or candidate.drive:
        raise ValueError(f"unsafe absolute video_id: {video_id!r}")
    if not candidate.parts or any(
        part in {"", ".", "..", "/", "\\"} for part in candidate.parts
    ):
        raise ValueError(f"unsafe video_id path: {video_id!r}")
    if any(separator in video_key for separator in ("/", "\\")):
        raise ValueError(f"unsafe video key: {video_key!r}")
    return candidate


def _classified(record: Mapping[str, Any]) -> ClassifiedInterval:
    indices = tuple(int(value) for value in record["frame_indices"])
    return ClassifiedInterval(
        video_id=str(record["video_id"]),
        start=indices[0],
        end=indices[-1],
        status=str(record.get("status", "review")),
        sample_id=str(record["sample_id"]),
        reasons=tuple(str(value) for value in record.get("reasons", ())),
    )


def build_segments(records: Sequence[Mapping[str, Any]]) -> tuple[FrameInterval, ...]:
    segments = merge_classified_intervals(
        (_classified(record) for record in records),
        min_length=3,
    )
    hard_by_video: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        if str(record.get("status", "")) not in HARD_STATUSES:
            continue
        indices = tuple(int(value) for value in record["frame_indices"])
        hard_by_video.setdefault(str(record["video_id"]), []).append(
            (indices[0], indices[-1])
        )
    retained: list[FrameInterval] = []
    for segment in segments:
        ranges = sorted(hard_by_video.get(segment.video_id, ()))
        starts = [item[0] for item in ranges]
        position = bisect_left(starts, segment.start)
        while position < len(ranges) and ranges[position][0] <= segment.end:
            start, end = ranges[position]
            if start >= segment.start and end <= segment.end:
                retained.append(segment)
                break
            position += 1
    return tuple(retained)


def build_frame_lookup(
    index_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, Path]]:
    lookup: dict[str, dict[int, Path]] = {}
    for record in index_records:
        video = lookup.setdefault(str(record["video_id"]), {})
        for role in ("img0", "gt", "img1"):
            frame = record[role]
            index = int(frame["frame_index"])
            path = Path(frame["path"]).resolve()
            previous = video.get(index)
            if previous is not None and previous != path:
                raise RuntimeError(
                    f"video {record['video_id']} frame {index} maps to two files"
                )
            video[index] = path
    return lookup


def plan_segment_mappings(
    segments: Sequence[FrameInterval],
    lookup: Mapping[str, Mapping[int, Path]],
    *,
    run_hash: str,
) -> tuple[list[tuple[Path, Path]], dict[str, str]]:
    """Plan a video-owned output container followed by segment leaf directories."""

    mappings: list[tuple[Path, Path]] = []
    output_directories: dict[str, str] = {}
    for segment in segments:
        identifier = segment_id(run_hash, segment)
        video_relative = safe_video_relative(segment.video_id)
        leaf = video_relative / f"segment_{segment.start}_{segment.end}_{identifier}"
        output_directories[identifier] = leaf.as_posix()
        ordered = tuple(sorted(lookup.get(segment.video_id, {}).items()))
        if not ordered:
            raise RuntimeError(f"segment video missing from index: {segment.video_id}")
        indices = [item[0] for item in ordered]
        start = bisect_left(indices, segment.start)
        end = bisect_right(indices, segment.end)
        selected = ordered[start:end]
        if len(selected) < 3:
            raise RuntimeError(
                f"merged segment has fewer than three indexed frames: {segment}"
            )
        for frame_index, source in selected:
            if not segment.start <= frame_index <= segment.end:
                raise RuntimeError("segment materialization escaped its closed interval")
            mappings.append((source, leaf / source.name))
    return mappings, output_directories


class IncrementalMaterializer:
    def __init__(
        self,
        config: AppConfig,
        *,
        execution_id: str,
        run_dir: Path,
        index_records: Sequence[Mapping[str, Any]],
    ) -> None:
        self.config = config
        self.execution_id = str(execution_id)
        self.run_dir = Path(run_dir).resolve()
        self.frame_lookup = build_frame_lookup(index_records)
        data_root = Path(config.data.root).expanduser().resolve()
        self.generation_root = data_root / STAGING_DIRECTORY / self.execution_id
        self.hard_staging = self.generation_root / "hard_case"
        self.status_dir = self.run_dir / "materialized_videos"
        self.hard_staging.mkdir(parents=True, exist_ok=True)
        root_marker = self.hard_staging / _OUTPUT_MARKER
        expected = {
            "format": 1,
            "generator": "vfi_hard_miner",
            "kind": "hard_case",
            "run_hash": config.run_hash(),
        }
        existing = _read_object(root_marker)
        if root_marker.exists() and existing is None:
            raise RuntimeError(
                f"incremental staging has an invalid ownership marker: {root_marker}"
            )
        if existing is not None and existing != expected:
            raise RuntimeError(
                f"incremental staging belongs to another run: {self.hard_staging}"
            )
        if existing is None:
            _atomic_json(root_marker, expected)
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def _status_path(self, video_id: str) -> Path:
        return self.status_dir / f"{_video_digest(video_id)}.json"

    def _status(self, video_id: str) -> dict[str, Any] | None:
        payload = _read_object(self._status_path(video_id))
        if not payload:
            return None
        if (
            payload.get("video_id") != video_id
            or payload.get("execution_id") != self.execution_id
            or payload.get("run_hash") != self.config.run_hash()
        ):
            raise RuntimeError(f"stale materialization status for video {video_id!r}")
        return payload

    def completed_video_ids(self) -> set[str]:
        completed: set[str] = set()
        for path in self.status_dir.glob("*.json"):
            payload = _read_object(path)
            if (
                payload
                and payload.get("execution_id") == self.execution_id
                and payload.get("run_hash") == self.config.run_hash()
                and isinstance(payload.get("video_id"), str)
            ):
                video_id = str(payload["video_id"])
                files = payload.get("files", ())
                files_valid = isinstance(files, list) and all(
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and (self.hard_staging / str(item["path"])).is_file()
                    and (self.hard_staging / str(item["path"])).stat().st_size
                    == int(item.get("size", -1))
                    for item in files
                )
                if int(payload.get("frames", 0)) == 0 or files_valid:
                    completed.add(video_id)
        return completed

    def records_from_tasks(self, tasks: Sequence[TaskRecord]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for task in sorted(tasks, key=lambda item: item.task_id):
            if task.result_path is None:
                raise RuntimeError(f"completed task has no result path: {task.task_id}")
            records.extend(read_jsonl(task.result_path))
        return records

    def materialize_tasks(self, video_id: str, tasks: Sequence[TaskRecord]) -> None:
        self.materialize_records(video_id, self.records_from_tasks(tasks))

    def materialize_records(
        self, video_id: str, records: Sequence[Mapping[str, Any]]
    ) -> None:
        existing_status = self._status(video_id)
        if any(str(record.get("video_id")) != video_id for record in records):
            raise RuntimeError(f"materialization records cross video boundary: {video_id}")
        if any(record.get("run_hash") != self.config.run_hash() for record in records):
            raise RuntimeError(f"materialization record belongs to another run: {video_id}")
        if any(record.get("execution_id") != self.execution_id for record in records):
            raise RuntimeError(
                f"materialization record belongs to another execution: {video_id}"
            )

        started = time.monotonic()
        segments = build_segments(records)
        mappings, output_directories = plan_segment_mappings(
            segments,
            self.frame_lookup,
            run_hash=self.config.run_hash(),
        )
        video_relative = safe_video_relative(video_id)
        stable_video_root = self.hard_staging / video_relative
        link_counts = {"hardlink": 0, "copy": 0, "existing": 0}

        if existing_status is not None and not mappings:
            return
        if mappings:
            existing_marker = _read_object(stable_video_root / _VIDEO_MARKER)
            if stable_video_root.exists():
                if not existing_marker or (
                    existing_marker.get("execution_id") != self.execution_id
                    or existing_marker.get("video_id") != video_id
                ):
                    raise RuntimeError(
                        f"unowned incremental video output: {stable_video_root}"
                    )
                complete = True
                for source, relative in mappings:
                    destination = self.hard_staging / relative
                    if (
                        not destination.is_file()
                        or destination.stat().st_size != source.stat().st_size
                    ):
                        complete = False
                        break
                if complete:
                    link_counts = {
                        key: int(existing_marker.get("link_counts", {}).get(key, 0))
                        for key in link_counts
                    }
                else:
                    shutil.rmtree(stable_video_root)
            if not stable_video_root.exists():
                temporary_parent = self.generation_root / ".video_tmp"
                temporary_parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f"{_video_digest(video_id)}.",
                        dir=temporary_parent,
                    )
                )
                try:
                    relative_mappings = [
                        (source, relative.relative_to(video_relative))
                        for source, relative in mappings
                    ]
                    link_counts = materialize_mapped_frames(
                        relative_mappings,
                        output_root=temporary,
                        mode=self.config.output.link_mode,
                    )
                    _atomic_json(
                        temporary / _VIDEO_MARKER,
                        {
                            "format": 1,
                            "generator": "vfi_hard_miner",
                            "run_hash": self.config.run_hash(),
                            "execution_id": self.execution_id,
                            "video_id": video_id,
                            "link_counts": link_counts,
                        },
                    )
                    stable_video_root.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, stable_video_root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)

        elapsed = time.monotonic() - started
        total_bytes = sum(source.stat().st_size for source, _ in mappings)
        payload = {
            "format": 1,
            "run_hash": self.config.run_hash(),
            "execution_id": self.execution_id,
            "video_id": video_id,
            "segments": [
                {
                    **asdict(segment),
                    "segment_id": segment_id(self.config.run_hash(), segment),
                    "output_directory": output_directories[
                        segment_id(self.config.run_hash(), segment)
                    ],
                }
                for segment in segments
            ],
            "frames": len(mappings),
            "files": [
                {"path": relative.as_posix(), "size": source.stat().st_size}
                for source, relative in mappings
            ],
            "bytes": total_bytes,
            "elapsed_seconds": elapsed,
            "mb_per_second": (
                total_bytes / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
            ),
            "link_counts": link_counts,
        }
        _atomic_json(self._status_path(video_id), payload)
        print(
            f"[materialize] video {video_id}: {len(segments)} segments"
            f"  {len(mappings)} frames  {elapsed:.1f}s"
            f"  hardlink={link_counts['hardlink']}"
            f" copy={link_counts['copy']} existing={link_counts['existing']}"
            f"  {payload['mb_per_second']:.1f} MiB/s",
            file=sys.stderr,
            flush=True,
        )

    def summary(self) -> MaterializationSummary:
        videos = segments = frames = 0
        counts = {"hardlink": 0, "copy": 0, "existing": 0}
        for path in self.status_dir.glob("*.json"):
            payload = _read_object(path)
            if (
                not payload
                or payload.get("execution_id") != self.execution_id
                or payload.get("run_hash") != self.config.run_hash()
            ):
                continue
            videos += 1
            segments += len(payload.get("segments", ()))
            frames += int(payload.get("frames", 0))
            for key in counts:
                counts[key] += int(payload.get("link_counts", {}).get(key, 0))
        return MaterializationSummary(
            strategy="per_video",
            staging_path=self.hard_staging,
            videos=videos,
            segments=segments,
            frames=frames,
            link_counts=counts,
        )

    def materialize_all(self, records: Sequence[Mapping[str, Any]]) -> None:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record["video_id"]), []).append(record)
        for video_id in sorted(grouped):
            self.materialize_records(video_id, grouped[video_id])

    def final_plan(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[
        tuple[FrameInterval, ...],
        list[tuple[Path, Path]],
        dict[str, str],
    ]:
        segments = build_segments(records)
        mappings, output_directories = plan_segment_mappings(
            segments,
            self.frame_lookup,
            run_hash=self.config.run_hash(),
        )
        for source, relative in mappings:
            destination = self.hard_staging / relative
            if (
                not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
            ):
                raise RuntimeError(
                    f"incremental materialization is incomplete: {destination}"
                )
        return segments, mappings, output_directories


__all__ = [
    "IncrementalMaterializer",
    "MaterializationSummary",
    "STAGING_DIRECTORY",
    "build_segments",
    "plan_segment_mappings",
    "safe_video_relative",
    "segment_id",
]
