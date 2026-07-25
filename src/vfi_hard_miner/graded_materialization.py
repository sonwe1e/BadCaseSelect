"""Crash-safe flat A/B materialization with original basenames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .config import AppConfig
from .grading import TRAINING_GRADES
from .materialization import STAGING_DIRECTORY
from .outputs import OutputCollisionError, materialize_mapped_frames


_OUTPUT_MARKER = ".vfi_hard_miner_output.json"


@dataclass(frozen=True, slots=True)
class GradedMaterializationSummary:
    strategy: str
    staging_path: Path
    videos: int
    centers: dict[str, int]
    frames: dict[str, int]
    copy_counts: dict[str, int]


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plan_graded_mappings(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    """Map every accepted center's three originals into flat A/B directories."""

    planned: dict[Path, Path] = {}
    centers = {"A": 0, "B": 0}
    for record in records:
        grade = str(record.get("grade", ""))
        if grade not in TRAINING_GRADES:
            continue
        centers[grade] += 1
        for role in ("img0", "gt", "img1"):
            frame = record.get(role)
            if not isinstance(frame, Mapping) or not isinstance(frame.get("path"), str):
                raise ValueError(
                    f"graded record {record.get('sample_id')} has invalid {role}"
                )
            source = Path(str(frame["path"])).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            relative = Path(grade) / source.name
            previous = planned.get(relative)
            if previous is not None and previous != source:
                raise OutputCollisionError(
                    f"multiple source frames map to {relative}: {previous} and {source}"
                )
            planned[relative] = source
    mappings = sorted(
        ((source, relative) for relative, source in planned.items()),
        key=lambda item: (item[1].as_posix(), item[0].as_posix()),
    )
    return mappings, centers


class GradedMaterializer:
    def __init__(
        self,
        config: AppConfig,
        *,
        execution_id: str,
        run_dir: Path,
    ) -> None:
        if config.output.layout != "graded_flat":
            raise ValueError("GradedMaterializer requires output.layout='graded_flat'")
        if config.output.link_mode != "copy":
            raise ValueError("graded training frames must use byte-copy mode")
        self.config = config
        self.execution_id = str(execution_id)
        self.run_dir = Path(run_dir).resolve()
        data_root = Path(config.data.root).expanduser().resolve()
        self.generation_root = data_root / STAGING_DIRECTORY / self.execution_id
        self.hard_staging = self.generation_root / "hard_case"
        self.status_dir = self.run_dir / "materialized_graded_videos"
        self.hard_staging.mkdir(parents=True, exist_ok=True)
        (self.hard_staging / "A").mkdir(exist_ok=True)
        (self.hard_staging / "B").mkdir(exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            "format": 2,
            "generator": "vfi_hard_miner",
            "kind": "hard_case",
            "layout": "graded_flat",
            "run_hash": config.run_hash(),
            "execution_id": self.execution_id,
        }
        marker = self.hard_staging / _OUTPUT_MARKER
        existing = _read_object(marker)
        if existing is not None and existing != expected:
            raise RuntimeError(
                f"graded staging belongs to another run: {self.hard_staging}"
            )
        if marker.exists() and existing is None:
            raise RuntimeError(f"invalid graded staging marker: {marker}")
        if existing is None:
            _atomic_json(marker, expected)

    def _status_path(self, video_id: str) -> Path:
        return self.status_dir / f"{_video_digest(video_id)}.json"

    def _statuses(self) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        for path in self.status_dir.glob("*.json"):
            payload = _read_object(path)
            if (
                payload
                and payload.get("run_hash") == self.config.run_hash()
                and payload.get("execution_id") == self.execution_id
            ):
                statuses.append(payload)
        return statuses

    def _owners(self) -> dict[str, str]:
        owners: dict[str, str] = {}
        for status in self._statuses():
            for item in status.get("files", ()):
                if not isinstance(item, Mapping):
                    continue
                relative = str(item.get("path", ""))
                source = str(item.get("source", ""))
                if not relative or not source:
                    continue
                previous = owners.get(relative)
                if previous is not None and previous != source:
                    raise OutputCollisionError(
                        f"staging ownership conflict for {relative}: "
                        f"{previous} and {source}"
                    )
                owners[relative] = source
        return owners

    def completed_video_ids(self) -> set[str]:
        completed: set[str] = set()
        for status in self._statuses():
            if status.get("state") != "completed":
                continue
            valid = all(
                isinstance(item, Mapping)
                and isinstance(item.get("path"), str)
                and (self.hard_staging / str(item["path"])).is_file()
                and (self.hard_staging / str(item["path"])).stat().st_size
                == int(item.get("size", -1))
                and _sha256_path(self.hard_staging / str(item["path"]))
                == str(item.get("sha256", ""))
                for item in status.get("files", ())
            )
            if valid and isinstance(status.get("video_id"), str):
                completed.add(str(status["video_id"]))
        return completed

    def materialize_video(
        self, video_id: str, records: Sequence[Mapping[str, Any]]
    ) -> None:
        if any(str(record.get("video_id")) != video_id for record in records):
            raise RuntimeError(f"graded materialization crossed video {video_id}")
        mappings, centers = plan_graded_mappings(records)
        owners = self._owners()
        files: list[dict[str, Any]] = []
        for source, relative in mappings:
            key = relative.as_posix()
            previous = owners.get(key)
            if previous is not None and Path(previous).resolve() != source:
                raise OutputCollisionError(
                    f"flat graded filename collision at {relative}: "
                    f"{previous} and {source}"
                )
            files.append(
                {
                    "path": key,
                    "source": str(source),
                    "size": source.stat().st_size,
                    "sha256": _sha256_path(source),
                }
            )
        status_path = self._status_path(video_id)
        planned = {
            "format": 1,
            "state": "materializing",
            "run_hash": self.config.run_hash(),
            "execution_id": self.execution_id,
            "video_id": video_id,
            "centers": centers,
            "files": files,
        }
        _atomic_json(status_path, planned)
        started = time.monotonic()
        counts = materialize_mapped_frames(
            mappings,
            output_root=self.hard_staging,
            mode="copy",
        )
        elapsed = time.monotonic() - started
        for item in files:
            destination = self.hard_staging / str(item["path"])
            if (
                not destination.is_file()
                or destination.stat().st_size != int(item["size"])
                or _sha256_path(destination) != str(item["sha256"])
            ):
                raise RuntimeError(
                    f"byte-copy verification failed for {destination}"
                )
        byte_count = sum(int(item["size"]) for item in files)
        completed = {
            **planned,
            "state": "completed",
            "frames": {
                grade: sum(
                    item["path"].startswith(f"{grade}/") for item in files
                )
                for grade in ("A", "B")
            },
            "copy_counts": counts,
            "bytes": byte_count,
            "elapsed_seconds": elapsed,
            "MBps": (
                byte_count / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
            ),
        }
        _atomic_json(status_path, completed)

    def materialize_all(self, records: Sequence[Mapping[str, Any]]) -> None:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record["video_id"]), []).append(record)
        completed = self.completed_video_ids()
        for video_id in sorted(grouped):
            if video_id not in completed:
                self.materialize_video(video_id, grouped[video_id])

    def final_plan(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[tuple[Path, Path]]:
        mappings, _ = plan_graded_mappings(records)
        for source, relative in mappings:
            destination = self.hard_staging / relative
            if (
                not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
                or _sha256_path(destination) != _sha256_path(source)
            ):
                raise RuntimeError(
                    f"graded materialization is incomplete: {destination}"
                )
        return mappings

    def summary(self) -> GradedMaterializationSummary:
        videos = 0
        centers = {"A": 0, "B": 0}
        frames = {"A": 0, "B": 0}
        counts = {"hardlink": 0, "copy": 0, "existing": 0}
        for status in self._statuses():
            if status.get("state") != "completed":
                continue
            videos += 1
            for grade in ("A", "B"):
                centers[grade] += int(status.get("centers", {}).get(grade, 0))
                frames[grade] += int(status.get("frames", {}).get(grade, 0))
            for key in counts:
                counts[key] += int(status.get("copy_counts", {}).get(key, 0))
        return GradedMaterializationSummary(
            strategy=self.config.output.materialize_strategy,
            staging_path=self.hard_staging,
            videos=videos,
            centers=centers,
            frames=frames,
            copy_counts=counts,
        )


__all__ = [
    "GradedMaterializationSummary",
    "GradedMaterializer",
    "plan_graded_mappings",
]
