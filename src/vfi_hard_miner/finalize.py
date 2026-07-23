"""Merge hard intervals and materialize final frames, diagnostics, and manifest."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import Any, Mapping, Sequence

from .config import AppConfig, load_config
from .diagnostics import run_diagnostic_stage
from .indexing import build_index
from .manifest import read_jsonl, write_jsonl_part
from .materialization import IncrementalMaterializer
from .offline import sha256_file
from .outputs import (
    link_or_copy,
    materialize_mapped_frames,
    materialize_original_frames,
)
from .pipeline import (
    execution_id,
    load_execution_snapshot,
    load_index_records,
    run_directory,
    run_state_path,
    stage_counts,
)
from .runtime import collect_runtime_info
from .segments import HARD_STATUSES, ClassifiedInterval, FrameInterval, merge_classified_intervals


_OUTPUT_MARKER = ".vfi_hard_miner_output.json"
_CURRENT_MARKER = ".vfi_hard_miner_current.json"


@dataclass(frozen=True, slots=True)
class FinalizeSummary:
    run_hash: str
    source_records: int
    accepted_records: int
    segments: int
    frames: int
    visualizations: int
    link_counts: dict[str, int]
    manifest_path: Path
    segment_path: Path


def _result_path(config: AppConfig) -> Path:
    run_dir = run_directory(config)
    if config.teacher is not None:
        path = run_dir / "teacher_results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"teacher is configured but teacher results are missing: {path}"
            )
        return path
    path = run_dir / "main_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"main results are missing: {path}")
    return path


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


def _segment_id(run_hash: str, segment: FrameInterval) -> str:
    payload = f"{run_hash}\0{segment.video_id}\0{segment.start}\0{segment.end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _safe_video_relative(video_id: str) -> Path:
    parent, separator, video_key = video_id.partition("::")
    if not separator:
        parent, video_key = "", video_id
    candidate = Path(parent) / (video_key or "video")
    if candidate.is_absolute() or candidate.drive:
        raise ValueError(f"unsafe absolute video_id: {video_id!r}")
    if not candidate.parts or any(part in {"", ".", "..", "/", "\\"} for part in candidate.parts):
        raise ValueError(f"unsafe video_id path: {video_id!r}")
    # A video key is a single stable identifier, never another path fragment.
    if any(separator in video_key for separator in ("/", "\\")):
        raise ValueError(f"unsafe video key: {video_key!r}")
    return candidate


def _frame_lookup(
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


def _segment_sources(
    segments: Sequence[FrameInterval],
    lookup: Mapping[str, Mapping[int, Path]],
) -> list[Path]:
    sources: list[Path] = []
    ordered_by_video = {
        video_id: tuple(sorted(frames.items())) for video_id, frames in lookup.items()
    }
    for segment in segments:
        ordered = ordered_by_video.get(segment.video_id)
        if ordered is None:
            raise RuntimeError(f"segment video missing from index: {segment.video_id}")
        indices = [item[0] for item in ordered]
        start = bisect_left(indices, segment.start)
        end = bisect_right(indices, segment.end)
        selected = [path for _, path in ordered[start:end]]
        if len(selected) < 3:
            raise RuntimeError(
                f"merged segment has fewer than three indexed frames: {segment}"
            )
        sources.extend(selected)
    return list(dict.fromkeys(sources))


def _segment_materialization_plan(
    segments: Sequence[FrameInterval],
    lookup: Mapping[str, Mapping[int, Path]],
    *,
    run_hash: str,
) -> tuple[list[tuple[Path, Path]], dict[str, str]]:
    """Map every segment to one leaf directory so disk readers cannot bridge it."""

    mappings: list[tuple[Path, Path]] = []
    output_directories: dict[str, str] = {}
    leaf_owners: dict[Path, str] = {}
    ordered_by_video = {
        video_id: tuple(sorted(frames.items())) for video_id, frames in lookup.items()
    }
    for segment in segments:
        segment_id = _segment_id(run_hash, segment)
        video_relative = _safe_video_relative(segment.video_id)
        leaf = video_relative.parent / (
            f"segment_{segment.start}_{segment.end}_{segment_id}"
        )
        previous_owner = leaf_owners.get(leaf)
        if previous_owner is not None and previous_owner != segment_id:
            raise RuntimeError(f"two segments map to output leaf {leaf}")
        leaf_owners[leaf] = segment_id
        output_directories[segment_id] = leaf.as_posix()

        ordered = ordered_by_video.get(segment.video_id)
        if ordered is None:
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


def _retain_segments_with_hard_centers(
    segments: Sequence[FrameInterval],
    records: Sequence[Mapping[str, Any]],
) -> tuple[FrameInterval, ...]:
    """Drop barrier-clipped pieces that no longer contain an accepted triplet."""

    hard_by_video: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        if str(record.get("status", "")) not in HARD_STATUSES:
            continue
        indices = tuple(int(value) for value in record["frame_indices"])
        hard_by_video.setdefault(str(record["video_id"]), []).append(
            (indices[0], indices[-1])
        )
    indexed: dict[str, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = {}
    for video_id, ranges in hard_by_video.items():
        ordered = tuple(sorted(ranges))
        indexed[video_id] = (tuple(item[0] for item in ordered), ordered)
    retained: list[FrameInterval] = []
    for segment in segments:
        candidates = indexed.get(segment.video_id)
        if candidates is None:
            continue
        starts, ranges = candidates
        position = bisect_left(starts, segment.start)
        while position < len(ranges) and ranges[position][0] <= segment.end:
            start, end = ranges[position]
            if start >= segment.start and end <= segment.end:
                retained.append(segment)
                break
            position += 1
    return tuple(retained)


def _validate_segment_relative_staging(
    hard_staging: Path,
    *,
    config: AppConfig,
    segments: Mapping[str, FrameInterval],
    output_directories: Mapping[str, str],
) -> dict[str, int]:
    """Re-scan disk output under the contract that each leaf is one video."""

    owners = {
        Path(relative).as_posix(): segment_id
        for segment_id, relative in output_directories.items()
    }
    counts = {segment_id: 0 for segment_id in segments}
    triplets = build_index(
        hard_staging,
        stride=config.data.stride,
        frame_regex=config.data.frame_regex,
        frame_digits=config.data.frame_digits,
        extensions=config.data.extensions,
        recursive=True,
    )
    for triplet in triplets:
        relative_parent = Path(triplet.img0.relative_path).parent.as_posix()
        if any(
            Path(frame.relative_path).parent.as_posix() != relative_parent
            for frame in (triplet.gt, triplet.img1)
        ):
            raise RuntimeError("staged triplet crosses output leaf directories")
        segment_id = owners.get(relative_parent)
        if segment_id is None or segment_id not in segments:
            raise RuntimeError(
                f"staged triplet belongs to an unknown segment leaf: {relative_parent}"
            )
        segment = segments[segment_id]
        indices = (triplet.img0.index, triplet.gt.index, triplet.img1.index)
        if indices[0] < segment.start or indices[-1] > segment.end:
            raise RuntimeError(f"staged triplet escapes segment {segment_id}")
        counts[segment_id] += 1
    empty = sorted(segment_id for segment_id, count in counts.items() if count == 0)
    if empty:
        raise RuntimeError(
            "segment-relative output contains segments with no trainable triplet: "
            + ", ".join(empty)
        )
    return counts


def _segment_lookup(
    segments: Mapping[str, FrameInterval],
) -> dict[str, tuple[tuple[int, ...], tuple[tuple[str, FrameInterval], ...]]]:
    grouped: dict[str, list[tuple[str, FrameInterval]]] = {}
    for segment_id, segment in segments.items():
        grouped.setdefault(segment.video_id, []).append((segment_id, segment))
    output: dict[str, tuple[tuple[int, ...], tuple[tuple[str, FrameInterval], ...]]] = {}
    for video_id, items in grouped.items():
        ordered = tuple(sorted(items, key=lambda item: (item[1].start, item[1].end)))
        output[video_id] = (tuple(item[1].start for item in ordered), ordered)
    return output


def _matching_segment_ids(
    record: Mapping[str, Any],
    lookup: Mapping[
        str, tuple[tuple[int, ...], tuple[tuple[str, FrameInterval], ...]]
    ],
) -> list[str]:
    indexed = lookup.get(str(record["video_id"]))
    if indexed is None:
        return []
    starts, items = indexed
    indices = tuple(int(value) for value in record["frame_indices"])
    position = bisect_right(starts, indices[0]) - 1
    if position < 0:
        return []
    segment_id, segment = items[position]
    return [segment_id] if indices[-1] <= segment.end else []


def _checkpoint_fingerprint(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"configured checkpoint is missing: {checkpoint}")
    return {
        "path": str(checkpoint),
        "size": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
    }


def _assert_stage_complete(config: AppConfig, stage: str) -> None:
    state_path = run_state_path(config, stage=stage)
    if not state_path.is_file():
        raise FileNotFoundError(f"{stage} stage state is missing: {state_path}")
    counts = stage_counts(config, stage=stage)
    if counts["failed"] or counts["pending"] or counts["running"]:
        raise RuntimeError(f"{stage} stage is not complete: {counts}")


def _checked_output_roots(config: AppConfig, data_root: Path) -> tuple[Path, Path]:
    hard_root = (data_root / config.output.hard_case_dir).resolve()
    visualization_root = (data_root / config.output.visualization_dir).resolve()
    for name, root in (("hard_case_dir", hard_root), ("visualization_dir", visualization_root)):
        try:
            root.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"output.{name} escapes data.root: {root}") from exc
        if root == data_root:
            raise ValueError(f"output.{name} must not resolve to data.root")
    if hard_root == visualization_root:
        raise ValueError("hard-case and visualization roots must differ")
    if hard_root in visualization_root.parents or visualization_root in hard_root.parents:
        raise ValueError("hard-case and visualization roots must not overlap")
    return hard_root, visualization_root


def _atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
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


def _write_output_marker(directory: Path, *, run_hash: str, kind: str) -> None:
    _atomic_json(
        directory / _OUTPUT_MARKER,
        {"format": 1, "generator": "vfi_hard_miner", "kind": kind, "run_hash": run_hash},
    )


def _output_marker(directory: Path) -> dict[str, Any] | None:
    marker = directory / _OUTPUT_MARKER
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_owned_output(directory: Path, *, kind: str | None = None) -> bool:
    payload = _output_marker(directory)
    return bool(
        payload is not None
        and payload.get("generator") == "vfi_hard_miner"
        and (kind is None or payload.get("kind") == kind)
        and isinstance(payload.get("run_hash"), str)
    )


def _backup_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.vfi-backup")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _publish_generation_unlocked(
    *,
    directory_pairs: Sequence[tuple[Path, Path, str]],
    file_pairs: Sequence[tuple[Path, Path]],
    current_path: Path,
    current_payload: Mapping[str, Any],
) -> None:
    """Commit every generated artifact, retaining all backups until CURRENT."""

    current = _read_json_object(current_path)
    current_owned = bool(current and current.get("generator") == "vfi_hard_miner")
    directories_owned = all(
        destination.is_dir() and _is_owned_output(destination, kind=kind)
        for _, destination, kind in directory_pairs
    )
    trusted_existing_files = current_owned or directories_owned

    # Recover a prior interrupted commit before starting another one.  A fully
    # committed generation is identified by CURRENT plus matching directory markers.
    committed = bool(
        current_owned
        and all(
            (_output_marker(destination) or {}).get("run_hash") == current.get("run_hash")
            for _, destination, _ in directory_pairs
        )
        and all(destination.is_file() for _, destination in file_pairs)
    )
    for _, destination, kind in directory_pairs:
        backup = _backup_path(destination)
        if not backup.exists():
            continue
        if not backup.is_dir() or not _is_owned_output(backup, kind=kind):
            raise RuntimeError(f"refusing to touch an unowned publish backup: {backup}")
        if committed:
            shutil.rmtree(backup)
        else:
            if destination.exists():
                if not destination.is_dir() or not _is_owned_output(destination, kind=kind):
                    raise RuntimeError(f"refusing to roll back unowned output: {destination}")
                shutil.rmtree(destination)
            os.replace(backup, destination)
    for _, destination in file_pairs:
        backup = _backup_path(destination)
        if not backup.exists():
            continue
        if not trusted_existing_files:
            raise RuntimeError(f"refusing to touch an unowned file backup: {backup}")
        if committed:
            backup.unlink()
        else:
            destination.unlink(missing_ok=True)
            os.replace(backup, destination)

    for staging, destination, kind in directory_pairs:
        if not staging.is_dir() or not _is_owned_output(staging, kind=kind):
            raise RuntimeError(f"staging directory is incomplete or unowned: {staging}")
        if destination.exists() and (
            not destination.is_dir() or not _is_owned_output(destination, kind=kind)
        ):
            raise RuntimeError(
                f"refusing to replace an output directory without {_OUTPUT_MARKER}: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
    for staging, destination in file_pairs:
        if not staging.is_file():
            raise RuntimeError(f"staged output file is missing: {staging}")
        if destination.exists() and not trusted_existing_files:
            raise RuntimeError(f"refusing to replace an unowned output file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    published: list[tuple[Path, Path, bool]] = []
    committed_current = False
    try:
        for staging, destination, _ in directory_pairs:
            backup = _backup_path(destination)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staging, destination)
            published.append((destination, backup, True))
        for staging, destination in file_pairs:
            backup = _backup_path(destination)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staging, destination)
            published.append((destination, backup, False))
        _atomic_json(current_path, current_payload)
        committed_current = True
    finally:
        if not committed_current:
            for destination, backup, is_directory in reversed(published):
                if destination.exists():
                    if is_directory:
                        if not _is_owned_output(destination):
                            raise RuntimeError(
                                f"cannot safely roll back unowned directory: {destination}"
                            )
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                if backup.exists():
                    os.replace(backup, destination)
        else:
            for _, backup, is_directory in published:
                if backup.exists():
                    if is_directory:
                        if not _is_owned_output(backup):
                            raise RuntimeError(f"committed backup is unowned: {backup}")
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()


def _publish_generation(
    *,
    directory_pairs: Sequence[tuple[Path, Path, str]],
    file_pairs: Sequence[tuple[Path, Path]],
    current_path: Path,
    current_payload: Mapping[str, Any],
) -> None:
    lock_path = current_path.parent / ".vfi_hard_miner_finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            owner = _read_json_object(lock_path)
            if not owner or owner.get("hostname") != socket.gethostname():
                raise RuntimeError(f"another finalize publication owns {lock_path}")
            try:
                os.kill(int(owner["pid"]), 0)
            except ProcessLookupError:
                lock_path.unlink()
                continue
            except (KeyError, TypeError, ValueError, PermissionError):
                raise RuntimeError(f"cannot safely recover finalize lock: {lock_path}")
            raise RuntimeError(f"another finalize publication is active: {owner}")
    if descriptor is None:
        raise RuntimeError(f"could not acquire finalize publication lock: {lock_path}")
    try:
        payload = json.dumps(
            {"generator": "vfi_hard_miner", "hostname": socket.gethostname(), "pid": os.getpid()},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        _publish_generation_unlocked(
            directory_pairs=directory_pairs,
            file_pairs=file_pairs,
            current_path=current_path,
            current_payload=current_payload,
        )
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _runtime_metadata(config: AppConfig) -> dict[str, Any]:
    metadata = collect_runtime_info(include_npu=False)
    # The standalone probe is intentionally run outside the coordinator so it
    # can import torch_npu without contaminating the parent's spawn state.
    probe_path = run_directory(config) / "runtime_probe.json"
    if probe_path.is_file():
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid runtime probe JSON: {probe_path}") from exc
        if not isinstance(probe, dict):
            raise ValueError(f"runtime probe must contain a JSON object: {probe_path}")
        metadata["external_probe"] = probe
    return metadata


def finalize_run(config_path: str | Path) -> FinalizeSummary:
    path = Path(config_path).resolve()
    config = load_config(path)
    if config.runtime.precision != "float32":
        raise RuntimeError("final diagnostics require the validated float32 baseline")
    _assert_stage_complete(config, "main")
    if config.teacher is not None:
        _assert_stage_complete(config, "teacher")
    records = list(read_jsonl(_result_path(config)))
    expected_hash = config.run_hash()
    expected_execution = execution_id(config)
    if any(record.get("run_hash") != expected_hash for record in records):
        raise RuntimeError("result manifest contains records from another configuration")
    if any(record.get("execution_id") != expected_execution for record in records):
        raise RuntimeError("result manifest contains records from another execution snapshot")
    index_records = load_index_records(config)
    index_by_id = {str(record["sample_id"]): record for record in index_records}
    expected_ids = set(index_by_id)
    result_ids = [str(record["sample_id"]) for record in records]
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("result manifest contains duplicate sample_id values")
    if set(result_ids) != expected_ids:
        missing = len(expected_ids - set(result_ids))
        extra = len(set(result_ids) - expected_ids)
        raise RuntimeError(
            f"result manifest does not cover the current index: missing={missing}, extra={extra}"
        )
    allowed_statuses = {
        "accept",
        "review",
        "reject",
        "invalid",
        "out_of_scope",
        "hard",
        "extremely_hard",
    }
    for record in records:
        sample_id = str(record["sample_id"])
        frozen = index_by_id[sample_id]
        for field in ("video_id", "stride", "frame_indices", "img0", "gt", "img1"):
            if record.get(field) != frozen.get(field):
                raise RuntimeError(
                    f"result changed frozen index field {field!r} for sample {sample_id}"
                )
        if record.get("status") not in allowed_statuses:
            raise RuntimeError(
                f"result has unknown status {record.get('status')!r} for sample {sample_id}"
            )
        if not isinstance(record.get("reasons", []), list) or not isinstance(
            record.get("regions", []), list
        ):
            raise RuntimeError(f"result reasons/regions schema is invalid for {sample_id}")
    incremental_materializer: IncrementalMaterializer | None = None
    if config.output.materialize_strategy == "per_video":
        incremental_materializer = IncrementalMaterializer(
            config,
            execution_id=expected_execution,
            run_dir=run_directory(config),
            index_records=index_records,
        )
        incremental_materializer.materialize_all(records)
        (
            segments,
            segment_mappings,
            segment_output_directories,
        ) = incremental_materializer.final_plan(records)
    else:
        segments = merge_classified_intervals(
            (_classified(record) for record in records),
            min_length=3,
        )
        segments = _retain_segments_with_hard_centers(segments, records)
        segment_mappings = []
        segment_output_directories = {}
    data_root = Path(config.data.root).expanduser().resolve()
    hard_root, visualization_root = _checked_output_roots(config, data_root)
    manifest_path = (data_root / config.output.manifest_name).resolve()
    try:
        manifest_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"output.manifest_name escapes data.root: {manifest_path}") from exc
    if manifest_path == data_root or hard_root in manifest_path.parents or visualization_root in manifest_path.parents:
        raise ValueError("manifest must be outside the generated frame and visualization trees")
    frame_lookup = _frame_lookup(index_records)
    if (
        incremental_materializer is None
        and segments
        and config.output.layout == "segment_relative"
    ):
        segment_mappings, segment_output_directories = _segment_materialization_plan(
            segments,
            frame_lookup,
            run_hash=config.run_hash(),
        )
        sources = [source for source, _ in segment_mappings]
    elif incremental_materializer is not None:
        sources = [source for source, _ in segment_mappings]
    else:
        sources = _segment_sources(segments, frame_lookup) if segments else []
    segment_ids = {_segment_id(config.run_hash(), segment): segment for segment in segments}
    segments_by_video = _segment_lookup(segment_ids)
    contributed_by_sample: dict[str, list[str]] = {}
    for segment_id, segment in segment_ids.items():
        for sample_id in segment.sample_ids:
            contributed_by_sample.setdefault(sample_id, []).append(segment_id)
    selected: list[dict[str, Any]] = []
    diagnostic_inputs: list[dict[str, Any]] = []
    for record in records:
        matching = _matching_segment_ids(record, segments_by_video)
        updated = dict(record)
        is_hard_center = str(record.get("status", "")) in HARD_STATUSES
        updated["covered_by_segment"] = bool(matching)
        updated["contributed_to_segment"] = str(record["sample_id"]) in contributed_by_sample
        updated["selected"] = bool(matching) and is_hard_center
        updated["segment_ids"] = matching
        updated["segment_output_directories"] = [
            segment_output_directories[segment_id]
            for segment_id in matching
            if segment_id in segment_output_directories
        ]
        updated["contributed_segment_ids"] = contributed_by_sample.get(
            str(record["sample_id"]), []
        )
        updated["visualization"] = None
        should_visualize = bool(updated["selected"]) or (
            config.output.save_review and record.get("status") == "review"
        )
        if should_visualize:
            relative_visualization = _safe_video_relative(str(record["video_id"]))
            relative_visualization /= f"{record['sample_id']}.png"
            updated["diagnostic_relative"] = relative_visualization.as_posix()
            diagnostic_inputs.append(updated)
        selected.append(updated)

    diagnostic_summary = run_diagnostic_stage(path, diagnostic_inputs)
    diagnostic_records = list(read_jsonl(diagnostic_summary.manifest_path))
    diagnostic_by_sample = {
        str(record["sample_id"]): record for record in diagnostic_records
    }
    if len(diagnostic_by_sample) != len(diagnostic_inputs):
        raise RuntimeError("diagnostic stage did not return one result per requested sample")

    if incremental_materializer is None:
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".vfi-finalize-{config.run_hash()}-", dir=data_root)
        )
        hard_staging = staging_root / "hard_case"
    else:
        staging_root = incremental_materializer.generation_root
        hard_staging = incremental_materializer.hard_staging
    visualization_staging = staging_root / "visualization"
    hard_staging.mkdir(parents=True, exist_ok=True)
    if visualization_staging.exists():
        shutil.rmtree(visualization_staging)
    visualization_staging.mkdir(parents=True)
    published = False
    try:
        segment_triplet_counts: dict[str, int] = {}
        if config.output.layout == "segment_relative":
            if incremental_materializer is None:
                link_counts = materialize_mapped_frames(
                    segment_mappings,
                    output_root=hard_staging,
                    mode=config.output.link_mode,
                )
            else:
                link_counts = incremental_materializer.summary().link_counts
            segment_triplet_counts = _validate_segment_relative_staging(
                hard_staging,
                config=config,
                segments=segment_ids,
                output_directories=segment_output_directories,
            )
        else:
            link_counts = materialize_original_frames(
                sources,
                source_root=data_root,
                output_root=hard_staging,
                mode=config.output.link_mode,
                layout=config.output.layout,
            )
        visualization_count = 0
        for record in selected:
            relative = record.pop("diagnostic_relative", None)
            if relative is None:
                continue
            diagnostic = diagnostic_by_sample.get(str(record["sample_id"]))
            if diagnostic is None:
                raise RuntimeError(f"missing diagnostic for {record['sample_id']}")
            if relative != diagnostic.get("visualization_relative"):
                raise RuntimeError(f"diagnostic path mismatch for {record['sample_id']}")
            destination = visualization_staging / Path(str(relative))
            link_or_copy(
                diagnostic["artifact_path"],
                destination,
                mode="hardlink_then_copy",
            )
            final_destination = (visualization_root / Path(str(relative))).resolve()
            try:
                final_destination.relative_to(visualization_root)
            except ValueError as exc:
                raise ValueError(f"visualization path escapes output root: {relative}") from exc
            record["visualization"] = str(final_destination)
            visualization_count += 1

        _write_output_marker(hard_staging, run_hash=config.run_hash(), kind="hard_case")
        _write_output_marker(
            visualization_staging,
            run_hash=config.run_hash(),
            kind="visualization",
        )
        metadata = {
            "run_hash": config.run_hash(),
            "execution": load_execution_snapshot(config),
            "current_checkpoint": _checkpoint_fingerprint(config.model.checkpoint),
            "teacher_checkpoint": (
                None
                if config.teacher is None
                else _checkpoint_fingerprint(config.teacher.checkpoint)
            ),
            "runtime": _runtime_metadata(config),
        }
        for record in selected:
            record["run_metadata"] = metadata

        segment_payload = [
            {
                **asdict(segment),
                "segment_id": _segment_id(config.run_hash(), segment),
                "output_directory": segment_output_directories.get(
                    _segment_id(config.run_hash(), segment)
                ),
                "trainable_triplets": segment_triplet_counts.get(
                    _segment_id(config.run_hash(), segment)
                ),
            }
            for segment in segments
        ]
        segment_path = run_directory(config) / "segments.json"
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, segment_staging_name = tempfile.mkstemp(
            prefix=".segments.", suffix=".staging", dir=segment_path.parent
        )
        os.close(descriptor)
        segment_staging = Path(segment_staging_name)
        manifest_staging = staging_root / "manifest.jsonl"
        _atomic_json(segment_staging, segment_payload)
        write_jsonl_part(manifest_staging, selected)
        _publish_generation(
            directory_pairs=(
                (hard_staging, hard_root, "hard_case"),
                (visualization_staging, visualization_root, "visualization"),
            ),
            file_pairs=(
                (manifest_staging, manifest_path),
                (segment_staging, segment_path),
            ),
            current_path=data_root / _CURRENT_MARKER,
            current_payload={
                "format": 1,
                "generator": "vfi_hard_miner",
                "run_hash": config.run_hash(),
                "execution_id": execution_id(config),
                "hard_case_dir": str(hard_root),
                "hard_case_layout": config.output.layout,
                "visualization_dir": str(visualization_root),
                "manifest": str(manifest_path),
                "segments": str(segment_path),
            },
        )
        published = True
    finally:
        # Preserve durable per-video frames after a failed finalize so a retry
        # never repeats inference or materialization.  A successful publication
        # renames both staged children away and leaves only the owned parent.
        if staging_root.exists() and (
            incremental_materializer is None or published
        ):
            shutil.rmtree(staging_root)
        elif incremental_materializer is not None:
            if visualization_staging.exists():
                shutil.rmtree(visualization_staging)
            (staging_root / "manifest.jsonl").unlink(missing_ok=True)
        if "segment_staging" in locals():
            segment_staging.unlink(missing_ok=True)

    return FinalizeSummary(
        run_hash=config.run_hash(),
        source_records=len(records),
        accepted_records=sum(bool(record["selected"]) for record in selected),
        segments=len(segments),
        frames=len(sources),
        visualizations=visualization_count,
        link_counts=link_counts,
        manifest_path=manifest_path,
        segment_path=segment_path,
    )


__all__ = ["FinalizeSummary", "finalize_run"]
