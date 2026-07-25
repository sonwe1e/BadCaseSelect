"""Candidate-only CGVQM refinement and durable graded-result generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from threading import Event, Thread
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from .cgvqm import CGVQM2Scorer
from .config import AppConfig, load_config
from .diagnostics import _hwc, _prefetched_diagnostic_batches
from .graded_materialization import (
    GradedMaterializationSummary,
    GradedMaterializer,
)
from .grading import apply_grade
from .manifest import read_jsonl, write_jsonl_part
from .model_adapter import ModelAdapter
from .pipeline import (
    execution_id,
    load_index_records,
    run_directory,
    run_state_path,
)
from .runtime import get_spawn_context
from .state import LeaseHeartbeat, LeaseLostError, TaskRecord, TaskStore
from .worker import (
    _infer_model_batch,
    _postproc_microbatch_size,
    _reconstruct_outputs,
    _resolve_reconstruction_device,
    _slice_model_outputs,
)


@dataclass(frozen=True, slots=True)
class CGVQMStageSummary:
    enabled: bool
    backend: str | None
    candidates: int
    refined: int
    videos: int
    reused_videos: int
    counts: dict[str, int]
    error_summary: dict[str, float]
    manifest_path: Path
    materialization: GradedMaterializationSummary | None = None


@dataclass(slots=True)
class _CandidateClip:
    record: Mapping[str, Any]
    context: tuple[Mapping[str, Any], ...]
    center_slot: int
    box: tuple[int, int, int, int]
    distorted: np.ndarray
    reference: np.ndarray
    filled: np.ndarray
    region_mask: np.ndarray | None


class _LogHeartbeat:
    def __init__(self, message: str, *, interval: float = 30.0) -> None:
        self.message = message
        self.interval = interval
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True, name="cgvqm-log-heartbeat")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(self.message, file=sys.stderr, flush=True)

    def __enter__(self) -> "_LogHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _source_result_path(config: AppConfig) -> Path:
    name = "teacher_results.jsonl" if config.teacher is not None else "main_results.jsonl"
    path = run_directory(config) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _video_digest(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:20]


def _sample_digest(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _task_id(config: AppConfig, execution: str, video_id: str) -> str:
    identity = json.dumps(
        {
            "run_hash": config.run_hash(),
            "execution_id": execution,
            "stage": "cgvqm",
            "video_id": video_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{config.run_hash()}:cgvqm:"
        f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
    )


def _task_part_is_valid(
    task: TaskRecord,
    *,
    config: AppConfig,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if task.status != "done" or task.result_path is None:
        return None
    return _valid_part(Path(task.result_path), candidates=candidates, config=config)


def _prepare_state(
    config: AppConfig,
    *,
    execution: str,
    by_video: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Path:
    state_path = run_state_path(config, stage="cgvqm")
    tasks = [
        (
            _task_id(config, execution, video_id),
            {
                "run_hash": config.run_hash(),
                "execution_id": execution,
                "stage": "cgvqm",
                "video_id": video_id,
                "sample_ids": [
                    str(record["sample_id"])
                    for record in sorted(
                        by_video[video_id],
                        key=lambda item: (
                            tuple(item["frame_indices"]),
                            str(item["sample_id"]),
                        ),
                    )
                ],
            },
        )
        for video_id in sorted(by_video)
    ]
    with TaskStore(state_path) as store:
        store.enqueue_many(tasks)
        store.recover_expired()
        for task in store.list_tasks():
            if task.status == "running" and task.owner is not None:
                owner_parts = task.owner.split(":", 2)
                if len(owner_parts) >= 2 and owner_parts[0] == socket.gethostname():
                    try:
                        os.kill(int(owner_parts[1]), 0)
                    except ProcessLookupError:
                        store.requeue_orphaned_running(task.task_id, task.owner)
                    except (PermissionError, ValueError):
                        pass
                continue
            if task.status == "failed":
                store.requeue_failed(task.task_id)
                continue
            if task.status != "done":
                continue
            video_id = str(task.payload.get("video_id", ""))
            candidates = [
                record
                for record in by_video.get(video_id, ())
                if _candidate_record(record, config)
            ]
            if _task_part_is_valid(
                task,
                config=config,
                candidates=candidates,
            ) is None:
                store.requeue_done(task.task_id)
    return state_path


def _candidate_record(record: Mapping[str, Any], config: AppConfig) -> bool:
    if str(record.get("validity_label")) != "accept":
        return False
    if str(record.get("in_scope_label")) != "accept":
        return False
    if float(record.get("p_solvable", 0.0)) < config.thresholds.solvable_accept_at:
        return False
    if float(
        record.get("mining_p_wrong", record.get("p_wrong", 0.0))
    ) < config.thresholds.wrong_accept_at:
        return False
    return bool(record.get("regions"))


def _primary_box(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    regions = record.get("regions", ())
    if not isinstance(regions, Sequence) or not regions:
        raise ValueError(f"candidate {record.get('sample_id')} has no regions")
    raw_index = int(record.get("primary_region_index", 0))
    index = raw_index if 0 <= raw_index < len(regions) else 0
    region = regions[index]
    box = region.get("box") if isinstance(region, Mapping) else None
    if not isinstance(box, Sequence) or len(box) != 4:
        raise ValueError(f"candidate {record.get('sample_id')} has invalid region box")
    return tuple(int(value) for value in box)


def _window_indices(length: int, center: int, frames: int) -> tuple[int, ...]:
    if length < 1:
        raise ValueError("video must contain at least one indexed sample")
    start = center - frames // 2
    return tuple(min(length - 1, max(0, start + offset)) for offset in range(frames))


def _expanded_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    region_width = x1 - x0
    region_height = y1 - y0
    side = max(region_width, region_height)
    side = max(side + max(8, int(round(side * 0.5))), 32)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(0, left), max(0, width - side))
    top = min(max(0, top), max(0, height - side))
    return left, top, min(width, left + side), min(height, top + side)


def _crop_resize_uint8(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = _expanded_box(box, array.shape)
    crop = Image.fromarray(array[y0:y1, x0:x1])
    resized = crop.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _build_candidate_clips(
    video_records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    clip_frames: int,
    crop_size: int,
) -> list[_CandidateClip]:
    ordered = sorted(
        video_records,
        key=lambda item: (tuple(item["frame_indices"]), str(item["sample_id"])),
    )
    positions = {str(record["sample_id"]): index for index, record in enumerate(ordered)}
    clips: list[_CandidateClip] = []
    for record in candidates:
        sample_id = str(record["sample_id"])
        if sample_id not in positions:
            raise RuntimeError(f"candidate is missing from frozen index: {sample_id}")
        indices = _window_indices(len(ordered), positions[sample_id], clip_frames)
        context = tuple(ordered[index] for index in indices)
        center_slot = clip_frames // 2
        clips.append(
            _CandidateClip(
                record=record,
                context=context,
                center_slot=center_slot,
                box=_primary_box(record),
                distorted=np.empty(
                    (clip_frames, crop_size, crop_size, 3), dtype=np.uint8
                ),
                reference=np.empty(
                    (clip_frames, crop_size, crop_size, 3), dtype=np.uint8
                ),
                filled=np.zeros(clip_frames, dtype=bool),
                region_mask=None,
            )
        )
    return clips


def _region_mask_in_crop(
    box: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int, int],
    crop_size: int,
) -> np.ndarray:
    crop_x0, crop_y0, crop_x1, crop_y1 = _expanded_box(box, image_shape)
    box_x0, box_y0, box_x1, box_y1 = box
    crop_width = max(1, crop_x1 - crop_x0)
    crop_height = max(1, crop_y1 - crop_y0)
    x0 = int(np.floor((box_x0 - crop_x0) * crop_size / crop_width))
    y0 = int(np.floor((box_y0 - crop_y0) * crop_size / crop_height))
    x1 = int(np.ceil((box_x1 - crop_x0) * crop_size / crop_width))
    y1 = int(np.ceil((box_y1 - crop_y0) * crop_size / crop_height))
    x0 = max(0, min(crop_size - 1, x0))
    y0 = max(0, min(crop_size - 1, y0))
    x1 = max(x0 + 1, min(crop_size, x1))
    y1 = max(y0 + 1, min(crop_size, y1))
    mask = np.zeros((crop_size, crop_size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _model_device(config: AppConfig) -> torch.device:
    if config.runtime.backend == "cpu":
        return torch.device("cpu")
    index = int(config.runtime.devices[0])
    if config.runtime.backend == "npu":
        import torch_npu  # type: ignore[import-not-found]  # noqa: F401

        device = torch.device(f"npu:{index}")
        torch.npu.set_device(device)
        return device
    if config.runtime.backend == "cuda":
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        return device
    raise ValueError(f"unsupported runtime backend: {config.runtime.backend}")


def _scorer_device(config: AppConfig, model_device: torch.device) -> torch.device:
    desired = (
        config.runtime.backend
        if config.cgvqm.backend == "auto"
        else config.cgvqm.backend
    )
    if desired == "cpu":
        return torch.device("cpu")
    if desired == model_device.type:
        return model_device
    if desired == "cuda":
        return torch.device(f"cuda:{int(config.runtime.devices[0])}")
    if desired == "npu":
        return torch.device(f"npu:{int(config.runtime.devices[0])}")
    raise ValueError(f"unsupported CGVQM backend: {desired}")


def _load_scorer(
    config: AppConfig, model_device: torch.device
) -> tuple[CGVQM2Scorer, str]:
    desired = _scorer_device(config, model_device)
    try:
        scorer = CGVQM2Scorer(
            config.cgvqm.backbone_checkpoint,
            config.cgvqm.calibration_checkpoint,
            device=desired,
        )
        scorer.probe(
            clip_frames=min(config.cgvqm.clip_frames, 4),
            crop_size=min(config.cgvqm.crop_size, 64),
        )
        return scorer, str(desired)
    except Exception as exc:
        if desired.type == "cpu" or not config.cgvqm.allow_cpu_fallback:
            raise
        print(
            f"[cgvqm] backend {desired} unavailable ({type(exc).__name__}: {exc}); "
            "using explicit CPU fallback",
            file=sys.stderr,
            flush=True,
        )
        scorer = CGVQM2Scorer(
            config.cgvqm.backbone_checkpoint,
            config.cgvqm.calibration_checkpoint,
            device="cpu",
        )
        scorer.probe(
            clip_frames=min(config.cgvqm.clip_frames, 4),
            crop_size=min(config.cgvqm.crop_size, 64),
        )
        return scorer, "cpu"


def _top_area_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, int(np.ceil(flat.size * fraction)))
    if count >= flat.size:
        return float(flat.mean())
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def _cgvqm_microbatch_size(config: AppConfig) -> int:
    # Conservative live-tensor estimate for distorted/reference input, stem,
    # layer1, their differences, and upsampled maps. A single sample is always
    # allowed even when it exceeds the configured budget.
    pixels = (
        int(config.cgvqm.clip_frames)
        * int(config.cgvqm.crop_size)
        * int(config.cgvqm.crop_size)
    )
    estimated_bytes = pixels * 80 * 4
    budget = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    return max(
        1,
        min(
            int(config.cgvqm.batch_size),
            budget // max(1, estimated_bytes),
        ),
    )


def _save_maps(
    path: Path,
    *,
    center: np.ndarray,
    temporal: np.ndarray,
    fused: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            center=np.asarray(center, dtype=np.float32),
            temporal=np.asarray(temporal, dtype=np.float32),
            fused=np.asarray(fused, dtype=np.float32),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _score_clip_batch(
    clips: Sequence[_CandidateClip],
    *,
    scorer: CGVQM2Scorer,
    config: AppConfig,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    batch_size = _cgvqm_microbatch_size(config)
    if batch_size < config.cgvqm.batch_size:
        print(
            f"[cgvqm] scorer batch reduced from {config.cgvqm.batch_size} "
            f"to {batch_size} by postproc_buffer_mb",
            file=sys.stderr,
            flush=True,
        )
    for start in range(0, len(clips), batch_size):
        batch = clips[start : start + batch_size]
        distorted = np.stack([clip.distorted for clip in batch])
        reference = np.stack([clip.reference for clip in batch])
        result = scorer.score(distorted, reference)
        for index, clip in enumerate(batch):
            maps = np.asarray(result.error_maps[index], dtype=np.float32)
            center_map = maps[clip.center_slot]
            temporal_map = (
                np.max(np.abs(np.diff(maps, axis=0)), axis=0)
                if maps.shape[0] > 1
                else np.zeros_like(center_map)
            )
            fused_map = np.maximum(center_map, temporal_map)
            frame_errors = maps.reshape(maps.shape[0], -1).mean(axis=1)
            if clip.region_mask is None or not bool(clip.region_mask.any()):
                raise RuntimeError(
                    f"candidate region mask is missing for "
                    f"{clip.record.get('sample_id')}"
                )
            support_threshold = float(np.quantile(center_map, 0.90))
            heat_support = center_map >= support_threshold
            spatial_overlap = float(
                np.logical_and(heat_support, clip.region_mask).sum()
                / max(1, int(heat_support.sum()))
            )
            region_error = float(center_map[clip.region_mask].mean())
            background = center_map[~clip.region_mask]
            background_error = (
                float(background.mean()) if background.size else region_error
            )
            center_error = max(
                float(center_map.mean()),
                _top_area_mean(center_map),
                0.85 * float(center_map.max()),
            )
            error = float(
                np.clip(max(float(result.errors[index]), center_error), 0.0, 100.0)
            )
            persistence = float(
                np.mean(frame_errors >= float(config.cgvqm.b_error_at))
            )
            temporal_change = float(
                np.max(np.abs(np.diff(frame_errors))) if len(frame_errors) > 1 else 0.0
            )
            sample_id = str(clip.record["sample_id"])
            artifact = artifact_root / f"{_sample_digest(sample_id)}.npz"
            _save_maps(
                artifact,
                center=center_map,
                temporal=temporal_map,
                fused=fused_map,
            )
            output.append(
                {
                    "sample_id": sample_id,
                    "error": error,
                    "clip_error": float(result.errors[index]),
                    "center_error": center_error,
                    "temporal_persistence": persistence,
                    "temporal_change": temporal_change,
                    "spatial_overlap": spatial_overlap,
                    "center_region_error": region_error,
                    "center_background_error": background_error,
                    "center_peak": float(center_map.max()),
                    "center_top_1pct_mean": _top_area_mean(center_map),
                    "temporal_mean": float(temporal_map.mean()),
                    "temporal_peak": float(temporal_map.max()),
                    "clip_frames": config.cgvqm.clip_frames,
                    "crop_size": config.cgvqm.crop_size,
                    "backend": result.backend,
                    "artifact_path": str(artifact.resolve()),
                }
            )
    return output


def _fill_and_score_clips(
    clips: Sequence[_CandidateClip],
    *,
    adapter: ModelAdapter,
    scorer: CGVQM2Scorer,
    config: AppConfig,
    model_device: torch.device,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    dependencies: dict[str, list[tuple[_CandidateClip, int]]] = defaultdict(list)
    context_records: dict[str, Mapping[str, Any]] = {}
    for clip in clips:
        for slot, record in enumerate(clip.context):
            sample_id = str(record["sample_id"])
            dependencies[sample_id].append((clip, slot))
            context_records[sample_id] = record
    ordered_context = sorted(
        context_records.values(),
        key=lambda item: (tuple(item["frame_indices"]), str(item["sample_id"])),
    )
    reconstruction_device = _resolve_reconstruction_device(config, model_device)
    buffer_bytes = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    completed = 0
    for items in _prefetched_diagnostic_batches(
        ordered_context,
        batch_size=config.model.batch_size,
        prefetch=config.runtime.prefetch,
        max_cache=config.runtime.chunk_triplets + 2,
        cache_budget_bytes=int(config.runtime.decode_cache_mb) * 1024 * 1024,
    ):
        img0_tensor, img1_tensor, outputs = _infer_model_batch(
            items,
            adapter=adapter,
            production_batch=config.model.batch_size,
        )
        microbatch = _postproc_microbatch_size(
            items,
            buffer_bytes=buffer_bytes,
            # This loop is synchronous and owns no CPU Future pool.  Use the
            # whole stage budget instead of dividing it by main-stage workers.
            postproc_workers=1,
        )
        for micro_start in range(0, len(items), microbatch):
            micro_end = min(len(items), micro_start + microbatch)
            reconstructed = _reconstruct_outputs(
                img0_tensor[micro_start:micro_end],
                img1_tensor[micro_start:micro_end],
                _slice_model_outputs(outputs, micro_start, micro_end),
                model_config=config.model,
                device=reconstruction_device,
            )
            for local, (record, _img0, gt, _img1) in enumerate(
                items[micro_start:micro_end]
            ):
                prediction = _hwc(reconstructed.prediction[local])
                sample_id = str(record["sample_id"])
                for clip, slot in dependencies[sample_id]:
                    if clip.region_mask is None:
                        clip.region_mask = _region_mask_in_crop(
                            clip.box,
                            image_shape=gt.shape,
                            crop_size=config.cgvqm.crop_size,
                        )
                    clip.distorted[slot] = _crop_resize_uint8(
                        prediction, clip.box, config.cgvqm.crop_size
                    )
                    clip.reference[slot] = _crop_resize_uint8(
                        gt, clip.box, config.cgvqm.crop_size
                    )
                    clip.filled[slot] = True
                completed += 1
        print(
            f"[cgvqm] reconstructed {completed}/{len(ordered_context)} "
            f"context samples",
            file=sys.stderr,
            flush=True,
        )
    incomplete = [
        str(clip.record["sample_id"]) for clip in clips if not bool(clip.filled.all())
    ]
    if incomplete:
        raise RuntimeError(f"CGVQM clips are incomplete: {incomplete[:3]}")
    return _score_clip_batch(
        clips,
        scorer=scorer,
        config=config,
        artifact_root=artifact_root,
    )


def _valid_part(
    part: Path,
    *,
    candidates: Sequence[Mapping[str, Any]],
    config: AppConfig,
) -> list[dict[str, Any]] | None:
    if not part.is_file():
        return None
    try:
        records = list(read_jsonl(part))
    except Exception:
        return None
    expected = {str(record["sample_id"]) for record in candidates}
    actual = [str(record.get("sample_id", "")) for record in records]
    if len(actual) != len(expected) or set(actual) != expected:
        return None
    for record in records:
        artifact = Path(str(record.get("artifact_path", "")))
        numeric = (
            record.get("error"),
            record.get("clip_error"),
            record.get("center_error"),
            record.get("temporal_persistence"),
            record.get("temporal_change"),
            record.get("spatial_overlap"),
        )
        try:
            numeric_valid = all(np.isfinite(float(value)) for value in numeric)
        except (TypeError, ValueError):
            numeric_valid = False
        if (
            record.get("run_hash") != config.run_hash()
            or record.get("execution_id") != execution_id(config)
            or not artifact.is_file()
            or not numeric_valid
        ):
            return None
        try:
            with np.load(artifact, allow_pickle=False) as payload:
                maps = [
                    np.asarray(payload[name])
                    for name in ("center", "temporal", "fused")
                ]
        except (OSError, ValueError, KeyError):
            return None
        if any(value.ndim != 2 or not np.isfinite(value).all() for value in maps):
            return None
    return records


def _run_cgvqm_stage_local(config_path: str | Path) -> CGVQMStageSummary:
    path = Path(config_path).resolve()
    config = load_config(path)
    source_records = list(read_jsonl(_source_result_path(config)))
    index_records = load_index_records(config)
    output_path = run_directory(config) / "graded_results.jsonl"
    execution = execution_id(config)
    source_by_id: dict[str, dict[str, Any]] = {}
    index_by_id = {str(record["sample_id"]): record for record in index_records}
    for record in source_records:
        sample_id = str(record.get("sample_id", ""))
        if (
            not sample_id
            or sample_id in source_by_id
            or record.get("run_hash") != config.run_hash()
            or record.get("execution_id") != execution
        ):
            raise RuntimeError(
                "source results contain duplicate, stale, or unidentified records"
            )
        frozen = index_by_id.get(sample_id)
        if frozen is None:
            raise RuntimeError(f"source result is absent from the frozen index: {sample_id}")
        for field in ("video_id", "stride", "frame_indices", "img0", "gt", "img1"):
            if record.get(field) != frozen.get(field):
                raise RuntimeError(
                    f"source result changed frozen field {field!r} for {sample_id}"
                )
        source_by_id[sample_id] = record
    if set(source_by_id) != set(index_by_id):
        raise RuntimeError("source results do not cover the frozen run index")
    by_video_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_video_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        by_video_source[str(record["video_id"])].append(record)
    for record in index_records:
        by_video_index[str(record["video_id"])].append(record)

    materializer = (
        GradedMaterializer(
            config,
            execution_id=execution,
            run_dir=run_directory(config),
        )
        if config.output.layout == "graded_flat"
        and config.output.materialize_strategy == "per_video"
        else None
    )

    if not config.cgvqm.enabled:
        graded = [apply_grade(record, None, config) for record in source_records]
        write_jsonl_part(output_path, graded)
        if materializer is not None:
            materializer.materialize_all(graded)
        return CGVQMStageSummary(
            False,
            None,
            0,
            0,
            len(by_video_source),
            0,
            {"pending": 0, "running": 0, "done": 0, "failed": 0},
            {},
            output_path,
            None if materializer is None else materializer.summary(),
        )

    model_device = _model_device(config)
    torch.set_num_threads(config.runtime.cpu_threads_per_worker)
    adapter = ModelAdapter.from_config(
        config.model, device=model_device, validate_values=False
    )
    scorer, scorer_backend = _load_scorer(config, model_device)
    parts_dir = run_directory(config) / "cgvqm_parts"
    artifacts_dir = run_directory(config) / "cgvqm_artifacts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    evidence_by_sample: dict[str, dict[str, Any]] = {}
    graded_by_sample: dict[str, dict[str, Any]] = {}
    candidates_total = sum(
        _candidate_record(record, config) for record in source_records
    )
    state_path = _prepare_state(
        config,
        execution=execution,
        by_video=by_video_source,
    )
    initially_done: set[str] = set()
    with TaskStore(state_path) as store:
        for task in store.list_tasks(status="done"):
            video_id = str(task.payload.get("video_id", ""))
            candidates = [
                record
                for record in by_video_source.get(video_id, ())
                if _candidate_record(record, config)
            ]
            if _task_part_is_valid(
                task,
                config=config,
                candidates=candidates,
            ) is not None:
                initially_done.add(video_id)

    owner = f"{socket.gethostname()}:{os.getpid()}:cgvqm:{model_device}"
    with TaskStore(state_path) as store:
        while task := store.claim(
            owner,
            lease_seconds=config.runtime.lease_seconds,
        ):
            video_id = str(task.payload["video_id"])
            video_source = sorted(
                by_video_source[video_id],
                key=lambda item: (
                    tuple(item["frame_indices"]),
                    str(item["sample_id"]),
                ),
            )
            if [str(record["sample_id"]) for record in video_source] != list(
                task.payload.get("sample_ids", ())
            ):
                raise RuntimeError(f"CGVQM task payload changed for video {video_id}")
            candidates = [
                record for record in video_source if _candidate_record(record, config)
            ]
            digest = _video_digest(video_id)
            short_id = task.task_id.rsplit(":", 1)[-1]
            part_path = (
                parts_dir / f"{digest}.{short_id[:12]}.attempt-{task.attempt}.jsonl"
            )
            started = time.monotonic()
            try:
                with LeaseHeartbeat(
                    state_path,
                    task.task_id,
                    owner,
                    lease_seconds=config.runtime.lease_seconds,
                    attempt=task.attempt,
                ) as lease:
                    evidence: list[dict[str, Any]] = []
                    artifact_root = artifacts_dir / digest
                    for start in range(
                        0,
                        len(candidates),
                        config.cgvqm.candidates_per_task,
                    ):
                        candidate_chunk = candidates[
                            start : start + config.cgvqm.candidates_per_task
                        ]
                        clips = _build_candidate_clips(
                            by_video_index[video_id],
                            candidate_chunk,
                            clip_frames=config.cgvqm.clip_frames,
                            crop_size=config.cgvqm.crop_size,
                        )
                        message = (
                            f"[cgvqm] video {video_id}: still refining "
                            f"{start}/{len(candidates)} candidates"
                        )
                        with _LogHeartbeat(message):
                            evidence.extend(
                                _fill_and_score_clips(
                                    clips,
                                    adapter=adapter,
                                    scorer=scorer,
                                    config=config,
                                    model_device=model_device,
                                    artifact_root=artifact_root,
                                )
                            )
                        lease.check()
                    for record in evidence:
                        record["run_hash"] = config.run_hash()
                        record["execution_id"] = execution
                        record["video_id"] = video_id
                    write_jsonl_part(part_path, evidence)
                    lease.check()
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
                elapsed = time.monotonic() - started
                rate = len(candidates) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[cgvqm] video {video_id}: refined={len(evidence)}/"
                    f"{len(candidates)}  {elapsed:.1f}s  {rate:.2f}/s; "
                    "JSON part and SQLite task committed",
                    file=sys.stderr,
                    flush=True,
                )
            except LeaseLostError:
                print(
                    f"[cgvqm] video {video_id}: lease lost",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            except Exception as exc:
                detail = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(
                    f"[cgvqm] video {video_id}: failed — {detail}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    store.fail(
                        task.task_id,
                        owner,
                        detail,
                        retry=task.attempt < 2,
                        attempt=task.attempt,
                    )
                except LeaseLostError:
                    continue
                continue

            graded_video = [
                apply_grade(
                    record,
                    next(
                        (
                            item
                            for item in evidence
                            if str(item["sample_id"]) == str(record["sample_id"])
                        ),
                        None,
                    ),
                    config,
                )
                for record in video_source
            ]
            if (
                materializer is not None
                and video_id not in materializer.completed_video_ids()
            ):
                materializer.materialize_video(video_id, graded_video)

    with TaskStore(state_path) as store:
        counts = store.counts()
        if counts["failed"] or counts["pending"] or counts["running"]:
            raise RuntimeError(f"CGVQM stage did not finish cleanly: {counts}")
        completed_tasks = store.list_tasks(status="done")

    for task in completed_tasks:
        video_id = str(task.payload["video_id"])
        video_source = sorted(
            by_video_source[video_id],
            key=lambda item: (tuple(item["frame_indices"]), str(item["sample_id"])),
        )
        candidates = [
            record for record in video_source if _candidate_record(record, config)
        ]
        evidence = _task_part_is_valid(
            task,
            config=config,
            candidates=candidates,
        )
        if evidence is None:
            raise RuntimeError(f"CGVQM winning part is invalid for video {video_id}")
        for record in evidence:
            evidence_by_sample[str(record["sample_id"])] = record
        graded_video = [
            apply_grade(
                record,
                evidence_by_sample.get(str(record["sample_id"])),
                config,
            )
            for record in video_source
        ]
        for record in graded_video:
            graded_by_sample[str(record["sample_id"])] = record
        if (
            materializer is not None
            and video_id not in materializer.completed_video_ids()
        ):
            materializer.materialize_video(video_id, graded_video)

    refined = len(evidence_by_sample)
    reused_videos = len(initially_done)
    error_values = np.asarray(
        [float(record["error"]) for record in evidence_by_sample.values()],
        dtype=np.float64,
    )
    error_summary = (
        {
            "min": float(error_values.min()),
            "median": float(np.median(error_values)),
            "p95": float(np.quantile(error_values, 0.95)),
            "max": float(error_values.max()),
        }
        if error_values.size
        else {}
    )

    graded = [graded_by_sample[str(record["sample_id"])] for record in source_records]
    write_jsonl_part(output_path, graded)
    return CGVQMStageSummary(
        True,
        scorer_backend,
        candidates_total,
        refined,
        len(by_video_source),
        reused_videos,
        counts,
        error_summary,
        output_path,
        None if materializer is None else materializer.summary(),
    )


def _write_summary(path: Path, summary: CGVQMStageSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(
            asdict(summary),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cgvqm_stage_entry(config_path: str, summary_path: str) -> None:
    summary = _run_cgvqm_stage_local(config_path)
    _write_summary(Path(summary_path), summary)


def _read_summary(path: Path) -> CGVQMStageSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    materialization_payload = payload.get("materialization")
    materialization = None
    if isinstance(materialization_payload, Mapping):
        materialization = GradedMaterializationSummary(
            strategy=str(materialization_payload["strategy"]),
            staging_path=Path(str(materialization_payload["staging_path"])),
            videos=int(materialization_payload["videos"]),
            centers={
                str(key): int(value)
                for key, value in materialization_payload["centers"].items()
            },
            frames={
                str(key): int(value)
                for key, value in materialization_payload["frames"].items()
            },
            copy_counts={
                str(key): int(value)
                for key, value in materialization_payload["copy_counts"].items()
            },
        )
    return CGVQMStageSummary(
        enabled=bool(payload["enabled"]),
        backend=None if payload["backend"] is None else str(payload["backend"]),
        candidates=int(payload["candidates"]),
        refined=int(payload["refined"]),
        videos=int(payload["videos"]),
        reused_videos=int(payload["reused_videos"]),
        counts={
            str(key): int(value) for key, value in payload["counts"].items()
        },
        error_summary={
            str(key): float(value)
            for key, value in payload["error_summary"].items()
        },
        manifest_path=Path(str(payload["manifest_path"])),
        materialization=materialization,
    )


def run_cgvqm_stage(config_path: str | Path) -> CGVQMStageSummary:
    """Run CGVQM in a clean child process when the main model uses an accelerator."""

    path = Path(config_path).resolve()
    config = load_config(path)
    if config.runtime.backend == "cpu":
        return _run_cgvqm_stage_local(path)
    summary_path = run_directory(config) / "cgvqm_stage_summary.json"
    summary_path.unlink(missing_ok=True)
    context = get_spawn_context()
    process = context.Process(
        target=_cgvqm_stage_entry,
        args=(str(path), str(summary_path)),
        name="vfi-cgvqm-refine",
    )
    process.start()
    last_log = time.monotonic()
    while process.is_alive():
        process.join(timeout=1.0)
        if time.monotonic() - last_log >= 30.0:
            print(
                "[cgvqm] refinement worker is still running",
                file=sys.stderr,
                flush=True,
            )
            last_log = time.monotonic()
    if process.exitcode != 0:
        raise RuntimeError(
            f"CGVQM refinement worker failed with exit code {process.exitcode}"
        )
    if not summary_path.is_file():
        raise RuntimeError("CGVQM refinement worker did not write its summary")
    return _read_summary(summary_path)


__all__ = [
    "CGVQMStageSummary",
    "run_cgvqm_stage",
]
