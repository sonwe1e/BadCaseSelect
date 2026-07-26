"""Candidate-only CGVQM refinement and durable graded-result generation."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
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
from .reconstruction import reconstruct_midpoint
from .runtime import get_spawn_context
from .state import LeaseHeartbeat, LeaseLostError, TaskRecord, TaskStore
from .worker import (
    _infer_model_batch,
    _postproc_microbatch_size,
    _prepare_reconstruction_microbatch,
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
    worker_backends: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class _CandidateClip:
    record: Mapping[str, Any]
    context: tuple[Mapping[str, Any], ...]
    center_slot: int
    box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
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


def _try_primary_box(
    record: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    """Lenient ``_primary_box``: return None instead of raising.

    Context frames in a clip window may be plain samples with no regions;
    they still contribute to the crop extent only if they expose a primary
    box, so a missing/invalid box is silently skipped rather than fatal.
    """

    try:
        return _primary_box(record)
    except (ValueError, TypeError, KeyError):
        return None


def _union_box(
    boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Smallest box covering every input box (min corner / max corner)."""

    if not boxes:
        raise ValueError("_union_box requires at least one box")
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return int(x0), int(y0), int(x1), int(y1)


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
    *,
    extent_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Crop to a padded square around ``extent_box`` (default ``box``).

    When ``extent_box`` is given (the 16-frame union), the crop follows the
    region's motion across the window while ``box`` still identifies the
    candidate's own region for the mask; passing ``box`` alone preserves the
    single-box behaviour.
    """

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    extent = box if extent_box is None else extent_box
    x0, y0, x1, y1 = _expanded_box(extent, array.shape)
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
        candidate_box = _primary_box(record)
        # Crop extent follows the region across the whole 16-frame window: the
        # union of every context frame's primary box (falling back to the
        # candidate's own box when no context frame exposes one).  The region
        # mask below still marks only ``candidate_box``.
        window_boxes = [
            box
            for box in (_try_primary_box(item) for item in context)
            if box is not None
        ]
        crop_box = _union_box(window_boxes) if window_boxes else candidate_box
        clips.append(
            _CandidateClip(
                record=record,
                context=context,
                center_slot=center_slot,
                box=candidate_box,
                crop_box=crop_box,
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
    crop_extent_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Mark the candidate's own ``box`` inside the (possibly larger) crop.

    ``crop_extent_box`` is the 16-frame union that defines the crop rectangle;
    the mask maps the candidate region into that same expanded crop so scoring
    still weights only the candidate's region even though the crop shows the
    full motion span.  Defaults to ``box`` (single-box behaviour).
    """

    extent = box if crop_extent_box is None else crop_extent_box
    crop_x0, crop_y0, crop_x1, crop_y1 = _expanded_box(extent, image_shape)
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


def _model_device(
    config: AppConfig, *, device_index: int | None = None
) -> torch.device:
    if config.runtime.backend == "cpu":
        return torch.device("cpu")
    index = int(config.runtime.devices[0]) if device_index is None else int(device_index)
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


def _scorer_device(
    config: AppConfig,
    model_device: torch.device,
    *,
    device_index: int | None = None,
) -> torch.device:
    desired = (
        config.runtime.backend
        if config.cgvqm.backend == "auto"
        else config.cgvqm.backend
    )
    if desired == "cpu":
        return torch.device("cpu")
    if desired == model_device.type:
        return model_device
    index = int(config.runtime.devices[0]) if device_index is None else int(device_index)
    if desired == "cuda":
        return torch.device(f"cuda:{index}")
    if desired == "npu":
        return torch.device(f"npu:{index}")
    raise ValueError(f"unsupported CGVQM backend: {desired}")


def _scorer_backend_label(config: AppConfig) -> str:
    """Configured scorer backend label, resolved without any torch import."""

    return str(
        config.runtime.backend
        if config.cgvqm.backend == "auto"
        else config.cgvqm.backend
    )


def _worker_backend_report_path(
    config: AppConfig, device_index: int | None
) -> Path:
    name = (
        "worker_cpu.json"
        if device_index is None
        else f"worker_{int(device_index)}.json"
    )
    return run_directory(config) / "cgvqm_workers" / name


def _write_worker_backend_report(
    config: AppConfig,
    device_index: int | None,
    info: Mapping[str, str | None],
) -> dict[str, Any]:
    """Persist this worker's real backend so the parent can aggregate it."""

    report = {
        "device": None if device_index is None else int(device_index),
        "model_backend": info.get("desired"),
        "scorer_backend": info.get("backend"),
        "fallback_reason": info.get("fallback_reason"),
    }
    path = _worker_backend_report_path(config, device_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _read_worker_backend_reports(config: AppConfig) -> list[dict[str, Any]]:
    directory = run_directory(config) / "cgvqm_workers"
    reports: list[dict[str, Any]] = []
    for path in sorted(directory.glob("worker_*.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict) and report.get("scorer_backend"):
            reports.append(report)
    return reports


def _aggregate_worker_backends(
    reports: Sequence[Mapping[str, Any]], config: AppConfig
) -> str:
    """Collapse per-worker real backends: uniform -> that backend, else mixed.

    Missing reports (e.g. a worker that died before writing one) fall back
    to the config-derived label with a warning rather than a wrong label.
    """

    if not reports:
        print(
            "[cgvqm] warning: no worker backend reports found;"
            " recording config-derived backend label",
            file=sys.stderr,
            flush=True,
        )
        return _scorer_backend_label(config)
    # Device-index suffixes (npu:0, npu:1, ...) still mean one backend kind.
    backends = {
        str(report["scorer_backend"]).split(":", 1)[0] for report in reports
    }
    if len(backends) == 1:
        return next(iter(backends))
    return "mixed"


class _ContextCache:
    """Per-video LRU cache of context-frame predictions and GT frames.

    Adjacent candidate chunks share most context samples; caching the
    reconstructed prediction (uint8 HWC, pre-quantized with the same formula
    the crop path applies) and the decoded GT frame lets the next chunk fill
    clip slots without decoding or inferring them again.  Storing the
    prediction as uint8 rather than float32 cuts its footprint to ~1/4, so a
    fixed ``context_cache_mb`` budget retains far more entries.  Eviction only
    costs a re-inference, so scores stay identical no matter when entries are
    dropped.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._budget = max(0, int(budget_bytes))
        self._entries: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._bytes = 0

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def get(self, sample_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        entry = self._entries.pop(sample_id, None)
        if entry is None:
            return None
        self._entries[sample_id] = entry
        return entry

    def put(self, sample_id: str, prediction: np.ndarray, gt: np.ndarray) -> None:
        if self._budget <= 0:
            return
        size = int(prediction.nbytes) + int(gt.nbytes)
        if size > self._budget:
            return
        existing = self._entries.pop(sample_id, None)
        if existing is not None:
            self._bytes -= int(existing[0].nbytes) + int(existing[1].nbytes)
        while self._bytes + size > self._budget and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= int(evicted[0].nbytes) + int(evicted[1].nbytes)
        self._entries[sample_id] = (prediction, gt)
        self._bytes += size


def _reconstruct_predictions_cpu(
    img0_tensor: torch.Tensor,
    img1_tensor: torch.Tensor,
    outputs: Any,
    *,
    config: AppConfig,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Reconstruct on device and transfer only prediction (3ch) to CPU.

    CGVQM consumes predictions exclusively, so the warps/masks/flows never
    need to cross the device boundary here.
    """

    reconstructed = reconstruct_midpoint(
        img0_tensor,
        img1_tensor,
        outputs.flow_t0,
        outputs.flow_t1,
        outputs.mask0,
        outputs.mask1,
        network_size=(config.model.input_height, config.model.input_width),
        mask0_role=config.model.mask0_role,
        align_corners=config.model.align_corners,
        padding_mode=config.model.padding_mode,
        device=device,
        validate=False,
    )
    return reconstructed.prediction.detach().to(device="cpu", dtype=torch.float32)


def _load_scorer(
    config: AppConfig,
    model_device: torch.device,
    *,
    device_index: int | None = None,
    force_cpu: bool = False,
    fallback_reason: str | None = None,
) -> tuple[CGVQM2Scorer, dict[str, str | None]]:
    """Load and probe the scorer, returning the backend that actually runs.

    The info dict carries ``backend`` (what the probe accepted), ``desired``
    (the configured target) and ``fallback_reason`` (why CPU was chosen, if
    it was), so the parent can record the real runtime backend instead of a
    config-derived label.

    When ``force_cpu`` is set the parent pool has already run the shared probe
    and decided to run the scorer on CPU; this loads CPU directly without
    re-attempting (and failing on) the unsupported accelerator backend.
    """

    if force_cpu:
        # The pool already ran the shared probe and chose CPU; build the
        # desired label as a string (no live accelerator torch.device, which
        # would require torch_npu just to name the fallback we are avoiding).
        backend = _scorer_backend_label(config)
        index = (
            int(config.runtime.devices[0])
            if device_index is None
            else int(device_index)
        )
        desired_label = "cpu" if backend == "cpu" else f"{backend}:{index}"
        scorer = CGVQM2Scorer(
            config.cgvqm.backbone_checkpoint,
            config.cgvqm.calibration_checkpoint,
            device="cpu",
        )
        scorer.probe(
            clip_frames=min(config.cgvqm.clip_frames, 4),
            crop_size=min(config.cgvqm.crop_size, 64),
        )
        return scorer, {
            "backend": "cpu",
            "desired": desired_label,
            "fallback_reason": fallback_reason or "pool selected CPU scorer",
        }
    desired = _scorer_device(config, model_device, device_index=device_index)
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
        return scorer, {
            "backend": str(desired),
            "desired": str(desired),
            "fallback_reason": None,
        }
    except Exception as exc:
        if desired.type == "cpu" or not config.cgvqm.allow_cpu_fallback:
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        print(
            f"[cgvqm] backend {desired} unavailable ({fallback_reason}); "
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
        return scorer, {
            "backend": "cpu",
            "desired": str(desired),
            "fallback_reason": fallback_reason,
        }


def _cgvqm_probe_entry(config_path: str, result_path: str) -> None:
    """Spawn target: probe CGVQM Conv3D support on every configured device.

    Runs in a clean child so the parent stays torch-free.  For each device it
    loads the R3D-18 backbone and runs a tiny Conv3D probe; the first failure's
    exception is recorded so the parent can decide (and log) the CPU fallback.
    Writes ``{supported, reason, devices: [{device, supported, reason}]}``.
    """

    config = load_config(config_path)
    per_device: list[dict[str, Any]] = []
    overall_supported = True
    first_reason: str | None = None
    for device_index in config.runtime.devices:
        device_index = int(device_index)
        try:
            model_device = _model_device(config, device_index=device_index)
            desired = _scorer_device(
                config, model_device, device_index=device_index
            )
            scorer = CGVQM2Scorer(
                config.cgvqm.backbone_checkpoint,
                config.cgvqm.calibration_checkpoint,
                device=desired,
            )
            scorer.probe(
                clip_frames=min(config.cgvqm.clip_frames, 4),
                crop_size=min(config.cgvqm.crop_size, 64),
            )
            per_device.append(
                {"device": device_index, "supported": True, "reason": None}
            )
            del scorer
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            reason = f"{type(exc).__name__}: {exc}"
            overall_supported = False
            first_reason = first_reason or reason
            per_device.append(
                {
                    "device": device_index,
                    "supported": False,
                    "reason": reason,
                }
            )
    result = {
        "supported": overall_supported,
        "reason": first_reason,
        "devices": per_device,
    }
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _probe_cgvqm_backend(config: AppConfig, config_path: Path) -> dict[str, Any]:
    """Run the Conv3D probe in a clean subprocess and return its verdict.

    A probe failure (crash / no output) is treated as unsupported so the pool
    errs toward the explicit, capped CPU fallback rather than spawning one
    accelerator worker per device that would each fail the same way.
    """

    result_path = run_directory(config) / "cgvqm_probe.json"
    spawn = get_spawn_context()
    process = spawn.Process(
        target=_cgvqm_probe_entry,
        args=(str(config_path), str(result_path)),
        name="vfi-cgvqm-probe",
    )
    process.start()
    process.join()
    if process.exitcode != 0 or not result_path.exists():
        reason = f"probe process exited with code {process.exitcode}"
        print(
            f"[cgvqm] Conv3D probe failed ({reason}); treating accelerator "
            "backend as unsupported",
            file=sys.stderr,
            flush=True,
        )
        return {"supported": False, "reason": reason, "devices": []}
    try:
        with open(result_path, encoding="utf-8") as handle:
            verdict = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "supported": False,
            "reason": f"unreadable probe result: {exc}",
            "devices": [],
        }
    return verdict


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


def _fill_clip_slots(
    dependencies: Mapping[str, list[tuple[_CandidateClip, int]]],
    sample_id: str,
    *,
    prediction: np.ndarray,
    gt: np.ndarray,
    config: AppConfig,
) -> None:
    for clip, slot in dependencies[sample_id]:
        if clip.region_mask is None:
            clip.region_mask = _region_mask_in_crop(
                clip.box,
                image_shape=gt.shape,
                crop_size=config.cgvqm.crop_size,
                crop_extent_box=clip.crop_box,
            )
        clip.distorted[slot] = _crop_resize_uint8(
            prediction,
            clip.box,
            config.cgvqm.crop_size,
            extent_box=clip.crop_box,
        )
        clip.reference[slot] = _crop_resize_uint8(
            gt,
            clip.box,
            config.cgvqm.crop_size,
            extent_box=clip.crop_box,
        )
        clip.filled[slot] = True


def _fill_and_score_clips(
    clips: Sequence[_CandidateClip],
    *,
    adapter: ModelAdapter,
    scorer: CGVQM2Scorer,
    config: AppConfig,
    model_device: torch.device,
    artifact_root: Path,
    context_cache: _ContextCache | None = None,
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
    # Context frames cached from earlier chunks of the same video fill their
    # clip slots directly; only cache misses are decoded and inferred.
    cached_count = 0
    uncached_context: list[Mapping[str, Any]] = []
    for record in ordered_context:
        sample_id = str(record["sample_id"])
        entry = None if context_cache is None else context_cache.get(sample_id)
        if entry is None:
            uncached_context.append(record)
            continue
        prediction, gt = entry
        _fill_clip_slots(
            dependencies,
            sample_id,
            prediction=prediction,
            gt=gt,
            config=config,
        )
        cached_count += 1
    reconstruction_device = _resolve_reconstruction_device(config, model_device)
    buffer_bytes = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    completed = cached_count
    for items in _prefetched_diagnostic_batches(
        uncached_context,
        batch_size=config.model.batch_size,
        prefetch=config.runtime.prefetch,
        max_cache=config.runtime.chunk_triplets + 2,
        cache_budget_bytes=int(config.runtime.decode_cache_mb) * 1024 * 1024,
    ):
        inference_batch = _infer_model_batch(
            items,
            adapter=adapter,
            production_batch=config.model.batch_size,
        )
        microbatch = _postproc_microbatch_size(
            items,
            buffer_bytes=buffer_bytes,
            # This loop is synchronous and owns no CPU Future pool.  Use the
            # whole stage budget and no main-stage scoring scratch.
            postproc_workers=1,
            scratch_channels=0,
        )
        for micro_start in range(0, len(items), microbatch):
            micro_end = min(len(items), micro_start + microbatch)
            item_slice = list(items[micro_start:micro_end])
            prepared = _prepare_reconstruction_microbatch(item_slice)
            predictions = _reconstruct_predictions_cpu(
                prepared.img0_tensor,
                prepared.img1_tensor,
                _slice_model_outputs(
                    inference_batch.outputs, micro_start, micro_end
                ),
                config=config,
                device=reconstruction_device,
            )
            for local, (record, _img0, gt, _img1) in enumerate(
                prepared.items
            ):
                # Quantize the prediction to uint8 up front with the exact
                # formula _crop_resize_uint8 uses at full resolution, so both
                # the cached and freshly reconstructed paths feed identical
                # uint8 pixels downstream (bit-exact) while the context cache
                # holds ~1/4 the bytes of the float32 prediction.
                prediction = np.clip(
                    np.rint(_hwc(predictions[local]) * 255.0), 0, 255
                ).astype(np.uint8)
                sample_id = str(record["sample_id"])
                if context_cache is not None:
                    context_cache.put(
                        sample_id, prediction, np.asarray(gt)
                    )
                _fill_clip_slots(
                    dependencies,
                    sample_id,
                    prediction=prediction,
                    gt=np.asarray(gt),
                    config=config,
                )
                completed += 1
            del prepared
        print(
            f"[cgvqm] reconstructed {completed}/{len(ordered_context)} "
            f"context samples ({cached_count} from context cache)",
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


@dataclass(slots=True)
class _CgvqmRunContext:
    """CPU-only stage inputs shared by the claim loop and finalization."""

    config: AppConfig
    source_records: list[dict[str, Any]]
    by_video_source: dict[str, list[dict[str, Any]]]
    by_video_index: dict[str, list[dict[str, Any]]]
    execution: str
    output_path: Path
    materializer: GradedMaterializer | None
    candidates_total: int
    state_path: Path
    initially_done: set[str]


def _prepare_cgvqm_context(config: AppConfig) -> _CgvqmRunContext:
    """Validate the frozen source results and ready the durable task state.

    Idempotent and torch-free: the parent runs it before spawning per-device
    workers, and every worker runs it again in its own address space
    (``_prepare_state`` skips tasks that already exist).
    """

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

    return _CgvqmRunContext(
        config=config,
        source_records=source_records,
        by_video_source=by_video_source,
        by_video_index=by_video_index,
        execution=execution,
        output_path=output_path,
        materializer=materializer,
        candidates_total=candidates_total,
        state_path=state_path,
        initially_done=initially_done,
    )


def _write_disabled_cgvqm_output(
    context: _CgvqmRunContext,
) -> CGVQMStageSummary:
    config = context.config
    graded = [
        apply_grade(record, None, config) for record in context.source_records
    ]
    write_jsonl_part(context.output_path, graded)
    if context.materializer is not None:
        context.materializer.materialize_all(graded)
    return CGVQMStageSummary(
        False,
        None,
        0,
        0,
        len(context.by_video_source),
        0,
        {"pending": 0, "running": 0, "done": 0, "failed": 0},
        {},
        context.output_path,
        None if context.materializer is None else context.materializer.summary(),
    )


def _run_cgvqm_stage_local(config_path: str | Path) -> CGVQMStageSummary:
    config = load_config(Path(config_path).resolve())
    context = _prepare_cgvqm_context(config)
    if not config.cgvqm.enabled:
        return _write_disabled_cgvqm_output(context)
    scorer_info = _run_cgvqm_claim_loop(context)
    report = _write_worker_backend_report(config, None, scorer_info)
    summary = _finalize_cgvqm_stage(
        context,
        backend_label=_aggregate_worker_backends([report], config),
        worker_backends=(report,),
    )
    _write_summary(
        run_directory(config) / "cgvqm_stage_summary.json", summary
    )
    return summary


def _run_cgvqm_claim_loop(
    context: _CgvqmRunContext,
    *,
    device_index: int | None = None,
    force_cpu_scorer: bool = False,
    scorer_fallback_reason: str | None = None,
) -> dict[str, str | None]:
    """Claim per-video tasks until the queue is empty.

    One call owns one accelerator device; several processes may run it
    concurrently against the shared SQLite state (the owner string is
    device- and pid-suffixed).  Returns the actual scorer backend info
    from ``_load_scorer`` (backend / desired / fallback_reason).
    """

    config = context.config
    by_video_source = context.by_video_source
    by_video_index = context.by_video_index
    execution = context.execution
    materializer = context.materializer
    state_path = context.state_path
    context_cache = (
        _ContextCache(int(config.cgvqm.context_cache_mb) * 1024 * 1024)
        if int(config.cgvqm.context_cache_mb) > 0
        else None
    )
    model_device = _model_device(config, device_index=device_index)
    torch.set_num_threads(config.runtime.cpu_threads_per_worker)
    adapter = ModelAdapter.from_config(
        config.model, device=model_device, validate_values=False
    )
    scorer, scorer_info = _load_scorer(
        config,
        model_device,
        device_index=device_index,
        force_cpu=force_cpu_scorer,
        fallback_reason=scorer_fallback_reason,
    )
    parts_dir = run_directory(config) / "cgvqm_parts"
    artifacts_dir = run_directory(config) / "cgvqm_artifacts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    owner = f"{socket.gethostname()}:{os.getpid()}:cgvqm:{model_device}"
    with TaskStore(state_path) as store:
        while task := store.claim(
            owner,
            lease_seconds=config.runtime.lease_seconds,
        ):
            video_id = str(task.payload["video_id"])
            if context_cache is not None:
                context_cache.clear()
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
                                    context_cache=context_cache,
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
    return scorer_info


def _finalize_cgvqm_stage(
    context: _CgvqmRunContext,
    *,
    backend_label: str | None,
    worker_backends: tuple[dict[str, Any], ...] = (),
) -> CGVQMStageSummary:
    """Grade every completed video from the shared state (torch-free)."""

    config = context.config
    by_video_source = context.by_video_source
    execution = context.execution
    output_path = context.output_path
    materializer = context.materializer
    initially_done = context.initially_done
    candidates_total = context.candidates_total
    source_records = context.source_records
    state_path = context.state_path
    evidence_by_sample: dict[str, dict[str, Any]] = {}
    graded_by_sample: dict[str, dict[str, Any]] = {}
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
        backend_label,
        candidates_total,
        refined,
        len(by_video_source),
        reused_videos,
        counts,
        error_summary,
        output_path,
        None if materializer is None else materializer.summary(),
        worker_backends,
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


def _cgvqm_stage_work_entry(
    config_path: str,
    device_index: int,
    force_cpu_scorer: bool = False,
    scorer_fallback_reason: str | None = None,
) -> None:
    """Spawn target: run the claim loop for exactly one accelerator device.

    Finalization (grading, materialization, summary) is the parent's job:
    it needs no torch context and reads the shared SQLite state after all
    workers have joined.

    ``force_cpu_scorer`` is set by the parent when the shared Conv3D probe
    found the accelerator unsupported: the VFI model still runs on the NPU
    (Conv2D is fine), only the CGVQM R3D-18 scorer drops to CPU, and the pool
    is capped at ``cgvqm.cpu_fallback_workers`` so CPU scoring is not
    oversubscribed.
    """

    config = load_config(config_path)
    context = _prepare_cgvqm_context(config)
    if not config.cgvqm.enabled:
        return
    scorer_info = _run_cgvqm_claim_loop(
        context,
        device_index=device_index,
        force_cpu_scorer=force_cpu_scorer,
        scorer_fallback_reason=scorer_fallback_reason,
    )
    _write_worker_backend_report(config, device_index, scorer_info)


def _resolve_cgvqm_worker_pool(
    config: AppConfig, config_path: Path
) -> list[dict[str, Any]]:
    """Decide the worker pool up front from a single shared Conv3D probe.

    Returns a list of worker specs ``{device, force_cpu_scorer, reason}``.

    * Scorer desired on the accelerator and the probe passes -> one worker per
      device, scorer on the accelerator (unchanged behaviour).
    * Scorer desired on CPU (config), or the probe finds Conv3D unsupported ->
      a pool capped at ``cgvqm.cpu_fallback_workers`` (never more than the
      device count) so CPU scoring is not oversubscribed by one worker per
      NPU.  Fail-loud instead if the probe fails and ``allow_cpu_fallback`` is
      off.
    """

    devices = [int(index) for index in config.runtime.devices]
    scorer_desired = _scorer_backend_label(config)
    force_cpu = False
    reason: str | None = None
    if scorer_desired == "cpu":
        force_cpu = True
        reason = "cgvqm.backend resolves to cpu"
    else:
        verdict = _probe_cgvqm_backend(config, config_path)
        if not bool(verdict.get("supported", False)):
            reason = str(verdict.get("reason") or "Conv3D probe unsupported")
            if not config.cgvqm.allow_cpu_fallback:
                raise RuntimeError(
                    "CGVQM Conv3D unsupported on "
                    f"{scorer_desired} and allow_cpu_fallback is off: {reason}"
                )
            force_cpu = True
            print(
                f"[cgvqm] Conv3D unsupported on {scorer_desired} ({reason}); "
                f"falling back to {config.cgvqm.cpu_fallback_workers} CPU "
                "scorer worker(s)",
                file=sys.stderr,
                flush=True,
            )

    if not force_cpu:
        return [
            {"device": device, "force_cpu_scorer": False, "reason": None}
            for device in devices
        ]
    worker_count = min(int(config.cgvqm.cpu_fallback_workers), len(devices))
    worker_count = max(1, worker_count)
    return [
        {
            "device": devices[i],
            "force_cpu_scorer": True,
            "reason": reason,
        }
        for i in range(worker_count)
    ]


def run_cgvqm_stage(config_path: str | Path) -> CGVQMStageSummary:
    """Run CGVQM with one worker process per accelerator device.

    Accelerator stages spawn one clean child process per configured device
    (each loads the model and R3D-18 once and claims videos from the shared
    SQLite state); the parent then finalizes from the shared state without
    initializing torch itself.  CPU runs stay in-process.

    A single shared Conv3D probe (``_resolve_cgvqm_worker_pool``) decides the
    pool before spawning: if the accelerator cannot run R3D-18, the scorer
    drops to a CPU pool capped at ``cgvqm.cpu_fallback_workers`` rather than
    one CPU worker per NPU.
    """

    path = Path(config_path).resolve()
    config = load_config(path)
    if config.runtime.backend == "cpu" or not config.cgvqm.enabled:
        return _run_cgvqm_stage_local(path)

    # Prepared before spawning so initially_done reflects the stage start.
    stage_context = _prepare_cgvqm_context(config)

    worker_specs = _resolve_cgvqm_worker_pool(config, path)
    spawn = get_spawn_context()
    processes: list[tuple[int, Any]] = []
    for spec in worker_specs:
        device_index = int(spec["device"])
        process = spawn.Process(
            target=_cgvqm_stage_work_entry,
            args=(
                str(path),
                device_index,
                bool(spec["force_cpu_scorer"]),
                spec["reason"],
            ),
            name=f"vfi-cgvqm-refine-{device_index}",
        )
        processes.append((device_index, process))
    for _, process in processes:
        process.start()
    last_log = time.monotonic()
    while any(process.is_alive() for _, process in processes):
        for _, process in processes:
            process.join(timeout=1.0)
        if time.monotonic() - last_log >= 30.0:
            running = sum(1 for _, process in processes if process.is_alive())
            print(
                f"[cgvqm] {running} refinement workers still running",
                file=sys.stderr,
                flush=True,
            )
            last_log = time.monotonic()
    failed = sorted(
        device_index
        for device_index, process in processes
        if process.exitcode != 0
    )
    if failed:
        raise RuntimeError(f"CGVQM refinement workers failed on devices {failed}")

    reports = _read_worker_backend_reports(config)
    summary = _finalize_cgvqm_stage(
        stage_context,
        backend_label=_aggregate_worker_backends(reports, config),
        worker_backends=tuple(reports),
    )
    _write_summary(
        run_directory(config) / "cgvqm_stage_summary.json", summary
    )
    return summary


__all__ = [
    "CGVQMStageSummary",
    "run_cgvqm_stage",
]
