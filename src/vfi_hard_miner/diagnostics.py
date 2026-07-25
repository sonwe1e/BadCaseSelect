"""Spawn-safe, multi-device generation of final diagnostic images.

The scheduler in this module never configures an accelerator.  Each spawned
worker owns one device, loads the current model once, and writes attempt-scoped
artifacts.  Only the result part recorded by SQLite is consumed by finalize,
which prevents an expired attempt from publishing over the winning attempt.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Full, Queue
import socket
import sys
from threading import Event, Thread
import time
import traceback
from typing import Any

import numpy as np
import torch

from .config import AppConfig, load_config
from .image_io import read_rgb_uint8, write_image_atomic
from .manifest import merge_jsonl_parts, read_jsonl, write_jsonl_part
from .model_adapter import ModelAdapter, ModelOutputs
from .pipeline import execution_id, run_directory, run_state_path
from .reconstruction import ReconstructionResult
from .runtime import get_spawn_context, spawn_device_workers
from .scoring import score_local_errors
from .state import LeaseHeartbeat, LeaseLostError, TaskRecord, TaskStore
from .visualization import make_diagnostic_grid
from .worker import (
    _FUTURE_WAIT_SECONDS,
    _TIMING_REPORT_SAMPLES,
    CpuTimingTotals,
    DecodedBatch,
    MemoryEstimate,
    PendingPostprocess,
    _ProgressLog,
    _decoded_batch_uint8_bytes,
    _infer_model_batch,
    _postproc_microbatch_size,
    _postproc_reservation,
    _prepare_reconstruction_microbatch,
    _print_timing_summary,
    _reconstruct_outputs,
    _resolve_postproc_workers,
    _resolve_reconstruction_device,
    _slice_model_outputs,
)


ImageCache = OrderedDict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class DiagnosticStageSummary:
    workers: int
    candidates: int
    records: int
    counts: dict[str, int]
    manifest_path: Path


def _tensor_from_hwc(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[0] in (1, 2, 3, 4):
        value = np.moveaxis(value, 0, -1)
    return np.asarray(value, dtype=np.float32)


def _load_cached_uint8(cache: ImageCache, path: str, *, max_items: int) -> np.ndarray:
    existing = cache.pop(path, None)
    if existing is not None:
        cache[path] = existing
        return existing
    image = read_rgb_uint8(path)
    cache[path] = image
    while len(cache) > max_items:
        cache.popitem(last=False)
    return image


def _pack_outputs_cpu(outputs: ModelOutputs, valid_count: int) -> ModelOutputs:
    """Trim padding and transfer the four outputs with one D2H operation."""

    if not 0 < valid_count <= outputs.flow_t0.shape[0]:
        raise ValueError("valid_count must be inside the model output batch")
    packed = torch.cat(
        (outputs.flow_t0, outputs.flow_t1, outputs.mask0, outputs.mask1), dim=1
    )[:valid_count]
    packed = packed.detach().to(device="cpu", dtype=torch.float32)
    flow_t0, flow_t1, mask0, mask1 = packed.split((2, 2, 1, 1), dim=1)
    return ModelOutputs(flow_t0, flow_t1, mask0, mask1)


def _region_boxes(record: Mapping[str, Any]) -> list[dict[str, int]]:
    boxes: list[dict[str, int]] = []
    for region in record.get("regions", ()):
        box = region.get("box") if isinstance(region, Mapping) else None
        if isinstance(box, Sequence) and len(box) == 4:
            boxes.append(
                {
                    "x0": int(box[0]),
                    "y0": int(box[1]),
                    "x1": int(box[2]),
                    "y1": int(box[3]),
                }
            )
    return boxes


def _safe_destination(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"unsafe diagnostic relative path: {relative!r}")
    destination = (root / value).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"diagnostic path escapes artifact root: {relative!r}") from exc
    return destination


def _load_cgvqm_maps(record: Mapping[str, Any]) -> tuple[np.ndarray, ...] | None:
    evidence = record.get("cgvqm")
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise ValueError(f"invalid CGVQM evidence for {record.get('sample_id')}")
    artifact = Path(str(evidence.get("artifact_path", ""))).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"CGVQM diagnostic artifact is missing for "
            f"{record.get('sample_id')}: {artifact}"
        )
    with np.load(artifact, allow_pickle=False) as payload:
        required = ("center", "temporal", "fused")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                f"CGVQM artifact {artifact} is missing arrays: {missing}"
            )
        maps = tuple(np.asarray(payload[name], dtype=np.float32) for name in required)
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in maps):
        raise ValueError(f"CGVQM artifact contains invalid heatmaps: {artifact}")
    return maps


def _finish_batch_diagnostics(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, (record, img0, gt, img1) in enumerate(items):
        prediction = _hwc(reconstructed.prediction[index])
        scoring = score_local_errors(prediction, gt, config.thresholds)
        cgvqm_maps = _load_cgvqm_maps(record)
        if config.output.layout == "graded_flat" and cgvqm_maps is None:
            raise RuntimeError(
                f"graded diagnostic lacks CGVQM maps for {record.get('sample_id')}"
            )
        if cgvqm_maps is None:
            empty = np.zeros(scoring.maps.structure.shape, dtype=np.float32)
            cgvqm_maps = (empty, empty, empty)
        grid = make_diagnostic_grid(
            img0,
            gt,
            prediction,
            img1,
            error_map=scoring.maps.structure,
            gt_only_edge=scoring.maps.gt_only_edges,
            pred_only_edge=scoring.maps.pred_only_edges,
            flow_t0=_hwc(reconstructed.flow_t0[index]),
            flow_t1=_hwc(reconstructed.flow_t1[index]),
            cgvqm_center=cgvqm_maps[0],
            cgvqm_temporal=cgvqm_maps[1],
            cgvqm_fused=cgvqm_maps[2],
            mask0=_hwc(reconstructed.mask0[index]),
            mask1=_hwc(reconstructed.mask1[index]),
            regions=_region_boxes(record),
            labels=record.get("reasons", ()),
            panel_width=config.output.visualization_width,
        )
        relative = str(record["diagnostic_relative"])
        destination = _safe_destination(artifact_root, relative)
        is_jpeg = destination.suffix.lower() in {".jpg", ".jpeg"}
        write_image_atomic(
            destination,
            grid,
            quality=config.output.visualization_quality if is_jpeg else None,
            subsampling=0 if is_jpeg else None,
        )
        results.append(
            {
                "sample_id": str(record["sample_id"]),
                "artifact_path": str(destination),
                "visualization_relative": Path(relative).as_posix(),
                "format": "JPEG" if is_jpeg else "PNG",
                "quality": (
                    config.output.visualization_quality if is_jpeg else None
                ),
                "subsampling": "4:4:4" if is_jpeg else None,
                "width": int(grid.shape[1]),
                "height": int(grid.shape[0]),
            }
        )
    return results


def _prefetched_diagnostic_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    prefetch: int,
    max_cache: int,
    cache_budget_bytes: int | None = None,
) -> Iterator[DecodedBatch]:
    """Decode and shape-group on a bounded CPU-only producer thread.

    Frames and queued batches remain uint8.  Native float32 conversion happens
    only after a reconstruction microbatch passes memory admission.
    """

    queue: Queue[tuple[str, Any]] = Queue(maxsize=max(1, prefetch))
    stopped = Event()

    def put(event: tuple[str, Any]) -> bool:
        while not stopped.is_set():
            try:
                queue.put(event, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        cache: ImageCache = OrderedDict()
        pending: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        pending_shape: tuple[int, int, int] | None = None
        pending_decode_seconds = 0.0
        capacity = max(1, int(max_cache))
        budget_resolved = cache_budget_bytes is None

        def load(path: str) -> np.ndarray:
            nonlocal capacity, budget_resolved
            image = _load_cached_uint8(cache, path, max_items=capacity)
            if not budget_resolved:
                budget_resolved = True
                capacity = max(
                    1,
                    min(capacity, max(8, int(cache_budget_bytes) // max(1, image.nbytes))),
                )
            return image

        def flush() -> bool:
            nonlocal pending, pending_shape, pending_decode_seconds
            if not pending:
                return True
            batch = tuple(pending)
            event = DecodedBatch(
                items=batch,
                decode_seconds=pending_decode_seconds,
                uint8_bytes=_decoded_batch_uint8_bytes(batch),
            )
            pending = []
            pending_shape = None
            pending_decode_seconds = 0.0
            return put(("batch", event))

        try:
            for record in records:
                if stopped.is_set():
                    return
                decode_started = time.perf_counter()
                first = load(str(record["img0"]["path"]))
                middle = load(str(record["gt"]["path"]))
                last = load(str(record["img1"]["path"]))
                if first.shape != middle.shape or first.shape != last.shape:
                    raise ValueError(
                        f"diagnostic triplet shapes differ for {record.get('sample_id')}: "
                        f"{first.shape}, {middle.shape}, {last.shape}"
                    )
                record_decode_seconds = time.perf_counter() - decode_started
                if pending and first.shape != pending_shape and not flush():
                    return
                pending_shape = first.shape
                pending_decode_seconds += record_decode_seconds
                pending.append((record, first, middle, last))
                if len(pending) == batch_size and not flush():
                    return
            if not flush():
                return
        except Exception as exc:
            if not put(("error", exc)):
                return
        finally:
            put(("done", None))

    producer = Thread(target=produce, name="vfi-diagnostic-decode", daemon=True)
    producer.start()
    try:
        while True:
            kind, value = queue.get()
            if kind == "done":
                break
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
        producer.join(timeout=1.0)


def process_diagnostic_payload(
    payload: Mapping[str, Any],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    artifact_root: Path,
    heartbeat: Any | None = None,
    reconstruction_device: torch.device | str | None = None,
    progress_prefix: str = "",
) -> list[dict[str, Any]]:
    if (
        payload.get("run_hash") != config.run_hash()
        or payload.get("execution_id") != execution_id(config)
        or payload.get("stage") != "diagnostic"
    ):
        raise RuntimeError("task payload belongs to another run or stage")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("diagnostic task payload must contain a records array")
    max_cache = int(config.runtime.chunk_triplets) + 2
    cache_budget_bytes = int(config.runtime.decode_cache_mb) * 1024 * 1024
    postproc_buffer_bytes = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    output: list[dict[str, Any]] = []
    postproc_workers = _resolve_postproc_workers(config)
    pending_futures: list[PendingPostprocess] = []
    pending_reserved_bytes = 0
    pending_retained_bytes = 0
    oversize_logged = False
    timings = CpuTimingTotals()
    scored_count = 0
    next_timing_report = _TIMING_REPORT_SAMPLES
    decode_uint8_bytes = 0
    network_bytes = 0
    reconstruction_transient_bytes = 0
    bar = _ProgressLog(len(records), progress_prefix) if progress_prefix else None

    def memory_estimate() -> MemoryEstimate:
        return MemoryEstimate(
            decode_uint8_bytes=decode_uint8_bytes,
            network_bytes=network_bytes,
            reconstruction_transient_bytes=reconstruction_transient_bytes,
        )

    def maybe_report_timings(*, force: bool = False) -> None:
        nonlocal next_timing_report
        if not progress_prefix:
            return
        if force:
            _print_timing_summary(
                progress_prefix,
                timings,
                scored=scored_count,
                final=True,
            )
            return
        if scored_count < next_timing_report:
            return
        _print_timing_summary(
            progress_prefix,
            timings,
            scored=scored_count,
            final=False,
        )
        while next_timing_report <= scored_count:
            next_timing_report += _TIMING_REPORT_SAMPLES

    def drain_one() -> None:
        nonlocal pending_reserved_bytes, pending_retained_bytes, scored_count
        current = pending_futures.pop(0)
        reservation = current.reservation
        wait_started = time.perf_counter()
        while True:
            try:
                completed = current.future.result(timeout=_FUTURE_WAIT_SECONDS)
                break
            except FutureTimeoutError:
                if heartbeat is not None:
                    heartbeat()
                if bar is not None:
                    bar.waiting(
                        pending_batches=len(pending_futures) + 1,
                        pending_bytes=pending_reserved_bytes,
                        pending_retained_bytes=pending_retained_bytes,
                        memory=memory_estimate(),
                    )
        timings.add_future_wait(
            time.perf_counter() - wait_started,
            reservation.sample_count,
        )
        output.extend(completed)
        pending_reserved_bytes -= reservation.reserved_bytes
        pending_retained_bytes -= reservation.retained_bytes
        scored_count += reservation.sample_count
        if heartbeat is not None:
            heartbeat()
        if bar is not None:
            bar.update_scored(
                reservation.sample_count,
                pending_batches=len(pending_futures),
                pending_bytes=pending_reserved_bytes,
                pending_retained_bytes=pending_retained_bytes,
                memory=memory_estimate(),
            )
        maybe_report_timings()

    with ThreadPoolExecutor(
        max_workers=postproc_workers, thread_name_prefix="vfi-diagnostic-cpu"
    ) as executor:
        for decoded_value in _prefetched_diagnostic_batches(
            records,
            batch_size=config.model.batch_size,
            prefetch=config.runtime.prefetch,
            max_cache=max_cache,
            cache_budget_bytes=cache_budget_bytes,
        ):
            decoded_batch = (
                decoded_value
                if isinstance(decoded_value, DecodedBatch)
                else DecodedBatch(
                    items=tuple(decoded_value),
                    decode_seconds=0.0,
                    uint8_bytes=_decoded_batch_uint8_bytes(decoded_value),
                )
            )
            items = decoded_batch.items
            decode_uint8_bytes = decoded_batch.uint8_bytes
            timings.add_decode(decoded_batch.decode_seconds, len(items))
            inference_started = time.perf_counter()
            inference_batch = _infer_model_batch(
                items,
                adapter=adapter,
                production_batch=config.model.batch_size,
            )
            timings.add_inference(
                time.perf_counter() - inference_started,
                len(items),
            )
            network_bytes = inference_batch.network_bytes
            if bar is not None:
                bar.update_inferred(
                    len(items),
                    pending_batches=len(pending_futures),
                    pending_bytes=pending_reserved_bytes,
                    pending_retained_bytes=pending_retained_bytes,
                    memory=memory_estimate(),
                )
            network_bytes = inference_batch.output_bytes
            microbatch_size = _postproc_microbatch_size(
                items,
                buffer_bytes=postproc_buffer_bytes,
                postproc_workers=postproc_workers,
            )
            for start in range(0, len(items), microbatch_size):
                end = min(len(items), start + microbatch_size)
                item_slice = list(items[start:end])
                reservation = _postproc_reservation(
                    item_slice,
                    buffer_bytes=postproc_buffer_bytes,
                )
                while pending_futures and (
                    len(pending_futures) >= postproc_workers
                    or reservation.oversize
                    or pending_reserved_bytes + reservation.pipeline_bytes
                    > postproc_buffer_bytes
                ):
                    drain_one()
                reconstruction_transient_bytes = (
                    reservation.reconstruction_transient_bytes
                )
                if bar is not None:
                    bar.waiting(
                        pending_batches=len(pending_futures),
                        pending_bytes=pending_reserved_bytes,
                        pending_retained_bytes=pending_retained_bytes,
                        memory=memory_estimate(),
                    )
                reconstruction_started = time.perf_counter()
                prepared = _prepare_reconstruction_microbatch(item_slice)
                reconstructed = _reconstruct_outputs(
                    prepared.img0_tensor,
                    prepared.img1_tensor,
                    _slice_model_outputs(inference_batch.outputs, start, end),
                    model_config=config.model,
                    device=reconstruction_device,
                )
                timings.add_reconstruction(
                    time.perf_counter() - reconstruction_started,
                    len(item_slice),
                )
                future = executor.submit(
                    _finish_batch_diagnostics,
                    prepared.items,
                    reconstructed,
                    config=config,
                    artifact_root=artifact_root,
                )
                del prepared
                reconstruction_transient_bytes = 0
                pending_futures.append(
                    PendingPostprocess(
                        future=future,
                        reservation=reservation,
                    )
                )
                pending_reserved_bytes += reservation.reserved_bytes
                pending_retained_bytes += reservation.retained_bytes
                if reservation.oversize and not oversize_logged:
                    print(
                        f"{progress_prefix}  pending reservation "
                        f"{reservation.pipeline_bytes / (1024 * 1024):.0f} MiB "
                        f"exceeds budget "
                        f"{postproc_buffer_bytes / (1024 * 1024):.0f} MiB; "
                        "running one sample exclusively (oversize=1)",
                        file=sys.stderr,
                        flush=True,
                    )
                    oversize_logged = True
            network_bytes = 0
            decode_uint8_bytes = 0
        while pending_futures:
            drain_one()
    if bar is not None:
        bar.close(memory=memory_estimate())
    maybe_report_timings(force=True)
    return output


def _configure_worker_threads(config: AppConfig) -> None:
    torch.set_num_threads(config.runtime.cpu_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before inter-op work starts.
        pass


def _warmup(adapter: ModelAdapter, config: AppConfig) -> None:
    if config.runtime.warmup_batches <= 0:
        return
    shape = (
        config.model.batch_size,
        3,
        config.model.input_height,
        config.model.input_width,
    )
    first = torch.zeros(shape, dtype=torch.float32)
    last = torch.zeros(shape, dtype=torch.float32)
    for _ in range(config.runtime.warmup_batches):
        _pack_outputs_cpu(adapter.infer(first, last), config.model.batch_size)


def diagnostic_worker_entry(worker_index: int, device: torch.device, config_path: str) -> None:
    config = load_config(config_path)
    if config.runtime.precision != "float32":
        raise RuntimeError("final diagnostics require runtime.precision=float32")
    _configure_worker_threads(config)
    adapter = ModelAdapter.from_config(config.model, device=device, validate_values=False)
    _warmup(adapter, config)
    prefix = f"[{device} W{worker_index}:diagnostic]"
    reconstruction_device = _resolve_reconstruction_device(config, device)
    if (
        device.type != "cpu"
        and reconstruction_device is None
        and config.runtime.reconstruction == "auto"
    ):
        print(
            f"{prefix} device reconstruction "
            "unavailable; using CPU reference",
            file=sys.stderr,
            flush=True,
        )
    owner = f"{socket.gethostname()}:{os.getpid()}:{worker_index}:{device}"
    state_path = run_state_path(config, stage="diagnostic")
    parts_dir = run_directory(config) / "diagnostic_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with TaskStore(state_path) as store:
        while task := store.claim(owner, lease_seconds=config.runtime.lease_seconds):
            suffix = task.task_id.rsplit(":", 1)[-1]
            artifact_root = (
                run_directory(config)
                / "diagnostic_artifacts"
                / suffix
                / f"attempt-{task.attempt}"
            )
            part_path = parts_dir / f"{suffix}.attempt-{task.attempt}.jsonl"

            try:
                with LeaseHeartbeat(
                    state_path,
                    task.task_id,
                    owner,
                    lease_seconds=config.runtime.lease_seconds,
                    attempt=task.attempt,
                ) as lease:
                    records = process_diagnostic_payload(
                        task.payload,
                        adapter=adapter,
                        config=config,
                        artifact_root=artifact_root,
                        heartbeat=lease.check,
                        reconstruction_device=reconstruction_device,
                        progress_prefix=prefix,
                    )
                    for record in records:
                        record["task_id"] = task.task_id
                        record["attempt"] = task.attempt
                        record["artifact_root"] = str(artifact_root.resolve())
                    write_jsonl_part(part_path, records)
                    lease.check()
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
            except LeaseLostError:
                # Another attempt owns the task; this attempt's scoped files are ignored.
                continue
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
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
                if device.type != "cpu" and isinstance(exc, RuntimeError):
                    raise


def _cpu_diagnostic_bootstrap(worker_index: int, config_path: str) -> None:
    diagnostic_worker_entry(worker_index, torch.device("cpu"), config_path)


def _task_id(
    run_hash: str,
    execution: str,
    video_id: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    identity = {
        "run_hash": run_hash,
        "execution_id": execution,
        "stage": "diagnostic",
        "video_id": video_id,
        "samples": [
            [str(record["sample_id"]), str(record["diagnostic_relative"])]
            for record in records
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{run_hash}:diagnostic:{digest}"


def _completed_task_is_valid(config: AppConfig, task: TaskRecord) -> bool:
    if task.status != "done" or task.result_path is None:
        return False
    suffix = task.task_id.rsplit(":", 1)[-1]
    expected_part = (
        run_directory(config)
        / "diagnostic_parts"
        / f"{suffix}.attempt-{task.attempt}.jsonl"
    ).resolve()
    if Path(task.result_path).resolve() != expected_part or not expected_part.is_file():
        return False
    expected = {
        str(record["sample_id"]): Path(str(record["diagnostic_relative"])).as_posix()
        for record in task.payload.get("records", ())
    }
    try:
        results = list(read_jsonl(expected_part))
    except Exception:
        return False
    if len(results) != len(expected) or {str(item.get("sample_id", "")) for item in results} != set(
        expected
    ):
        return False
    artifact_root = (
        run_directory(config)
        / "diagnostic_artifacts"
        / suffix
        / f"attempt-{task.attempt}"
    ).resolve()
    for item in results:
        sample_id = str(item.get("sample_id", ""))
        relative = Path(str(item.get("visualization_relative", ""))).as_posix()
        if (
            item.get("task_id") != task.task_id
            or int(item.get("attempt", -1)) != task.attempt
            or Path(str(item.get("artifact_root", ""))).resolve() != artifact_root
            or relative != expected.get(sample_id)
            or Path(str(item.get("artifact_path", ""))).resolve()
            != (artifact_root / Path(relative)).resolve()
            or not Path(str(item.get("artifact_path", ""))).is_file()
        ):
            return False
    return True


def _prepare_tasks(
    config: AppConfig, records: Sequence[Mapping[str, Any]]
) -> tuple[int, tuple[str, ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["video_id"])].append(record)
    tasks: list[tuple[str, dict[str, Any]]] = []
    execution = execution_id(config)
    target_task_count = max(1, config.runtime.workers * 4)
    dynamic_chunk = max(1, (len(records) + target_task_count - 1) // target_task_count)
    chunk_size = min(config.runtime.chunk_triplets, dynamic_chunk)
    for video_id in sorted(grouped):
        ordered = sorted(
            grouped[video_id],
            key=lambda item: (tuple(item["frame_indices"]), str(item["sample_id"])),
        )
        for start in range(0, len(ordered), chunk_size):
            chunk = ordered[start : start + chunk_size]
            tasks.append(
                (
                    _task_id(config.run_hash(), execution, video_id, chunk),
                    {
                        "run_hash": config.run_hash(),
                        "execution_id": execution,
                        "stage": "diagnostic",
                        "video_id": video_id,
                        "records": [dict(record) for record in chunk],
                    },
                )
            )
    with TaskStore(run_state_path(config, stage="diagnostic")) as store:
        store.enqueue_many(tasks)
        store.recover_expired()
        for task_id, _ in tasks:
            task = store.get(task_id)
            if task is None:
                raise RuntimeError(f"diagnostic task disappeared after enqueue: {task_id}")
            if task.status == "running" and task.owner is not None:
                owner_parts = task.owner.split(":", 2)
                if len(owner_parts) >= 2 and owner_parts[0] == socket.gethostname():
                    try:
                        os.kill(int(owner_parts[1]), 0)
                    except ProcessLookupError:
                        store.requeue_orphaned_running(task_id, task.owner)
                    except (ValueError, PermissionError):
                        pass
            elif task.status == "failed":
                store.requeue_failed(task_id)
            elif task.status == "done" and not _completed_task_is_valid(config, task):
                store.requeue_done(task_id)
    return len(records), tuple(task_id for task_id, _ in tasks)


def run_diagnostic_stage(
    config_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> DiagnosticStageSummary:
    """Generate diagnostics on all configured devices and merge winning parts."""

    path = Path(config_path).resolve()
    config = load_config(path)
    candidates, task_ids = _prepare_tasks(config, records)
    worker_count = min(config.runtime.workers, len(task_ids))
    manifest_path = run_directory(config) / "diagnostic_results.jsonl"
    if candidates == 0:
        write_jsonl_part(manifest_path, [])
        return DiagnosticStageSummary(
            0,
            0,
            0,
            {"done": 0, "failed": 0, "pending": 0, "running": 0},
            manifest_path,
        )
    if config.runtime.backend == "cpu":
        context = get_spawn_context()
        processes = [
            context.Process(
                target=_cpu_diagnostic_bootstrap,
                args=(index, str(path)),
                name=f"vfi-diagnostic-cpu-{index}",
            )
            for index in range(worker_count)
        ]
        for process in processes:
            process.start()
        failures: list[str] = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append(f"{process.name}: exit code {process.exitcode}")
        if failures:
            raise RuntimeError("one or more diagnostic CPU workers failed: " + "; ".join(failures))
    else:
        devices = [
            f"{config.runtime.backend}:{index}"
            for index in config.runtime.devices[:worker_count]
        ]
        spawn_device_workers(
            diagnostic_worker_entry,
            devices,
            args=(str(path),),
            join=True,
        )
    with TaskStore(run_state_path(config, stage="diagnostic")) as store:
        maybe_tasks = [store.get(task_id) for task_id in task_ids]
        if any(task is None or task.status != "done" for task in maybe_tasks):
            current_counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for task in maybe_tasks:
                if task is not None:
                    current_counts[task.status] += 1
            raise RuntimeError(f"diagnostic stage did not finish cleanly: {current_counts}")
        tasks = [task for task in maybe_tasks if task is not None]
        counts = {
            status: sum(task is not None and task.status == status for task in tasks)
            for status in ("done", "failed", "pending", "running")
        }
        missing = [task.task_id for task in tasks if task.result_path is None]
        if missing:
            raise RuntimeError(f"diagnostic tasks completed without result parts: {missing[:3]}")
        parts: list[Path] = []
        tasks_by_id = {task.task_id: task for task in tasks}
        sample_task: dict[str, str] = {}
        for task in tasks:
            suffix = task.task_id.rsplit(":", 1)[-1]
            expected_part = (
                run_directory(config)
                / "diagnostic_parts"
                / f"{suffix}.attempt-{task.attempt}.jsonl"
            ).resolve()
            actual_part = Path(str(task.result_path)).resolve()
            if actual_part != expected_part or not actual_part.is_file():
                raise RuntimeError(
                    f"diagnostic winning part is missing or outside its attempt: {task.task_id}"
                )
            parts.append(actual_part)
            for source in task.payload["records"]:
                sample_id = str(source["sample_id"])
                if sample_id in sample_task:
                    raise RuntimeError(f"sample was assigned to two diagnostic tasks: {sample_id}")
                sample_task[sample_id] = task.task_id
    count = merge_jsonl_parts(parts, manifest_path)
    results = list(read_jsonl(manifest_path))
    expected_relative = {
        str(record["sample_id"]): Path(str(record["diagnostic_relative"])).as_posix()
        for record in records
    }
    result_ids = [str(item.get("sample_id", "")) for item in results]
    if (
        count != candidates
        or len(set(result_ids)) != candidates
        or set(result_ids) != set(expected_relative)
    ):
        raise RuntimeError(
            f"diagnostic result cardinality mismatch: expected {candidates}, got {count}"
        )
    for item in results:
        sample_id = str(item["sample_id"])
        task_id = str(item.get("task_id", ""))
        if sample_task.get(sample_id) != task_id or task_id not in tasks_by_id:
            raise RuntimeError(f"diagnostic result is not bound to its winning task: {sample_id}")
        task = tasks_by_id[task_id]
        if int(item.get("attempt", -1)) != task.attempt:
            raise RuntimeError(f"diagnostic result has stale attempt metadata: {sample_id}")
        suffix = task_id.rsplit(":", 1)[-1]
        expected_root = (
            run_directory(config)
            / "diagnostic_artifacts"
            / suffix
            / f"attempt-{task.attempt}"
        ).resolve()
        if Path(str(item.get("artifact_root", ""))).resolve() != expected_root:
            raise RuntimeError(f"diagnostic result has an invalid artifact root: {sample_id}")
        relative = Path(str(item.get("visualization_relative", ""))).as_posix()
        if relative != expected_relative[sample_id]:
            raise RuntimeError(f"diagnostic result path changed for {sample_id}")
        artifact = Path(item["artifact_path"]).resolve()
        if artifact != (expected_root / Path(relative)).resolve():
            raise RuntimeError(f"diagnostic artifact escaped its winning attempt: {sample_id}")
        if not artifact.is_file():
            raise FileNotFoundError(f"diagnostic artifact is missing: {item['artifact_path']}")
    return DiagnosticStageSummary(
        worker_count,
        candidates,
        count,
        counts,
        manifest_path,
    )


__all__ = [
    "DiagnosticStageSummary",
    "diagnostic_worker_entry",
    "process_diagnostic_payload",
    "run_diagnostic_stage",
]
