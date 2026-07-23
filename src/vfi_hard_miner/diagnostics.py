"""Spawn-safe, multi-device generation of final diagnostic images.

The scheduler in this module never configures an accelerator.  Each spawned
worker owns one device, loads the current model once, and writes attempt-scoped
artifacts.  Only the result part recorded by SQLite is consumed by finalize,
which prevents an expired attempt from publishing over the winning attempt.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Full, Queue
import socket
import sys
from threading import Event, Thread
import traceback
from typing import Any

import numpy as np
import torch

from .config import AppConfig, load_config
from .image_io import read_rgb_uint8, rgb_uint8_to_float32, write_image_atomic
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
    _ProgressLog,
    _infer_model_batch,
    _postproc_microbatch_size,
    _reconstruct_outputs,
    _reconstruction_bytes_per_sample,
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
            mask0=_hwc(reconstructed.mask0[index]),
            mask1=_hwc(reconstructed.mask1[index]),
            regions=_region_boxes(record),
            labels=record.get("reasons", ()),
            panel_width=config.output.visualization_width,
        )
        relative = str(record["diagnostic_relative"])
        destination = _safe_destination(artifact_root, relative)
        write_image_atomic(destination, grid)
        results.append(
            {
                "sample_id": str(record["sample_id"]),
                "artifact_path": str(destination),
                "visualization_relative": Path(relative).as_posix(),
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
) -> Iterator[list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]]]:
    """Decode and shape-group on a bounded CPU-only producer thread.

    Frames are cached as uint8 (a quarter of float32 memory) and converted to
    float32 when each triplet item is built.
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
            return rgb_uint8_to_float32(image)

        def flush() -> bool:
            nonlocal pending, pending_shape
            if not pending:
                return True
            batch = pending
            pending = []
            pending_shape = None
            return put(("batch", batch))

        try:
            for record in records:
                if stopped.is_set():
                    return
                first = load(str(record["img0"]["path"]))
                middle = load(str(record["gt"]["path"]))
                last = load(str(record["img1"]["path"]))
                if first.shape != middle.shape or first.shape != last.shape:
                    raise ValueError(
                        f"diagnostic triplet shapes differ for {record.get('sample_id')}: "
                        f"{first.shape}, {middle.shape}, {last.shape}"
                    )
                if pending and first.shape != pending_shape and not flush():
                    return
                pending_shape = first.shape
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
    pending_futures: list[
        tuple[Future[list[dict[str, Any]]], int, int]
    ] = []
    pending_bytes = 0
    bar = _ProgressLog(len(records), progress_prefix) if progress_prefix else None

    def drain_one() -> None:
        nonlocal pending_bytes
        future, sample_count, estimated_bytes = pending_futures.pop(0)
        while True:
            try:
                completed = future.result(timeout=_FUTURE_WAIT_SECONDS)
                break
            except FutureTimeoutError:
                if heartbeat is not None:
                    heartbeat()
                if bar is not None:
                    bar.waiting(
                        pending_batches=len(pending_futures) + 1,
                        pending_bytes=pending_bytes,
                    )
        output.extend(completed)
        pending_bytes -= estimated_bytes
        if heartbeat is not None:
            heartbeat()
        if bar is not None:
            bar.update_scored(
                sample_count,
                pending_batches=len(pending_futures),
                pending_bytes=pending_bytes,
            )

    with ThreadPoolExecutor(
        max_workers=postproc_workers, thread_name_prefix="vfi-diagnostic-cpu"
    ) as executor:
        for items in _prefetched_diagnostic_batches(
            records,
            batch_size=config.model.batch_size,
            prefetch=config.runtime.prefetch,
            max_cache=max_cache,
            cache_budget_bytes=cache_budget_bytes,
        ):
            img0_tensor, img1_tensor, outputs = _infer_model_batch(
                items,
                adapter=adapter,
                production_batch=config.model.batch_size,
            )
            if bar is not None:
                bar.update_inferred(
                    len(items),
                    pending_batches=len(pending_futures),
                    pending_bytes=pending_bytes,
                )
            microbatch_size = _postproc_microbatch_size(
                items,
                buffer_bytes=postproc_buffer_bytes,
                postproc_workers=postproc_workers,
            )
            bytes_per_sample = _reconstruction_bytes_per_sample(items)
            for start in range(0, len(items), microbatch_size):
                end = min(len(items), start + microbatch_size)
                item_slice = list(items[start:end])
                estimated_bytes = len(item_slice) * bytes_per_sample
                while pending_futures and (
                    len(pending_futures) >= postproc_workers
                    or pending_bytes + estimated_bytes > postproc_buffer_bytes
                ):
                    drain_one()
                reconstructed = _reconstruct_outputs(
                    img0_tensor[start:end],
                    img1_tensor[start:end],
                    _slice_model_outputs(outputs, start, end),
                    model_config=config.model,
                    device=reconstruction_device,
                )
                future = executor.submit(
                    _finish_batch_diagnostics,
                    item_slice,
                    reconstructed,
                    config=config,
                    artifact_root=artifact_root,
                )
                pending_futures.append(
                    (future, len(item_slice), estimated_bytes)
                )
                pending_bytes += estimated_bytes
        while pending_futures:
            drain_one()
    if bar is not None:
        bar.close()
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
