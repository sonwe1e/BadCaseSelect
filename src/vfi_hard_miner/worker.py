"""Worker-local model execution and per-triplet main-stage evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
from queue import Full, Queue
import socket
from threading import Event, Thread
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import AppConfig, load_config
from .diagnosis import (
    REASON_LABELS,
    SolvabilityResult,
    diagnose_sample,
    estimate_solvability,
)
from .gates import (
    compute_scope_metrics,
    compute_validity_metrics,
    decide_hard_case,
    evaluate_in_scope,
    evaluate_validity,
)
from .image_io import read_rgb_uint8, rgb_uint8_to_float32
from .manifest import write_jsonl_part
from .model_adapter import ModelAdapter, ModelOutputs
from .pipeline import run_directory
from .reconstruction import (
    ReconstructionResult,
    pack_reconstruction_to_cpu,
    reconstruct_midpoint,
)
from .scoring import score_local_errors, score_region
from .state import LeaseHeartbeat, LeaseLostError, TaskStore


ImageCache = OrderedDict[str, np.ndarray]
DecodedItem = tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]
DecodeEvent = tuple[str, Any]


def _tensor_from_hwc(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[0] in (1, 2, 3, 4):
        value = np.moveaxis(value, 0, -1)
    return np.asarray(value, dtype=np.float32)


def _load_cached_uint8(cache: ImageCache, path: str, *, max_items: int) -> np.ndarray:
    """LRU frame cache in uint8 form (4x less memory than float32)."""

    existing = cache.pop(path, None)
    if existing is not None:
        cache[path] = existing
        return existing
    image = read_rgb_uint8(path)
    cache[path] = image
    while len(cache) > max_items:
        cache.popitem(last=False)
    return image


def _label_value(label: str) -> bool | None:
    if label == "accept":
        return True
    if label == "review":
        return None
    if label == "reject":
        return False
    raise ValueError(f"gate label must be accept, review, or reject, got {label!r}")


def _invalid_record(record: Mapping[str, Any], error: Exception | str) -> dict[str, Any]:
    message = str(error)
    return {
        **dict(record),
        "status": "invalid",
        "validity_label": "reject",
        "in_scope_label": "review",
        "valid": False,
        "in_scope": None,
        "p_wrong": 0.0,
        "mining_p_wrong": 0.0,
        "p_solvable": 0.0,
        "reasons": ["invalid_decode_or_shape"],
        "regions": [],
        "metrics": {},
        "error": message,
    }


def _sample_record(
    source: Mapping[str, Any],
    *,
    img0: np.ndarray,
    gt: np.ndarray,
    img1: np.ndarray,
    reconstructed: ReconstructionResult,
    batch_index: int,
    thresholds: Any,
) -> dict[str, Any]:
    prediction = _hwc(reconstructed.prediction[batch_index])
    warp0 = _hwc(reconstructed.warp0[batch_index])
    warp1 = _hwc(reconstructed.warp1[batch_index])
    warp_blend = _hwc(reconstructed.warp_blend[batch_index])
    scoring = score_local_errors(
        prediction,
        gt,
        thresholds,
        img0=img0,
        img1=img1,
    )
    diagnosis = diagnose_sample(
        prediction,
        gt,
        warp0=warp0,
        warp1=warp1,
        warp_blend=warp_blend,
        img0=img0,
        img1=img1,
        regions=scoring.regions,
        scoring_config=thresholds,
        scoring_result=scoring,
        config=thresholds,
    )
    indices = tuple(int(value) for value in source["frame_indices"])
    stride = int(source["stride"])
    contiguous = indices == (indices[0], indices[0] + stride, indices[0] + 2 * stride)
    validity_metrics = compute_validity_metrics(
        img0,
        gt,
        img1,
        sequence_contiguous=contiguous,
    )
    validity = evaluate_validity(validity_metrics, thresholds)
    scope_metrics = compute_scope_metrics(
        reconstructed.flow_t0[batch_index],
        reconstructed.flow_t1[batch_index],
    )
    scope = evaluate_in_scope(scope_metrics, thresholds)
    decision = decide_hard_case(
        validity,
        scope,
        diagnosis.mining_p_wrong,
        diagnosis.p_solvable,
        thresholds,
    )
    if validity.label == "reject":
        status = "invalid"
    elif scope.label == "reject":
        status = "out_of_scope"
    else:
        status = decision.label
    reasons = list(
        dict.fromkeys((*diagnosis.reasons, *validity.reasons, *scope.reasons, *decision.reasons))
    )
    return {
        **dict(source),
        "status": status,
        "validity_label": validity.label,
        "in_scope_label": scope.label,
        "valid": _label_value(validity.label),
        "in_scope": _label_value(scope.label),
        "p_wrong": float(diagnosis.p_wrong),
        "mining_p_wrong": float(diagnosis.mining_p_wrong),
        "p_solvable": float(diagnosis.p_solvable),
        "reasons": reasons,
        "regions": [region.to_dict() for region in diagnosis.regions],
        "metrics": {
            "scoring": scoring.metrics,
            "diagnosis": diagnosis.metrics,
            "validity": validity.metrics,
            "scope": scope.metrics,
            "decision": decision.metrics,
        },
        "primary_region_index": diagnosis.primary_region_index,
        "error": None,
    }


def _pack_outputs_to_cpu(outputs: ModelOutputs, valid_count: int) -> ModelOutputs:
    """Truncate a padded batch and copy all model outputs in one D2H transfer."""

    batch = int(outputs.flow_t0.shape[0])
    if valid_count < 1 or valid_count > batch:
        raise ValueError(
            f"valid_count must be between 1 and output batch {batch}, got {valid_count}"
        )
    packed_device = torch.cat(
        (outputs.flow_t0, outputs.flow_t1, outputs.mask0, outputs.mask1),
        dim=1,
    )
    packed_cpu = packed_device[:valid_count].detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    flow_t0, flow_t1, mask0, mask1 = packed_cpu.split((2, 2, 1, 1), dim=1)
    return ModelOutputs(flow_t0, flow_t1, mask0, mask1)


def _warmup_adapter(
    adapter: ModelAdapter,
    model_config: Any,
    warmup_batches: int,
) -> None:
    """Run fixed-shape inference and synchronize through the production D2H path."""

    if warmup_batches <= 0:
        return
    shape = (
        int(model_config.batch_size),
        3,
        int(model_config.input_height),
        int(model_config.input_width),
    )
    img0 = torch.zeros(shape, dtype=torch.float32)
    img1 = torch.zeros(shape, dtype=torch.float32)
    for _ in range(warmup_batches):
        outputs = adapter.infer(img0, img1)
        _pack_outputs_to_cpu(outputs, shape[0])


def _configure_cpu_threads(config: AppConfig) -> None:
    configured = config.runtime.cpu_threads_per_worker
    raw_value = os.environ.get("VFI_CPU_THREADS_PER_WORKER", str(configured))
    try:
        thread_count = int(raw_value)
    except ValueError as exc:
        raise ValueError("VFI_CPU_THREADS_PER_WORKER must be an integer") from exc
    if thread_count < 1:
        raise ValueError("CPU threads per worker must be >= 1")
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits changing the inter-op pool before parallel work
        # starts.  Re-entry can happen in unit tests and embedded launchers.
        pass


def _probe_device_reconstruction(device: torch.device) -> bool:
    """Return True if ``device`` can run the reconstruction's grid_sample."""

    try:
        image = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device=device)
        grid = torch.zeros((1, 8, 8, 2), dtype=torch.float32, device=device)
        warped = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return bool(torch.isfinite(warped).all())
    except Exception:
        return False


def _resolve_reconstruction_device(
    config: AppConfig, device: torch.device
) -> torch.device | None:
    """Pick the reconstruction device per ``runtime.reconstruction``.

    Returns ``None`` for the CPU reference path.  ``"auto"`` probes the
    accelerator once and silently degrades to CPU when grid_sample is
    unavailable; ``"device"`` forces the accelerator and fails loudly.
    """

    mode = config.runtime.reconstruction
    if mode == "cpu" or device.type == "cpu":
        return None
    if mode == "device":
        return device
    if _probe_device_reconstruction(device):
        return device
    return None


def _resolve_postproc_workers(config: AppConfig) -> int:
    """Resolve the CPU postprocess thread count for one worker process.

    ``runtime.postproc_workers == 0`` selects an automatic count derived from
    the per-worker CPU budget; scipy/numpy release the GIL on the heavy paths,
    so these threads achieve real parallelism on reconstruction-free scoring.

    When ``cpu_threads_per_worker`` is the default value of 1 (meaning the
    user left it unconfigured), the formula ``1 // 4 == 0`` would collapse to
    a single postproc thread and negate the entire overlap design.  Fall back
    to the physical core count in that case so the thread pool actually fires.
    """

    configured = int(config.runtime.postproc_workers)
    if configured > 0:
        return configured
    cpu = int(config.runtime.cpu_threads_per_worker)
    if cpu <= 1:
        # cpu_threads_per_worker=1 is the "not explicitly configured" default;
        # derive from physical cores instead of dividing 1 by 4.
        cpu = os.cpu_count() or 4
    automatic = max(1, cpu // 4)
    return min(automatic, 8)


class _ProgressLog:
    """Periodic stderr progress reporter, safe for concurrent worker processes.

    Each worker process writes complete lines so output from different processes
    does not interleave mid-line.  A log line is emitted at construction time
    (task start), every ``_INTERVAL`` seconds during processing, and once on
    ``close()`` (task end).
    """

    _INTERVAL: float = 30.0

    def __init__(self, total: int, prefix: str) -> None:
        self._total = total
        self._prefix = prefix
        self._inferred = 0
        self._scored = 0
        self._start = time.monotonic()
        self._last = self._start
        print(
            f"{self._prefix}  inferred 0/{self._total}"
            f"  scored 0/{self._total}  starting",
            file=sys.stderr,
            flush=True,
        )

    def update_inferred(
        self, n: int, *, pending_batches: int, pending_bytes: int
    ) -> None:
        self._inferred += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
        )

    def update_scored(
        self, n: int, *, pending_batches: int, pending_bytes: int
    ) -> None:
        self._scored += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
        )

    def update_invalid(
        self, n: int, *, pending_batches: int, pending_bytes: int
    ) -> None:
        self._inferred += n
        self._scored += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
        )

    def waiting(self, *, pending_batches: int, pending_bytes: int) -> None:
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
        )

    def _maybe_emit(self, *, pending_batches: int, pending_bytes: int) -> None:
        now = time.monotonic()
        if now - self._last >= self._INTERVAL:
            self._emit(
                now,
                pending_batches=pending_batches,
                pending_bytes=pending_bytes,
            )
            self._last = now

    def close(self, *, pending_batches: int = 0, pending_bytes: int = 0) -> None:
        self._emit(
            time.monotonic(),
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            final=True,
        )

    def _emit(
        self,
        now: float,
        *,
        pending_batches: int,
        pending_bytes: int,
        final: bool = False,
    ) -> None:
        elapsed = now - self._start
        rate = self._scored / elapsed if elapsed > 0 else 0.0
        label = "done" if final else "..."
        print(
            f"{self._prefix}  inferred {self._inferred}/{self._total}"
            f"  scored {self._scored}/{self._total}"
            f"  pending {pending_batches} batches/{pending_bytes / (1024 * 1024):.0f} MiB"
            f"  {elapsed:.0f}s  {rate:.1f}/s  {label}",
            file=sys.stderr,
            flush=True,
        )


def _prefetched_decode_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    prefetch: int,
    max_cache: int,
    cache_budget_bytes: int | None = None,
) -> Iterator[DecodeEvent]:
    """Decode/group triplets on one bounded producer thread.

    The producer only performs CPU file I/O and shape grouping.  Model calls
    remain on the worker's main thread, so the producer never touches a CANN,
    CUDA, or NPU context.

    Frames are cached as uint8 (a quarter of float32 memory) so a large cache
    can cover a whole stride=1 chunk; conversion to float32 happens when each
    triplet item is built, so downstream consumers still see the usual
    float32 arrays.  ``cache_budget_bytes`` caps cache memory once the first
    frame's size is known.
    """

    queue: Queue[DecodeEvent] = Queue(maxsize=max(1, prefetch))
    stopped = Event()

    def put(event: DecodeEvent) -> bool:
        while not stopped.is_set():
            try:
                queue.put(event, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        cache: ImageCache = OrderedDict()
        # Separate float32 layer so stride=1 frames aren't re-converted on each
        # triplet overlap.  For stride=1 data each unique frame appears in up to
        # three consecutive triplets; without this cache the uint8 LRU hit still
        # costs one full-frame float32 allocation + divide per access.
        f32_cache: ImageCache = OrderedDict()
        pending: list[DecodedItem] = []
        pending_shape: tuple[int, int, int] | None = None
        capacity = max(1, int(max_cache))
        # float32 is 4x larger than uint8; keep proportionally fewer entries.
        f32_capacity = max(4, capacity // 4)
        budget_resolved = cache_budget_bytes is None

        def load(path: str) -> np.ndarray:
            nonlocal capacity, f32_capacity, budget_resolved
            # Fast path: float32 already computed for this path (stride=1 hit).
            existing = f32_cache.pop(path, None)
            if existing is not None:
                f32_cache[path] = existing
                return existing
            image = _load_cached_uint8(cache, path, max_items=capacity)
            if not budget_resolved:
                budget_resolved = True
                capacity = max(
                    1,
                    min(capacity, max(8, int(cache_budget_bytes) // max(1, image.nbytes))),
                )
                # Re-derive f32 capacity now that per-frame size is known.
                f32_capacity = max(4, capacity // 4)
            f32 = rgb_uint8_to_float32(image)
            f32_cache[path] = f32
            while len(f32_cache) > f32_capacity:
                f32_cache.popitem(last=False)
            return f32

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
                try:
                    first = load(str(record["img0"]["path"]))
                    middle = load(str(record["gt"]["path"]))
                    last = load(str(record["img1"]["path"]))
                    if first.shape != middle.shape or first.shape != last.shape:
                        raise ValueError(
                            "triplet image shapes differ: "
                            f"{first.shape}, {middle.shape}, {last.shape}"
                        )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if not flush() or not put(("invalid", (record, exc))):
                        return
                    continue
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

    producer = Thread(target=produce, name="vfi-decode-prefetch", daemon=True)
    producer.start()
    try:
        while True:
            kind, value = queue.get()
            if kind == "done":
                break
            if kind == "error":
                raise value
            yield kind, value
    finally:
        stopped.set()
        producer.join(timeout=1.0)


def _infer_output_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    *,
    adapter: ModelAdapter,
    production_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, ModelOutputs]:
    if not items:
        raise ValueError("inference batch must not be empty")
    valid_count = len(items)
    padded = list(items)
    while len(padded) < production_batch:
        padded.append(padded[-1])
    img0_tensor = torch.stack([_tensor_from_hwc(item[1]) for item in padded])
    img1_tensor = torch.stack([_tensor_from_hwc(item[3]) for item in padded])
    outputs = _pack_outputs_to_cpu(
        adapter.infer(img0_tensor, img1_tensor),
        valid_count,
    )
    return img0_tensor[:valid_count], img1_tensor[:valid_count], outputs


def _trim_model_outputs(outputs: ModelOutputs, valid_count: int) -> ModelOutputs:
    """Drop tail padding on-device (a cheap slice, no host synchronization)."""

    if valid_count < 1 or valid_count > outputs.flow_t0.shape[0]:
        raise ValueError(
            f"valid_count must be between 1 and output batch "
            f"{outputs.flow_t0.shape[0]}, got {valid_count}"
        )
    if valid_count == outputs.flow_t0.shape[0]:
        return outputs
    return ModelOutputs(
        outputs.flow_t0[:valid_count],
        outputs.flow_t1[:valid_count],
        outputs.mask0[:valid_count],
        outputs.mask1[:valid_count],
    )


def _reconstruct_outputs(
    img0_tensor: torch.Tensor,
    img1_tensor: torch.Tensor,
    outputs: ModelOutputs,
    *,
    model_config: Any,
    device: torch.device | str | None = None,
) -> ReconstructionResult:
    """Run reconstruction on ``device`` and return CPU tensors.

    ``device=None`` keeps the calibrated CPU reference path.  On accelerator
    devices the native-resolution warps run there and the result returns to
    the host in one packed transfer, so CPU threads only ever score.
    """

    reconstructed = reconstruct_midpoint(
        img0_tensor,
        img1_tensor,
        outputs.flow_t0,
        outputs.flow_t1,
        outputs.mask0,
        outputs.mask1,
        network_size=(model_config.input_height, model_config.input_width),
        mask0_role=model_config.mask0_role,
        align_corners=model_config.align_corners,
        padding_mode=model_config.padding_mode,
        device=device,
    )
    if reconstructed.prediction.device.type != "cpu":
        reconstructed = pack_reconstruction_to_cpu(reconstructed)
    return reconstructed


def _infer_and_reconstruct(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    *,
    adapter: ModelAdapter,
    model_config: Any,
    production_batch: int,
    reconstruction_device: torch.device | str | None = None,
) -> ReconstructionResult:
    """Inference + reconstruction on the worker's main thread.

    Model outputs never round-trip to the host before reconstruction: on
    accelerator devices the reconstruction consumes them in place, and only
    the finished native-resolution result is transferred back (packed).
    """

    if not items:
        raise ValueError("inference batch must not be empty")
    valid_count = len(items)
    padded = list(items)
    while len(padded) < production_batch:
        padded.append(padded[-1])
    img0_tensor = torch.stack([_tensor_from_hwc(item[1]) for item in padded])
    img1_tensor = torch.stack([_tensor_from_hwc(item[3]) for item in padded])
    outputs = _trim_model_outputs(
        adapter.infer(img0_tensor, img1_tensor), valid_count
    )
    return _reconstruct_outputs(
        img0_tensor[:valid_count],
        img1_tensor[:valid_count],
        outputs,
        model_config=model_config,
        device=reconstruction_device,
    )


_RECONSTRUCTION_CHANNELS = 15
_FUTURE_WAIT_SECONDS = 5.0


def _infer_model_batch(
    items: Sequence[DecodedItem],
    *,
    adapter: ModelAdapter,
    production_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, ModelOutputs]:
    """Run one production-sized model batch without full-resolution reconstruction."""

    if not items:
        raise ValueError("inference batch must not be empty")
    valid_count = len(items)
    padded = list(items)
    while len(padded) < production_batch:
        padded.append(padded[-1])
    img0_tensor = torch.stack([_tensor_from_hwc(item[1]) for item in padded])
    img1_tensor = torch.stack([_tensor_from_hwc(item[3]) for item in padded])
    outputs = _trim_model_outputs(
        adapter.infer(img0_tensor, img1_tensor),
        valid_count,
    )
    return img0_tensor[:valid_count], img1_tensor[:valid_count], outputs


def _slice_model_outputs(
    outputs: ModelOutputs, start: int, end: int
) -> ModelOutputs:
    return ModelOutputs(
        outputs.flow_t0[start:end],
        outputs.flow_t1[start:end],
        outputs.mask0[start:end],
        outputs.mask1[start:end],
    )


def _reconstruction_bytes_per_sample(items: Sequence[DecodedItem]) -> int:
    if not items:
        raise ValueError("cannot estimate reconstruction bytes for an empty batch")
    height, width = items[0][1].shape[:2]
    return int(height) * int(width) * _RECONSTRUCTION_CHANNELS * 4


def _postproc_microbatch_size(
    items: Sequence[DecodedItem],
    *,
    buffer_bytes: int,
    postproc_workers: int,
) -> int:
    """Choose a slice so all active CPU futures fit inside one worker budget."""

    bytes_per_sample = _reconstruction_bytes_per_sample(items)
    per_future_budget = max(1, int(buffer_bytes) // max(1, int(postproc_workers)))
    return max(1, min(len(items), per_future_budget // max(1, bytes_per_sample)))


def _process_payload_records(
    records: Sequence[Mapping[str, Any]],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    model_config: Any,
    finish_batch: Callable[..., list[dict[str, Any]]],
    invalid_record: Callable[[Mapping[str, Any], Exception], dict[str, Any]],
    heartbeat: Callable[[], None] | None,
    progress_prefix: str,
    reconstruction_device: torch.device | str | None,
    thread_name_prefix: str,
) -> list[dict[str, Any]]:
    max_cache = int(config.runtime.chunk_triplets) + 2
    cache_budget_bytes = int(config.runtime.decode_cache_mb) * 1024 * 1024
    postproc_buffer_bytes = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    postproc_workers = _resolve_postproc_workers(config)
    output: list[dict[str, Any]] = []
    pending: list[tuple[Future[list[dict[str, Any]]], int, int]] = []
    pending_bytes = 0
    bar = _ProgressLog(len(records), progress_prefix) if progress_prefix else None

    def drain_one() -> None:
        nonlocal pending_bytes
        future, sample_count, estimated_bytes = pending.pop(0)
        while True:
            try:
                completed = future.result(timeout=_FUTURE_WAIT_SECONDS)
                break
            except FutureTimeoutError:
                if heartbeat is not None:
                    heartbeat()
                if bar is not None:
                    bar.waiting(
                        pending_batches=len(pending) + 1,
                        pending_bytes=pending_bytes,
                    )
        output.extend(completed)
        pending_bytes -= estimated_bytes
        if heartbeat is not None:
            heartbeat()
        if bar is not None:
            bar.update_scored(
                sample_count,
                pending_batches=len(pending),
                pending_bytes=pending_bytes,
            )

    with ThreadPoolExecutor(
        max_workers=postproc_workers,
        thread_name_prefix=thread_name_prefix,
    ) as executor:
        for kind, value in _prefetched_decode_batches(
            records,
            batch_size=model_config.batch_size,
            prefetch=config.runtime.prefetch,
            max_cache=max_cache,
            cache_budget_bytes=cache_budget_bytes,
        ):
            if kind == "batch":
                items: Sequence[DecodedItem] = value
                img0_tensor, img1_tensor, outputs = _infer_model_batch(
                    items,
                    adapter=adapter,
                    production_batch=model_config.batch_size,
                )
                if bar is not None:
                    bar.update_inferred(
                        len(items),
                        pending_batches=len(pending),
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
                    while pending and (
                        len(pending) >= postproc_workers
                        or pending_bytes + estimated_bytes > postproc_buffer_bytes
                    ):
                        drain_one()
                    reconstructed = _reconstruct_outputs(
                        img0_tensor[start:end],
                        img1_tensor[start:end],
                        _slice_model_outputs(outputs, start, end),
                        model_config=model_config,
                        device=reconstruction_device,
                    )
                    future = executor.submit(
                        finish_batch,
                        item_slice,
                        reconstructed,
                        config=config,
                    )
                    pending.append((future, len(item_slice), estimated_bytes))
                    pending_bytes += estimated_bytes
            elif kind == "invalid":
                while pending:
                    drain_one()
                record, error = value
                output.append(invalid_record(record, error))
                if bar is not None:
                    bar.update_invalid(
                        1,
                        pending_batches=0,
                        pending_bytes=0,
                    )
            else:  # pragma: no cover - producer owns this internal protocol
                raise RuntimeError(f"unexpected decode event {kind!r}")
            if heartbeat is not None:
                heartbeat()
        while pending:
            drain_one()
    if bar is not None:
        bar.close()
    return output


def _finish_main_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
) -> list[dict[str, Any]]:
    return [
        _sample_record(
            item[0],
            img0=item[1],
            gt=item[2],
            img1=item[3],
            reconstructed=reconstructed,
            batch_index=index,
            thresholds=config.thresholds,
        )
        for index, item in enumerate(items)
    ]


def _evaluate_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    reconstruction_device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    reconstructed = _infer_and_reconstruct(
        items,
        adapter=adapter,
        model_config=config.model,
        production_batch=config.model.batch_size,
        reconstruction_device=reconstruction_device,
    )
    return _finish_main_batch(
        items,
        reconstructed,
        config=config,
    )


def _validate_payload_identity(
    payload: Mapping[str, Any], config: AppConfig, *, stage: str
) -> None:
    if payload.get("run_hash") != config.run_hash() or payload.get("stage") != stage:
        raise RuntimeError("task payload belongs to another run or stage")
    from .pipeline import execution_id

    if payload.get("execution_id") != execution_id(config):
        raise RuntimeError("task payload belongs to another execution snapshot")


def process_main_payload(
    payload: Mapping[str, Any],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    heartbeat: Callable[[], None] | None = None,
    progress_prefix: str = "",
    reconstruction_device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    _validate_payload_identity(payload, config, stage="main")
    records = payload.get("triplets")
    if not isinstance(records, list):
        raise TypeError("main task payload must contain a triplets array")
    return _process_payload_records(
        records,
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=_finish_main_batch,
        invalid_record=_invalid_record,
        heartbeat=heartbeat,
        progress_prefix=progress_prefix,
        reconstruction_device=reconstruction_device,
        thread_name_prefix="vfi-main-cpu",
    )


def main_worker_entry(
    worker_index: int,
    device: torch.device,
    config_path: str,
) -> None:
    """Top-level spawn target: bind device, load once, claim until empty."""

    config = load_config(config_path)
    if config.runtime.precision != "float32":
        raise RuntimeError(
            "the first production baseline requires runtime.precision=float32; "
            "enable mixed precision only after target-side parity validation"
        )
    _configure_cpu_threads(config)
    adapter = ModelAdapter.from_config(
        config.model,
        device=device,
        validate_values=False,
    )
    _warmup_adapter(adapter, config.model, config.runtime.warmup_batches)
    _prefix = f"[{device} W{worker_index}]"
    reconstruction_device = _resolve_reconstruction_device(config, device)
    if (
        device.type != "cpu"
        and reconstruction_device is None
        and config.runtime.reconstruction == "auto"
    ):
        print(
            f"{_prefix} device reconstruction unavailable; using CPU reference",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"{_prefix} ready (reconstruction: {reconstruction_device or 'cpu'})",
        file=sys.stderr,
        flush=True,
    )
    owner = f"{socket.gethostname()}:{os.getpid()}:{worker_index}:{device}"
    state_path = __import__("vfi_hard_miner.pipeline", fromlist=["run_state_path"]).run_state_path(
        config, stage="main"
    )
    parts_dir = run_directory(config) / "main_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with TaskStore(state_path) as store:
        while task := store.claim(owner, lease_seconds=config.runtime.lease_seconds):
            short_id = task.task_id.rsplit(":", 1)[-1]
            n_triplets = len(task.payload.get("triplets", ()))
            part_path = parts_dir / f"{short_id}.attempt-{task.attempt}.jsonl"
            print(
                f"{_prefix} task {short_id}: {n_triplets} triplets",
                file=sys.stderr,
                flush=True,
            )
            _t0 = time.monotonic()
            try:
                with LeaseHeartbeat(
                    state_path,
                    task.task_id,
                    owner,
                    lease_seconds=config.runtime.lease_seconds,
                    attempt=task.attempt,
                ) as lease:
                    records = process_main_payload(
                        task.payload,
                        adapter=adapter,
                        config=config,
                        heartbeat=lease.check,
                        progress_prefix=_prefix,
                        reconstruction_device=reconstruction_device,
                    )
                    print(
                        f"{_prefix} task {short_id}: scoring complete; writing JSON part",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_jsonl_part(part_path, records)
                    lease.check()
                _elapsed = time.monotonic() - _t0
                _rate = n_triplets / _elapsed if _elapsed > 0 else 0.0
                print(
                    f"{_prefix} task {short_id}: JSON part committed"
                    f"  {n_triplets} triplets  {_elapsed:.1f}s  {_rate:.1f}/s",
                    file=sys.stderr,
                    flush=True,
                )
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
                print(
                    f"{_prefix} task {short_id}: SQLite task committed",
                    file=sys.stderr,
                    flush=True,
                )
            except LeaseLostError:
                print(f"{_prefix} task {short_id}: lease lost", file=sys.stderr, flush=True)
                continue
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"{_prefix} task {short_id}: failed — {detail}", file=sys.stderr, flush=True)
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


def _main_decision_snapshot(
    source: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    existing = source.get("main_decision")
    if isinstance(existing, Mapping):
        return dict(existing)
    decision = metrics.get("decision", {})
    return {
        "status": source.get("status"),
        "p_wrong": float(source.get("p_wrong", 0.0)),
        "mining_p_wrong": float(
            source.get("mining_p_wrong", source.get("p_wrong", 0.0))
        ),
        "p_solvable": float(source.get("p_solvable", 0.0)),
        "reasons": [str(reason) for reason in source.get("reasons", ())],
        "decision": dict(decision) if isinstance(decision, Mapping) else {},
    }


def _teacher_region_update(
    region: Mapping[str, Any],
    *,
    teacher_structure: np.ndarray,
    thresholds: Any,
) -> tuple[dict[str, Any], SolvabilityResult, float]:
    box = tuple(int(value) for value in region["box"])
    teacher_error = score_region(teacher_structure, box)
    raw_metrics = region.get("metrics", {})
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
    current_error = float(region.get("p_wrong", metrics.get("current_error", 0.0)))
    warp_errors: dict[str, float] = {}
    for name in ("warp0_error", "warp1_error", "warp_blend_error"):
        value = metrics.get(name)
        if value is not None and float(value) >= 0.0:
            warp_errors[name] = float(value)
    solvability = estimate_solvability(
        current_error,
        teacher_error,
        warp_errors=warp_errors,
        config=thresholds,
    )
    evidence = solvability.to_dict()
    metrics.update(
        {
            "teacher_error": float(teacher_error),
            "teacher_gain": (
                float(solvability.teacher_gain)
                if solvability.teacher_gain is not None
                else -1.0
            ),
            "teacher_best_warp_error": (
                float(solvability.best_warp_error)
                if solvability.best_warp_error is not None
                else -1.0
            ),
            "teacher_warp_gain": (
                float(solvability.warp_gain)
                if solvability.warp_gain is not None
                else -1.0
            ),
            "p_solvable": float(solvability.p_solvable),
        }
    )
    updated = dict(region)
    updated["p_wrong"] = current_error
    priority_weight = float(np.clip(metrics.get("priority_weight", 1.0), 0.0, 1.0))
    metrics["priority_weight"] = priority_weight
    metrics["ui_likelihood"] = float(
        np.clip(metrics.get("ui_likelihood", 0.0), 0.0, 1.0)
    )
    metrics["mining_p_wrong"] = float(current_error * priority_weight)
    updated["mining_p_wrong"] = float(current_error * priority_weight)
    updated["p_solvable"] = float(solvability.p_solvable)
    updated["reasons"] = [
        str(reason)
        for reason in region.get("reasons", ())
        if str(reason) in REASON_LABELS
    ]
    updated["metrics"] = metrics
    updated["teacher"] = {
        "local_error": float(teacher_error),
        "solvability": evidence,
    }
    return updated, solvability, float(teacher_error)


def _teacher_update_record(
    source: Mapping[str, Any],
    *,
    gt: np.ndarray,
    reconstructed: ReconstructionResult,
    batch_index: int,
    thresholds: Any,
) -> dict[str, Any]:
    teacher_prediction = _hwc(reconstructed.prediction[batch_index])
    teacher_scoring = score_local_errors(teacher_prediction, gt, thresholds)
    raw_regions = source.get("regions")
    if isinstance(raw_regions, list) and raw_regions:
        evaluated = [
            _teacher_region_update(
                region,
                teacher_structure=teacher_scoring.maps.structure,
                thresholds=thresholds,
            )
            for region in raw_regions
        ]
        regions = [item[0] for item in evaluated]
        primary_index = max(
            range(len(regions)),
            key=lambda index: (
                float(regions[index].get("p_wrong", 0.0))
                * float(regions[index].get("p_solvable", 0.0))
                * float(
                    regions[index].get("metrics", {}).get("priority_weight", 1.0)
                ),
                float(regions[index].get("p_wrong", 0.0))
                * float(
                    regions[index].get("metrics", {}).get("priority_weight", 1.0)
                ),
                float(regions[index].get("p_wrong", 0.0)),
                -index,
            ),
        )
        primary = regions[primary_index]
        solvability = evaluated[primary_index][1]
        teacher_error = evaluated[primary_index][2]
        p_wrong = float(primary.get("p_wrong", 0.0))
        priority_weight = float(
            np.clip(primary.get("metrics", {}).get("priority_weight", 1.0), 0.0, 1.0)
        )
        mining_p_wrong = float(p_wrong * priority_weight)
        box = tuple(int(value) for value in primary["box"])
    else:
        regions = []
        primary_index = None
        box = None
        teacher_error = teacher_scoring.p_wrong
        p_wrong = float(source.get("p_wrong", 0.0))
        mining_p_wrong = float(source.get("mining_p_wrong", p_wrong))
        priority_weight = float(
            np.clip(mining_p_wrong / max(p_wrong, 1e-8) if p_wrong > 0.0 else 1.0, 0.0, 1.0)
        )
        solvability = estimate_solvability(
            p_wrong,
            teacher_error,
            config=thresholds,
        )
    raw_metrics = source.get("metrics", {})
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
    validity = evaluate_validity(metrics.get("validity", {}), thresholds)
    scope = evaluate_in_scope(metrics.get("scope", {}), thresholds)
    decision = decide_hard_case(
        validity,
        scope,
        mining_p_wrong,
        solvability.p_solvable,
        thresholds,
    )
    if validity.label == "reject":
        status = "invalid"
    elif scope.label == "reject":
        status = "out_of_scope"
    else:
        status = decision.label
    if regions:
        diagnostic_reasons = [
            str(reason)
            for region in regions
            for reason in region.get("reasons", ())
            if str(reason) in REASON_LABELS
        ]
    else:
        diagnostic_reasons = [
            str(reason)
            for reason in source.get("reasons", ())
            if str(reason) in REASON_LABELS
        ]
    reasons = list(
        dict.fromkeys(
            (
                *diagnostic_reasons,
                *validity.reasons,
                *scope.reasons,
                *decision.reasons,
            )
        )
    )
    updated = dict(source)
    updated["main_decision"] = _main_decision_snapshot(source, metrics)
    updated["status"] = status
    updated["validity_label"] = validity.label
    updated["in_scope_label"] = scope.label
    updated["valid"] = _label_value(validity.label)
    updated["in_scope"] = _label_value(scope.label)
    updated["p_wrong"] = p_wrong
    updated["mining_p_wrong"] = mining_p_wrong
    updated["p_solvable"] = float(solvability.p_solvable)
    updated["reasons"] = reasons
    updated["regions"] = regions
    updated["primary_region_index"] = primary_index
    updated["teacher"] = {
        "local_error": float(teacher_error),
        "global_local_error": float(teacher_scoring.p_wrong),
        "region": None if box is None else list(box),
        "solvability": solvability.to_dict(),
    }
    updated_metrics = dict(metrics)
    raw_diagnosis = metrics.get("diagnosis", {})
    diagnosis_metrics = (
        dict(raw_diagnosis) if isinstance(raw_diagnosis, Mapping) else {}
    )
    diagnosis_metrics["selected_p_wrong"] = p_wrong
    diagnosis_metrics["selected_mining_p_wrong"] = mining_p_wrong
    diagnosis_metrics["selected_priority_weight"] = priority_weight
    diagnosis_metrics["selected_ui_likelihood"] = float(
        primary.get("metrics", {}).get("ui_likelihood", 0.0) if regions else 0.0
    )
    diagnosis_metrics["selected_p_solvable"] = float(solvability.p_solvable)
    updated_metrics["diagnosis"] = diagnosis_metrics
    updated_metrics["decision"] = dict(decision.metrics)
    updated_metrics["teacher_decision"] = decision.to_dict()
    updated["metrics"] = updated_metrics
    return updated


def _evaluate_teacher_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    reconstruction_device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    if not items or config.teacher is None:
        return []
    reconstructed = _infer_and_reconstruct(
        items,
        adapter=adapter,
        model_config=config.teacher,
        production_batch=config.teacher.batch_size,
        reconstruction_device=reconstruction_device,
    )
    return _finish_teacher_batch(
        items,
        reconstructed,
        config=config,
    )


def _finish_teacher_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
) -> list[dict[str, Any]]:
    if config.teacher is None:
        raise RuntimeError("teacher postprocess requires config.teacher")
    return [
        _teacher_update_record(
            item[0],
            gt=item[2],
            reconstructed=reconstructed,
            batch_index=index,
            thresholds=config.thresholds,
        )
        for index, item in enumerate(items)
    ]


def process_teacher_payload(
    payload: Mapping[str, Any],
    *,
    adapter: ModelAdapter,
    config: AppConfig,
    heartbeat: Callable[[], None] | None = None,
    progress_prefix: str = "",
    reconstruction_device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    if config.teacher is None:
        raise RuntimeError("teacher stage requires config.teacher")
    _validate_payload_identity(payload, config, stage="teacher")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("teacher task payload must contain a records array")

    def invalid_teacher_record(
        record: Mapping[str, Any], error: Exception
    ) -> dict[str, Any]:
        failed = dict(record)
        failed["status"] = "review"
        failed["teacher"] = {"error": str(error)}
        return failed

    return _process_payload_records(
        records,
        adapter=adapter,
        config=config,
        model_config=config.teacher,
        finish_batch=_finish_teacher_batch,
        invalid_record=invalid_teacher_record,
        heartbeat=heartbeat,
        progress_prefix=progress_prefix,
        reconstruction_device=reconstruction_device,
        thread_name_prefix="vfi-teacher-cpu",
    )


def teacher_worker_entry(
    worker_index: int,
    device: torch.device,
    config_path: str,
) -> None:
    config = load_config(config_path)
    if config.teacher is None:
        raise RuntimeError("teacher worker requires config.teacher")
    if config.runtime.precision != "float32":
        raise RuntimeError("teacher baseline requires runtime.precision=float32")
    _configure_cpu_threads(config)
    adapter = ModelAdapter.from_config(
        config.teacher,
        device=device,
        validate_values=False,
    )
    _warmup_adapter(adapter, config.teacher, config.runtime.warmup_batches)
    _prefix = f"[{device} W{worker_index}:teacher]"
    reconstruction_device = _resolve_reconstruction_device(config, device)
    if (
        device.type != "cpu"
        and reconstruction_device is None
        and config.runtime.reconstruction == "auto"
    ):
        print(
            f"{_prefix} device reconstruction unavailable; using CPU reference",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"{_prefix} ready (reconstruction: {reconstruction_device or 'cpu'})",
        file=sys.stderr,
        flush=True,
    )
    owner = f"{socket.gethostname()}:{os.getpid()}:{worker_index}:{device}:teacher"
    state_path = __import__("vfi_hard_miner.pipeline", fromlist=["run_state_path"]).run_state_path(
        config, stage="teacher"
    )
    parts_dir = run_directory(config) / "teacher_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with TaskStore(state_path) as store:
        while task := store.claim(owner, lease_seconds=config.runtime.lease_seconds):
            short_id = task.task_id.rsplit(":", 1)[-1]
            n_records = len(task.payload.get("records", ()))
            part_path = parts_dir / f"{short_id}.attempt-{task.attempt}.jsonl"
            print(
                f"{_prefix} task {short_id}: {n_records} records",
                file=sys.stderr,
                flush=True,
            )
            _t0 = time.monotonic()
            try:
                with LeaseHeartbeat(
                    state_path,
                    task.task_id,
                    owner,
                    lease_seconds=config.runtime.lease_seconds,
                    attempt=task.attempt,
                ) as lease:
                    records = process_teacher_payload(
                        task.payload,
                        adapter=adapter,
                        config=config,
                        heartbeat=lease.check,
                        progress_prefix=_prefix,
                        reconstruction_device=reconstruction_device,
                    )
                    print(
                        f"{_prefix} task {short_id}: scoring complete; writing JSON part",
                        file=sys.stderr,
                        flush=True,
                    )
                    write_jsonl_part(part_path, records)
                    lease.check()
                _elapsed = time.monotonic() - _t0
                _rate = n_records / _elapsed if _elapsed > 0 else 0.0
                print(
                    f"{_prefix} task {short_id}: JSON part committed"
                    f"  {n_records} records  {_elapsed:.1f}s  {_rate:.1f}/s",
                    file=sys.stderr,
                    flush=True,
                )
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
                print(
                    f"{_prefix} task {short_id}: SQLite task committed",
                    file=sys.stderr,
                    flush=True,
                )
            except LeaseLostError:
                print(f"{_prefix} task {short_id}: lease lost", file=sys.stderr, flush=True)
                continue
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"{_prefix} task {short_id}: failed — {detail}", file=sys.stderr, flush=True)
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


__all__ = [
    "main_worker_entry",
    "process_main_payload",
    "process_teacher_payload",
    "teacher_worker_entry",
]
