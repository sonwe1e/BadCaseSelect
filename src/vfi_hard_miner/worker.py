"""Worker-local model execution and per-triplet main-stage evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
import os
from queue import Full, Queue
import socket
from threading import Event, Thread
import traceback
from typing import Any

import numpy as np
import torch

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
from .image_io import read_rgb01
from .manifest import write_jsonl_part
from .model_adapter import ModelAdapter, ModelOutputs
from .pipeline import run_directory
from .reconstruction import ReconstructionResult, reconstruct_midpoint
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


def _load_cached(cache: ImageCache, path: str, *, max_items: int) -> np.ndarray:
    existing = cache.pop(path, None)
    if existing is not None:
        cache[path] = existing
        return existing
    image = read_rgb01(path)
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


def _prefetched_decode_batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    prefetch: int,
    max_cache: int,
) -> Iterator[DecodeEvent]:
    """Decode/group triplets on one bounded producer thread.

    The producer only performs CPU file I/O and shape grouping.  Model calls
    remain on the worker's main thread, so the producer never touches a CANN,
    CUDA, or NPU context.
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
        pending: list[DecodedItem] = []
        pending_shape: tuple[int, int, int] | None = None

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
                    first = _load_cached(
                        cache,
                        str(record["img0"]["path"]),
                        max_items=max_cache,
                    )
                    middle = _load_cached(
                        cache,
                        str(record["gt"]["path"]),
                        max_items=max_cache,
                    )
                    last = _load_cached(
                        cache,
                        str(record["img1"]["path"]),
                        max_items=max_cache,
                    )
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


def _reconstruct_outputs(
    img0_tensor: torch.Tensor,
    img1_tensor: torch.Tensor,
    outputs: ModelOutputs,
    *,
    model_config: Any,
) -> ReconstructionResult:
    return reconstruct_midpoint(
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
    )


def _finish_main_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    img0_tensor: torch.Tensor,
    img1_tensor: torch.Tensor,
    outputs: ModelOutputs,
    *,
    config: AppConfig,
) -> list[dict[str, Any]]:
    reconstructed = _reconstruct_outputs(
        img0_tensor, img1_tensor, outputs, model_config=config.model
    )
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
) -> list[dict[str, Any]]:
    if not items:
        return []
    img0_tensor, img1_tensor, outputs = _infer_output_batch(
        items, adapter=adapter, production_batch=config.model.batch_size
    )
    return _finish_main_batch(
        items,
        img0_tensor,
        img1_tensor,
        outputs,
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
) -> list[dict[str, Any]]:
    _validate_payload_identity(payload, config, stage="main")
    records = payload.get("triplets")
    if not isinstance(records, list):
        raise TypeError("main task payload must contain a triplets array")
    max_cache = max(8, config.model.batch_size * 2 + 3)
    output: list[dict[str, Any]] = []
    max_pending = max(1, config.runtime.prefetch)
    pending: list[Future[list[dict[str, Any]]]] = []

    def drain_one() -> None:
        output.extend(pending.pop(0).result())
        if heartbeat is not None:
            heartbeat()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vfi-main-cpu") as executor:
        for kind, value in _prefetched_decode_batches(
            records,
            batch_size=config.model.batch_size,
            prefetch=config.runtime.prefetch,
            max_cache=max_cache,
        ):
            if kind == "batch":
                img0_tensor, img1_tensor, outputs = _infer_output_batch(
                    value,
                    adapter=adapter,
                    production_batch=config.model.batch_size,
                )
                pending.append(
                    executor.submit(
                        _finish_main_batch,
                        value,
                        img0_tensor,
                        img1_tensor,
                        outputs,
                        config=config,
                    )
                )
                # Keep at least one CPU job in flight while the main thread
                # launches the following accelerator batch.
                while len(pending) > max_pending:
                    drain_one()
            elif kind == "invalid":
                # Preserve source order across a decode barrier.
                while pending:
                    drain_one()
                record, error = value
                output.append(_invalid_record(record, error))
            else:  # pragma: no cover - producer owns this internal protocol
                raise RuntimeError(f"unexpected decode event {kind!r}")
            if heartbeat is not None:
                heartbeat()
        while pending:
            drain_one()
    return output


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
    owner = f"{socket.gethostname()}:{os.getpid()}:{worker_index}:{device}"
    state_path = __import__("vfi_hard_miner.pipeline", fromlist=["run_state_path"]).run_state_path(
        config, stage="main"
    )
    parts_dir = run_directory(config) / "main_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with TaskStore(state_path) as store:
        while task := store.claim(owner, lease_seconds=config.runtime.lease_seconds):
            part_path = parts_dir / (
                f"{task.task_id.rsplit(':', 1)[-1]}.attempt-{task.attempt}.jsonl"
            )

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
                    )
                    write_jsonl_part(part_path, records)
                    lease.check()
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
            except LeaseLostError:
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
) -> list[dict[str, Any]]:
    if not items or config.teacher is None:
        return []
    img0_tensor, img1_tensor, outputs = _infer_output_batch(
        items,
        adapter=adapter,
        production_batch=config.teacher.batch_size,
    )
    return _finish_teacher_batch(
        items,
        img0_tensor,
        img1_tensor,
        outputs,
        config=config,
    )


def _finish_teacher_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    img0_tensor: torch.Tensor,
    img1_tensor: torch.Tensor,
    outputs: ModelOutputs,
    *,
    config: AppConfig,
) -> list[dict[str, Any]]:
    if config.teacher is None:
        raise RuntimeError("teacher postprocess requires config.teacher")
    reconstructed = _reconstruct_outputs(
        img0_tensor,
        img1_tensor,
        outputs,
        model_config=config.teacher,
    )
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
) -> list[dict[str, Any]]:
    if config.teacher is None:
        raise RuntimeError("teacher stage requires config.teacher")
    _validate_payload_identity(payload, config, stage="teacher")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("teacher task payload must contain a records array")
    max_cache = max(8, config.teacher.batch_size * 2 + 3)
    output: list[dict[str, Any]] = []
    max_pending = max(1, config.runtime.prefetch)
    pending: list[Future[list[dict[str, Any]]]] = []

    def drain_one() -> None:
        output.extend(pending.pop(0).result())
        if heartbeat is not None:
            heartbeat()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vfi-teacher-cpu") as executor:
        for kind, value in _prefetched_decode_batches(
            records,
            batch_size=config.teacher.batch_size,
            prefetch=config.runtime.prefetch,
            max_cache=max_cache,
        ):
            if kind == "batch":
                img0_tensor, img1_tensor, outputs = _infer_output_batch(
                    value,
                    adapter=adapter,
                    production_batch=config.teacher.batch_size,
                )
                pending.append(
                    executor.submit(
                        _finish_teacher_batch,
                        value,
                        img0_tensor,
                        img1_tensor,
                        outputs,
                        config=config,
                    )
                )
                while len(pending) > max_pending:
                    drain_one()
            elif kind == "invalid":
                while pending:
                    drain_one()
                record, error = value
                failed = dict(record)
                failed["status"] = "review"
                failed["teacher"] = {"error": str(error)}
                output.append(failed)
            else:  # pragma: no cover - producer owns this internal protocol
                raise RuntimeError(f"unexpected decode event {kind!r}")
            if heartbeat is not None:
                heartbeat()
        while pending:
            drain_one()
    return output


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
    owner = f"{socket.gethostname()}:{os.getpid()}:{worker_index}:{device}:teacher"
    state_path = __import__("vfi_hard_miner.pipeline", fromlist=["run_state_path"]).run_state_path(
        config, stage="teacher"
    )
    parts_dir = run_directory(config) / "teacher_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with TaskStore(state_path) as store:
        while task := store.claim(owner, lease_seconds=config.runtime.lease_seconds):
            part_path = parts_dir / (
                f"{task.task_id.rsplit(':', 1)[-1]}.attempt-{task.attempt}.jsonl"
            )

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
                    )
                    write_jsonl_part(part_path, records)
                    lease.check()
                store.complete(
                    task.task_id,
                    owner,
                    result_path=part_path,
                    attempt=task.attempt,
                )
            except LeaseLostError:
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


__all__ = [
    "main_worker_entry",
    "process_main_payload",
    "process_teacher_payload",
    "teacher_worker_entry",
]
