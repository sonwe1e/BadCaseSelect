"""Worker-local model execution and per-triplet main-stage evaluation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait as wait_futures,
)
from dataclasses import dataclass, replace as _replace_dc
import os
from queue import Full, Queue
import socket
from threading import Event, Lock, Thread
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
    MotionEvidence,
    ScopeMetrics,
    compute_motion_evidence,
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
    RECONSTRUCTION_CHANNELS,
    TIER1_CHANNELS,
    TIER2_CHANNELS,
    ReconstructionResult,
    Tier2Residue,
    merge_tier2,
    pack_prediction_to_cpu,
    pack_reconstruction_to_cpu,
    pack_tier1_to_cpu,
    reconstruct_midpoint,
    slice_reconstruction_cpu,
)
from .scoring import (
    ImageBasis,
    build_image_basis,
    complete_score_metrics,
    luminance,
    score_local_errors,
    score_region,
)
from .state import LeaseHeartbeat, LeaseLostError, TaskStore


ImageCache = OrderedDict[str, np.ndarray]
DecodedItem = tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]
ScoringItem = tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]
DecodeEvent = tuple[str, Any]


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    items: tuple[DecodedItem, ...]
    decode_seconds: float
    uint8_bytes: int

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int | slice) -> DecodedItem | tuple[DecodedItem, ...]:
        return self.items[index]


@dataclass(frozen=True, slots=True)
class InferenceBatch:
    outputs: ModelOutputs
    input_bytes: int
    output_bytes: int

    @property
    def network_bytes(self) -> int:
        return self.input_bytes + self.output_bytes


@dataclass(slots=True)
class PreparedMicrobatch:
    items: list[ScoringItem]
    img0_tensor: torch.Tensor
    img1_tensor: torch.Tensor
    float32_bytes: int
    stack_bytes: int


@dataclass
class CpuTimingTotals:
    decode_seconds: float = 0.0
    inference_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    future_wait_seconds: float = 0.0
    scoring_seconds: float = 0.0
    branch_evidence_seconds: float = 0.0
    motion_gates_seconds: float = 0.0
    serialization_seconds: float = 0.0
    decode_samples: int = 0
    inference_samples: int = 0
    reconstruction_samples: int = 0
    future_wait_samples: int = 0
    samples: int = 0

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._phase_seconds: dict[str, float] = {}
        self._phase_samples: dict[str, int] = {}

    def add(
        self,
        *,
        scoring: float,
        branch_evidence: float,
        motion_gates: float,
        serialization: float,
    ) -> None:
        with self._lock:
            self.scoring_seconds += float(scoring)
            self.branch_evidence_seconds += float(branch_evidence)
            self.motion_gates_seconds += float(motion_gates)
            self.serialization_seconds += float(serialization)
            self.samples += 1

    def add_decode(self, seconds: float, samples: int) -> None:
        with self._lock:
            self.decode_seconds += float(seconds)
            self.decode_samples += int(samples)

    def add_inference(self, seconds: float, samples: int) -> None:
        with self._lock:
            self.inference_seconds += float(seconds)
            self.inference_samples += int(samples)

    def add_reconstruction(self, seconds: float, samples: int) -> None:
        with self._lock:
            self.reconstruction_seconds += float(seconds)
            self.reconstruction_samples += int(samples)

    def add_future_wait(self, seconds: float, samples: int) -> None:
        with self._lock:
            self.future_wait_seconds += float(seconds)
            self.future_wait_samples += int(samples)

    def add_phases(
        self,
        phases: Mapping[str, float] | None,
        *,
        samples: int = 1,
    ) -> None:
        if not phases:
            return
        count = max(1, int(samples))
        with self._lock:
            for name, seconds in phases.items():
                self._phase_seconds[name] = self._phase_seconds.get(
                    name, 0.0
                ) + float(seconds)
                self._phase_samples[name] = self._phase_samples.get(
                    name, 0
                ) + count

    def milliseconds_per_sample(self) -> dict[str, float]:
        with self._lock:
            values = {
                "decode": (
                    self.decode_seconds,
                    self.decode_samples,
                ),
                "inference": (
                    self.inference_seconds,
                    self.inference_samples,
                ),
                "reconstruction": (
                    self.reconstruction_seconds,
                    self.reconstruction_samples,
                ),
                "future_wait": (
                    self.future_wait_seconds,
                    self.future_wait_samples,
                ),
                "scoring": (self.scoring_seconds, self.samples),
                "branch_evidence": (
                    self.branch_evidence_seconds,
                    self.samples,
                ),
                "motion_gates": (
                    self.motion_gates_seconds,
                    self.samples,
                ),
                "serialization": (
                    self.serialization_seconds,
                    self.samples,
                ),
            }
            values.update(
                {
                    name: (
                        seconds,
                        self._phase_samples.get(name, 0),
                    )
                    for name, seconds in self._phase_seconds.items()
                }
            )
        return {
            name: (seconds * 1000.0 / count if count > 0 else 0.0)
            for name, (seconds, count) in values.items()
        }


_TIMING_REPORT_SAMPLES = 16
_DETAILED_TIMING_FIELDS = (
    "validity_difference",
    "validity_histogram",
    "validity_quantile",
    "flow_oob",
    "flow_background_median",
    "flow_gradient",
    "flow_quantile",
    "flow_cpu_fallback",
    "error_map_generation",
    "candidate_quantile",
    "native_component_label",
    "integral_windows",
    "summary_metrics",
    "phase2_diagnosis",
)


def _print_timing_summary(
    prefix: str,
    timings: CpuTimingTotals,
    *,
    scored: int,
    final: bool,
) -> None:
    if not prefix:
        return
    values = timings.milliseconds_per_sample()
    label = "final" if final else "periodic"
    print(
        f"{prefix}  timing {label} scored={scored}"
        f"  decode_ms/sample={values['decode']:.1f}"
        f"  inference_ms/sample={values['inference']:.1f}"
        f"  reconstruction_ms/sample={values['reconstruction']:.1f}"
        f"  future_wait_ms/sample={values['future_wait']:.1f}"
        f"  scoring_ms/sample={values['scoring']:.1f}"
        f"  branch_evidence_ms/sample={values['branch_evidence']:.1f}"
        f"  motion_gates_ms/sample={values['motion_gates']:.1f}"
        f"  serialization_ms/sample={values['serialization']:.1f}"
        + "".join(
            f"  {name}_ms/sample={values.get(name, 0.0):.1f}"
            for name in _DETAILED_TIMING_FIELDS
        ),
        file=sys.stderr,
        flush=True,
    )


def _tensor_from_hwc(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array).permute(2, 0, 1)


def _validate_source_frame(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} spatial dimensions must be positive")
    if array.dtype == np.uint8:
        return array
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be uint8 or normalized floating point")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if array.size and (
        float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6
    ):
        raise ValueError(f"{name} floating values must be normalized to [0,1]")
    return array


def _decoded_batch_uint8_bytes(items: Sequence[DecodedItem]) -> int:
    """Count unique uint8 frame allocations referenced by a decoded batch."""

    seen: set[int] = set()
    total = 0
    for _, img0, gt, img1 in items:
        for image in (img0, gt, img1):
            identity = id(image)
            if identity in seen:
                continue
            seen.add(identity)
            total += int(image.nbytes)
    return total


def _network_frame_tensor(
    image: np.ndarray,
    *,
    network_size: tuple[int, int],
) -> torch.Tensor:
    """Convert one uint8 frame directly into one network-sized CHW tensor."""

    array = _validate_source_frame(image, name="network input")
    tensor = _tensor_from_hwc(array).to(dtype=torch.float32)
    if array.dtype == np.uint8:
        tensor.div_(255.0)
    if tuple(tensor.shape[-2:]) != tuple(network_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=network_size,
            mode="bilinear",
            align_corners=False,
        )[0]
    return tensor


def _network_input_batch(
    items: Sequence[DecodedItem],
    *,
    production_batch: int,
    network_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build only fixed-resolution model inputs, one source frame at a time."""

    if not items:
        raise ValueError("inference batch must not be empty")
    if production_batch < len(items):
        raise ValueError(
            f"production batch {production_batch} is smaller than "
            f"valid item count {len(items)}"
        )
    height, width = (int(network_size[0]), int(network_size[1]))
    if height < 1 or width < 1:
        raise ValueError(f"network_size must be positive, got {network_size}")
    img0_batch = torch.empty(
        (production_batch, 3, height, width),
        dtype=torch.float32,
    )
    img1_batch = torch.empty_like(img0_batch)
    for index, item in enumerate(items):
        img0 = _validate_source_frame(item[1], name=f"items[{index}].img0")
        img1 = _validate_source_frame(item[3], name=f"items[{index}].img1")
        if img0.shape != img1.shape:
            raise ValueError(
                f"items[{index}] endpoint shapes differ: "
                f"{img0.shape} and {img1.shape}"
            )
        img0_batch[index].copy_(
            _network_frame_tensor(img0, network_size=(height, width))
        )
        img1_batch[index].copy_(
            _network_frame_tensor(img1, network_size=(height, width))
        )
    valid_count = len(items)
    if valid_count < production_batch:
        img0_batch[valid_count:].copy_(img0_batch[valid_count - 1])
        img1_batch[valid_count:].copy_(img1_batch[valid_count - 1])
    return img0_batch, img1_batch


def _prepare_reconstruction_microbatch(
    items: Sequence[DecodedItem],
) -> PreparedMicrobatch:
    """Convert only the current reconstruction slice to native float32."""

    if not items:
        raise ValueError("reconstruction microbatch must not be empty")
    converted: dict[int, np.ndarray] = {}

    def as_float32(image: np.ndarray, *, name: str) -> np.ndarray:
        array = _validate_source_frame(image, name=name)
        identity = id(array)
        cached = converted.get(identity)
        if cached is not None:
            return cached
        value = (
            rgb_uint8_to_float32(array)
            if array.dtype == np.uint8
            else np.ascontiguousarray(array, dtype=np.float32)
        )
        converted[identity] = value
        return value

    scoring_items: list[ScoringItem] = []
    for index, (record, img0, gt, img1) in enumerate(items):
        first = as_float32(img0, name=f"items[{index}].img0")
        middle = as_float32(gt, name=f"items[{index}].gt")
        last = as_float32(img1, name=f"items[{index}].img1")
        if first.shape != middle.shape or first.shape != last.shape:
            raise ValueError(
                f"items[{index}] triplet shapes differ: "
                f"{first.shape}, {middle.shape}, {last.shape}"
            )
        scoring_items.append((record, first, middle, last))
    img0_tensor = torch.stack(
        [_tensor_from_hwc(item[1]) for item in scoring_items]
    )
    img1_tensor = torch.stack(
        [_tensor_from_hwc(item[3]) for item in scoring_items]
    )
    return PreparedMicrobatch(
        items=scoring_items,
        img0_tensor=img0_tensor,
        img1_tensor=img1_tensor,
        float32_bytes=sum(int(value.nbytes) for value in converted.values()),
        stack_bytes=(
            int(img0_tensor.numel() * img0_tensor.element_size())
            + int(img1_tensor.numel() * img1_tensor.element_size())
        ),
    )


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[0] in (1, 2, 3, 4):
        value = np.moveaxis(value, 0, -1)
    return np.asarray(value, dtype=np.float32)


def _motion_evidence_for_sample(
    reconstructed: ReconstructionResult,
    batch_index: int,
) -> MotionEvidence:
    """Use compact device-computed evidence, with a legacy CPU fallback."""

    if reconstructed.scope_metrics is not None:
        values = (
            reconstructed.scope_metrics[batch_index]
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .numpy()
        )
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("device scope metrics must contain six finite values")
        support: np.ndarray | None = None
        if reconstructed.flow_discontinuity_map is not None:
            raw_support = (
                reconstructed.flow_discontinuity_map[batch_index]
                .detach()
                .cpu()
                .numpy()
            )
            support = np.squeeze(raw_support).astype(np.float32, copy=False)
            if support.ndim != 2:
                raise ValueError(
                    "device flow discontinuity support must be a 2-D map"
                )
        return MotionEvidence(
            scope_metrics=ScopeMetrics(
                out_of_bounds_ratio=float(values[0]),
                flow_discontinuity_ratio=float(values[1]),
                foreground_large_motion_ratio=float(values[2]),
                occlusion_ratio=float(values[3]),
                unexplained_motion_ratio=float(values[4]),
                background_motion=float(values[5]),
            ),
            flow_discontinuity_map=support,
        )
    if reconstructed.flow_t0 is None or reconstructed.flow_t1 is None:
        raise RuntimeError(
            "reconstruction has neither compact scope evidence nor CPU flow"
        )
    return compute_motion_evidence(
        _hwc(reconstructed.flow_t0[batch_index]),
        _hwc(reconstructed.flow_t1[batch_index]),
    )


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
    timings: CpuTimingTotals | None = None,
    tier2_residue: Tier2Residue | None = None,
) -> dict[str, Any]:
    scoring_basis_started = time.perf_counter()
    gt_basis = build_image_basis(gt, name="gt")
    img0_luminance = luminance(img0)
    img1_luminance = luminance(img1)
    endpoint_change = np.asarray(
        np.abs(img0 - img1).mean(axis=-1),
        dtype=np.float32,
    )
    scoring_basis_elapsed = time.perf_counter() - scoring_basis_started

    motion_gates_started = time.perf_counter()
    motion_evidence = _motion_evidence_for_sample(
        reconstructed, batch_index
    )
    indices = tuple(int(value) for value in source["frame_indices"])
    stride = int(source["stride"])
    contiguous = indices == (indices[0], indices[0] + stride, indices[0] + 2 * stride)
    validity_metrics = compute_validity_metrics(
        img0,
        gt,
        img1,
        sequence_contiguous=contiguous,
        luminance_triplet=(
            img0_luminance,
            gt_basis.luminance,
            img1_luminance,
        ),
    )
    validity = evaluate_validity(validity_metrics, thresholds)
    scope_metrics = motion_evidence.scope_metrics
    scope = evaluate_in_scope(scope_metrics, thresholds)
    motion_gates_elapsed = time.perf_counter() - motion_gates_started

    scoring_started = time.perf_counter()
    prediction = _hwc(reconstructed.prediction[batch_index])
    scoring = score_local_errors(
        prediction,
        gt,
        thresholds,
        img0=img0,
        img1=img1,
        reference_basis=gt_basis,
        endpoint_change_map=endpoint_change,
    )
    scoring_elapsed = (
        scoring_basis_elapsed + time.perf_counter() - scoring_started
    )

    fast_reject_reason: str | None = None
    if validity.label == "reject":
        fast_reject_reason = "validity_reject"
    elif scope.label == "reject":
        fast_reject_reason = "scope_reject"
    elif scoring.mining_p_wrong < float(thresholds.wrong_reject_below):
        fast_reject_reason = "prediction_not_wrong"

    branch_started = time.perf_counter()
    if fast_reject_reason is None:
        if tier2_residue is not None:
            # Candidate: transfer the held-back warps + masks (11ch) now.
            # Rejected samples never reach this line, so they never pay the
            # tier-2 D2H.  Candidates within one microbatch share the cached
            # materialization (single packed transfer).
            reconstructed = merge_tier2(
                reconstructed, tier2_residue.materialize()
            )
        diagnosis = diagnose_sample(
            prediction,
            gt,
            warp0=_hwc(reconstructed.warp0[batch_index]),
            warp1=_hwc(reconstructed.warp1[batch_index]),
            warp_blend=_hwc(reconstructed.warp_blend[batch_index]),
            img0=img0,
            img1=img1,
            flow_discontinuity_map=motion_evidence.flow_discontinuity_map,
            mask0=_hwc(reconstructed.mask0[batch_index]),
            mask1=_hwc(reconstructed.mask1[batch_index]),
            regions=scoring.regions,
            scoring_config=thresholds,
            scoring_result=scoring,
            config=thresholds,
        )
        p_wrong = float(diagnosis.p_wrong)
        mining_p_wrong = float(diagnosis.mining_p_wrong)
        p_solvable = float(diagnosis.p_solvable)
        diagnosis_reasons = diagnosis.reasons
        diagnosis_regions = [region.to_dict() for region in diagnosis.regions]
        diagnosis_metrics: dict[str, Any] = dict(diagnosis.metrics)
        primary_region_index = diagnosis.primary_region_index
    else:
        p_wrong = float(scoring.p_wrong)
        mining_p_wrong = float(scoring.mining_p_wrong)
        p_solvable = 0.0
        diagnosis_reasons = ()
        diagnosis_regions = []
        primary_region_index = None
        priority_weight = float(
            np.clip(
                mining_p_wrong / max(p_wrong, 1e-8)
                if p_wrong > 0.0
                else 1.0,
                0.0,
                1.0,
            )
        )
        diagnosis_metrics = {
            "candidate_region_count": 0.0,
            "scoring_p_wrong": p_wrong,
            "scoring_mining_p_wrong": mining_p_wrong,
            "selected_p_wrong": p_wrong,
            "selected_mining_p_wrong": mining_p_wrong,
            "selected_priority_weight": priority_weight,
            "selected_ui_likelihood": 0.0,
            "selected_p_solvable": 0.0,
            "skipped": 1.0,
            "skip_reason": fast_reject_reason,
        }
    branch_elapsed = time.perf_counter() - branch_started

    decision = decide_hard_case(
        validity,
        scope,
        mining_p_wrong,
        p_solvable,
        thresholds,
    )

    serialization_started = time.perf_counter()
    if validity.label == "reject":
        status = "invalid"
    elif scope.label == "reject":
        status = "out_of_scope"
    else:
        status = decision.label
    reasons = list(
        dict.fromkeys(
            (
                *diagnosis_reasons,
                *validity.reasons,
                *scope.reasons,
                *decision.reasons,
            )
        )
    )
    result = {
        **dict(source),
        "status": status,
        "validity_label": validity.label,
        "in_scope_label": scope.label,
        "valid": _label_value(validity.label),
        "in_scope": _label_value(scope.label),
        "p_wrong": p_wrong,
        "mining_p_wrong": mining_p_wrong,
        "p_solvable": p_solvable,
        "reasons": reasons,
        "regions": diagnosis_regions,
        "metrics": {
            "scoring": scoring.metrics,
            "diagnosis": diagnosis_metrics,
            "validity": validity.metrics,
            "scope": scope.metrics,
            "decision": decision.metrics,
        },
        "primary_region_index": primary_region_index,
        "error": None,
    }
    serialization_elapsed = time.perf_counter() - serialization_started
    if timings is not None:
        timings.add(
            scoring=scoring_elapsed,
            branch_evidence=branch_elapsed,
            motion_gates=motion_gates_elapsed,
            serialization=serialization_elapsed,
        )
    return result


@dataclass(frozen=True, slots=True)
class Phase1Reject:
    """A fast-rejected sample: its final record needs no tier-2 channels."""

    sample_index: int
    record: dict[str, Any]


class _BatchImageBasisCache:
    """Tiny rolling cache for overlapping stride-1 triplets in one Future."""

    def __init__(self, max_items: int = 3) -> None:
        self._max_items = max(1, int(max_items))
        self._items: OrderedDict[str, ImageBasis] = OrderedDict()

    def get(
        self,
        key: str,
        image: np.ndarray,
        *,
        name: str,
    ) -> ImageBasis:
        existing = self._items.pop(key, None)
        if existing is not None:
            self._items[key] = existing
            return existing
        basis = build_image_basis(image, name=name)
        self._items[key] = basis
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)
        return basis


@dataclass(frozen=True, slots=True)
class Phase1Candidate:
    """Live tier-1 intermediates for a sample that earned a full diagnosis.

    Phase 2 consumes these on the CPU after the worker's main thread has
    transferred the sample's tier-2 payload; pool threads never touch the
    device-held residue themselves.
    """

    sample_index: int
    source: Mapping[str, Any]
    img0: np.ndarray
    gt: np.ndarray
    img1: np.ndarray
    img1_basis: ImageBasis
    prediction: np.ndarray
    scoring: Any
    motion_evidence: Any
    validity: Any
    scope: Any
    scoring_elapsed: float
    motion_gates_elapsed: float


@dataclass(frozen=True, slots=True)
class Phase1BatchResult:
    """Per-sample phase-1 outcomes in the microbatch's original order."""

    outcomes: list[Phase1Reject | Phase1Candidate]


def _score_sample_phase1(
    source: Mapping[str, Any],
    *,
    img0: np.ndarray,
    gt: np.ndarray,
    img1: np.ndarray,
    reconstructed: ReconstructionResult,
    batch_index: int,
    thresholds: Any,
    timings: CpuTimingTotals | None = None,
    basis_cache: _BatchImageBasisCache | None = None,
) -> Phase1Reject | Phase1Candidate:
    """Tier-1 scoring for one sample: pure numpy / CPU-torch work.

    Runs in a postprocess pool thread on tier-1 data only (prediction +
    compact scope evidence, already on the CPU). Rejected samples return finished
    records bitwise identical to the legacy single-phase path; candidates
    carry their intermediates into phase 2.
    """

    phase_timings: dict[str, float] = {}
    scoring_basis_started = time.perf_counter()
    if basis_cache is None:
        gt_basis = build_image_basis(gt, name="gt")
        img0_luminance = luminance(img0)
        img1_luminance = luminance(img1)
        img1_basis = build_image_basis(img1, name="img1")
    else:
        def basis_key(field: str, image: np.ndarray) -> str:
            value = source.get(field, {})
            if isinstance(value, Mapping) and value.get("path"):
                return str(value["path"])
            return f"{field}:{id(image)}"

        img0_basis = basis_cache.get(
            basis_key("img0", img0),
            img0,
            name="img0",
        )
        gt_basis = basis_cache.get(
            basis_key("gt", gt),
            gt,
            name="gt",
        )
        img1_basis = basis_cache.get(
            basis_key("img1", img1),
            img1,
            name="img1",
        )
        img0_luminance = img0_basis.luminance
        img1_luminance = img1_basis.luminance
    endpoint_change = np.asarray(
        np.abs(img0 - img1).mean(axis=-1),
        dtype=np.float32,
    )
    scoring_basis_elapsed = time.perf_counter() - scoring_basis_started

    motion_gates_started = time.perf_counter()
    motion_evidence = _motion_evidence_for_sample(
        reconstructed, batch_index
    )
    indices = tuple(int(value) for value in source["frame_indices"])
    stride = int(source["stride"])
    contiguous = indices == (indices[0], indices[0] + stride, indices[0] + 2 * stride)
    validity_metrics = compute_validity_metrics(
        img0,
        gt,
        img1,
        sequence_contiguous=contiguous,
        luminance_triplet=(
            img0_luminance,
            gt_basis.luminance,
            img1_luminance,
        ),
        timings=phase_timings,
    )
    validity = evaluate_validity(validity_metrics, thresholds)
    scope_metrics = motion_evidence.scope_metrics
    scope = evaluate_in_scope(scope_metrics, thresholds)
    motion_gates_elapsed = time.perf_counter() - motion_gates_started

    scoring_started = time.perf_counter()
    prediction = _hwc(reconstructed.prediction[batch_index])
    scoring = score_local_errors(
        prediction,
        gt,
        thresholds,
        img0=img0,
        img1=img1,
        reference_basis=gt_basis,
        endpoint_change_map=endpoint_change,
        detailed_metrics=False,
        timings=phase_timings,
    )
    scoring_elapsed = (
        scoring_basis_elapsed + time.perf_counter() - scoring_started
    )

    fast_reject_reason: str | None = None
    if validity.label == "reject":
        fast_reject_reason = "validity_reject"
    elif scope.label == "reject":
        fast_reject_reason = "scope_reject"
    elif scoring.mining_p_wrong < float(thresholds.wrong_reject_below):
        fast_reject_reason = "prediction_not_wrong"

    if fast_reject_reason is None:
        if timings is not None:
            timings.add_phases(phase_timings)
        return Phase1Candidate(
            sample_index=batch_index,
            source=source,
            img0=img0,
            gt=gt,
            img1=img1,
            img1_basis=img1_basis,
            prediction=prediction,
            scoring=scoring,
            motion_evidence=motion_evidence,
            validity=validity,
            scope=scope,
            scoring_elapsed=scoring_elapsed,
            motion_gates_elapsed=motion_gates_elapsed,
        )

    branch_started = time.perf_counter()
    p_wrong = float(scoring.p_wrong)
    mining_p_wrong = float(scoring.mining_p_wrong)
    p_solvable = 0.0
    diagnosis_reasons = ()
    diagnosis_regions = []
    primary_region_index = None
    priority_weight = float(
        np.clip(
            mining_p_wrong / max(p_wrong, 1e-8)
            if p_wrong > 0.0
            else 1.0,
            0.0,
            1.0,
        )
    )
    diagnosis_metrics = {
        "candidate_region_count": 0.0,
        "scoring_p_wrong": p_wrong,
        "scoring_mining_p_wrong": mining_p_wrong,
        "selected_p_wrong": p_wrong,
        "selected_mining_p_wrong": mining_p_wrong,
        "selected_priority_weight": priority_weight,
        "selected_ui_likelihood": 0.0,
        "selected_p_solvable": 0.0,
        "skipped": 1.0,
        "skip_reason": fast_reject_reason,
    }
    branch_elapsed = time.perf_counter() - branch_started

    decision = decide_hard_case(
        validity,
        scope,
        mining_p_wrong,
        p_solvable,
        thresholds,
    )

    serialization_started = time.perf_counter()
    if validity.label == "reject":
        status = "invalid"
    elif scope.label == "reject":
        status = "out_of_scope"
    else:
        status = decision.label
    reasons = list(
        dict.fromkeys(
            (
                *diagnosis_reasons,
                *validity.reasons,
                *scope.reasons,
                *decision.reasons,
            )
        )
    )
    result = {
        **dict(source),
        "status": status,
        "validity_label": validity.label,
        "in_scope_label": scope.label,
        "valid": _label_value(validity.label),
        "in_scope": _label_value(scope.label),
        "p_wrong": p_wrong,
        "mining_p_wrong": mining_p_wrong,
        "p_solvable": p_solvable,
        "reasons": reasons,
        "regions": diagnosis_regions,
        "metrics": {
            "scoring": scoring.metrics,
            "diagnosis": diagnosis_metrics,
            "validity": validity.metrics,
            "scope": scope.metrics,
            "decision": decision.metrics,
        },
        "primary_region_index": primary_region_index,
        "error": None,
    }
    serialization_elapsed = time.perf_counter() - serialization_started
    if timings is not None:
        timings.add_phases(phase_timings)
        timings.add(
            scoring=scoring_elapsed,
            branch_evidence=branch_elapsed,
            motion_gates=motion_gates_elapsed,
            serialization=serialization_elapsed,
        )
    return Phase1Reject(sample_index=batch_index, record=result)


def _diagnose_sample_phase2(
    candidate: Phase1Candidate,
    *,
    reconstructed: ReconstructionResult,
    thresholds: Any,
    timings: CpuTimingTotals | None = None,
) -> dict[str, Any]:
    """Candidate diagnosis for one sample: pure CPU work.

    Runs in a postprocess pool thread on the merged reconstruction; every
    tensor here is CPU-resident (the tier-2 D2H already happened on the
    worker's main thread).
    """

    index = candidate.sample_index
    phase_timings: dict[str, float] = {}
    complete_score_metrics(
        candidate.scoring,
        thresholds,
        timings=phase_timings,
    )
    branch_started = time.perf_counter()
    diagnosis = diagnose_sample(
        candidate.prediction,
        candidate.gt,
        warp0=_hwc(reconstructed.warp0[index]),
        warp1=_hwc(reconstructed.warp1[index]),
        warp_blend=_hwc(reconstructed.warp_blend[index]),
        img0=candidate.img0,
        img1=candidate.img1,
        flow_discontinuity_map=candidate.motion_evidence.flow_discontinuity_map,
        mask0=_hwc(reconstructed.mask0[index]),
        mask1=_hwc(reconstructed.mask1[index]),
        regions=candidate.scoring.regions,
        scoring_config=thresholds,
        scoring_result=candidate.scoring,
        img1_basis=candidate.img1_basis,
        config=thresholds,
    )
    p_wrong = float(diagnosis.p_wrong)
    mining_p_wrong = float(diagnosis.mining_p_wrong)
    p_solvable = float(diagnosis.p_solvable)
    diagnosis_reasons = diagnosis.reasons
    diagnosis_regions = [region.to_dict() for region in diagnosis.regions]
    diagnosis_metrics: dict[str, Any] = dict(diagnosis.metrics)
    primary_region_index = diagnosis.primary_region_index
    branch_elapsed = time.perf_counter() - branch_started
    phase_timings["phase2_diagnosis"] = (
        phase_timings.get("phase2_diagnosis", 0.0)
        + branch_elapsed
    )

    decision = decide_hard_case(
        candidate.validity,
        candidate.scope,
        mining_p_wrong,
        p_solvable,
        thresholds,
    )

    serialization_started = time.perf_counter()
    if candidate.validity.label == "reject":
        status = "invalid"
    elif candidate.scope.label == "reject":
        status = "out_of_scope"
    else:
        status = decision.label
    reasons = list(
        dict.fromkeys(
            (
                *diagnosis_reasons,
                *candidate.validity.reasons,
                *candidate.scope.reasons,
                *decision.reasons,
            )
        )
    )
    result = {
        **dict(candidate.source),
        "status": status,
        "validity_label": candidate.validity.label,
        "in_scope_label": candidate.scope.label,
        "valid": _label_value(candidate.validity.label),
        "in_scope": _label_value(candidate.scope.label),
        "p_wrong": p_wrong,
        "mining_p_wrong": mining_p_wrong,
        "p_solvable": p_solvable,
        "reasons": reasons,
        "regions": diagnosis_regions,
        "metrics": {
            "scoring": candidate.scoring.metrics,
            "diagnosis": diagnosis_metrics,
            "validity": candidate.validity.metrics,
            "scope": candidate.scope.metrics,
            "decision": decision.metrics,
        },
        "primary_region_index": primary_region_index,
        "error": None,
    }
    serialization_elapsed = time.perf_counter() - serialization_started
    if timings is not None:
        timings.add_phases(phase_timings)
        timings.add(
            scoring=candidate.scoring_elapsed,
            branch_evidence=branch_elapsed,
            motion_gates=candidate.motion_gates_elapsed,
            serialization=serialization_elapsed,
        )
    return result


def _finish_phase1_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
    timings: CpuTimingTotals | None = None,
) -> Phase1BatchResult:
    """Pool-thread entry: tier-1 fast-reject scoring for a whole microbatch."""

    outcomes: list[Phase1Reject | Phase1Candidate] = []
    basis_cache = _BatchImageBasisCache(max_items=3)
    for index, item in enumerate(items):
        outcomes.append(
            _score_sample_phase1(
                item[0],
                img0=item[1],
                gt=item[2],
                img1=item[3],
                reconstructed=reconstructed,
                batch_index=index,
                thresholds=config.thresholds,
                timings=timings,
                basis_cache=basis_cache,
            )
        )
    return Phase1BatchResult(outcomes=outcomes)


def _finish_phase2_batch(
    candidates: Sequence[Phase1Candidate],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
    timings: CpuTimingTotals | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Pool-thread entry: diagnose candidates on the merged CPU result.

    Returns ``(sample_index, record)`` pairs so the main thread can restore
    the original sample order without touching record contents.
    """

    return [
        (
            candidate.sample_index,
            _diagnose_sample_phase2(
                candidate,
                reconstructed=reconstructed,
                thresholds=config.thresholds,
                timings=timings,
            ),
        )
        for candidate in candidates
    ]


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

    Automatic mode is deliberately memory-bandwidth oriented.  Full-resolution
    NumPy/SciPy diagnostics gain little from an unbounded number of concurrent
    scans, so each accelerator worker gets at most two scoring Futures.
    """

    configured = int(config.runtime.postproc_workers)
    if configured > 0:
        return configured
    cpu_budget = max(1, int(config.runtime.cpu_threads_per_worker))
    logical_cpus = max(1, int(os.cpu_count() or 1))
    configured_workers = max(1, int(config.runtime.workers))
    fair_share = max(1, logical_cpus // configured_workers)
    bandwidth_share = max(1, fair_share // 8)
    return min(2, cpu_budget, bandwidth_share)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    decode_uint8_bytes: int = 0
    network_bytes: int = 0
    reconstruction_transient_bytes: int = 0

    def resident_bytes(
        self,
        pending_reserved_bytes: int,
        pending_device_retained_bytes: int = 0,
    ) -> int:
        return (
            max(0, int(self.decode_uint8_bytes))
            + max(0, int(self.network_bytes))
            + max(0, int(self.reconstruction_transient_bytes))
            + max(0, int(pending_reserved_bytes))
            + max(0, int(pending_device_retained_bytes))
        )


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
        self._resolved_postproc_workers = 0
        self._resolved_microbatch_size: int | str = "auto_pending"
        self._postproc_buffer_mb = 0
        self._candidate_ratio = 0.0
        self._phase1_queue_depth = 0
        self._phase2_queue_depth = 0
        print(
            f"{self._prefix}  inferred 0/{self._total}"
            f"  scored 0/{self._total}  starting",
            file=sys.stderr,
            flush=True,
        )

    def configure(
        self,
        *,
        resolved_postproc_workers: int,
        resolved_microbatch_size: int | str,
        postproc_buffer_mb: int,
    ) -> None:
        self._resolved_postproc_workers = int(resolved_postproc_workers)
        self._resolved_microbatch_size = resolved_microbatch_size
        self._postproc_buffer_mb = int(postproc_buffer_mb)
        print(
            f"{self._prefix}  pipeline_config"
            f"  resolved_postproc_workers "
            f"{self._resolved_postproc_workers}"
            f"  resolved_microbatch_size "
            f"{self._resolved_microbatch_size}"
            f"  postproc_buffer_mb {self._postproc_buffer_mb}"
            f"  candidate_ratio {self._candidate_ratio:.4f}"
            f"  actual_tier2_d2h_bytes 0"
            f"  phase1_queue_depth 0"
            f"  phase2_queue_depth 0",
            file=sys.stderr,
            flush=True,
        )

    def set_pipeline_state(
        self,
        *,
        candidate_ratio: float,
        phase1_queue_depth: int,
        phase2_queue_depth: int,
        resolved_microbatch_size: int | None = None,
    ) -> None:
        self._candidate_ratio = float(np.clip(candidate_ratio, 0.0, 1.0))
        self._phase1_queue_depth = max(0, int(phase1_queue_depth))
        self._phase2_queue_depth = max(0, int(phase2_queue_depth))
        if resolved_microbatch_size is not None:
            first_resolution = self._resolved_microbatch_size == "auto_pending"
            self._resolved_microbatch_size = int(resolved_microbatch_size)
            if first_resolution:
                print(
                    f"{self._prefix}  pipeline_resolved"
                    f"  resolved_microbatch_size "
                    f"{self._resolved_microbatch_size}",
                    file=sys.stderr,
                    flush=True,
                )

    def update_inferred(
        self,
        n: int,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        self._inferred += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            pending_retained_bytes=pending_retained_bytes,
            pending_device_retained_bytes=pending_device_retained_bytes,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory,
        )

    def update_scored(
        self,
        n: int,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        self._scored += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            pending_retained_bytes=pending_retained_bytes,
            pending_device_retained_bytes=pending_device_retained_bytes,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory,
        )

    def update_invalid(
        self,
        n: int,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        self._inferred += n
        self._scored += n
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            pending_retained_bytes=pending_retained_bytes,
            pending_device_retained_bytes=pending_device_retained_bytes,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory,
        )

    def waiting(
        self,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        self._maybe_emit(
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            pending_retained_bytes=pending_retained_bytes,
            pending_device_retained_bytes=pending_device_retained_bytes,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory,
        )

    def _maybe_emit(
        self,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        now = time.monotonic()
        if now - self._last >= self._INTERVAL:
            self._emit(
                now,
                pending_batches=pending_batches,
                pending_bytes=pending_bytes,
                pending_retained_bytes=pending_retained_bytes,
                pending_device_retained_bytes=pending_device_retained_bytes,
                tier2_d2h_bytes=tier2_d2h_bytes,
                tier2_materialized_batches=tier2_materialized_batches,
                tier2_candidate_samples=tier2_candidate_samples,
                memory=memory,
            )
            self._last = now

    def close(
        self,
        *,
        pending_batches: int = 0,
        pending_bytes: int = 0,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
    ) -> None:
        self._emit(
            time.monotonic(),
            pending_batches=pending_batches,
            pending_bytes=pending_bytes,
            pending_retained_bytes=pending_retained_bytes,
            pending_device_retained_bytes=pending_device_retained_bytes,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory,
            final=True,
        )

    def _emit(
        self,
        now: float,
        *,
        pending_batches: int,
        pending_bytes: int,
        pending_retained_bytes: int | None = None,
        pending_device_retained_bytes: int = 0,
        tier2_d2h_bytes: int = 0,
        tier2_materialized_batches: int = 0,
        tier2_candidate_samples: int = 0,
        memory: MemoryEstimate | None = None,
        final: bool = False,
    ) -> None:
        elapsed = now - self._start
        rate = self._scored / elapsed if elapsed > 0 else 0.0
        label = "done" if final else "..."
        retained_bytes = (
            pending_bytes
            if pending_retained_bytes is None
            else pending_retained_bytes
        )
        resolved_memory = memory or MemoryEstimate()
        resident_estimate = resolved_memory.resident_bytes(
            pending_bytes, pending_device_retained_bytes
        )
        print(
            f"{self._prefix}  inferred {self._inferred}/{self._total}"
            f"  scored {self._scored}/{self._total}"
            f"  pending {pending_batches} batches"
            f"  retained {retained_bytes / (1024 * 1024):.0f} MiB"
            f"  reserved {pending_bytes / (1024 * 1024):.0f} MiB"
            f"  device_retained "
            f"{pending_device_retained_bytes / (1024 * 1024):.0f} MiB"
            f"  tier2_d2h {tier2_d2h_bytes / (1024 * 1024):.0f} MiB"
            f"  actual_tier2_d2h_bytes {tier2_d2h_bytes}"
            f"  tier2_materialized_batches {tier2_materialized_batches}"
            f"  tier2_candidate_samples {tier2_candidate_samples}"
            f"  candidate_ratio {self._candidate_ratio:.4f}"
            f"  resolved_postproc_workers "
            f"{self._resolved_postproc_workers}"
            f"  resolved_microbatch_size "
            f"{self._resolved_microbatch_size}"
            f"  postproc_buffer_mb {self._postproc_buffer_mb}"
            f"  phase1_queue_depth {self._phase1_queue_depth}"
            f"  phase2_queue_depth {self._phase2_queue_depth}"
            f"  decode_uint8 "
            f"{resolved_memory.decode_uint8_bytes / (1024 * 1024):.0f} MiB"
            f"  network {resolved_memory.network_bytes / (1024 * 1024):.0f} MiB"
            f"  reconstruction_transient "
            f"{resolved_memory.reconstruction_transient_bytes / (1024 * 1024):.0f} MiB"
            f"  resident_estimate {resident_estimate / (1024 * 1024):.0f} MiB"
            f"  {elapsed:.0f}s  {rate:.2f}/s  {label}",
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
    decode_workers: int = 1,
) -> Iterator[DecodeEvent]:
    """Decode/group triplets on one bounded producer thread.

    The producer only performs CPU file I/O and shape grouping.  Model calls
    remain on the worker's main thread, so the producer never touches a CANN,
    CUDA, or NPU context.

    Frames and queued batches remain uint8.  Native-resolution float32
    conversion happens only for the reconstruction microbatch that has already
    passed memory admission.  ``cache_budget_bytes`` caps the uint8 LRU once
    the first frame's size is known.

    ``decode_workers > 1`` fans cache-MISS PNG decodes out to a thread pool
    inside the producer (zlib releases the GIL, so this is true parallelism);
    LRU hits, cache writes, shape grouping, and the ("invalid", ...) event
    protocol stay serial, so the event stream is identical to the
    single-threaded path.
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
        pending_decode_seconds = 0.0
        capacity = max(1, int(max_cache))
        budget_resolved = cache_budget_bytes is None
        workers = max(1, int(decode_workers))
        pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="vfi-decode-io"
            )
            if workers > 1
            else None
        )

        def resolve_budget(image: np.ndarray) -> None:
            nonlocal capacity, budget_resolved
            if not budget_resolved:
                budget_resolved = True
                capacity = max(
                    1,
                    min(capacity, max(8, int(cache_budget_bytes) // max(1, image.nbytes))),
                )

        def insert(path: str, image: np.ndarray) -> None:
            cache[path] = image
            while len(cache) > capacity:
                cache.popitem(last=False)

        def load(path: str) -> np.ndarray:
            image = _load_cached_uint8(cache, path, max_items=capacity)
            resolve_budget(image)
            return image

        inflight: dict[str, Future] = {}
        window = max(4, int(batch_size))

        def triplet_paths(record: Mapping[str, Any]) -> tuple[str, str, str]:
            return (
                str(record["img0"]["path"]),
                str(record["gt"]["path"]),
                str(record["img1"]["path"]),
            )

        def prefetch_upto(scan_index: int, consume_index: int) -> int:
            """Batch-level look-ahead: submit unique cache-miss paths for the
            next ``window`` triplets so the decode pool stays busy across
            overlapping triplets instead of stalling on one triplet's misses.

            Decoded frames wait in ``inflight`` futures — never in the LRU
            cache — until their record is consumed in order, so cache writes,
            eviction, and budget resolution match the serial path exactly.
            Malformed records are skipped here (no early raise); the consume
            loop hits their error and routes it through the ("invalid", ...)
            protocol just as before.
            """

            assert pool is not None
            target = min(len(records), consume_index + window)
            while scan_index < target:
                record = records[scan_index]
                scan_index += 1
                try:
                    paths = triplet_paths(record)
                except (KeyError, TypeError):
                    continue
                for path in paths:
                    if path in cache or path in inflight:
                        continue
                    inflight[path] = pool.submit(read_rgb_uint8, path)
            return scan_index

        def consume_load(path: str) -> np.ndarray:
            """Consume-time load mirroring serial ``load`` semantics.

            Cache hit -> LRU touch; otherwise take the prefetched future (or
            decode inline if the look-ahead has not reached it yet) and insert
            into the cache in consume order.
            """

            existing = cache.pop(path, None)
            if existing is not None:
                cache[path] = existing
                resolve_budget(existing)
                return existing
            future = inflight.pop(path, None)
            image = future.result() if future is not None else read_rgb_uint8(path)
            insert(path, image)
            resolve_budget(image)
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
            scan_index = 0
            for consume_index, record in enumerate(records):
                if stopped.is_set():
                    return
                if pool is not None:
                    scan_index = prefetch_upto(scan_index, consume_index)
                try:
                    decode_started = time.perf_counter()
                    paths = triplet_paths(record)
                    if pool is None:
                        first = load(paths[0])
                        middle = load(paths[1])
                        last = load(paths[2])
                    else:
                        first = consume_load(paths[0])
                        middle = consume_load(paths[1])
                        last = consume_load(paths[2])
                    if first.shape != middle.shape or first.shape != last.shape:
                        raise ValueError(
                            "triplet image shapes differ: "
                            f"{first.shape}, {middle.shape}, {last.shape}"
                        )
                    record_decode_seconds = time.perf_counter() - decode_started
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if not flush() or not put(("invalid", (record, exc))):
                        return
                    continue
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
            if pool is not None:
                pool.shutdown(wait=False)
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
    source_shape = tuple(int(value) for value in items[0][1].shape[:2])
    network_size = tuple(
        int(value)
        for value in getattr(adapter, "network_size", source_shape)
    )
    img0_tensor, img1_tensor = _network_input_batch(
        items,
        production_batch=production_batch,
        network_size=network_size,
    )
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
    validate: bool = True,
    tiered: bool = False,
    prediction_only: bool = False,
) -> ReconstructionResult | tuple[ReconstructionResult, Tier2Residue]:
    """Run reconstruction on ``device`` and return CPU tensors.

    ``device=None`` keeps the calibrated CPU reference path.  On accelerator
    devices the native-resolution warps run there and the result returns to
    the host in one packed transfer, so CPU threads only ever score.

    ``validate=False`` disables the device-synchronizing NaN/range scans for
    the production hot path; results are bitwise identical either way.

    ``tiered=True`` returns ``(partial, residue)``: tier-1 transfers prediction,
    six scope scalars, and one uint8 discontinuity support plane; tier-2
    (warps + masks, 11ch) stays resident until the main thread materializes
    selected candidates.

    ``prediction_only=True`` transfers only the prediction (3ch) and leaves
    every other field on the device (unpacked): the teacher path scores the
    prediction alone, so compact scope evidence and the held-back tier-2
    residue are both pure waste there.
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
        validate=validate,
        compute_motion_metrics=tiered,
    )
    if prediction_only:
        return pack_prediction_to_cpu(reconstructed)
    if tiered:
        return pack_tier1_to_cpu(reconstructed)
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
    validate: bool = True,
) -> ReconstructionResult:
    """Inference + reconstruction on the worker's main thread.

    Model outputs never round-trip to the host before reconstruction: on
    accelerator devices the reconstruction consumes them in place, and only
    the finished native-resolution result is transferred back (packed).
    """

    if not items:
        raise ValueError("inference batch must not be empty")
    inference_batch = _infer_model_batch(
        items,
        adapter=adapter,
        production_batch=production_batch,
    )
    prepared = _prepare_reconstruction_microbatch(items)
    return _reconstruct_outputs(
        prepared.img0_tensor,
        prepared.img1_tensor,
        inference_batch.outputs,
        model_config=model_config,
        device=reconstruction_device,
        validate=validate,
    )


_INPUT_FRAME_CHANNELS = 9
_MAIN_SCRATCH_CHANNELS = 24
_RECONSTRUCTION_TRANSIENT_CHANNELS = 6
_PREDICTION_CHANNELS = 3
_FUTURE_FIXED_BYTES = 1024 * 1024
_FUTURE_WAIT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class PendingReservation:
    """Memory owned or temporarily required by one postprocess Future."""

    sample_count: int
    retained_bytes: int
    device_retained_bytes: int
    scratch_bytes: int
    fixed_bytes: int
    reserved_bytes: int
    reconstruction_transient_bytes: int
    pipeline_bytes: int
    oversize: bool = False


@dataclass(slots=True)
class PendingPostprocess:
    future: Future[list[dict[str, Any]]]
    reservation: PendingReservation
    sequence_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class PendingPhase1:
    """Two-phase pending entry: phase-1 future + main-thread-owned residue.

    The tier-2 residue and the tier-1 partial stay referenced here (not in
    the pool thread) so the main thread alone performs the on-demand D2H
    once phase-1 scoring flags candidates.
    """

    future: Future[Phase1BatchResult]
    reservation: PendingReservation
    tier2_residue: Tier2Residue | None
    reconstructed_tier1: ReconstructionResult
    sequence_ids: tuple[int, ...]


@dataclass(slots=True)
class PendingPhase2:
    """Phase-2 diagnosis future created after the main-thread tier-2 D2H.

    It lives in a separate ready-scanned queue; final records are restored
    through the global sequence reorder buffer.
    """

    future: Future[list[tuple[int, dict[str, Any]]]]
    record_by_index: dict[int, dict[str, Any]]
    reservation: PendingReservation
    candidate_count: int
    sequence_ids: tuple[int, ...]
    candidate_indices: tuple[int, ...]


def _infer_model_batch(
    items: Sequence[DecodedItem],
    *,
    adapter: ModelAdapter,
    production_batch: int,
) -> InferenceBatch:
    """Infer from a production-sized fixed-network-resolution input batch."""

    if not items:
        raise ValueError("inference batch must not be empty")
    valid_count = len(items)
    source_shape = tuple(int(value) for value in items[0][1].shape[:2])
    network_size = tuple(
        int(value)
        for value in getattr(adapter, "network_size", source_shape)
    )
    img0_tensor, img1_tensor = _network_input_batch(
        items,
        production_batch=production_batch,
        network_size=network_size,
    )
    outputs = _trim_model_outputs(
        adapter.infer(img0_tensor, img1_tensor),
        valid_count,
    )
    input_bytes = int(
        img0_tensor.numel() * img0_tensor.element_size()
        + img1_tensor.numel() * img1_tensor.element_size()
    )
    output_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in (
            outputs.flow_t0,
            outputs.flow_t1,
            outputs.mask0,
            outputs.mask1,
        )
    )
    return InferenceBatch(
        outputs=outputs,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )


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
    return int(height) * int(width) * RECONSTRUCTION_CHANNELS * 4


def _reconstruction_tier1_bytes_per_sample(items: Sequence[DecodedItem]) -> int:
    """CPU tier-1: prediction + uint8 support map + six float scalars."""

    if not items:
        raise ValueError("cannot estimate reconstruction bytes for an empty batch")
    height, width = items[0][1].shape[:2]
    pixels = int(height) * int(width)
    return pixels * TIER1_CHANNELS * 4 + pixels + 6 * 4


def _reconstruction_tier2_bytes_per_sample(items: Sequence[DecodedItem]) -> int:
    """Device-held tier-2 payload (warps + masks, 11ch) awaiting on-demand D2H."""

    if not items:
        raise ValueError("cannot estimate reconstruction bytes for an empty batch")
    height, width = items[0][1].shape[:2]
    return int(height) * int(width) * TIER2_CHANNELS * 4


def _prediction_bytes_per_sample(items: Sequence[DecodedItem]) -> int:
    """CPU-held prediction payload (3ch float32) for the teacher path."""

    if not items:
        raise ValueError("cannot estimate reconstruction bytes for an empty batch")
    height, width = items[0][1].shape[:2]
    return int(height) * int(width) * _PREDICTION_CHANNELS * 4


def _postproc_reservation(
    items: Sequence[DecodedItem],
    *,
    sample_count: int | None = None,
    buffer_bytes: int | None = None,
    scratch_channels: int = _MAIN_SCRATCH_CHANNELS,
    tiered: bool = False,
    mode: str = "full",
) -> PendingReservation:
    """Budget one Future's memory.

    ``mode="tiered"`` (or legacy ``tiered=True``) matches the two-tier
    production pipeline: prediction + compact scope evidence are CPU-retained
    per sample while
    tier-2 (11ch) stays on the reconstruction device until candidates
    materialize it; both count against ``pipeline_bytes``.  ``mode="full"``
    keeps the legacy 18-channel all-on-CPU accounting.  ``mode="prediction"``
    matches the teacher path: only the prediction (3ch) is retained and
    nothing stays on the device. ``mode="phase2"`` is the candidate-only CPU
    diagnosis reservation after selected tier-2 rows have transferred.
    """

    if not items:
        raise ValueError("cannot reserve postprocess memory for an empty batch")
    if tiered:
        mode = "tiered"
    if mode not in ("full", "tiered", "prediction", "phase2"):
        raise ValueError(f"unknown postproc reservation mode: {mode!r}")
    count = len(items) if sample_count is None else int(sample_count)
    if count < 1 or count > len(items):
        raise ValueError("sample_count must be within the supplied item count")
    height, width = items[0][1].shape[:2]
    plane_bytes = int(height) * int(width) * 4
    if mode == "tiered":
        reconstruction_retained_bytes = (
            count * _reconstruction_tier1_bytes_per_sample(items)
        )
        device_retained_bytes = (
            count * _reconstruction_tier2_bytes_per_sample(items)
        )
    elif mode == "phase2":
        reconstruction_retained_bytes = count * (
            _reconstruction_tier1_bytes_per_sample(items)
            + _reconstruction_tier2_bytes_per_sample(items)
        )
        device_retained_bytes = 0
    elif mode == "prediction":
        reconstruction_retained_bytes = (
            count * _prediction_bytes_per_sample(items)
        )
        device_retained_bytes = 0
    else:
        reconstruction_retained_bytes = (
            count * _reconstruction_bytes_per_sample(items)
        )
        device_retained_bytes = 0
    retained_bytes = (
        reconstruction_retained_bytes + count * plane_bytes * _INPUT_FRAME_CHANNELS
    )
    scratch_bytes = plane_bytes * max(0, int(scratch_channels))
    reserved_bytes = retained_bytes + scratch_bytes + _FUTURE_FIXED_BYTES
    reconstruction_transient_bytes = (
        count * plane_bytes * _RECONSTRUCTION_TRANSIENT_CHANNELS
    )
    pipeline_bytes = (
        reserved_bytes + reconstruction_transient_bytes + device_retained_bytes
    )
    return PendingReservation(
        sample_count=count,
        retained_bytes=retained_bytes,
        device_retained_bytes=device_retained_bytes,
        scratch_bytes=scratch_bytes,
        fixed_bytes=_FUTURE_FIXED_BYTES,
        reserved_bytes=reserved_bytes,
        reconstruction_transient_bytes=reconstruction_transient_bytes,
        pipeline_bytes=pipeline_bytes,
        oversize=(
            buffer_bytes is not None and pipeline_bytes > max(0, int(buffer_bytes))
        ),
    )


def _postproc_microbatch_size(
    items: Sequence[DecodedItem],
    *,
    buffer_bytes: int,
    postproc_workers: int,
    scratch_channels: int = _MAIN_SCRATCH_CHANNELS,
    tiered: bool = False,
    mode: str = "full",
    override: int = 0,
) -> int:
    """Choose a slice whose complete Future reservation fits its fair share.

    When ``override > 0`` the auto-derived value is bypassed; the admission
    gate in ``_process_payload_records`` still limits total in-flight bytes.
    """

    if not items:
        raise ValueError("cannot size an empty postprocess batch")
    if override > 0:
        return min(len(items), override)
    per_future_budget = max(1, int(buffer_bytes) // max(1, int(postproc_workers)))
    single = _postproc_reservation(
        items,
        sample_count=1,
        buffer_bytes=per_future_budget,
        scratch_channels=scratch_channels,
        tiered=tiered,
        mode=mode,
    )
    retained_per_sample = single.retained_bytes
    transient_per_sample = single.reconstruction_transient_bytes
    device_per_sample = single.device_retained_bytes
    fixed_and_scratch = single.scratch_bytes + single.fixed_bytes
    available = max(0, per_future_budget - fixed_and_scratch)
    return max(
        1,
        min(
            len(items),
            available
            // max(
                1,
                retained_per_sample + transient_per_sample + device_per_sample,
            ),
        ),
    )


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
    tiered_reconstruction: bool = False,
    prediction_only: bool = False,
) -> list[dict[str, Any]]:
    max_cache = int(config.runtime.chunk_triplets) + 2
    cache_budget_bytes = int(config.runtime.decode_cache_mb) * 1024 * 1024
    postproc_buffer_bytes = int(config.runtime.postproc_buffer_mb) * 1024 * 1024
    postproc_workers = _resolve_postproc_workers(config)
    output: list[dict[str, Any]] = []
    legacy_pending: list[PendingPostprocess] = []
    phase1_pending: list[PendingPhase1] = []
    phase2_pending: list[PendingPhase2] = []
    completed_reorder_buffer: dict[int, dict[str, Any]] = {}
    next_sequence_to_assign = 0
    next_sequence_to_publish = 0
    pending_reserved_bytes = 0
    pending_retained_bytes = 0
    pending_device_retained_bytes = 0
    tier2_d2h_bytes = 0
    tier2_materialized_batches = 0
    tier2_candidate_samples = 0
    phase1_evaluated_samples = 0
    timings = CpuTimingTotals()
    oversize_logged = False
    scored_count = 0
    next_timing_report = _TIMING_REPORT_SAMPLES
    decode_uint8_bytes = 0
    network_bytes = 0
    reconstruction_transient_bytes = 0
    bar = _ProgressLog(len(records), progress_prefix) if progress_prefix else None
    if bar is not None:
        bar.configure(
            resolved_postproc_workers=postproc_workers,
            resolved_microbatch_size=(
                config.runtime.postproc_microbatch_size
                if config.runtime.postproc_microbatch_size > 0
                else "auto_pending"
            ),
            postproc_buffer_mb=config.runtime.postproc_buffer_mb,
        )

    def sync_bar_state(*, microbatch_size: int | None = None) -> None:
        if bar is None:
            return
        bar.set_pipeline_state(
            candidate_ratio=(
                tier2_candidate_samples
                / max(1, phase1_evaluated_samples)
            ),
            phase1_queue_depth=len(phase1_pending),
            phase2_queue_depth=len(phase2_pending),
            resolved_microbatch_size=microbatch_size,
        )

    def pending_count() -> int:
        return (
            len(legacy_pending)
            + len(phase1_pending)
            + len(phase2_pending)
        )

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

    def publish_completed(
        sequence_ids: Sequence[int],
        completed: Sequence[dict[str, Any]],
        *,
        reservation: PendingReservation,
        release_device: bool,
    ) -> None:
        """Release memory now and publish records later in source order."""

        nonlocal pending_reserved_bytes, pending_retained_bytes
        nonlocal pending_device_retained_bytes, scored_count
        nonlocal next_sequence_to_publish
        if len(sequence_ids) != len(completed):
            raise RuntimeError(
                "completed record count does not match sequence ids"
            )
        pending_reserved_bytes -= reservation.reserved_bytes
        pending_retained_bytes -= reservation.retained_bytes
        if release_device:
            pending_device_retained_bytes -= (
                reservation.device_retained_bytes
            )
        for sequence_id, record in zip(sequence_ids, completed):
            completed_reorder_buffer[int(sequence_id)] = record
        while next_sequence_to_publish in completed_reorder_buffer:
            output.append(
                completed_reorder_buffer.pop(next_sequence_to_publish)
            )
            next_sequence_to_publish += 1
        scored_count += len(completed)
        if heartbeat is not None:
            heartbeat()
        sync_bar_state()
        if bar is not None:
            bar.update_scored(
                len(completed),
                pending_batches=pending_count(),
                pending_bytes=pending_reserved_bytes,
                pending_retained_bytes=pending_retained_bytes,
                pending_device_retained_bytes=pending_device_retained_bytes,
                tier2_d2h_bytes=tier2_d2h_bytes,
                tier2_materialized_batches=tier2_materialized_batches,
                tier2_candidate_samples=tier2_candidate_samples,
                memory=memory_estimate(),
            )
        maybe_report_timings()

    def process_phase1(entry: PendingPhase1) -> None:
        """Convert one completed Phase-1 result into output or Phase 2."""

        nonlocal pending_reserved_bytes, pending_retained_bytes
        nonlocal pending_device_retained_bytes
        nonlocal tier2_d2h_bytes, tier2_materialized_batches
        nonlocal tier2_candidate_samples, phase1_evaluated_samples
        reservation = entry.reservation
        phase1: Phase1BatchResult = entry.future.result()
        candidates = [
            outcome
            for outcome in phase1.outcomes
            if isinstance(outcome, Phase1Candidate)
        ]
        phase1_evaluated_samples += reservation.sample_count
        record_by_index: dict[int, dict[str, Any]] = {
            outcome.sample_index: outcome.record
            for outcome in phase1.outcomes
            if isinstance(outcome, Phase1Reject)
        }
        if not candidates:
            pending_device_retained_bytes -= (
                reservation.device_retained_bytes
            )
            completed = [
                record_by_index[index]
                for index in range(reservation.sample_count)
            ]
            publish_completed(
                entry.sequence_ids,
                completed,
                reservation=reservation,
                release_device=False,
            )
            return

        residue = entry.tier2_residue
        if residue is None:
            raise RuntimeError("two-phase drain requires a tier-2 residue")
        candidate_indices = tuple(
            candidate.sample_index for candidate in candidates
        )
        candidate_items: list[DecodedItem] = [
            (
                candidate.source,
                candidate.img0,
                candidate.gt,
                candidate.img1,
            )
            for candidate in candidates
        ]
        phase2_reservation = _postproc_reservation(
            candidate_items,
            buffer_bytes=postproc_buffer_bytes,
            mode="phase2",
        )
        d2h_started = time.perf_counter()
        tier2_fields = residue.materialize(list(candidate_indices))
        d2h_elapsed = time.perf_counter() - d2h_started
        timings.add_reconstruction(d2h_elapsed, len(candidates))
        actual_d2h = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in tier2_fields.values()
        )
        tier2_d2h_bytes += actual_d2h
        tier2_materialized_batches += 1
        tier2_candidate_samples += len(candidates)

        # Phase 1 held every sample plus full device residue.  After selected
        # D2H, keep only candidate CPU inputs/tier-1/tier-2 and recompute the
        # reservation before Phase 2 enters the pool.
        pending_reserved_bytes += (
            phase2_reservation.reserved_bytes - reservation.reserved_bytes
        )
        pending_retained_bytes += (
            phase2_reservation.retained_bytes - reservation.retained_bytes
        )
        pending_device_retained_bytes -= reservation.device_retained_bytes
        compacted_tier1 = slice_reconstruction_cpu(
            entry.reconstructed_tier1,
            list(candidate_indices),
        )
        merged = merge_tier2(compacted_tier1, tier2_fields)
        local_candidates = [
            _replace_dc(candidate, sample_index=index)
            for index, candidate in enumerate(candidates)
        ]
        future = executor.submit(
            _finish_phase2_batch,
            local_candidates,
            merged,
            config=config,
            timings=timings,
        )
        phase2_pending.append(
            PendingPhase2(
                future=future,
                record_by_index=record_by_index,
                reservation=phase2_reservation,
                candidate_count=len(candidates),
                sequence_ids=entry.sequence_ids,
                candidate_indices=candidate_indices,
            )
        )
        sync_bar_state()

    def process_phase2(entry: PendingPhase2) -> None:
        record_by_index = dict(entry.record_by_index)
        for local_index, record in entry.future.result():
            try:
                original_index = entry.candidate_indices[local_index]
            except IndexError as exc:
                raise RuntimeError(
                    "phase-2 result index is outside candidate mapping"
                ) from exc
            record_by_index[original_index] = record
        completed = [
            record_by_index[index]
            for index in range(len(entry.sequence_ids))
        ]
        publish_completed(
            entry.sequence_ids,
            completed,
            reservation=entry.reservation,
            release_device=False,
        )

    def process_legacy(entry: PendingPostprocess) -> None:
        completed = entry.future.result()
        publish_completed(
            entry.sequence_ids,
            completed,
            reservation=entry.reservation,
            release_device=True,
        )

    def reap_ready() -> bool:
        """Reap any completed Future, preferring Phase 2 then Phase 1."""

        for entries, processor in (
            (phase2_pending, process_phase2),
            (phase1_pending, process_phase1),
            (legacy_pending, process_legacy),
        ):
            for index, entry in enumerate(entries):
                if not entry.future.done():
                    continue
                entries.pop(index)
                processor(entry)
                return True
        return False

    def drain_any(*, block: bool) -> bool:
        """Reap ready work without FIFO head-of-line blocking."""

        if reap_ready():
            return True
        if not block or pending_count() == 0:
            return False
        futures = [
            entry.future
            for entry in (
                *phase2_pending,
                *phase1_pending,
                *legacy_pending,
            )
        ]
        wait_started = time.perf_counter()
        while True:
            done, _ = wait_futures(
                futures,
                timeout=_FUTURE_WAIT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if done:
                break
            if heartbeat is not None:
                heartbeat()
            sync_bar_state()
            if bar is not None:
                bar.waiting(
                    pending_batches=pending_count(),
                    pending_bytes=pending_reserved_bytes,
                    pending_retained_bytes=pending_retained_bytes,
                    pending_device_retained_bytes=(
                        pending_device_retained_bytes
                    ),
                    tier2_d2h_bytes=tier2_d2h_bytes,
                    tier2_materialized_batches=tier2_materialized_batches,
                    tier2_candidate_samples=tier2_candidate_samples,
                    memory=memory_estimate(),
                )
        timings.add_future_wait(
            time.perf_counter() - wait_started,
            max(1, pending_count()),
        )
        return reap_ready()

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
            decode_workers=config.runtime.decode_workers,
        ):
            if kind == "batch":
                decoded_batch = (
                    value
                    if isinstance(value, DecodedBatch)
                    else DecodedBatch(
                        items=tuple(value),
                        decode_seconds=0.0,
                        uint8_bytes=_decoded_batch_uint8_bytes(value),
                    )
                )
                items = decoded_batch.items
                decode_uint8_bytes = decoded_batch.uint8_bytes
                timings.add_decode(decoded_batch.decode_seconds, len(items))
                inference_started = time.perf_counter()
                inference_batch = _infer_model_batch(
                    items,
                    adapter=adapter,
                    production_batch=model_config.batch_size,
                )
                timings.add_inference(
                    time.perf_counter() - inference_started,
                    len(items),
                )
                network_bytes = inference_batch.network_bytes
                sync_bar_state()
                if bar is not None:
                    bar.update_inferred(
                        len(items),
                        pending_batches=pending_count(),
                        pending_bytes=pending_reserved_bytes,
                        pending_retained_bytes=pending_retained_bytes,
                        pending_device_retained_bytes=(
                            pending_device_retained_bytes
                        ),
                        tier2_d2h_bytes=tier2_d2h_bytes,
                        tier2_materialized_batches=tier2_materialized_batches,
                        tier2_candidate_samples=tier2_candidate_samples,
                        memory=memory_estimate(),
                    )
                # Network-size input tensors are released when inference
                # returns; only low-resolution outputs survive reconstruction.
                network_bytes = inference_batch.output_bytes
                pack_mode = (
                    "prediction"
                    if prediction_only
                    else ("tiered" if tiered_reconstruction else "full")
                )
                microbatch_size = _postproc_microbatch_size(
                    items,
                    buffer_bytes=postproc_buffer_bytes,
                    postproc_workers=postproc_workers,
                    mode=pack_mode,
                    override=config.runtime.postproc_microbatch_size,
                )
                sync_bar_state(microbatch_size=microbatch_size)
                for start in range(0, len(items), microbatch_size):
                    while drain_any(block=False):
                        pass
                    end = min(len(items), start + microbatch_size)
                    item_slice = list(items[start:end])
                    sequence_ids = tuple(
                        range(
                            next_sequence_to_assign,
                            next_sequence_to_assign + len(item_slice),
                        )
                    )
                    next_sequence_to_assign += len(item_slice)
                    reservation = _postproc_reservation(
                        item_slice,
                        buffer_bytes=postproc_buffer_bytes,
                        mode=pack_mode,
                    )
                    while pending_count() and (
                        pending_count() >= postproc_workers
                        or reservation.oversize
                        or pending_reserved_bytes
                        + pending_device_retained_bytes
                        + reservation.pipeline_bytes
                        > postproc_buffer_bytes
                    ):
                        if not drain_any(block=True):
                            raise RuntimeError(
                                "pending work could not make progress"
                            )
                    reconstruction_transient_bytes = (
                        reservation.reconstruction_transient_bytes
                    )
                    sync_bar_state()
                    if bar is not None:
                        bar.waiting(
                            pending_batches=pending_count(),
                            pending_bytes=pending_reserved_bytes,
                            pending_retained_bytes=pending_retained_bytes,
                            pending_device_retained_bytes=(
                                pending_device_retained_bytes
                            ),
                            tier2_d2h_bytes=tier2_d2h_bytes,
                            tier2_materialized_batches=(
                                tier2_materialized_batches
                            ),
                            tier2_candidate_samples=tier2_candidate_samples,
                            memory=memory_estimate(),
                        )
                    reconstruction_started = time.perf_counter()
                    prepared = _prepare_reconstruction_microbatch(item_slice)
                    tier2_residue: Tier2Residue | None = None
                    if prediction_only:
                        # Teacher scores the prediction alone: transfer 3ch
                        # and leave flows/warps/masks unpacked on the device.
                        reconstructed = _reconstruct_outputs(
                            prepared.img0_tensor,
                            prepared.img1_tensor,
                            _slice_model_outputs(inference_batch.outputs, start, end),
                            model_config=model_config,
                            device=reconstruction_device,
                            validate=False,
                            prediction_only=True,
                        )
                    elif tiered_reconstruction:
                        # Tier 1 transfers prediction + six scope scalars +
                        # one uint8 discontinuity support plane.  Full-
                        # resolution flows stay on the device and are freed;
                        # warps + masks transfer only for candidates.
                        reconstructed, tier2_residue = _reconstruct_outputs(
                            prepared.img0_tensor,
                            prepared.img1_tensor,
                            _slice_model_outputs(inference_batch.outputs, start, end),
                            model_config=model_config,
                            device=reconstruction_device,
                            # Production hot path: skip device-synchronizing
                            # NaN/range scans (~30 per batch).  The scoring
                            # stage remains the NaN safety net.
                            validate=False,
                            tiered=True,
                        )
                    else:
                        reconstructed = _reconstruct_outputs(
                            prepared.img0_tensor,
                            prepared.img1_tensor,
                            _slice_model_outputs(inference_batch.outputs, start, end),
                            model_config=model_config,
                            device=reconstruction_device,
                            validate=False,
                        )
                    timings.add_phases(
                        reconstructed.flow_metric_timings,
                        samples=len(item_slice),
                    )
                    timings.add_reconstruction(
                        time.perf_counter() - reconstruction_started,
                        len(item_slice),
                    )
                    use_two_phase = (
                        finish_batch is _ORIGINAL_FINISH_MAIN_BATCH
                        and tiered_reconstruction
                    )
                    if use_two_phase:
                        # Two-phase: pool threads score tier-1 only and
                        # return candidate decisions; the main thread runs
                        # the selected tier-2 D2H at drain time and then
                        # dispatches diagnosis (see drain_two_phase).
                        phase1_future = executor.submit(
                            _finish_phase1_batch,
                            prepared.items,
                            reconstructed,
                            config=config,
                            timings=timings,
                        )
                        pending_entry: PendingPostprocess | PendingPhase1 = (
                            PendingPhase1(
                                future=phase1_future,
                                reservation=reservation,
                                tier2_residue=tier2_residue,
                                reconstructed_tier1=reconstructed,
                                sequence_ids=sequence_ids,
                            )
                        )
                    else:
                        finish_kwargs: dict[str, Any] = {"config": config}
                        if finish_batch is _ORIGINAL_FINISH_MAIN_BATCH:
                            finish_kwargs["timings"] = timings
                        if tier2_residue is not None:
                            finish_kwargs["tier2_residue"] = tier2_residue
                        future = executor.submit(
                            finish_batch,
                            prepared.items,
                            reconstructed,
                            **finish_kwargs,
                        )
                        pending_entry = PendingPostprocess(
                            future=future,
                            reservation=reservation,
                            sequence_ids=sequence_ids,
                        )
                    del prepared
                    reconstruction_transient_bytes = 0
                    if isinstance(pending_entry, PendingPhase1):
                        phase1_pending.append(pending_entry)
                    else:
                        legacy_pending.append(pending_entry)
                    pending_reserved_bytes += reservation.reserved_bytes
                    pending_retained_bytes += reservation.retained_bytes
                    pending_device_retained_bytes += (
                        reservation.device_retained_bytes
                    )
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
            elif kind == "invalid":
                record, error = value
                sequence_id = next_sequence_to_assign
                next_sequence_to_assign += 1
                completed_reorder_buffer[sequence_id] = invalid_record(
                    record, error
                )
                while next_sequence_to_publish in completed_reorder_buffer:
                    output.append(
                        completed_reorder_buffer.pop(
                            next_sequence_to_publish
                        )
                    )
                    next_sequence_to_publish += 1
                scored_count += 1
                sync_bar_state()
                if bar is not None:
                    bar.update_invalid(
                        1,
                        pending_batches=pending_count(),
                        pending_bytes=pending_reserved_bytes,
                        pending_retained_bytes=pending_retained_bytes,
                        pending_device_retained_bytes=(
                            pending_device_retained_bytes
                        ),
                        tier2_d2h_bytes=tier2_d2h_bytes,
                        tier2_materialized_batches=tier2_materialized_batches,
                        tier2_candidate_samples=tier2_candidate_samples,
                        memory=memory_estimate(),
                    )
                maybe_report_timings()
            else:  # pragma: no cover - producer owns this internal protocol
                raise RuntimeError(f"unexpected decode event {kind!r}")
            if heartbeat is not None:
                heartbeat()
        while pending_count():
            if not drain_any(block=True):
                raise RuntimeError("pending work could not make progress")
    if completed_reorder_buffer:
        raise RuntimeError(
            "completed reorder buffer has an unresolved sequence gap"
        )
    if next_sequence_to_publish != next_sequence_to_assign:
        raise RuntimeError(
            "processed record count does not match task payload"
        )
    sync_bar_state()
    if bar is not None:
        bar.close(
            pending_batches=0,
            pending_bytes=0,
            pending_retained_bytes=0,
            pending_device_retained_bytes=0,
            tier2_d2h_bytes=tier2_d2h_bytes,
            tier2_materialized_batches=tier2_materialized_batches,
            tier2_candidate_samples=tier2_candidate_samples,
            memory=memory_estimate(),
        )
    maybe_report_timings(force=True)
    return output


def _finish_main_batch(
    items: Sequence[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    reconstructed: ReconstructionResult,
    *,
    config: AppConfig,
    timings: CpuTimingTotals | None = None,
    tier2_residue: Tier2Residue | None = None,
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
            timings=timings,
            tier2_residue=tier2_residue,
        )
        for index, item in enumerate(items)
    ]


_ORIGINAL_FINISH_MAIN_BATCH = _finish_main_batch


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
        tiered_reconstruction=True,
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
    timings: CpuTimingTotals | None = None,
    tier2_residue: Tier2Residue | None = None,
) -> list[dict[str, Any]]:
    # Teacher records only consume the prediction (tier-1); the residue is
    # accepted for interface parity and released without materialization.
    del tier2_residue
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
        prediction_only=True,
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
