"""Local recoverability estimates and stable diagnostic reason labels.

GT defines correctness.  Teacher and best-of-warp branches are used only to
estimate whether the current failure appears recoverable; a teacher failure
does not invalidate the source triplet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import RegionBox
from .scoring import (
    ErrorMaps,
    LocalScoreResult,
    compute_error_maps,
    robust_local_score,
    score_local_errors,
    score_region,
    top_area_mean,
)


REASON_LABELS: tuple[str, ...] = (
    "missing_part",
    "broken_structure",
    "ghosting",
    "edge_tearing",
    "flicker",
    "blur",
    "endpoint_copy",
    "blend_mask_error",
)


@dataclass(frozen=True, slots=True)
class DiagnosisConfig:
    good_error: float = 0.12
    minimum_wrong_error: float = 0.05
    teacher_weight: float = 0.65
    warp_weight: float = 0.35
    edge_reason_threshold: float = 0.14
    missing_ratio: float = 1.25
    branch_improvement: float = 0.08
    endpoint_copy_ratio: float = 0.65
    flicker_threshold: float = 0.18

    @classmethod
    def from_value(cls, value: Any | None) -> "DiagnosisConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        defaults = cls()

        def read(name: str, fallback: Any, *aliases: str) -> Any:
            for candidate in (name, *aliases):
                if isinstance(value, Mapping) and candidate in value:
                    return value[candidate]
                if not isinstance(value, Mapping) and hasattr(value, candidate):
                    return getattr(value, candidate)
            return fallback

        return cls(
            good_error=float(read("good_error", defaults.good_error)),
            minimum_wrong_error=float(
                read("minimum_wrong_error", defaults.minimum_wrong_error)
            ),
            teacher_weight=float(read("teacher_weight", defaults.teacher_weight)),
            warp_weight=float(read("warp_weight", defaults.warp_weight)),
            edge_reason_threshold=float(
                read(
                    "edge_reason_threshold",
                    defaults.edge_reason_threshold,
                    "edge_threshold",
                )
            ),
            missing_ratio=float(read("missing_ratio", defaults.missing_ratio)),
            branch_improvement=float(
                read("branch_improvement", defaults.branch_improvement)
            ),
            endpoint_copy_ratio=float(
                read("endpoint_copy_ratio", defaults.endpoint_copy_ratio)
            ),
            flicker_threshold=float(
                read("flicker_threshold", defaults.flicker_threshold)
            ),
        )


@dataclass(frozen=True, slots=True)
class SolvabilityResult:
    p_solvable: float
    current_error: float
    teacher_error: float | None
    best_warp_error: float | None
    teacher_gain: float | None
    warp_gain: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegionDiagnosis:
    box: tuple[int, int, int, int]
    p_wrong: float
    p_solvable: float
    reasons: tuple[str, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "p_wrong": float(self.p_wrong),
            "p_solvable": float(self.p_solvable),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class DiagnosisResult:
    p_wrong: float
    mining_p_wrong: float
    p_solvable: float
    reasons: tuple[str, ...]
    regions: tuple[RegionDiagnosis, ...]
    metrics: dict[str, float]
    primary_region_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_wrong": float(self.p_wrong),
            "mining_p_wrong": float(self.mining_p_wrong),
            "p_solvable": float(self.p_solvable),
            "reasons": list(self.reasons),
            "regions": [region.to_dict() for region in self.regions],
            "metrics": dict(self.metrics),
            "primary_region_index": self.primary_region_index,
        }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    numpy_method = getattr(current, "numpy", None)
    return np.asarray(numpy_method() if callable(numpy_method) else current)


def _error_scalar(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        number = float(value)
    elif isinstance(value, Mapping):
        for key in ("p_wrong", "local_error", "score", "error"):
            if key in value:
                return _error_scalar(value[key])
        raise ValueError("error mapping must contain p_wrong, local_error, score, or error")
    elif hasattr(value, "p_wrong"):
        return _error_scalar(getattr(value, "p_wrong"))
    elif hasattr(value, "score") and not isinstance(value, np.ndarray):
        return _error_scalar(getattr(value, "score"))
    else:
        array = np.asarray(_to_numpy(value), dtype=np.float32)
        if array.ndim >= 3 and array.shape[-1] in (1, 2, 3, 4):
            array = np.abs(array).mean(axis=-1)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError("array error evidence must reduce to a 2-D map")
        number = robust_local_score(array)
    if not np.isfinite(number):
        raise ValueError("error evidence must be finite")
    return float(np.clip(number, 0.0, 1.0))


def _gain(current: float, alternative: float, floor: float) -> float:
    return float(np.clip((current - alternative) / max(current, floor), -1.0, 1.0))


def _recoverability_evidence(
    current: float, alternative: float, cfg: DiagnosisConfig
) -> tuple[float, float]:
    gain = _gain(current, alternative, cfg.minimum_wrong_error)
    relative_score = np.clip(0.5 + 0.5 * gain, 0.0, 1.0)
    absolute_quality = np.clip(1.0 - alternative / max(cfg.good_error, 1e-6), 0.0, 1.0)
    evidence = 0.80 * relative_score + 0.20 * absolute_quality
    return float(np.clip(evidence, 0.0, 1.0)), gain


def estimate_solvability(
    current_error: Any,
    teacher_error: Any | None = None,
    best_warp_error: Any | None = None,
    *,
    warp_errors: Mapping[str, Any] | Sequence[Any] | None = None,
    config: DiagnosisConfig | Mapping[str, Any] | Any | None = None,
) -> SolvabilityResult:
    """Estimate recoverability from local errors, never correctness itself."""

    cfg = DiagnosisConfig.from_value(config)
    current = _error_scalar(current_error)
    teacher = _error_scalar(teacher_error) if teacher_error is not None else None
    warp_values: list[float] = []
    if best_warp_error is not None:
        warp_values.append(_error_scalar(best_warp_error))
    if warp_errors is not None:
        source = warp_errors.values() if isinstance(warp_errors, Mapping) else warp_errors
        warp_values.extend(_error_scalar(item) for item in source if item is not None)
    best_warp = min(warp_values) if warp_values else None

    weighted: list[tuple[float, float]] = []
    reasons: list[str] = []
    teacher_gain: float | None = None
    warp_gain: float | None = None
    if teacher is not None:
        evidence, teacher_gain = _recoverability_evidence(current, teacher, cfg)
        weighted.append((evidence, max(0.0, cfg.teacher_weight)))
        if teacher_gain >= 0.25:
            reasons.append("teacher_recovers")
    if best_warp is not None:
        evidence, warp_gain = _recoverability_evidence(current, best_warp, cfg)
        weighted.append((evidence, max(0.0, cfg.warp_weight)))
        if warp_gain >= 0.25:
            reasons.append("warp_branch_recovers")

    total_weight = sum(weight for _, weight in weighted)
    if total_weight > 0.0:
        probability = sum(score * weight for score, weight in weighted) / total_weight
    else:
        probability = 0.5
        reasons.append("solvability_evidence_missing")
    if current < cfg.minimum_wrong_error:
        probability = min(probability, 0.5)
        reasons.append("wrongness_too_low_for_solvability")
    if probability < 0.55:
        reasons.append("solvability_uncertain")
    return SolvabilityResult(
        p_solvable=float(np.clip(probability, 0.0, 1.0)),
        current_error=current,
        teacher_error=teacher,
        best_warp_error=best_warp,
        teacher_gain=teacher_gain,
        warp_gain=warp_gain,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def compute_p_solvable(*args: Any, **kwargs: Any) -> float:
    return estimate_solvability(*args, **kwargs).p_solvable


def localized_error_score(
    candidate: Any,
    gt: Any,
    box: RegionBox | tuple[int, int, int, int] | None = None,
) -> float:
    maps = compute_error_maps(candidate, gt)
    return score_region(maps.structure, box)


def _normalize_box(
    box: RegionBox | Sequence[int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    if isinstance(box, RegionBox):
        raw = (box.x0, box.y0, box.x1, box.y1)
    else:
        if len(box) != 4:
            raise ValueError("region box must contain x0, y0, x1, y1")
        raw = tuple(int(item) for item in box)
    height, width = shape
    x0 = max(0, min(width, raw[0]))
    y0 = max(0, min(height, raw[1]))
    x1 = max(0, min(width, raw[2]))
    y1 = max(0, min(height, raw[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("region box must have positive image intersection")
    return x0, y0, x1, y1


def _crop(values: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return values[y0:y1, x0:x1]


def _edge_endpoint_count(edge_map: np.ndarray, threshold: float) -> int:
    binary = edge_map >= threshold
    if not np.any(binary):
        return 0
    padded = np.pad(binary.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
    neighbors = np.zeros(binary.shape, dtype=np.uint8)
    for y_offset in range(3):
        for x_offset in range(3):
            if y_offset == 1 and x_offset == 1:
                continue
            neighbors += padded[
                y_offset : y_offset + binary.shape[0],
                x_offset : x_offset + binary.shape[1],
            ]
    return int(np.count_nonzero(binary & (neighbors <= 1)))


def _binary_component_count(edge_map: np.ndarray, threshold: float) -> int:
    """Count eight-connected edge islands inside a small candidate crop."""

    remaining = edge_map >= threshold
    count = 0
    height, width = remaining.shape
    while np.any(remaining):
        start_y, start_x = np.argwhere(remaining)[0]
        remaining[start_y, start_x] = False
        stack = [(int(start_y), int(start_x))]
        count += 1
        while stack:
            y, x = stack.pop()
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if remaining[next_y, next_x]:
                        remaining[next_y, next_x] = False
                        stack.append((next_y, next_x))
    return count


def _branch_error(
    branch: Any | None,
    gt: Any,
    box: tuple[int, int, int, int],
) -> float | None:
    if branch is None:
        return None
    return localized_error_score(branch, gt, box)


def _ordered_labels(labels: Sequence[str]) -> tuple[str, ...]:
    present = set(labels)
    return tuple(label for label in REASON_LABELS if label in present)


def _diagnose_region(
    maps: ErrorMaps,
    box: tuple[int, int, int, int],
    *,
    prediction: Any,
    gt: Any,
    teacher_prediction: Any | None,
    warp0: Any | None,
    warp1: Any | None,
    warp_blend: Any | None,
    img1: Any | None,
    temporal_error: float | None,
    priority_metrics: Mapping[str, float] | None,
    cfg: DiagnosisConfig,
) -> RegionDiagnosis:
    structure = _crop(maps.structure, box)
    missing_map = _crop(maps.gt_only_edges, box)
    extra_map = _crop(maps.pred_only_edges, box)
    gt_edge_map = _crop(maps.sobel_gt, box)
    pred_edge_map = _crop(maps.sobel_prediction, box)
    current_error = robust_local_score(structure)
    missing = robust_local_score(missing_map)
    extra = robust_local_score(extra_map)
    edge_error = max(missing, extra)
    gt_edge = top_area_mean(gt_edge_map, 0.05) if gt_edge_map.size else 0.0
    pred_edge = top_area_mean(pred_edge_map, 0.05) if pred_edge_map.size else 0.0
    gt_endpoints = _edge_endpoint_count(gt_edge_map, cfg.edge_reason_threshold)
    pred_endpoints = _edge_endpoint_count(pred_edge_map, cfg.edge_reason_threshold)
    gt_components = _binary_component_count(gt_edge_map, cfg.edge_reason_threshold)
    pred_components = _binary_component_count(pred_edge_map, cfg.edge_reason_threshold)

    teacher_error = _branch_error(teacher_prediction, gt, box)
    warp0_error = _branch_error(warp0, gt, box)
    warp1_error = _branch_error(warp1, gt, box)
    blend_error = _branch_error(warp_blend, gt, box)
    endpoint1_error = _branch_error(img1, gt, box)
    endpoint_copy_distance = (
        localized_error_score(prediction, img1, box) if img1 is not None else None
    )
    warp_errors = {
        name: value
        for name, value in (
            ("warp0", warp0_error),
            ("warp1", warp1_error),
            ("warp_blend", blend_error),
        )
        if value is not None
    }
    solvability = estimate_solvability(
        current_error,
        teacher_error,
        warp_errors=warp_errors,
        config=cfg,
    )

    labels: list[str] = []
    if (
        missing >= cfg.edge_reason_threshold
        and missing >= extra * cfg.missing_ratio + 0.01
    ):
        labels.append("missing_part")
    width = box[2] - box[0]
    height = box[3] - box[1]
    # Tight native proposals include a small padding halo, so a severed thin
    # structure commonly has a box aspect ratio around 1.8 rather than the
    # much larger ratio of its original full extent.
    elongated = max(width, height) / max(1, min(width, height)) >= 1.75
    if missing >= cfg.edge_reason_threshold and (
        pred_endpoints > gt_endpoints
        or pred_components > gt_components
        or elongated
    ):
        labels.append("broken_structure")
    if missing >= cfg.edge_reason_threshold and extra >= cfg.edge_reason_threshold:
        ratio = missing / max(extra, 1e-6)
        if 0.35 <= ratio <= 2.85:
            labels.append("ghosting")
    if edge_error >= cfg.edge_reason_threshold:
        labels.append("edge_tearing")
    if temporal_error is not None and float(temporal_error) >= cfg.flicker_threshold:
        labels.append("flicker")
    if (
        missing >= cfg.edge_reason_threshold
        and gt_edge - pred_edge >= cfg.branch_improvement
    ):
        labels.append("blur")
    if (
        endpoint_copy_distance is not None
        and current_error >= cfg.minimum_wrong_error
        and endpoint_copy_distance
        <= max(
            cfg.good_error * 0.25,
            current_error * (1.0 - cfg.endpoint_copy_ratio),
        )
    ):
        labels.append("endpoint_copy")
    direct_warps = [value for value in (warp0_error, warp1_error) if value is not None]
    if (
        direct_warps
        and blend_error is not None
        and min(direct_warps) + cfg.branch_improvement < blend_error
    ):
        labels.append("blend_mask_error")

    metrics = {
        "current_error": current_error,
        "missing_edge_error": missing,
        "extra_edge_error": extra,
        "gt_edge_strength": gt_edge,
        "prediction_edge_strength": pred_edge,
        "gt_edge_endpoints": float(gt_endpoints),
        "prediction_edge_endpoints": float(pred_endpoints),
        "gt_edge_components": float(gt_components),
        "prediction_edge_components": float(pred_components),
        "teacher_error": float(teacher_error) if teacher_error is not None else -1.0,
        "warp0_error": float(warp0_error) if warp0_error is not None else -1.0,
        "warp1_error": float(warp1_error) if warp1_error is not None else -1.0,
        "warp_blend_error": float(blend_error) if blend_error is not None else -1.0,
        "endpoint1_error": float(endpoint1_error) if endpoint1_error is not None else -1.0,
        "endpoint_copy_distance": (
            float(endpoint_copy_distance) if endpoint_copy_distance is not None else -1.0
        ),
        "p_solvable": solvability.p_solvable,
    }
    if priority_metrics:
        for name in (
            "ui_context_available",
            "border_overlap",
            "border_likelihood",
            "endpoint_change_mean",
            "endpoint_change_q90",
            "endpoint_static_likelihood",
            "gt_edge_density",
            "gt_edge_density_likelihood",
            "ui_likelihood",
            "priority_weight",
            "priority_score",
        ):
            if name in priority_metrics:
                metrics[name] = float(priority_metrics[name])
    metrics["ui_likelihood"] = float(
        np.clip(metrics.get("ui_likelihood", 0.0), 0.0, 1.0)
    )
    metrics["priority_weight"] = float(
        np.clip(metrics.get("priority_weight", 1.0), 0.0, 1.0)
    )
    metrics["mining_p_wrong"] = float(current_error * metrics["priority_weight"])
    return RegionDiagnosis(
        box=box,
        p_wrong=current_error,
        p_solvable=solvability.p_solvable,
        reasons=_ordered_labels(labels),
        metrics=metrics,
    )


def diagnose_sample(
    prediction: Any,
    gt: Any,
    *,
    teacher_prediction: Any | None = None,
    warp0: Any | None = None,
    warp1: Any | None = None,
    warp_blend: Any | None = None,
    img0: Any | None = None,
    img1: Any | None = None,
    regions: Sequence[RegionBox | Sequence[int]] | None = None,
    temporal_error: float | None = None,
    scoring_config: Any | None = None,
    scoring_result: LocalScoreResult | None = None,
    config: DiagnosisConfig | Mapping[str, Any] | Any | None = None,
) -> DiagnosisResult:
    """Diagnose candidate regions and select the best wrong-and-solvable one."""

    cfg = DiagnosisConfig.from_value(config)
    if scoring_result is not None and not isinstance(scoring_result, LocalScoreResult):
        raise TypeError("scoring_result must be a LocalScoreResult")
    scoring = scoring_result or score_local_errors(
        prediction,
        gt,
        scoring_config,
        img0=img0 if img0 is not None and img1 is not None else None,
        img1=img1 if img0 is not None and img1 is not None else None,
    )
    height, width = scoring.maps.structure.shape
    source_regions: Sequence[RegionBox | Sequence[int]]
    source_regions = scoring.regions if regions is None else regions
    boxes = tuple(_normalize_box(region, (height, width)) for region in source_regions)
    priority_metrics = tuple(
        dict(region.metrics) if isinstance(region, RegionBox) else {}
        for region in source_regions
    )
    region_results = tuple(
        _diagnose_region(
            scoring.maps,
            box,
            prediction=prediction,
            gt=gt,
            teacher_prediction=teacher_prediction,
            warp0=warp0,
            warp1=warp1,
            warp_blend=warp_blend,
            img1=img1,
            temporal_error=temporal_error,
            priority_metrics=region_priority,
            cfg=cfg,
        )
        for box, region_priority in zip(boxes, priority_metrics)
    )

    if region_results:
        primary_index = max(
            range(len(region_results)),
            key=lambda index: (
                region_results[index].p_wrong
                * region_results[index].p_solvable
                * region_results[index].metrics.get("priority_weight", 1.0),
                region_results[index].p_wrong
                * region_results[index].metrics.get("priority_weight", 1.0),
                region_results[index].p_wrong,
                -index,
            ),
        )
        primary = region_results[primary_index]
        p_wrong = primary.p_wrong
        priority_weight = float(primary.metrics.get("priority_weight", 1.0))
        mining_p_wrong = float(p_wrong * priority_weight)
        p_solvable = primary.p_solvable
    else:
        primary_index = None
        p_wrong = scoring.p_wrong
        priority_weight = float(
            np.clip(
                scoring.mining_p_wrong / max(scoring.p_wrong, 1e-8)
                if scoring.p_wrong > 0.0
                else 1.0,
                0.0,
                1.0,
            )
        )
        mining_p_wrong = scoring.mining_p_wrong
        p_solvable = estimate_solvability(scoring.p_wrong, config=cfg).p_solvable
    reasons = _ordered_labels(
        [label for region in region_results for label in region.reasons]
    )
    metrics = {
        "candidate_region_count": float(len(region_results)),
        "scoring_p_wrong": float(scoring.p_wrong),
        "scoring_mining_p_wrong": float(scoring.mining_p_wrong),
        "selected_p_wrong": float(p_wrong),
        "selected_mining_p_wrong": float(mining_p_wrong),
        "selected_priority_weight": float(priority_weight),
        "selected_ui_likelihood": float(
            region_results[primary_index].metrics.get("ui_likelihood", 0.0)
            if primary_index is not None
            else 0.0
        ),
        "selected_p_solvable": float(p_solvable),
    }
    return DiagnosisResult(
        p_wrong=float(p_wrong),
        mining_p_wrong=float(mining_p_wrong),
        p_solvable=float(p_solvable),
        reasons=reasons,
        regions=region_results,
        metrics=metrics,
        primary_region_index=primary_index,
    )


# Stable aliases for pipeline code.
diagnose = diagnose_sample
solvability_from_errors = estimate_solvability


__all__ = [
    "DiagnosisConfig",
    "DiagnosisResult",
    "REASON_LABELS",
    "RegionDiagnosis",
    "SolvabilityResult",
    "compute_p_solvable",
    "diagnose",
    "diagnose_sample",
    "estimate_solvability",
    "localized_error_score",
    "solvability_from_errors",
]
