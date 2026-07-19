"""Conservative validity, scope, and final three-way decisions.

The gates separate bad/out-of-scope data from prediction correctness.  Missing
critical evidence produces ``review`` rather than an optimistic acceptance or
an unsupported rejection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

import numpy as np

from .scoring import as_rgb01


Decision = str
_LARGE_MOTION_NORMALIZED = 0.08
_BACKGROUND_BORDER_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class GateConfig:
    duplicate_reject_at: float = 0.002
    duplicate_review_at: float = 0.006
    scene_cut_review_at: float = 0.45
    scene_cut_reject_at: float = 0.62
    temporal_asymmetry_review_at: float = 0.65
    temporal_asymmetry_reject_at: float = 0.90
    menu_transition_review_at: float = 0.40
    menu_transition_reject_at: float = 0.70
    out_of_bounds_review_at: float = 0.45
    out_of_bounds_reject_at: float = 0.72
    occlusion_review_at: float = 0.40
    occlusion_reject_at: float = 0.70
    foreground_motion_review_at: float = 0.35
    foreground_motion_reject_at: float = 0.65
    flow_discontinuity_review_at: float = 0.30
    flow_discontinuity_reject_at: float = 0.60
    unexplained_motion_review_at: float = 0.35
    unexplained_motion_reject_at: float = 0.70
    wrong_reject_below: float = 0.20
    wrong_accept_at: float = 0.45
    solvable_review_below: float = 0.30
    solvable_accept_at: float = 0.55
    missing_metrics_to_review: bool = True

    @classmethod
    def from_value(cls, value: Any | None) -> "GateConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        defaults = cls()

        def read(name: str, fallback: Any, *aliases: str) -> Any:
            names = (name, *aliases)
            for candidate in names:
                if isinstance(value, Mapping) and candidate in value:
                    current = value[candidate]
                    if current is not None:
                        return current
                if not isinstance(value, Mapping) and hasattr(value, candidate):
                    current = getattr(value, candidate)
                    if current is not None:
                        return current
            return fallback

        duplicate_reject = float(
            read("duplicate_reject_at", defaults.duplicate_reject_at, "reject_duplicate")
        )
        scene_reject = float(
            read("scene_cut_reject_at", defaults.scene_cut_reject_at, "reject_scene_cut")
        )
        out_reject = float(
            read(
                "out_of_bounds_reject_at",
                defaults.out_of_bounds_reject_at,
                "reject_out_of_bounds",
            )
        )
        return cls(
            duplicate_reject_at=duplicate_reject,
            duplicate_review_at=float(
                read("duplicate_review_at", max(defaults.duplicate_review_at, duplicate_reject * 3.0))
            ),
            scene_cut_review_at=float(
                read("scene_cut_review_at", min(defaults.scene_cut_review_at, scene_reject * 0.75))
            ),
            scene_cut_reject_at=scene_reject,
            temporal_asymmetry_review_at=float(
                read("temporal_asymmetry_review_at", defaults.temporal_asymmetry_review_at)
            ),
            temporal_asymmetry_reject_at=float(
                read("temporal_asymmetry_reject_at", defaults.temporal_asymmetry_reject_at)
            ),
            menu_transition_review_at=float(
                read("menu_transition_review_at", defaults.menu_transition_review_at)
            ),
            menu_transition_reject_at=float(
                read("menu_transition_reject_at", defaults.menu_transition_reject_at)
            ),
            out_of_bounds_review_at=float(
                read("out_of_bounds_review_at", min(defaults.out_of_bounds_review_at, out_reject * 0.65))
            ),
            out_of_bounds_reject_at=out_reject,
            occlusion_review_at=float(
                read("occlusion_review_at", defaults.occlusion_review_at)
            ),
            occlusion_reject_at=float(
                read("occlusion_reject_at", defaults.occlusion_reject_at)
            ),
            foreground_motion_review_at=float(
                read("foreground_motion_review_at", defaults.foreground_motion_review_at)
            ),
            foreground_motion_reject_at=float(
                read("foreground_motion_reject_at", defaults.foreground_motion_reject_at)
            ),
            flow_discontinuity_review_at=float(
                read("flow_discontinuity_review_at", defaults.flow_discontinuity_review_at)
            ),
            flow_discontinuity_reject_at=float(
                read("flow_discontinuity_reject_at", defaults.flow_discontinuity_reject_at)
            ),
            unexplained_motion_review_at=float(
                read("unexplained_motion_review_at", defaults.unexplained_motion_review_at)
            ),
            unexplained_motion_reject_at=float(
                read("unexplained_motion_reject_at", defaults.unexplained_motion_reject_at)
            ),
            wrong_reject_below=float(
                read("wrong_reject_below", defaults.wrong_reject_below)
            ),
            wrong_accept_at=float(read("wrong_accept_at", defaults.wrong_accept_at)),
            solvable_review_below=float(
                read(
                    "solvable_review_below",
                    defaults.solvable_review_below,
                    "solvable_reject_below",
                )
            ),
            solvable_accept_at=float(
                read("solvable_accept_at", defaults.solvable_accept_at)
            ),
            missing_metrics_to_review=bool(
                read("missing_metrics_to_review", defaults.missing_metrics_to_review)
            ),
        )


@dataclass(frozen=True, slots=True)
class FrameValidityMetrics:
    decode_ok: bool = True
    finite: bool = True
    sequence_contiguous: bool = True
    duplicate_distance: float | None = None
    scene_cut_score: float | None = None
    histogram_jump: float | None = None
    temporal_asymmetry: float | None = None
    max_adjacent_difference: float | None = None
    menu_transition_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScopeMetrics:
    out_of_bounds_ratio: float | None = None
    flow_discontinuity_ratio: float | None = None
    foreground_large_motion_ratio: float | None = None
    occlusion_ratio: float | None = None
    unexplained_motion_ratio: float | None = None
    background_motion: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateResult:
    label: Decision
    reasons: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def decision(self) -> Decision:
        return self.label

    @property
    def passed(self) -> bool:
        return self.label == "accept"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError("metrics must be a mapping or dataclass")


def _numeric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _luma(image: Any) -> np.ndarray:
    rgb = as_rgb01(image)
    return np.asarray(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        dtype=np.float32,
    )


def _histogram_distance(left: np.ndarray, right: np.ndarray, bins: int = 64) -> float:
    left_hist, _ = np.histogram(left, bins=bins, range=(0.0, 1.0))
    right_hist, _ = np.histogram(right, bins=bins, range=(0.0, 1.0))
    left_norm = left_hist.astype(np.float64) / max(1, int(left_hist.sum()))
    right_norm = right_hist.astype(np.float64) / max(1, int(right_hist.sum()))
    return float(0.5 * np.abs(left_norm - right_norm).sum())


def _repeat_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.abs(left - right).reshape(-1)
    if difference.size == 0:
        return 0.0
    return float(
        max(
            difference.mean(),
            0.10 * np.quantile(difference, 0.95),
            0.05 * np.quantile(difference, 0.99),
        )
    )


def _boundary_overlay_score(difference: np.ndarray) -> float:
    """Score a dense rectangular change band attached to an image boundary."""

    height, width = difference.shape
    threshold = max(0.08, 0.5 * float(np.quantile(difference, 0.75)))
    changed = difference >= threshold
    if not np.any(changed):
        return 0.0
    strength = np.where(changed, difference, 0.0)
    total_pixels = float(height * width)

    def candidate_scores(
        counts: np.ndarray, totals: np.ndarray, areas: np.ndarray
    ) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        totals = np.asarray(totals, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)
        density = np.divide(counts, areas, out=np.zeros_like(counts), where=areas > 0)
        coverage = counts / total_pixels
        contrast = np.divide(
            totals,
            counts,
            out=np.zeros_like(totals),
            where=counts > 0,
        )
        density_score = np.clip((density - 0.45) / 0.55, 0.0, 1.0)
        coverage_score = np.clip(coverage / 0.40, 0.0, 1.0)
        contrast_score = np.clip((contrast - 0.08) / 0.32, 0.0, 1.0)
        scores = (
            0.40 * density_score
            + 0.40 * coverage_score
            + 0.20 * contrast_score
        )
        return np.where((counts > 0.0) & (density >= 0.45), scores, 0.0)

    row_counts = changed.sum(axis=1, dtype=np.int64)
    row_strength = strength.sum(axis=1, dtype=np.float64)
    column_counts = changed.sum(axis=0, dtype=np.int64)
    column_strength = strength.sum(axis=0, dtype=np.float64)
    row_count_prefix = np.concatenate(([0], np.cumsum(row_counts, dtype=np.int64)))
    row_strength_prefix = np.concatenate(
        ([0.0], np.cumsum(row_strength, dtype=np.float64))
    )
    column_count_prefix = np.concatenate(
        ([0], np.cumsum(column_counts, dtype=np.int64))
    )
    column_strength_prefix = np.concatenate(
        ([0.0], np.cumsum(column_strength, dtype=np.float64))
    )
    minimum_rows = max(1, int(np.ceil(height * 0.04)))
    minimum_columns = max(1, int(np.ceil(width * 0.04)))
    edge_density = 0.35
    candidates: list[np.ndarray] = []
    row_sizes = np.arange(minimum_rows, height + 1, dtype=np.int64)
    row_areas = row_sizes * width
    if row_counts[0] / float(width) >= edge_density:
        candidates.append(
            candidate_scores(
                row_count_prefix[row_sizes],
                row_strength_prefix[row_sizes],
                row_areas,
            )
        )
    if row_counts[-1] / float(width) >= edge_density:
        row_starts = height - row_sizes
        candidates.append(
            candidate_scores(
                row_count_prefix[-1] - row_count_prefix[row_starts],
                row_strength_prefix[-1] - row_strength_prefix[row_starts],
                row_areas,
            )
        )
    column_sizes = np.arange(minimum_columns, width + 1, dtype=np.int64)
    column_areas = column_sizes * height
    if column_counts[0] / float(height) >= edge_density:
        candidates.append(
            candidate_scores(
                column_count_prefix[column_sizes],
                column_strength_prefix[column_sizes],
                column_areas,
            )
        )
    if column_counts[-1] / float(height) >= edge_density:
        column_starts = width - column_sizes
        candidates.append(
            candidate_scores(
                column_count_prefix[-1] - column_count_prefix[column_starts],
                column_strength_prefix[-1]
                - column_strength_prefix[column_starts],
                column_areas,
            )
        )
    best = max((float(scores.max()) for scores in candidates), default=0.0)
    return float(np.clip(best, 0.0, 1.0))


def _automatic_menu_transition_score(
    difference_0t: np.ndarray, difference_t1: np.ndarray
) -> float:
    first_score = _boundary_overlay_score(difference_0t)
    second_score = _boundary_overlay_score(difference_t1)
    dominant = max(first_score, second_score)
    if dominant <= 0.0:
        return 0.0
    geometry_asymmetry = abs(first_score - second_score) / dominant
    first_energy = float(difference_0t.mean())
    second_energy = float(difference_t1.mean())
    energy_asymmetry = abs(first_energy - second_energy) / (
        first_energy + second_energy + 1e-8
    )
    one_sidedness = 0.60 * geometry_asymmetry + 0.40 * energy_asymmetry
    return float(np.clip(dominant * one_sidedness, 0.0, 1.0))


def compute_validity_metrics(
    img0: Any,
    gt: Any,
    img1: Any,
    *,
    decode_ok: bool = True,
    sequence_contiguous: bool = True,
    menu_transition_score: float | None = None,
) -> FrameValidityMetrics:
    """Compute cheap temporal validity evidence from an aligned triplet."""

    first = _luma(img0)
    middle = _luma(gt)
    last = _luma(img1)
    if first.shape != middle.shape or first.shape != last.shape:
        raise ValueError("img0, gt, and img1 must have the same spatial shape")

    difference_0t = np.abs(first - middle)
    difference_t1 = np.abs(middle - last)
    adjacent_0t = float(difference_0t.mean())
    adjacent_t1 = float(difference_t1.mean())
    denominator = adjacent_0t + adjacent_t1 + 1e-8
    asymmetry = float(abs(adjacent_0t - adjacent_t1) / denominator)
    histogram_0t = _histogram_distance(first, middle)
    histogram_t1 = _histogram_distance(middle, last)
    histogram_jump = max(histogram_0t, histogram_t1)
    # A cut is both a large distribution jump and temporally one-sided.
    scene_cut_score = float(histogram_jump * (0.5 + 0.5 * asymmetry))
    resolved_menu_score = (
        _automatic_menu_transition_score(difference_0t, difference_t1)
        if menu_transition_score is None
        else float(menu_transition_score)
    )
    return FrameValidityMetrics(
        decode_ok=bool(decode_ok),
        finite=True,
        sequence_contiguous=bool(sequence_contiguous),
        duplicate_distance=min(
            _repeat_distance(first, middle), _repeat_distance(middle, last)
        ),
        scene_cut_score=scene_cut_score,
        histogram_jump=histogram_jump,
        temporal_asymmetry=asymmetry,
        max_adjacent_difference=max(adjacent_0t, adjacent_t1),
        menu_transition_score=resolved_menu_score,
    )


def evaluate_validity(
    metrics: FrameValidityMetrics | Mapping[str, Any],
    config: GateConfig | Mapping[str, Any] | Any | None = None,
) -> GateResult:
    cfg = GateConfig.from_value(config)
    values = _to_mapping(metrics)
    reject: list[str] = []
    review: list[str] = []
    missing: list[str] = []

    for name, reason in (
        ("decode_ok", "invalid_decode"),
        ("finite", "invalid_nonfinite"),
        ("sequence_contiguous", "invalid_sequence_gap"),
    ):
        if name not in values or values[name] is None:
            missing.append(name)
        elif not bool(values[name]):
            reject.append(reason)

    duplicate = _numeric(values, "duplicate_distance")
    if duplicate is None:
        missing.append("duplicate_distance")
    elif duplicate <= cfg.duplicate_reject_at:
        reject.append("duplicate_frame")
    elif duplicate <= cfg.duplicate_review_at:
        review.append("possible_duplicate_frame")

    scene_cut = _numeric(values, "scene_cut_score")
    if scene_cut is None:
        missing.append("scene_cut_score")
    elif scene_cut >= cfg.scene_cut_reject_at:
        reject.append("scene_cut")
    elif scene_cut >= cfg.scene_cut_review_at:
        review.append("possible_scene_cut")

    asymmetry = _numeric(values, "temporal_asymmetry")
    adjacent = _numeric(values, "max_adjacent_difference")
    if asymmetry is None:
        missing.append("temporal_asymmetry")
    elif adjacent is None or adjacent >= cfg.duplicate_review_at:
        if asymmetry >= cfg.temporal_asymmetry_reject_at:
            reject.append("temporal_discontinuity")
        elif asymmetry >= cfg.temporal_asymmetry_review_at:
            review.append("temporal_asymmetry")

    menu = _numeric(values, "menu_transition_score")
    if menu is not None:
        if menu >= cfg.menu_transition_reject_at:
            reject.append("menu_transition")
        elif menu >= cfg.menu_transition_review_at:
            review.append("possible_menu_transition")

    if reject:
        label = "reject"
        reasons = tuple(dict.fromkeys(reject + review))
    elif review or (missing and cfg.missing_metrics_to_review):
        label = "review"
        reasons = tuple(dict.fromkeys(review + (["validity_metrics_missing"] if missing else [])))
    else:
        label = "accept"
        reasons = ()
    output_metrics = dict(values)
    output_metrics["missing_metric_count"] = len(set(missing))
    return GateResult(label, reasons, output_metrics)


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


def _as_flow(flow: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(flow)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} must contain one flow field")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be HxWx2 or 2xHxW")
    if array.shape[-1] != 2 and array.shape[0] == 2:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 2:
        raise ValueError(f"{name} must have two flow channels")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_mask(mask: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    array = _to_numpy(mask)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.shape != shape:
        raise ValueError(f"{name} must match flow spatial shape {shape}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _flow_metrics(flow: np.ndarray) -> dict[str, float | np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    sample_x = xx + flow[..., 0]
    sample_y = yy + flow[..., 1]
    out_of_bounds = (
        (sample_x < 0.0)
        | (sample_x > width - 1)
        | (sample_y < 0.0)
        | (sample_y > height - 1)
    )
    diagonal = max(1.0, float(np.hypot(height, width)))
    dx = np.linalg.norm(np.diff(flow, axis=1), axis=-1) / diagonal
    dy = np.linalg.norm(np.diff(flow, axis=0), axis=-1) / diagonal
    discontinuous_count = int(np.count_nonzero(dx > 0.02)) + int(
        np.count_nonzero(dy > 0.02)
    )
    discontinuity_total = dx.size + dy.size
    border = max(
        1,
        min(
            min(height, width),
            int(round(min(height, width) * _BACKGROUND_BORDER_FRACTION)),
        ),
    )
    background_mask = np.zeros((height, width), dtype=bool)
    background_mask[:border, :] = True
    background_mask[-border:, :] = True
    background_mask[:, :border] = True
    background_mask[:, -border:] = True
    background_motion = np.median(flow[background_mask], axis=0)
    residual = np.linalg.norm(flow - background_motion, axis=-1) / diagonal
    # A camera pan naturally sends a border band outside the source image.
    # Only OOB pixels whose motion differs from the boundary/background model
    # are evidence of unsupported foreground motion or occlusion.
    unexplained_out_of_bounds = out_of_bounds & (
        residual > _LARGE_MOTION_NORMALIZED
    )
    return {
        "out_of_bounds_ratio": float(unexplained_out_of_bounds.mean()),
        "out_of_bounds": unexplained_out_of_bounds,
        "flow_discontinuity_ratio": float(
            discontinuous_count / max(1, discontinuity_total)
        ),
        "residual": residual,
        "background_motion": float(np.linalg.norm(background_motion) / diagonal),
    }


def compute_scope_metrics(
    flow_t0: Any,
    flow_t1: Any | None = None,
    *,
    foreground_mask: Any | None = None,
    occlusion_mask: Any | None = None,
) -> ScopeMetrics:
    """Compute scope evidence while explicitly allowing global camera motion."""

    flows = [_as_flow(flow_t0, name="flow_t0")]
    if flow_t1 is not None:
        flows.append(_as_flow(flow_t1, name="flow_t1"))
    if any(item.shape != flows[0].shape for item in flows[1:]):
        raise ValueError("flow_t0 and flow_t1 must have the same shape")
    per_flow = [_flow_metrics(item) for item in flows]
    residual = np.maximum.reduce([item["residual"] for item in per_flow])
    shape = flows[0].shape[:2]

    if foreground_mask is None:
        # Boundary-median translation is a robust camera/background estimate.
        # Counting only residual motion keeps a pure global pan in scope while
        # exposing the image area occupied by independently moving parts.
        foreground_ratio = float(np.mean(residual > _LARGE_MOTION_NORMALIZED))
    else:
        foreground = _as_mask(foreground_mask, shape, name="foreground_mask") > 0.5
        if np.any(foreground):
            foreground_ratio = float(
                np.mean(residual[foreground] > _LARGE_MOTION_NORMALIZED)
            )
        else:
            foreground_ratio = 0.0

    if occlusion_mask is not None:
        occlusion = _as_mask(occlusion_mask, shape, name="occlusion_mask")
        occlusion_ratio = float(np.clip(occlusion, 0.0, 1.0).mean())
    elif len(flows) == 2:
        diagonal = max(1.0, float(np.hypot(*shape)))
        backward_inconsistency = (
            np.linalg.norm(flows[0] + flows[1], axis=-1) / diagonal
        )
        out_of_bounds = np.logical_or.reduce(
            [item["out_of_bounds"] for item in per_flow]
        )
        occlusion_ratio = float(
            np.mean(
                (backward_inconsistency > _LARGE_MOTION_NORMALIZED)
                | out_of_bounds
            )
        )
    else:
        occlusion_ratio = None

    return ScopeMetrics(
        out_of_bounds_ratio=max(float(item["out_of_bounds_ratio"]) for item in per_flow),
        flow_discontinuity_ratio=max(
            float(item["flow_discontinuity_ratio"]) for item in per_flow
        ),
        foreground_large_motion_ratio=foreground_ratio,
        occlusion_ratio=occlusion_ratio,
        unexplained_motion_ratio=float(
            np.mean(residual > _LARGE_MOTION_NORMALIZED)
        ),
        background_motion=max(float(item["background_motion"]) for item in per_flow),
    )


def evaluate_in_scope(
    metrics: ScopeMetrics | Mapping[str, Any],
    config: GateConfig | Mapping[str, Any] | Any | None = None,
) -> GateResult:
    cfg = GateConfig.from_value(config)
    values = _to_mapping(metrics)
    reject: list[str] = []
    review: list[str] = []
    missing: list[str] = []

    checks = (
        (
            "out_of_bounds_ratio",
            cfg.out_of_bounds_review_at,
            cfg.out_of_bounds_reject_at,
            "flow_out_of_bounds",
        ),
        (
            "flow_discontinuity_ratio",
            cfg.flow_discontinuity_review_at,
            cfg.flow_discontinuity_reject_at,
            "flow_discontinuous",
        ),
        (
            "foreground_large_motion_ratio",
            cfg.foreground_motion_review_at,
            cfg.foreground_motion_reject_at,
            "foreground_motion_extreme",
        ),
        (
            "occlusion_ratio",
            cfg.occlusion_review_at,
            cfg.occlusion_reject_at,
            "occlusion_extreme",
        ),
        (
            "unexplained_motion_ratio",
            cfg.unexplained_motion_review_at,
            cfg.unexplained_motion_reject_at,
            "unexplained_motion",
        ),
    )
    for name, review_at, reject_at, reason in checks:
        number = _numeric(values, name)
        if number is None:
            if name in ("out_of_bounds_ratio", "flow_discontinuity_ratio"):
                missing.append(name)
            continue
        if number >= reject_at:
            reject.append(reason)
        elif number >= review_at:
            review.append(f"possible_{reason}")

    if (
        _numeric(values, "foreground_large_motion_ratio") is None
        and _numeric(values, "occlusion_ratio") is None
        and cfg.missing_metrics_to_review
    ):
        review.append("scope_semantics_missing")

    if reject:
        label = "reject"
        reasons = tuple(dict.fromkeys(reject + review))
    elif review or (missing and cfg.missing_metrics_to_review):
        label = "review"
        reasons = tuple(dict.fromkeys(review + (["scope_metrics_missing"] if missing else [])))
    else:
        label = "accept"
        reasons = ()
    output_metrics = dict(values)
    output_metrics["missing_metric_count"] = len(set(missing))
    return GateResult(label, reasons, output_metrics)


def _gate_label(value: GateResult | Mapping[str, Any] | str | bool, name: str) -> tuple[str, tuple[str, ...]]:
    if isinstance(value, GateResult):
        return value.label, value.reasons
    if isinstance(value, str):
        if value not in ("accept", "review", "reject"):
            raise ValueError(f"{name} label must be accept, review, or reject")
        return value, ()
    if isinstance(value, bool):
        return ("accept" if value else "reject"), ()
    mapping = _to_mapping(value)
    label = str(mapping.get("label", mapping.get("decision", "review")))
    reasons = tuple(str(item) for item in mapping.get("reasons", ()))
    if label not in ("accept", "review", "reject"):
        raise ValueError(f"{name} label must be accept, review, or reject")
    return label, reasons


def decide_hard_case(
    validity: GateResult | Mapping[str, Any] | str | bool,
    in_scope: GateResult | Mapping[str, Any] | str | bool,
    p_wrong: float,
    p_solvable: float,
    config: GateConfig | Mapping[str, Any] | Any | None = None,
) -> GateResult:
    """Combine independent evidence while preserving threshold gray zones."""

    cfg = GateConfig.from_value(config)
    wrong = float(p_wrong)
    solvable = float(p_solvable)
    if not np.isfinite(wrong) or not np.isfinite(solvable):
        raise ValueError("p_wrong and p_solvable must be finite")
    wrong = float(np.clip(wrong, 0.0, 1.0))
    solvable = float(np.clip(solvable, 0.0, 1.0))
    validity_label, validity_reasons = _gate_label(validity, "validity")
    scope_label, scope_reasons = _gate_label(in_scope, "in_scope")
    metrics = {"p_wrong": wrong, "p_solvable": solvable}

    if validity_label == "reject":
        return GateResult(
            "reject",
            tuple(dict.fromkeys((*validity_reasons, "invalid_data"))),
            metrics,
        )
    if scope_label == "reject":
        return GateResult(
            "reject",
            tuple(dict.fromkeys((*scope_reasons, "out_of_scope"))),
            metrics,
        )
    if wrong < cfg.wrong_reject_below:
        return GateResult("reject", ("prediction_not_wrong",), metrics)

    review: list[str] = []
    if validity_label == "review":
        review.extend((*validity_reasons, "validity_review"))
    if scope_label == "review":
        review.extend((*scope_reasons, "scope_review"))
    if wrong < cfg.wrong_accept_at:
        review.append("wrongness_gray_zone")
    # Low solvability never claims that the source data is invalid.  It is
    # retained for review, especially when the only weak evidence is teacher.
    if solvable < cfg.solvable_accept_at:
        review.append(
            "solvability_low" if solvable < cfg.solvable_review_below else "solvability_gray_zone"
        )
    if review:
        return GateResult("review", tuple(dict.fromkeys(review)), metrics)
    return GateResult("accept", (), metrics)


# Integration-friendly aliases.
evaluate_scope = evaluate_in_scope
combine_gates = decide_hard_case
classify_sample = decide_hard_case


__all__ = [
    "FrameValidityMetrics",
    "GateConfig",
    "GateResult",
    "ScopeMetrics",
    "classify_sample",
    "combine_gates",
    "compute_scope_metrics",
    "compute_validity_metrics",
    "decide_hard_case",
    "evaluate_in_scope",
    "evaluate_scope",
    "evaluate_validity",
]
