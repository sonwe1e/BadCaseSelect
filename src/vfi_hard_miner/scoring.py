"""Local, full-reference error scoring without heavyweight vision dependencies.

The scorer deliberately keeps native-resolution evidence.  A global mean is
reported for diagnostics, but it is never the sole source of ``p_wrong``:
small high-contrast structures are retained through top-area statistics,
native connected components, and multi-scale windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import RegionBox


ArrayLike = Any
BoxLike = tuple[int, int, int, int] | RegionBox


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Thresholds for native-resolution candidate generation.

    ``min_region_pixels`` only suppresses weak isolated noise.  A component
    with at least two pixels and a sufficiently high peak is retained by the
    small-structure exception.
    """

    edge_threshold: float = 0.12
    candidate_quantile: float = 0.96
    min_region_pixels: int = 9
    max_regions: int = 8
    window_sizes: tuple[int, ...] = (16, 32, 64, 128)
    window_threshold: float = 0.10
    small_structure_peak: float = 0.50
    small_structure_min_pixels: int = 2
    component_padding: int = 2
    top_area_fractions: tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.005, 0.01)
    # Conservative HUD/subtitle prior.  It only activates when both endpoints
    # are supplied and all three cues agree: edge location, endpoint-static
    # appearance, and dense GT edges.  A floor keeps UI failures available for
    # review rather than deleting them from the candidate set.
    ui_border_fraction: float = 0.20
    ui_border_min_overlap: float = 0.50
    ui_static_threshold: float = 0.08
    ui_gt_edge_threshold: float = 0.12
    ui_edge_density_target: float = 0.12
    ui_priority_floor: float = 0.35
    ui_likelihood_threshold: float = 0.55
    non_ui_region_reserve: int = 1

    @classmethod
    def from_value(cls, value: Any | None) -> "ScoringConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        defaults = cls()

        def read(name: str, fallback: Any) -> Any:
            if isinstance(value, Mapping):
                return value.get(name, fallback)
            return getattr(value, name, fallback)

        edge_threshold = float(read("edge_threshold", defaults.edge_threshold))
        return cls(
            edge_threshold=edge_threshold,
            candidate_quantile=float(
                read("candidate_quantile", defaults.candidate_quantile)
            ),
            min_region_pixels=int(
                read("min_region_pixels", defaults.min_region_pixels)
            ),
            max_regions=int(read("max_regions", defaults.max_regions)),
            window_sizes=tuple(read("window_sizes", defaults.window_sizes)),
            window_threshold=float(
                read("window_threshold", max(defaults.window_threshold, edge_threshold * 0.75))
            ),
            small_structure_peak=float(
                read("small_structure_peak", defaults.small_structure_peak)
            ),
            small_structure_min_pixels=int(
                read(
                    "small_structure_min_pixels",
                    defaults.small_structure_min_pixels,
                )
            ),
            component_padding=int(
                read("component_padding", defaults.component_padding)
            ),
            top_area_fractions=tuple(
                read("top_area_fractions", defaults.top_area_fractions)
            ),
            ui_border_fraction=float(
                read("ui_border_fraction", defaults.ui_border_fraction)
            ),
            ui_border_min_overlap=float(
                read("ui_border_min_overlap", defaults.ui_border_min_overlap)
            ),
            ui_static_threshold=float(
                read("ui_static_threshold", defaults.ui_static_threshold)
            ),
            ui_gt_edge_threshold=float(
                read("ui_gt_edge_threshold", defaults.ui_gt_edge_threshold)
            ),
            ui_edge_density_target=float(
                read("ui_edge_density_target", defaults.ui_edge_density_target)
            ),
            ui_priority_floor=float(
                read("ui_priority_floor", defaults.ui_priority_floor)
            ),
            ui_likelihood_threshold=float(
                read("ui_likelihood_threshold", defaults.ui_likelihood_threshold)
            ),
            non_ui_region_reserve=int(
                read("non_ui_region_reserve", defaults.non_ui_region_reserve)
            ),
        )


@dataclass(frozen=True, slots=True)
class ErrorMaps:
    """Native-resolution maps in approximately ``[0, 1]``."""

    rgb: np.ndarray
    luminance: np.ndarray
    sobel_gt: np.ndarray
    sobel_prediction: np.ndarray
    gt_only_edges: np.ndarray
    pred_only_edges: np.ndarray
    structure: np.ndarray

    @property
    def edge_difference(self) -> np.ndarray:
        return np.maximum(self.gt_only_edges, self.pred_only_edges)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb,
            "luminance": self.luminance,
            "sobel_gt": self.sobel_gt,
            "sobel_prediction": self.sobel_prediction,
            "gt_only_edges": self.gt_only_edges,
            "pred_only_edges": self.pred_only_edges,
            "structure": self.structure,
        }


@dataclass(frozen=True, slots=True)
class LocalScoreResult:
    maps: ErrorMaps
    regions: tuple[RegionBox, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    p_wrong: float = 0.0
    mining_p_wrong: float = 0.0

    def to_dict(self, *, include_maps: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "regions": [region.to_dict() for region in self.regions],
            "metrics": dict(self.metrics),
            "p_wrong": float(self.p_wrong),
            "mining_p_wrong": float(self.mining_p_wrong),
        }
        if include_maps:
            payload["maps"] = self.maps.as_dict()
        return payload


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Convert ndarray or tensor-like objects without importing torch."""

    if isinstance(value, np.ndarray):
        return value
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        return np.asarray(numpy_method())
    return np.asarray(current)


def as_rgb01(value: ArrayLike, *, name: str = "image") -> np.ndarray:
    """Return a finite ``H x W x 3`` float32 image in ``[0, 1]``.

    Single-item NCHW/NHWC batches, CHW/HWC arrays, and grayscale arrays are
    accepted.  Values outside the normalized-image contract fail fast rather
    than being silently re-scaled.
    """

    array = _to_numpy(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} must be a single image, got batch {array.shape[0]}")
        array = array[0]
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"{name} must have 2, 3, or single-batch 4 dimensions")

    if array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if array.size and (float(array.min()) < -1e-4 or float(array.max()) > 1.0001):
        raise ValueError(f"{name} must be normalized to [0, 1]")
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def luminance(image: ArrayLike) -> np.ndarray:
    rgb = as_rgb01(image)
    return np.asarray(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        dtype=np.float32,
    )


def sobel_magnitude(image_or_luma: ArrayLike) -> np.ndarray:
    """Compute a normalized Sobel magnitude with NumPy only."""

    array = _to_numpy(image_or_luma)
    if array.ndim != 2:
        array = luminance(array)
    else:
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("luminance contains NaN or infinite values")
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]
    gx = (top_right + 2.0 * right + bottom_right) - (
        top_left + 2.0 * left + bottom_left
    )
    gy = (bottom_left + 2.0 * bottom + bottom_right) - (
        top_left + 2.0 * top + top_right
    )
    return np.asarray(np.clip(np.hypot(gx, gy) / 4.0, 0.0, 1.0), dtype=np.float32)


def compute_error_maps(prediction: ArrayLike, gt: ArrayLike) -> ErrorMaps:
    """Compute aligned RGB, luminance, and directional edge evidence."""

    pred = as_rgb01(prediction, name="prediction")
    reference = as_rgb01(gt, name="gt")
    if pred.shape != reference.shape:
        raise ValueError(
            f"prediction and gt must have the same shape, got {pred.shape} and {reference.shape}"
        )
    absolute = np.abs(pred - reference)
    rgb_error = np.asarray(absolute.mean(axis=-1), dtype=np.float32)
    pred_luma = (
        0.2126 * pred[..., 0] + 0.7152 * pred[..., 1] + 0.0722 * pred[..., 2]
    )
    gt_luma = (
        0.2126 * reference[..., 0]
        + 0.7152 * reference[..., 1]
        + 0.0722 * reference[..., 2]
    )
    luma_error = np.asarray(np.abs(pred_luma - gt_luma), dtype=np.float32)
    gt_edges = sobel_magnitude(gt_luma)
    pred_edges = sobel_magnitude(pred_luma)
    gt_only = np.asarray(np.maximum(gt_edges - pred_edges, 0.0), dtype=np.float32)
    pred_only = np.asarray(np.maximum(pred_edges - gt_edges, 0.0), dtype=np.float32)
    edge_error = np.maximum(gt_only, pred_only)

    # Max fusion intentionally preserves a thin high-confidence structure.
    structure = np.maximum.reduce(
        (
            rgb_error,
            luma_error,
            np.clip(edge_error * 1.25, 0.0, 1.0),
        )
    ).astype(np.float32, copy=False)
    return ErrorMaps(
        rgb=rgb_error,
        luminance=luma_error,
        sobel_gt=gt_edges,
        sobel_prediction=pred_edges,
        gt_only_edges=gt_only,
        pred_only_edges=pred_only,
        structure=structure,
    )


def top_area_mean(values: ArrayLike, fraction: float) -> float:
    """Mean of the largest area fraction, always retaining at least one pixel."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    flat = np.asarray(_to_numpy(values), dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    if not np.isfinite(flat).all():
        raise ValueError("values contain NaN or infinite values")
    count = max(1, int(np.ceil(flat.size * float(fraction))))
    if count >= flat.size:
        return float(flat.mean())
    start = flat.size - count
    return float(np.partition(flat, start)[start:].mean())


def _fraction_name(fraction: float) -> str:
    percent = fraction * 100.0
    text = f"{percent:.4f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"top_{text}pct_mean"


def summarize_error_map(
    values: ArrayLike,
    *,
    top_area_fractions: Sequence[float] = (0.0001, 0.0005, 0.001, 0.005, 0.01),
) -> dict[str, float]:
    array = np.asarray(_to_numpy(values), dtype=np.float32)
    if array.size == 0:
        return {"mean": 0.0, "max": 0.0, "q95": 0.0, "q99": 0.0, "q999": 0.0}
    if not np.isfinite(array).all():
        raise ValueError("error map contains NaN or infinite values")
    flat = array.reshape(-1)
    metrics = {
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "q999": float(np.quantile(flat, 0.999)),
    }
    for fraction in top_area_fractions:
        metrics[_fraction_name(float(fraction))] = top_area_mean(flat, float(fraction))
    return metrics


def robust_local_score(values: ArrayLike) -> float:
    """A local severity score that cannot be dominated by background area."""

    array = np.asarray(_to_numpy(values), dtype=np.float32)
    if array.size == 0:
        return 0.0
    peak = float(array.max())
    score = max(
        float(array.mean()),
        0.90 * top_area_mean(array, min(0.10, 1.0)),
        0.95 * top_area_mean(array, min(0.01, 1.0)),
        0.85 * peak,
    )
    return float(np.clip(score, 0.0, 1.0))


def _normalize_box(box: BoxLike, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if isinstance(box, RegionBox):
        raw = (box.x0, box.y0, box.x1, box.y1)
    else:
        raw = tuple(int(item) for item in box)
        if len(raw) != 4:
            raise ValueError("box must contain x0, y0, x1, y1")
    height, width = shape
    x0 = max(0, min(width, int(raw[0])))
    y0 = max(0, min(height, int(raw[1])))
    x1 = max(0, min(width, int(raw[2])))
    y1 = max(0, min(height, int(raw[3])))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must have positive area inside the image")
    return x0, y0, x1, y1


def score_region(values: ArrayLike, box: BoxLike | None = None) -> float:
    array = np.asarray(_to_numpy(values), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("score_region expects a 2-D error map")
    if box is not None:
        x0, y0, x1, y1 = _normalize_box(box, array.shape)
        array = array[y0:y1, x0:x1]
    return robust_local_score(array)


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        label = len(self.parent)
        self.parent.append(label)
        self.rank.append(0)
        return label

    def find(self, label: int) -> int:
        while self.parent[label] != label:
            self.parent[label] = self.parent[self.parent[label]]
            label = self.parent[label]
        return label

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(slots=True)
class _Component:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    total: float
    peak: float
    perimeter: float


def _extract_components(mask: np.ndarray, values: np.ndarray) -> list[_Component]:
    """Eight-connected components using row runs instead of per-pixel Python BFS."""

    if mask.shape != values.shape or mask.ndim != 2:
        raise ValueError("mask and values must be equally-shaped 2-D arrays")
    dsu = _DisjointSet()
    all_runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    for y, row in enumerate(mask):
        padded = np.concatenate((np.array([False]), row, np.array([False])))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        current: list[tuple[int, int, int]] = []
        previous_start = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            label = dsu.add()
            while previous_start < len(previous) and previous[previous_start][1] < x0 - 1:
                previous_start += 1
            index = previous_start
            while index < len(previous) and previous[index][0] <= x1:
                px0, px1, previous_label = previous[index]
                if px1 >= x0 - 1 and px0 <= x1:
                    dsu.union(label, previous_label)
                index += 1
            current.append((x0, x1, label))
            all_runs.append((y, x0, x1, label))
        previous = current

    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for y, x0, x1, label in all_runs:
        grouped.setdefault(dsu.find(label), []).append((y, x0, x1))

    components: list[_Component] = []
    for runs in grouped.values():
        min_x = min(run[1] for run in runs)
        max_x = max(run[2] for run in runs)
        min_y = min(run[0] for run in runs)
        max_y = max(run[0] for run in runs) + 1
        area = 0
        total = 0.0
        peak = 0.0
        perimeter = 0.0
        by_row: dict[int, list[tuple[int, int]]] = {}
        for y, x0, x1 in runs:
            run_values = values[y, x0:x1]
            length = x1 - x0
            area += length
            total += float(run_values.sum())
            if run_values.size:
                peak = max(peak, float(run_values.max()))
            perimeter += 2.0 + 2.0 * length
            by_row.setdefault(y, []).append((x0, x1))
        for y, row_runs in by_row.items():
            previous_runs = by_row.get(y - 1, ())
            for x0, x1 in row_runs:
                for px0, px1 in previous_runs:
                    overlap = max(0, min(x1, px1) - max(x0, px0))
                    perimeter -= 2.0 * overlap
        components.append(
            _Component(min_x, min_y, max_x, max_y, area, total, peak, perimeter)
        )
    return components


def _padded_box(
    box: tuple[int, int, int, int],
    padding: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    height, width = shape
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _integral_image(values: np.ndarray) -> np.ndarray:
    integral = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = values.cumsum(axis=0, dtype=np.float64).cumsum(
        axis=1, dtype=np.float64
    )
    return integral


def _window_grid(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = values.shape
    size_y = min(max(1, int(window)), height)
    size_x = min(max(1, int(window)), width)
    stride_y = max(1, size_y // 2)
    stride_x = max(1, size_x // 2)
    ys = np.arange(0, max(1, height - size_y + 1), stride_y, dtype=np.int32)
    xs = np.arange(0, max(1, width - size_x + 1), stride_x, dtype=np.int32)
    if ys[-1] != height - size_y:
        ys = np.append(ys, height - size_y)
    if xs[-1] != width - size_x:
        xs = np.append(xs, width - size_x)
    y1 = ys + size_y
    x1 = xs + size_x
    integral = _integral_image(values)
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[ys[:, None], x1[None, :]]
        - integral[y1[:, None], xs[None, :]]
        + integral[ys[:, None], xs[None, :]]
    )
    means = np.asarray(sums / float(size_y * size_x), dtype=np.float32)
    return means, ys, xs, y1, x1


def _region_from_box(
    values: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    source: str,
    extra_metrics: Mapping[str, float] | None = None,
) -> RegionBox:
    x0, y0, x1, y1 = box
    crop = values[y0:y1, x0:x1]
    stats = summarize_error_map(crop, top_area_fractions=(0.01, 0.05, 0.10))
    metrics = {f"local_{key}": value for key, value in stats.items()}
    metrics["area"] = float((x1 - x0) * (y1 - y0))
    metrics["source_native"] = 1.0 if source == "native_component" else 0.0
    if extra_metrics:
        metrics.update({key: float(value) for key, value in extra_metrics.items()})
    return RegionBox(x0, y0, x1, y1, robust_local_score(crop), metrics)


def _intersection_metrics(left: RegionBox, right: RegionBox) -> tuple[float, float]:
    ix0 = max(left.x0, right.x0)
    iy0 = max(left.y0, right.y0)
    ix1 = min(left.x1, right.x1)
    iy1 = min(left.y1, right.y1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    left_area = (left.x1 - left.x0) * (left.y1 - left.y0)
    right_area = (right.x1 - right.x0) * (right.y1 - right.y0)
    union = left_area + right_area - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(left_area, right_area) if intersection else 0.0
    return float(iou), float(containment)


def _region_priority(region: RegionBox) -> float:
    return float(np.clip(region.metrics.get("priority_weight", 1.0), 0.0, 1.0))


def _deduplicate_regions(
    regions: Sequence[RegionBox],
    max_regions: int,
    *,
    non_ui_region_reserve: int,
    ui_likelihood_threshold: float,
) -> list[RegionBox]:
    # Native components keep the tight support of thin structures.  Process
    # them before broader window proposals so containment de-duplication does
    # not replace a weapon/limb-sized region with a large generic tile.
    ranked = sorted(
        regions,
        key=lambda item: (
            -item.metrics.get("source_native", 0.0),
            -(item.score * _region_priority(item)),
            -_region_priority(item),
            -item.score,
            item.y0,
            item.x0,
            item.y1,
            item.x1,
        ),
    )

    def overlaps_existing(region: RegionBox, existing_regions: Sequence[RegionBox]) -> bool:
        for existing in existing_regions:
            iou, containment = _intersection_metrics(region, existing)
            if iou >= 0.50 or containment >= 0.85:
                return True
        return False

    if max_regions <= 0:
        kept: list[RegionBox] = []
        for region in ranked:
            if not overlaps_existing(region, kept):
                kept.append(region)
        return kept

    reserve = min(max(0, int(non_ui_region_reserve)), max_regions)
    threshold = float(np.clip(ui_likelihood_threshold, 0.0, 1.0))
    selected: list[RegionBox] = []
    if reserve > 0:
        for native_only in (True, False):
            for region in ranked:
                is_native = region.metrics.get("source_native", 0.0) >= 0.5
                if native_only != is_native:
                    continue
                if float(region.metrics.get("ui_likelihood", 0.0)) >= threshold:
                    continue
                if not is_native:
                    # A central low-contrast structure may only produce a
                    # window proposal.  Admit compact, fully non-UI windows,
                    # while keeping broad HUD/background windows out of the
                    # reserved slot.
                    if _region_priority(region) < 0.95:
                        continue
                    area_fraction = float(
                        region.metrics.get("region_area_fraction", 1.0)
                    )
                    if area_fraction > 0.35:
                        continue
                if not overlaps_existing(region, selected):
                    selected.append(region)
                    if len(selected) >= reserve:
                        break
            if len(selected) >= reserve:
                break
    if len(selected) >= max_regions:
        return selected
    selected_ids = {id(region) for region in selected}
    for region in ranked:
        if id(region) in selected_ids:
            continue
        if overlaps_existing(region, selected):
            continue
        selected.append(region)
        selected_ids.add(id(region))
        if len(selected) >= max_regions:
            break
    return selected


def _border_overlap(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    border_fraction: float,
) -> float:
    """Return the fraction of a box that lies in any outer image band."""

    x0, y0, x1, y1 = box
    height, width = shape
    fraction = float(np.clip(border_fraction, 0.0, 0.49))
    band_x = int(np.ceil(width * fraction))
    band_y = int(np.ceil(height * fraction))
    inner_x0, inner_x1 = band_x, max(band_x, width - band_x)
    inner_y0, inner_y1 = band_y, max(band_y, height - band_y)
    inner_width = max(0, min(x1, inner_x1) - max(x0, inner_x0))
    inner_height = max(0, min(y1, inner_y1) - max(y0, inner_y0))
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(np.clip(1.0 - (inner_width * inner_height) / area, 0.0, 1.0))


def _ui_priority_metrics(
    box: tuple[int, int, int, int],
    *,
    gt_edge_map: np.ndarray | None,
    endpoint_change_map: np.ndarray | None,
    cfg: ScoringConfig,
) -> dict[str, float]:
    metrics = {
        "ui_context_available": 0.0,
        "border_overlap": 0.0,
        "border_likelihood": 0.0,
        "endpoint_change_mean": -1.0,
        "endpoint_change_q90": -1.0,
        "endpoint_static_likelihood": 0.0,
        "gt_edge_density": 0.0,
        "gt_edge_density_likelihood": 0.0,
        "region_area_fraction": -1.0,
        "ui_likelihood": 0.0,
        "priority_weight": 1.0,
    }
    if gt_edge_map is None or endpoint_change_map is None:
        return metrics
    x0, y0, x1, y1 = box
    endpoint_crop = endpoint_change_map[y0:y1, x0:x1]
    edge_crop = gt_edge_map[y0:y1, x0:x1]
    if not endpoint_crop.size or not edge_crop.size:
        return metrics

    border_overlap = _border_overlap(box, gt_edge_map.shape, cfg.ui_border_fraction)
    border_floor = float(np.clip(cfg.ui_border_min_overlap, 0.0, 0.99))
    border_likelihood = float(
        np.clip((border_overlap - border_floor) / (1.0 - border_floor), 0.0, 1.0)
    )
    endpoint_mean = float(endpoint_crop.mean())
    endpoint_q90 = float(np.quantile(endpoint_crop, 0.90))
    endpoint_motion = 0.60 * endpoint_mean + 0.40 * endpoint_q90
    static_threshold = max(float(cfg.ui_static_threshold), 1e-6)
    static_likelihood = float(np.clip(1.0 - endpoint_motion / static_threshold, 0.0, 1.0))
    edge_density = float(
        np.count_nonzero(edge_crop >= float(cfg.ui_gt_edge_threshold)) / edge_crop.size
    )
    region_area_fraction = float(
        ((x1 - x0) * (y1 - y0)) / max(1, gt_edge_map.shape[0] * gt_edge_map.shape[1])
    )
    density_target = max(float(cfg.ui_edge_density_target), 1e-6)
    density_likelihood = float(np.clip(edge_density / density_target, 0.0, 1.0))
    ui_likelihood = float(
        np.clip(border_likelihood * static_likelihood * density_likelihood, 0.0, 1.0)
    )
    priority_floor = float(np.clip(cfg.ui_priority_floor, 0.0, 1.0))
    priority_weight = float(1.0 - ui_likelihood * (1.0 - priority_floor))
    metrics.update(
        {
            "ui_context_available": 1.0,
            "border_overlap": border_overlap,
            "border_likelihood": border_likelihood,
            "endpoint_change_mean": endpoint_mean,
            "endpoint_change_q90": endpoint_q90,
            "endpoint_static_likelihood": static_likelihood,
            "gt_edge_density": edge_density,
            "gt_edge_density_likelihood": density_likelihood,
            "region_area_fraction": region_area_fraction,
            "ui_likelihood": ui_likelihood,
            "priority_weight": priority_weight,
        }
    )
    return metrics


def _with_ui_priority(
    region: RegionBox,
    *,
    gt_edge_map: np.ndarray | None,
    endpoint_change_map: np.ndarray | None,
    cfg: ScoringConfig,
) -> RegionBox:
    metrics = dict(region.metrics)
    metrics.update(
        _ui_priority_metrics(
            (region.x0, region.y0, region.x1, region.y1),
            gt_edge_map=gt_edge_map,
            endpoint_change_map=endpoint_change_map,
            cfg=cfg,
        )
    )
    metrics["priority_score"] = float(region.score * metrics["priority_weight"])
    return RegionBox(region.x0, region.y0, region.x1, region.y1, region.score, metrics)


def _native_region_candidates(
    mask: np.ndarray,
    values: np.ndarray,
    support: np.ndarray | None,
    cfg: ScoringConfig,
    *,
    candidate_threshold: float,
) -> list[RegionBox]:
    candidates: list[RegionBox] = []
    for component in _extract_components(mask, values):
        support_pixels = component.area
        if support is not None:
            support_pixels = int(
                np.count_nonzero(
                    support[component.y0 : component.y1, component.x0 : component.x1]
                    >= cfg.edge_threshold
                )
            )
        strong_small = (
            component.area >= cfg.small_structure_min_pixels
            and support_pixels >= cfg.small_structure_min_pixels
            and component.peak >= cfg.small_structure_peak
        )
        if support is not None and support_pixels < cfg.small_structure_min_pixels:
            continue
        if component.area < cfg.min_region_pixels and not strong_small:
            continue
        box = _padded_box(
            (component.x0, component.y0, component.x1, component.y1),
            cfg.component_padding,
            values.shape,
        )
        candidates.append(
            _region_from_box(
                values,
                box,
                source="native_component",
                extra_metrics={
                    "component_area": float(component.area),
                    "component_mean": component.total / max(1, component.area),
                    "component_peak": component.peak,
                    "component_energy": component.total,
                    "component_boundary_length": component.perimeter,
                    "small_structure_exception": float(strong_small),
                    "support_pixels": float(support_pixels),
                    "candidate_threshold": float(candidate_threshold),
                },
            )
        )
    return candidates


def _find_candidate_regions_and_raw_score(
    error_map: ArrayLike,
    config: ScoringConfig | Mapping[str, Any] | Any | None = None,
    *,
    support_map: ArrayLike | None = None,
    gt_edge_map: ArrayLike | None = None,
    endpoint_change_map: ArrayLike | None = None,
) -> tuple[tuple[RegionBox, ...], float]:
    """Generate prioritized regions plus context-independent raw severity."""

    cfg = ScoringConfig.from_value(config)
    values = np.asarray(_to_numpy(error_map), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("error_map must be 2-D")
    if values.size == 0 or not np.isfinite(values).all():
        if values.size and not np.isfinite(values).all():
            raise ValueError("error_map contains NaN or infinite values")
        return (), 0.0
    if float(values.max()) <= 0.0:
        return (), 0.0
    support = None
    if support_map is not None:
        support = np.asarray(_to_numpy(support_map), dtype=np.float32)
        if support.shape != values.shape:
            raise ValueError("support_map must match error_map")
    gt_edges = None
    if gt_edge_map is not None:
        gt_edges = np.asarray(_to_numpy(gt_edge_map), dtype=np.float32)
        if gt_edges.shape != values.shape:
            raise ValueError("gt_edge_map must match error_map")
    endpoint_change = None
    if endpoint_change_map is not None:
        endpoint_change = np.asarray(_to_numpy(endpoint_change_map), dtype=np.float32)
        if endpoint_change.shape != values.shape:
            raise ValueError("endpoint_change_map must match error_map")
    if (gt_edges is None) != (endpoint_change is None):
        raise ValueError("GT edges and endpoint change must be provided together")

    quantile_threshold = float(np.quantile(values, float(cfg.candidate_quantile)))
    adaptive = max(float(cfg.edge_threshold), quantile_threshold)
    native_mask = values >= adaptive
    native_threshold_floor = adaptive
    # Dense edge HUDs can dominate the global quantile and hide a weaker
    # central character/weapon component.  Recover the non-border core for
    # every call, irrespective of endpoint availability, so raw GT evidence
    # remains exactly context-independent.
    height, width = values.shape
    fraction = float(np.clip(cfg.ui_border_fraction, 0.0, 0.49))
    band_x = int(np.ceil(width * fraction))
    band_y = int(np.ceil(height * fraction))
    x0, x1 = band_x, max(band_x, width - band_x)
    y0, y1 = band_y, max(band_y, height - band_y)
    interior = values[y0:y1, x0:x1]
    if interior.size:
        interior_threshold = max(
            float(cfg.edge_threshold),
            float(np.quantile(interior, float(cfg.candidate_quantile))),
        )
        native_threshold_floor = min(native_threshold_floor, interior_threshold)
        native_mask[y0:y1, x0:x1] |= interior >= interior_threshold
    raw_native_candidates = _native_region_candidates(
        native_mask,
        values,
        support,
        cfg,
        candidate_threshold=native_threshold_floor,
    )
    # Candidate generation is deliberately independent of endpoint/UI context.
    # UI evidence may reprioritize candidates but must never change raw GT
    # wrongness or make the candidate pool depend on whether endpoints exist.
    contextual_native_candidates = raw_native_candidates

    window_candidates: list[RegionBox] = []
    seen_sizes: set[int] = set()
    for requested_size in cfg.window_sizes:
        size = min(max(1, int(requested_size)), min(values.shape))
        if size in seen_sizes:
            continue
        seen_sizes.add(size)
        means, ys, xs, y1s, x1s = _window_grid(values, size)
        qualified = means >= float(cfg.window_threshold)
        for component in _extract_components(qualified, means):
            grid_x0, grid_y0 = component.x0, component.y0
            grid_x1, grid_y1 = component.x1, component.y1
            box = (
                int(xs[grid_x0]),
                int(ys[grid_y0]),
                int(x1s[grid_x1 - 1]),
                int(y1s[grid_y1 - 1]),
            )
            window_candidates.append(
                _region_from_box(
                    values,
                    box,
                    source=f"window_{size}",
                    extra_metrics={
                        "window_size": float(size),
                        "window_grid_peak": component.peak,
                        "window_grid_mean": component.total / max(1, component.area),
                    },
                )
            )

    raw_candidates = (*raw_native_candidates, *window_candidates)
    raw_regions = _deduplicate_regions(
        raw_candidates,
        cfg.max_regions,
        non_ui_region_reserve=0,
        ui_likelihood_threshold=cfg.ui_likelihood_threshold,
    )
    raw_region_local = max((region.score for region in raw_regions), default=0.0)
    candidates = (*contextual_native_candidates, *window_candidates)
    prioritized = [
        _with_ui_priority(
            region,
            gt_edge_map=gt_edges,
            endpoint_change_map=endpoint_change,
            cfg=cfg,
        )
        for region in candidates
    ]
    regions = tuple(
        _deduplicate_regions(
            prioritized,
            cfg.max_regions,
            non_ui_region_reserve=cfg.non_ui_region_reserve,
            ui_likelihood_threshold=cfg.ui_likelihood_threshold,
        )
    )
    return regions, float(raw_region_local)


def find_candidate_regions(
    error_map: ArrayLike,
    config: ScoringConfig | Mapping[str, Any] | Any | None = None,
    *,
    support_map: ArrayLike | None = None,
    gt_edge_map: ArrayLike | None = None,
    endpoint_change_map: ArrayLike | None = None,
) -> tuple[RegionBox, ...]:
    """Generate native and multi-scale regions without a Top-K sample policy."""

    regions, _ = _find_candidate_regions_and_raw_score(
        error_map,
        config,
        support_map=support_map,
        gt_edge_map=gt_edge_map,
        endpoint_change_map=endpoint_change_map,
    )
    return regions


def score_local_errors(
    prediction: ArrayLike,
    gt: ArrayLike,
    config: ScoringConfig | Mapping[str, Any] | Any | None = None,
    *,
    img0: ArrayLike | None = None,
    img1: ArrayLike | None = None,
) -> LocalScoreResult:
    """Score a prediction against GT and retain localized evidence."""

    cfg = ScoringConfig.from_value(config)
    maps = compute_error_maps(prediction, gt)
    if (img0 is None) != (img1 is None):
        raise ValueError("img0 and img1 must be supplied together for UI prioritization")
    endpoint_change: np.ndarray | None = None
    if img0 is not None and img1 is not None:
        first = as_rgb01(img0, name="img0")
        last = as_rgb01(img1, name="img1")
        expected_shape = (*maps.structure.shape, 3)
        if first.shape != expected_shape or last.shape != expected_shape:
            raise ValueError(
                f"img0/img1 must match GT shape {expected_shape}, got "
                f"{first.shape} and {last.shape}"
            )
        endpoint_change = np.asarray(np.abs(first - last).mean(axis=-1), dtype=np.float32)
    regions, raw_region_local = _find_candidate_regions_and_raw_score(
        maps.structure,
        cfg,
        support_map=maps.rgb,
        gt_edge_map=maps.sobel_gt if endpoint_change is not None else None,
        endpoint_change_map=endpoint_change,
    )
    metrics: dict[str, float] = {}
    for map_name, values in (
        ("rgb", maps.rgb),
        ("luminance", maps.luminance),
        ("edge", maps.edge_difference),
        ("gt_only_edge", maps.gt_only_edges),
        ("pred_only_edge", maps.pred_only_edges),
        ("structure", maps.structure),
    ):
        for name, value in summarize_error_map(
            values, top_area_fractions=cfg.top_area_fractions
        ).items():
            metrics[f"{map_name}_{name}"] = value

    for size in cfg.window_sizes:
        clipped_size = min(max(1, int(size)), min(maps.structure.shape))
        window_means, *_ = _window_grid(maps.structure, clipped_size)
        metrics[f"window_{clipped_size}_mean_max"] = float(window_means.max())

    global_local = max(
        metrics.get("structure_top_0_01pct_mean", 0.0),
        metrics.get("structure_top_0_05pct_mean", 0.0),
        metrics.get("structure_top_0_1pct_mean", 0.0),
        metrics.get("structure_q999", 0.0),
    )
    if raw_region_local > 0.0:
        p_wrong_source = max(global_local, raw_region_local)
    else:
        # A lone hot pixel may dominate a top-area statistic on a small image.
        # Without a retained native component, fall back to evidence that has
        # spatial support instead of re-introducing the pixel noise we removed.
        window_support = max(
            (
                value
                for name, value in metrics.items()
                if name.startswith("window_") and name.endswith("_mean_max")
            ),
            default=0.0,
        )
        p_wrong_source = max(metrics.get("structure_q99", 0.0), window_support)
    p_wrong = float(np.clip(p_wrong_source, 0.0, 1.0))
    if regions and raw_region_local > 0.0:
        weighted_region_local = max(
            region.score * _region_priority(region) for region in regions
        )
        priority_ratio = float(
            np.clip(
                weighted_region_local / max(raw_region_local, 1e-8),
                0.0,
                1.0,
            )
        )
        mining_p_wrong = float(np.clip(p_wrong * priority_ratio, 0.0, 1.0))
    else:
        mining_p_wrong = p_wrong
    metrics["p_wrong"] = p_wrong
    metrics["mining_p_wrong"] = mining_p_wrong
    metrics["raw_region_local"] = float(raw_region_local)
    metrics["region_count"] = float(len(regions))
    metrics["ui_context_available"] = float(endpoint_change is not None)
    metrics["max_ui_likelihood"] = max(
        (region.metrics.get("ui_likelihood", 0.0) for region in regions),
        default=0.0,
    )
    metrics["min_priority_weight"] = min(
        (region.metrics.get("priority_weight", 1.0) for region in regions),
        default=1.0,
    )
    return LocalScoreResult(
        maps=maps,
        regions=regions,
        metrics=metrics,
        p_wrong=p_wrong,
        mining_p_wrong=mining_p_wrong,
    )


# Stable descriptive aliases for callers that prefer more explicit names.
compute_local_error_maps = compute_error_maps
generate_candidate_regions = find_candidate_regions
score_prediction = score_local_errors
score_pair = score_local_errors


__all__ = [
    "ErrorMaps",
    "LocalScoreResult",
    "ScoringConfig",
    "as_rgb01",
    "compute_error_maps",
    "compute_local_error_maps",
    "find_candidate_regions",
    "generate_candidate_regions",
    "luminance",
    "robust_local_score",
    "score_local_errors",
    "score_pair",
    "score_prediction",
    "score_region",
    "sobel_magnitude",
    "summarize_error_map",
    "top_area_mean",
]
