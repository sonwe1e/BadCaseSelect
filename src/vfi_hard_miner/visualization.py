"""CPU diagnostic visualizations for selected local failures."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _rgb_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in (1, 3, 4):
        raise ValueError(f"invalid visualization image shape: {value.shape}")
    if value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating):
        value = np.rint(np.clip(value, 0.0, 1.0) * 255.0)
    return np.asarray(value, dtype=np.uint8)


def colorize_error(error_map: np.ndarray) -> np.ndarray:
    """Colorize a normalized error map with a perceptually ordered blue-red ramp."""
    value = np.clip(np.asarray(error_map, dtype=np.float32), 0.0, 1.0)
    if value.ndim == 3:
        value = value.mean(axis=2)
    red = np.clip(1.8 * value, 0.0, 1.0)
    green = np.clip(1.8 - 3.0 * np.abs(value - 0.5), 0.0, 1.0)
    blue = np.clip(1.8 * (1.0 - value), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def colorize_flow(flow: np.ndarray) -> np.ndarray:
    """Visualize HxWx2 pixel displacement using angle and robust magnitude."""
    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 2:
        raise ValueError(f"flow must be HxWx2, got {value.shape}")
    angle = (np.arctan2(value[..., 1], value[..., 0]) + np.pi) / (2.0 * np.pi)
    magnitude = np.linalg.norm(value, axis=2)
    scale = max(float(np.quantile(magnitude, 0.99)), 1e-6)
    saturation = np.clip(magnitude / scale, 0.0, 1.0)
    hue = angle * 6.0
    sector = np.floor(hue).astype(np.int32) % 6
    fraction = hue - np.floor(hue)
    p = 1.0 - saturation
    q = 1.0 - saturation * fraction
    t = 1.0 - saturation * (1.0 - fraction)
    choices = (
        (np.ones_like(hue), t, p),
        (q, np.ones_like(hue), p),
        (p, np.ones_like(hue), t),
        (p, q, np.ones_like(hue)),
        (t, p, np.ones_like(hue)),
        (np.ones_like(hue), p, q),
    )
    result = np.zeros((*value.shape[:2], 3), dtype=np.float32)
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, component in enumerate(channels):
            result[..., channel][mask] = component[mask]
    return result


def _fit_panel(array: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(_rgb_uint8(array))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _normalized_heatmaps(*maps: np.ndarray) -> tuple[np.ndarray, ...]:
    values = [np.asarray(item, dtype=np.float32) for item in maps]
    finite_parts = [
        item[np.isfinite(item)].reshape(-1) for item in values if item.size
    ]
    if not finite_parts:
        raise ValueError("CGVQM heatmaps do not contain finite values")
    finite = np.concatenate(finite_parts)
    if not finite.size:
        raise ValueError("CGVQM heatmaps do not contain finite values")
    scale = max(float(np.quantile(np.maximum(finite, 0.0), 0.995)), 1e-8)
    return tuple(np.clip(np.nan_to_num(item, nan=0.0) / scale, 0.0, 1.0) for item in values)


def _region_boxes(regions: Iterable[Any]) -> tuple[tuple[int, int, int, int], ...]:
    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        getter = (
            (lambda key: region[key])
            if isinstance(region, dict)
            else (lambda key: getattr(region, key))
        )
        try:
            boxes.append(
                tuple(int(getter(key)) for key in ("x0", "y0", "x1", "y1"))
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return tuple(boxes)


def _labeled_panel(
    array: np.ndarray,
    *,
    label: str,
    width: int,
    image_height: int,
    boxes: Iterable[tuple[int, int, int, int]] = (),
    source_shape: tuple[int, int] | None = None,
) -> Image.Image:
    label_height = 28
    image = _fit_panel(array, width, image_height)
    if source_shape is not None:
        source_height, source_width = source_shape
        draw = ImageDraw.Draw(image)
        scale_x = width / max(1, source_width)
        scale_y = image_height / max(1, source_height)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle(
                (
                    round(x0 * scale_x),
                    round(y0 * scale_y),
                    round(x1 * scale_x),
                    round(y1 * scale_y),
                ),
                outline=(255, 224, 64),
                width=max(1, width // 240),
            )
    panel = Image.new("RGB", (width, label_height + image_height), (14, 18, 24))
    panel.paste(image, (0, label_height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), label, fill=(244, 247, 252))
    return panel


def make_diagnostic_grid(
    img0: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    img1: np.ndarray,
    *,
    error_map: np.ndarray,
    gt_only_edge: np.ndarray | None = None,
    pred_only_edge: np.ndarray | None = None,
    flow_t0: np.ndarray | None = None,
    flow_t1: np.ndarray | None = None,
    cgvqm_center: np.ndarray | None = None,
    cgvqm_temporal: np.ndarray | None = None,
    cgvqm_fused: np.ndarray | None = None,
    mask0: np.ndarray | None = None,
    mask1: np.ndarray | None = None,
    regions: Iterable[Any] = (),
    labels: Iterable[str] = (),
    panel_width: int = 320,
) -> np.ndarray:
    """Build the fixed two-row by five-column diagnostic contract."""

    base = np.asarray(img0)
    if base.ndim != 3:
        raise ValueError("img0 must be HxWxC")
    source_shape = (int(base.shape[0]), int(base.shape[1]))
    image_height = max(1, round(source_shape[0] * panel_width / source_shape[1]))
    boxes = _region_boxes(regions)
    if flow_t0 is None:
        flow_t0 = np.zeros((*source_shape, 2), dtype=np.float32)
    if flow_t1 is None:
        flow_t1 = np.zeros((*source_shape, 2), dtype=np.float32)
    missing_cgvqm = (
        cgvqm_center is None or cgvqm_temporal is None or cgvqm_fused is None
    )
    if missing_cgvqm:
        zeros = np.zeros(source_shape, dtype=np.float32)
        cgvqm_center, cgvqm_temporal, cgvqm_fused = zeros, zeros, zeros
    normalized_cgvqm = _normalized_heatmaps(
        np.asarray(cgvqm_center),
        np.asarray(cgvqm_temporal),
        np.asarray(cgvqm_fused),
    )
    reason_text = ", ".join(str(item) for item in labels)
    structure_label = "structure error"
    if reason_text:
        structure_label += f" · {reason_text[:48]}"
    rows = (
        (
            (img0, "img0", True),
            (gt, "GT", True),
            (prediction, "prediction", True),
            (img1, "img1", True),
            (colorize_error(error_map), structure_label, True),
        ),
        (
            (colorize_flow(flow_t0), "flow_t0", True),
            (colorize_flow(flow_t1), "flow_t1", True),
            (colorize_error(normalized_cgvqm[0]), "CGVQM center error", False),
            (
                colorize_error(normalized_cgvqm[1]),
                "CGVQM temporal sensitivity",
                False,
            ),
            (
                colorize_error(normalized_cgvqm[2]),
                "CGVQM fused confidence",
                False,
            ),
        ),
    )
    panel_height = image_height + 28
    canvas = Image.new(
        "RGB",
        (panel_width * 5, panel_height * 2),
        (0, 0, 0),
    )
    for row_index, row_items in enumerate(rows):
        for column_index, (array, label, show_boxes) in enumerate(row_items):
            panel = _labeled_panel(
                np.asarray(array),
                label=label,
                width=panel_width,
                image_height=image_height,
                boxes=boxes if show_boxes else (),
                source_shape=source_shape if show_boxes else None,
            )
            canvas.paste(
                panel,
                (column_index * panel_width, row_index * panel_height),
            )
    return np.asarray(canvas)
