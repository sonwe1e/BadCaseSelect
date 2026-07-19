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


def _fit_panel(array: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(_rgb_uint8(array))
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BILINEAR)


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
    mask0: np.ndarray | None = None,
    mask1: np.ndarray | None = None,
    regions: Iterable[Any] = (),
    labels: Iterable[str] = (),
    panel_width: int = 320,
) -> np.ndarray:
    """Build a two-row global diagnostic plus top local GT/pred crops."""
    top_arrays = [img0, gt, prediction, img1]
    edge = np.zeros_like(np.asarray(error_map), dtype=np.float32)
    if gt_only_edge is not None:
        edge += np.asarray(gt_only_edge, dtype=np.float32)
    if pred_only_edge is not None:
        edge -= np.asarray(pred_only_edge, dtype=np.float32)
    edge_rgb = np.stack((np.clip(edge, 0, 1), np.zeros_like(edge), np.clip(-edge, 0, 1)), axis=-1)
    diagnostic_arrays: list[np.ndarray] = [colorize_error(error_map), edge_rgb]
    if flow_t0 is not None:
        diagnostic_arrays.append(colorize_flow(flow_t0))
    elif mask0 is not None:
        diagnostic_arrays.append(np.asarray(mask0))
    else:
        diagnostic_arrays.append(np.zeros_like(img0))
    if flow_t1 is not None:
        diagnostic_arrays.append(colorize_flow(flow_t1))
    elif mask1 is not None:
        diagnostic_arrays.append(np.asarray(mask1))
    else:
        diagnostic_arrays.append(np.zeros_like(img0))
    rows: list[Image.Image] = []
    for arrays in (top_arrays, diagnostic_arrays):
        panels = [_fit_panel(array, panel_width) for array in arrays]
        row_height = max(panel.height for panel in panels)
        row = Image.new("RGB", (panel_width * 4, row_height), "black")
        for index, panel in enumerate(panels):
            row.paste(panel, (index * panel_width, 0))
        rows.append(row)
    local_panels: list[Image.Image] = []
    for region in list(regions)[:4]:
        getter = (lambda key: region[key]) if isinstance(region, dict) else (lambda key: getattr(region, key))
        x0, y0, x1, y1 = (int(getter(key)) for key in ("x0", "y0", "x1", "y1"))
        for array in (gt, prediction):
            crop = np.asarray(array)[y0:y1, x0:x1]
            if crop.size:
                local_panels.append(_fit_panel(crop, panel_width // 2))
    if local_panels:
        local_height = max(panel.height for panel in local_panels)
        local_row = Image.new("RGB", (panel_width * 4, local_height), "black")
        x = 0
        for panel in local_panels:
            if x + panel.width > local_row.width:
                break
            local_row.paste(panel, (x, 0))
            x += panel.width
        rows.append(local_row)
    header_height = 28
    canvas = Image.new("RGB", (panel_width * 4, header_height + sum(row.height for row in rows)), "black")
    draw = ImageDraw.Draw(canvas)
    title = " | ".join(["img0", "GT", "prediction", "img1"])
    reason_text = ", ".join(labels)
    draw.text((8, 6), f"{title}    reasons: {reason_text}", fill="white")
    y = header_height
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return np.asarray(canvas)
