"""Small, deterministic RGB image I/O helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb01(path: str | Path) -> np.ndarray:
    """Read an image as contiguous float32 HWC RGB in [0, 1]."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(array)


def read_rgb_uint8(path: str | Path) -> np.ndarray:
    """Read an image as contiguous uint8 HWC RGB (a 4x smaller cache form)."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(array)


def rgb_uint8_to_float32(array: np.ndarray) -> np.ndarray:
    """Convert cached uint8 RGB to float32 [0, 1], bit-identical to read_rgb01."""
    return np.asarray(array, dtype=np.float32) / 255.0


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected HxW, HxWx1, HxWx3 or HxWx4 image, got {value.shape}")
    if value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    if np.issubdtype(value.dtype, np.floating):
        value = np.rint(np.clip(value, 0.0, 1.0) * 255.0)
    return np.asarray(value, dtype=np.uint8)


def write_image_atomic(path: str | Path, array: np.ndarray) -> None:
    """Write a PNG/JPEG through a same-directory temporary file and replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(_to_uint8_rgb(array)).save(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
