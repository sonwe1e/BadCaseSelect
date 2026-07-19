from __future__ import annotations

import numpy as np

from vfi_hard_miner.image_io import read_rgb01, write_image_atomic


def test_rgb_round_trip(tmp_path):
    source = np.zeros((5, 7, 3), dtype=np.float32)
    source[2, 3] = (1.0, 0.5, 0.0)
    path = tmp_path / "image.png"
    write_image_atomic(path, source)
    restored = read_rgb01(path)
    assert restored.shape == source.shape
    assert restored.dtype == np.float32
    assert np.max(np.abs(restored - source)) <= 1.0 / 255.0
