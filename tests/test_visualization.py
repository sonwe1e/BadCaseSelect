from __future__ import annotations

import numpy as np

from vfi_hard_miner.visualization import make_diagnostic_grid


def test_diagnostic_grid_has_expected_width():
    image = np.zeros((24, 32, 3), dtype=np.float32)
    error = np.zeros((24, 32), dtype=np.float32)
    error[4:8, 6:12] = 1.0
    result = make_diagnostic_grid(
        image,
        image,
        image,
        image,
        error_map=error,
        regions=[{"x0": 5, "y0": 3, "x1": 13, "y1": 9}],
        labels=["broken_structure"],
        panel_width=64,
    )
    assert result.ndim == 3
    assert result.shape[1] == 256
    assert result.shape[0] > 96
