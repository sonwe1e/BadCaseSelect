from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfi_hard_miner.cgvqm import CGVQM2Scorer


def test_pinned_cgvqm_scorer_returns_zero_for_identical_clips():
    root = Path(__file__).resolve().parents[1]
    backbone = root / "third_party" / "weights" / "cgvqm" / "r3d_18-b3b3357e.pth"
    calibration = root / "third_party" / "weights" / "cgvqm" / "cgvqm-2.pickle"
    if not backbone.is_file() or not calibration.is_file():
        pytest.skip("offline CGVQM weights are not installed")
    scorer = CGVQM2Scorer(backbone, calibration, device="cpu")
    clip = np.zeros((1, 3, 32, 32, 3), dtype=np.uint8)
    result = scorer.score(clip, clip)
    assert result.errors.shape == (1,)
    assert result.error_maps.shape == (1, 3, 32, 32)
    np.testing.assert_allclose(result.errors, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.error_maps, 0.0, atol=1e-6)
