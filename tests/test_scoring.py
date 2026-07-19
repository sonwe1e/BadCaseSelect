from __future__ import annotations

import numpy as np
import pytest

from vfi_hard_miner.scoring import (
    ScoringConfig,
    compute_error_maps,
    score_local_errors,
    top_area_mean,
)


def _overlaps(region, box):
    x0, y0, x1, y1 = box
    return (
        min(region.x1, x1) > max(region.x0, x0)
        and min(region.y1, y1) > max(region.y0, y0)
    )


def test_identical_images_have_zero_local_error():
    image = np.zeros((64, 96, 3), dtype=np.float32)
    result = score_local_errors(image, image)
    assert result.p_wrong == 0.0
    assert result.regions == ()
    assert np.count_nonzero(result.maps.structure) == 0


def test_small_missing_structure_is_not_diluted_by_global_mean():
    gt = np.zeros((128, 128, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[20:108, 63:65] = 1.0

    result = score_local_errors(prediction, gt)

    assert result.metrics["structure_mean"] < 0.03
    assert result.p_wrong > 0.70
    assert any(_overlaps(region, (63, 20, 65, 108)) for region in result.regions)
    assert result.metrics["gt_only_edge_max"] > 0.5
    assert result.metrics["pred_only_edge_max"] == 0.0


def test_extra_structure_is_recorded_as_prediction_only_edge():
    gt = np.zeros((64, 64, 3), dtype=np.float32)
    prediction = gt.copy()
    prediction[30:33, 8:56] = 1.0
    maps = compute_error_maps(prediction, gt)
    assert float(maps.pred_only_edges.max()) > 0.5
    assert float(maps.gt_only_edges.max()) == 0.0


def test_high_confidence_two_pixel_structure_survives_area_filter():
    gt = np.zeros((32, 32, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[14, 15:17] = 1.0
    config = ScoringConfig(min_region_pixels=20, max_regions=8)
    result = score_local_errors(prediction, gt, config)
    assert result.regions
    assert any(region.metrics.get("small_structure_exception") == 1.0 for region in result.regions)


def test_two_separate_errors_produce_spatially_separate_candidates():
    gt = np.zeros((128, 128, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[15:35, 16:18] = 1.0
    gt[90:110, 104:106] = 1.0
    result = score_local_errors(
        prediction,
        gt,
        ScoringConfig(max_regions=16, window_threshold=0.08),
    )
    assert any(_overlaps(region, (16, 15, 18, 35)) for region in result.regions)
    assert any(_overlaps(region, (104, 90, 106, 110)) for region in result.regions)


def test_top_area_mean_retains_tiny_peak():
    error = np.zeros((100, 100), dtype=np.float32)
    error[0, 0] = 1.0
    assert float(error.mean()) == pytest.approx(0.0001)
    assert top_area_mean(error, 0.0001) == pytest.approx(1.0)


def test_isolated_pixel_noise_does_not_become_a_hard_case():
    gt = np.zeros((64, 64, 3), dtype=np.float32)
    prediction = gt.copy()
    prediction[20, 20] = 1.0
    result = score_local_errors(prediction, gt)
    assert result.regions == ()
    assert result.p_wrong < 0.05


def test_chw_tensor_like_input_matches_hwc_numpy():
    torch = pytest.importorskip("torch")
    gt = np.zeros((48, 64, 3), dtype=np.float32)
    prediction = gt.copy()
    prediction[20:24, 30:34] = 0.75
    numpy_result = score_local_errors(prediction, gt)
    tensor_result = score_local_errors(
        torch.from_numpy(prediction).permute(2, 0, 1),
        torch.from_numpy(gt).permute(2, 0, 1),
    )
    assert tensor_result.p_wrong == pytest.approx(numpy_result.p_wrong, abs=1e-6)
    assert np.allclose(tensor_result.maps.structure, numpy_result.maps.structure)


def test_invalid_range_fails_fast():
    gt = np.zeros((8, 8, 3), dtype=np.float32)
    prediction = np.full_like(gt, 255.0)
    with pytest.raises(ValueError, match="normalized"):
        score_local_errors(prediction, gt)


def _static_hud_and_central_structure():
    gt = np.zeros((128, 128, 3), dtype=np.float32)
    # Dense, endpoint-static subtitle/HUD strokes inside the bottom edge band.
    for x0 in range(8, 120, 8):
        gt[108:121, x0 : x0 + 3] = 1.0
    # A lower-contrast character/weapon-like structure in the image centre.
    gt[30:98, 62:66] = 0.70
    gt[60:65, 43:85] = 0.70
    prediction = gt.copy()
    prediction[108:121] = 0.0
    prediction[30:98, 62:66] = 0.0
    prediction[60:65, 43:85] = 0.0
    return prediction, gt, gt.copy(), gt.copy()


def test_static_edge_hud_gets_lower_mining_priority_without_changing_gt_wrongness():
    prediction = np.zeros((128, 128, 3), dtype=np.float32)
    gt = np.zeros_like(prediction)
    for x0 in range(8, 120, 8):
        gt[108:121, x0 : x0 + 3] = 1.0

    legacy = score_local_errors(prediction, gt)
    prioritized = score_local_errors(prediction, gt, img0=gt, img1=gt)

    assert prioritized.p_wrong == pytest.approx(legacy.p_wrong)
    assert prioritized.mining_p_wrong < prioritized.p_wrong
    assert prioritized.metrics["ui_context_available"] == 1.0
    assert any(region.metrics["ui_likelihood"] > 0.6 for region in prioritized.regions)
    assert any(region.metrics["priority_weight"] < 0.7 for region in prioritized.regions)


def test_ui_context_cannot_change_raw_wrongness_with_one_region_budget():
    prediction = np.zeros((256, 256, 3), dtype=np.float32)
    gt = np.zeros_like(prediction)
    for x0 in range(8, 248, 8):
        gt[220:246, x0 : x0 + 3] = 1.0
    gt[70:190, 126:130] = 0.65
    config = ScoringConfig(max_regions=1, non_ui_region_reserve=1)

    legacy = score_local_errors(prediction, gt, config)
    prioritized = score_local_errors(
        prediction,
        gt,
        config,
        img0=gt,
        img1=gt,
    )

    assert prioritized.p_wrong == pytest.approx(legacy.p_wrong, abs=1e-7)
    assert prioritized.metrics["raw_region_local"] == pytest.approx(
        legacy.metrics["raw_region_local"], abs=1e-7
    )


def test_candidate_budget_reserves_central_non_ui_structure():
    prediction, gt, img0, img1 = _static_hud_and_central_structure()

    result = score_local_errors(
        prediction,
        gt,
        ScoringConfig(max_regions=2, non_ui_region_reserve=1),
        img0=img0,
        img1=img1,
    )

    central = [region for region in result.regions if _overlaps(region, (43, 30, 85, 98))]
    assert central, "the edge HUD must not consume every candidate slot"
    assert central[0].metrics["ui_likelihood"] < 0.1
    assert central[0].metrics["priority_weight"] == pytest.approx(1.0)


def test_non_ui_reserve_can_keep_a_central_window_only_candidate():
    prediction = np.zeros((128, 128, 3), dtype=np.float32)
    gt = np.zeros_like(prediction)
    for x0 in range(4, 124, 6):
        gt[108:123, x0 : x0 + 2] = 1.0
    gt[48:80, 48:80] = 0.07
    config = ScoringConfig(
        max_regions=2,
        non_ui_region_reserve=1,
        edge_threshold=0.12,
        window_threshold=0.05,
    )

    result = score_local_errors(
        prediction,
        gt,
        config,
        img0=gt,
        img1=gt,
    )

    central = [region for region in result.regions if _overlaps(region, (48, 48, 80, 80))]
    assert central, "a compact non-UI window must survive a native HUD candidate"
    assert central[0].metrics["source_native"] == 0.0
    assert central[0].metrics["priority_weight"] == pytest.approx(1.0)
