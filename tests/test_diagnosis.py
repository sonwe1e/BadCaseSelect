from __future__ import annotations

import numpy as np
import pytest

import vfi_hard_miner.diagnosis as diagnosis_module
from vfi_hard_miner.diagnosis import (
    REASON_LABELS,
    diagnose_sample,
    estimate_solvability,
)
from vfi_hard_miner.scoring import score_local_errors


def test_teacher_that_recovers_local_error_raises_solvability():
    result = estimate_solvability(current_error=0.80, teacher_error=0.05)
    assert result.p_solvable > 0.80
    assert result.teacher_gain is not None and result.teacher_gain > 0.8
    assert "teacher_recovers" in result.reasons


def test_teacher_failure_does_not_redefine_correctness():
    result = estimate_solvability(current_error=0.80, teacher_error=0.90)
    assert result.current_error == pytest.approx(0.80)
    assert result.p_solvable < 0.50
    assert "solvability_uncertain" in result.reasons


def test_best_warp_can_prove_failure_is_locally_recoverable():
    result = estimate_solvability(
        current_error=0.75,
        warp_errors={"warp0": 0.08, "warp1": 0.70, "warp_blend": 0.60},
    )
    assert result.best_warp_error == pytest.approx(0.08)
    assert result.p_solvable > 0.75
    assert "warp_branch_recovers" in result.reasons


def test_missing_thin_part_gets_stable_nonsemantic_reason():
    gt = np.zeros((96, 96, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[15:82, 47:49] = 1.0
    teacher = gt.copy()
    result = diagnose_sample(prediction, gt, teacher_prediction=teacher)
    assert result.p_wrong > 0.7
    assert result.p_solvable > 0.8
    assert "missing_part" in result.reasons
    assert "edge_tearing" in result.reasons
    assert all(reason in REASON_LABELS for reason in result.reasons)


def test_endpoint_copy_and_blend_error_are_diagnosed_from_branch_evidence():
    gt = np.zeros((80, 80, 3), dtype=np.float32)
    gt[25:55, 38:42] = 1.0
    img1 = np.zeros_like(gt)
    prediction = img1.copy()
    warp0 = gt.copy()
    warp1 = img1.copy()
    warp_blend = img1.copy()
    result = diagnose_sample(
        prediction,
        gt,
        img1=img1,
        warp0=warp0,
        warp1=warp1,
        warp_blend=warp_blend,
    )
    assert "endpoint_copy" in result.reasons
    assert "blend_mask_error" in result.reasons


def test_broken_line_produces_structure_reason_and_order_is_stable():
    gt = np.zeros((96, 96, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[46:49, 10:86] = 1.0
    prediction[:] = gt
    prediction[46:49, 43:54] = 0.0
    result = diagnose_sample(prediction, gt)
    assert "broken_structure" in result.reasons
    order = [REASON_LABELS.index(reason) for reason in result.reasons]
    assert order == sorted(order)


def test_identical_prediction_has_no_regions_or_error_reasons():
    image = np.zeros((32, 48, 3), dtype=np.float32)
    result = diagnose_sample(image, image)
    assert result.p_wrong == 0.0
    assert result.reasons == ()
    assert result.regions == ()
    assert result.primary_region_index is None


def test_precomputed_local_score_avoids_rescoring_and_matches_legacy_path(monkeypatch):
    gt = np.zeros((64, 64, 3), dtype=np.float32)
    prediction = gt.copy()
    gt[12:52, 31:33] = 1.0
    scoring_result = score_local_errors(prediction, gt)
    expected = diagnose_sample(prediction, gt)

    def unexpected_rescore(*args, **kwargs):
        raise AssertionError("score_local_errors must not be called")

    monkeypatch.setattr(diagnosis_module, "score_local_errors", unexpected_rescore)
    actual = diagnose_sample(
        prediction,
        gt,
        scoring_result=scoring_result,
    )

    assert actual == expected


def test_primary_region_prefers_central_structure_over_static_edge_hud():
    gt = np.zeros((128, 128, 3), dtype=np.float32)
    for x0 in range(8, 120, 8):
        gt[108:121, x0 : x0 + 3] = 1.0
    gt[30:98, 62:66] = 0.70
    gt[60:65, 43:85] = 0.70
    prediction = np.zeros_like(gt)
    scoring = score_local_errors(
        prediction,
        gt,
        img0=gt,
        img1=gt,
    )

    result = diagnose_sample(
        prediction,
        gt,
        img0=gt,
        img1=gt,
        scoring_result=scoring,
    )

    assert result.primary_region_index is not None
    primary = result.regions[result.primary_region_index]
    x0, y0, x1, y1 = primary.box
    assert x0 < 66 and x1 > 62 and y0 < 65 and y1 > 60
    assert primary.metrics["ui_likelihood"] < 0.1
    assert primary.metrics["priority_weight"] == pytest.approx(1.0)
    assert result.mining_p_wrong == pytest.approx(result.p_wrong)
    assert result.metrics["selected_mining_p_wrong"] == pytest.approx(result.p_wrong)


# ---------------------------------------------------------------------------
# _binary_component_count / _edge_endpoint_count vs brute-force references
# ---------------------------------------------------------------------------

from vfi_hard_miner.diagnosis import (  # noqa: E402
    _binary_component_count,
    _edge_endpoint_count,
)


def _reference_component_count(binary):
    remaining = binary.copy()
    height, width = remaining.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not remaining[y, x]:
                continue
            count += 1
            remaining[y, x] = False
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and remaining[ny, nx]
                        ):
                            remaining[ny, nx] = False
                            stack.append((ny, nx))
    return count


def _reference_endpoint_count(binary):
    height, width = binary.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not binary[y, x]:
                continue
            neighbours = 0
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if (ny, nx) == (y, x):
                        continue
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx]:
                        neighbours += 1
            if neighbours <= 1:
                count += 1
    return count


def test_binary_component_count_matches_brute_force_on_known_shapes():
    edge_map = np.zeros((12, 12), dtype=np.float32)
    # Two eight-connected islands, one of them diagonal-linked.
    edge_map[1, 1] = edge_map[2, 2] = edge_map[2, 3] = 1.0
    edge_map[8:10, 8:10] = 1.0
    assert _binary_component_count(edge_map, 0.5) == 2
    assert _binary_component_count(np.zeros((4, 4), dtype=np.float32), 0.5) == 0


@pytest.mark.parametrize("density", [0.02, 0.1, 0.3])
def test_edge_feature_counts_match_brute_force_on_random_maps(density):
    rng = np.random.default_rng(987)
    edge_map = np.asarray(rng.random((33, 41)), dtype=np.float32)
    binary = edge_map >= 0.5
    assert _binary_component_count(edge_map, 0.5) == _reference_component_count(binary)
    assert _edge_endpoint_count(edge_map, 0.5) == _reference_endpoint_count(binary)
