from __future__ import annotations

import numpy as np
import pytest

from vfi_hard_miner.gates import (
    FrameValidityMetrics,
    GateResult,
    ScopeMetrics,
    compute_scope_metrics,
    compute_validity_metrics,
    decide_hard_case,
    evaluate_in_scope,
    evaluate_validity,
)


def _clean_validity() -> FrameValidityMetrics:
    return FrameValidityMetrics(
        decode_ok=True,
        finite=True,
        sequence_contiguous=True,
        duplicate_distance=0.10,
        scene_cut_score=0.10,
        temporal_asymmetry=0.10,
        max_adjacent_difference=0.10,
    )


def _clean_scope() -> ScopeMetrics:
    return ScopeMetrics(
        out_of_bounds_ratio=0.05,
        flow_discontinuity_ratio=0.05,
        foreground_large_motion_ratio=0.05,
        occlusion_ratio=0.05,
        unexplained_motion_ratio=0.05,
        background_motion=0.80,
    )


def test_clean_validity_is_accepted():
    result = evaluate_validity(_clean_validity())
    assert result.label == "accept"
    assert result.reasons == ()


def test_sequence_gap_and_exact_duplicate_are_rejected():
    gap = _clean_validity().to_dict()
    gap["sequence_contiguous"] = False
    assert evaluate_validity(gap).label == "reject"
    assert "invalid_sequence_gap" in evaluate_validity(gap).reasons

    duplicate = _clean_validity().to_dict()
    duplicate["duplicate_distance"] = 0.0
    result = evaluate_validity(duplicate)
    assert result.label == "reject"
    assert "duplicate_frame" in result.reasons


def test_scene_cut_gray_zone_is_reviewed_and_extreme_cut_rejected():
    gray = _clean_validity().to_dict()
    gray["scene_cut_score"] = 0.52
    assert evaluate_validity(gray).label == "review"
    extreme = _clean_validity().to_dict()
    extreme["scene_cut_score"] = 0.90
    result = evaluate_validity(extreme)
    assert result.label == "reject"
    assert "scene_cut" in result.reasons


def test_triplet_metrics_detect_one_sided_cut():
    black = np.zeros((32, 32, 3), dtype=np.float32)
    white = np.ones_like(black)
    metrics = compute_validity_metrics(black, white, white)
    assert metrics.scene_cut_score is not None and metrics.scene_cut_score > 0.9
    assert evaluate_validity(metrics).label == "reject"


def _gradient_frame(height=100, width=120):
    horizontal = np.linspace(0.10, 0.30, width, dtype=np.float32)
    luma = np.broadcast_to(horizontal, (height, width))
    return np.repeat(luma[..., None], 3, axis=-1).copy()


def test_large_one_sided_bottom_overlay_is_a_menu_transition_reject():
    first = _gradient_frame()
    middle = first.copy()
    middle[70:, 8:112] = 0.90
    last = np.clip(middle + 0.02, 0.0, 1.0)

    metrics = compute_validity_metrics(first, middle, last)
    result = evaluate_validity(metrics)

    assert metrics.menu_transition_score is not None
    assert metrics.menu_transition_score >= 0.70
    assert result.label == "reject"
    assert "menu_transition" in result.reasons


def test_disappearing_bottom_overlay_is_detected_symmetrically():
    middle = _gradient_frame()
    first = middle.copy()
    first[70:, 8:112] = 0.90
    last = np.clip(middle + 0.02, 0.0, 1.0)

    metrics = compute_validity_metrics(first, middle, last)

    assert metrics.menu_transition_score is not None
    assert metrics.menu_transition_score >= 0.70


def test_small_one_sided_bottom_overlay_is_a_menu_transition_review():
    first = _gradient_frame()
    middle = first.copy()
    middle[90:, 8:112] = 0.90
    last = np.clip(middle + 0.03, 0.0, 1.0)

    metrics = compute_validity_metrics(first, middle, last)
    result = evaluate_validity(metrics)

    assert metrics.menu_transition_score is not None
    assert 0.40 <= metrics.menu_transition_score < 0.70
    assert result.label == "review"
    assert "possible_menu_transition" in result.reasons


def test_consistent_camera_motion_and_gradient_do_not_look_like_menu_transitions():
    base = _gradient_frame()
    shifted_once = np.roll(base, 4, axis=1)
    shifted_twice = np.roll(base, 8, axis=1)
    camera = compute_validity_metrics(base, shifted_once, shifted_twice)

    dark = np.full_like(base, 0.10)
    medium = np.full_like(base, 0.30)
    light = np.full_like(base, 0.50)
    gradient = compute_validity_metrics(dark, medium, light)

    assert camera.menu_transition_score is not None
    assert camera.menu_transition_score < 0.40
    assert gradient.menu_transition_score is not None
    assert gradient.menu_transition_score < 0.40


def test_explicit_menu_transition_score_overrides_automatic_value():
    first = _gradient_frame()
    middle = first.copy()
    middle[70:, :] = 0.90

    automatic = compute_validity_metrics(first, middle, middle)
    explicit = compute_validity_metrics(
        first,
        middle,
        middle,
        menu_transition_score=0.05,
    )

    assert automatic.menu_transition_score is not None
    assert automatic.menu_transition_score >= 0.70
    assert explicit.menu_transition_score == 0.05


def test_large_global_background_motion_is_not_a_scope_reject():
    flow = np.zeros((64, 96, 2), dtype=np.float32)
    flow[..., 0] = 18.0
    metrics = compute_scope_metrics(flow, -flow)
    result = evaluate_in_scope(metrics)
    assert metrics.background_motion is not None and metrics.background_motion > 0.1
    assert metrics.foreground_large_motion_ratio == 0.0
    assert metrics.occlusion_ratio is not None and metrics.occlusion_ratio < 0.40
    assert result.label == "accept"


def test_very_large_global_translation_does_not_become_boundary_occlusion():
    flow_t0 = np.zeros((100, 100, 2), dtype=np.float32)
    flow_t0[..., 0] = 36.0

    metrics = compute_scope_metrics(flow_t0, -flow_t0)
    result = evaluate_in_scope(metrics)

    assert metrics.foreground_large_motion_ratio == 0.0
    assert metrics.out_of_bounds_ratio == 0.0
    assert metrics.occlusion_ratio == 0.0
    assert result.label == "accept"


def test_relative_motion_area_is_derived_without_a_foreground_mask():
    flow_t0 = np.zeros((100, 100, 2), dtype=np.float32)
    flow_t1 = np.zeros_like(flow_t0)
    flow_t0[..., 0] = 5.0
    flow_t1[..., 0] = -5.0
    flow_t0[:, 30:70, 0] = 25.0
    flow_t1[:, 30:70, 0] = -25.0

    metrics = compute_scope_metrics(flow_t0, flow_t1)
    result = evaluate_in_scope(metrics)

    assert metrics.foreground_large_motion_ratio == pytest.approx(0.40)
    assert metrics.occlusion_ratio is not None and metrics.occlusion_ratio < 0.40
    assert result.label == "review"
    assert "possible_foreground_motion_extreme" in result.reasons


def test_majority_central_foreground_is_not_mistaken_for_background_motion():
    flow_t0 = np.zeros((100, 100, 2), dtype=np.float32)
    flow_t1 = np.zeros_like(flow_t0)
    flow_t0[:, 15:85, 0] = 20.0
    flow_t1[:, 15:85, 0] = -20.0

    metrics = compute_scope_metrics(flow_t0, flow_t1)
    result = evaluate_in_scope(metrics)

    assert metrics.foreground_large_motion_ratio == pytest.approx(0.70)
    assert result.label == "reject"
    assert "foreground_motion_extreme" in result.reasons


def test_large_backward_flow_inconsistency_derives_occlusion_reject():
    flow_t0 = np.zeros((100, 100, 2), dtype=np.float32)
    flow_t1 = np.zeros_like(flow_t0)
    flow_t1[:, 12:88, 0] = 12.0

    metrics = compute_scope_metrics(flow_t0, flow_t1)
    result = evaluate_in_scope(metrics)

    assert metrics.occlusion_ratio == pytest.approx(0.76)
    assert result.label == "reject"
    assert "occlusion_extreme" in result.reasons


def test_explicit_scope_masks_override_flow_derived_fallbacks():
    flow_t0 = np.zeros((80, 80, 2), dtype=np.float32)
    flow_t1 = np.zeros_like(flow_t0)
    flow_t0[..., 0] = 10.0
    flow_t1[..., 0] = 10.0
    empty_mask = np.zeros((80, 80), dtype=np.float32)

    derived = compute_scope_metrics(flow_t0, flow_t1)
    explicit = compute_scope_metrics(
        flow_t0,
        flow_t1,
        foreground_mask=empty_mask,
        occlusion_mask=empty_mask,
    )

    assert derived.occlusion_ratio == 1.0
    assert explicit.foreground_large_motion_ratio == 0.0
    assert explicit.occlusion_ratio == 0.0
    assert evaluate_in_scope(explicit).label == "accept"


def test_extreme_foreground_motion_and_occlusion_are_out_of_scope():
    result = evaluate_in_scope(
        ScopeMetrics(
            out_of_bounds_ratio=0.10,
            flow_discontinuity_ratio=0.10,
            foreground_large_motion_ratio=0.90,
            occlusion_ratio=0.80,
            unexplained_motion_ratio=0.20,
        )
    )
    assert result.label == "reject"
    assert "foreground_motion_extreme" in result.reasons
    assert "occlusion_extreme" in result.reasons


def test_missing_scope_semantics_goes_to_review():
    result = evaluate_in_scope(
        {"out_of_bounds_ratio": 0.05, "flow_discontinuity_ratio": 0.05}
    )
    assert result.label == "review"
    assert "scope_semantics_missing" in result.reasons


def test_final_decision_has_accept_review_reject_gray_zone():
    valid = GateResult("accept", (), {})
    scope = GateResult("accept", (), {})
    assert decide_hard_case(valid, scope, 0.80, 0.80).label == "accept"
    assert decide_hard_case(valid, scope, 0.30, 0.80).label == "review"
    assert decide_hard_case(valid, scope, 0.10, 0.80).label == "reject"


def test_low_solvability_is_review_not_invalid_data_rejection():
    result = decide_hard_case(True, True, p_wrong=0.90, p_solvable=0.05)
    assert result.label == "review"
    assert "solvability_low" in result.reasons


def test_rejected_validity_or_scope_overrides_correctness():
    invalid = GateResult("reject", ("scene_cut",), {})
    scope = GateResult("accept", (), {})
    result = decide_hard_case(invalid, scope, 0.95, 0.95)
    assert result.label == "reject"
    assert "invalid_data" in result.reasons
