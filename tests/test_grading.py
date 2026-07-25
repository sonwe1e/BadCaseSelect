from __future__ import annotations

from vfi_hard_miner.config import (
    AppConfig,
    CGVQMConfig,
    DataConfig,
    ModelConfig,
    ThresholdConfig,
)
from vfi_hard_miner.grading import grade_candidate


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        data=DataConfig(root=str(tmp_path)),
        model=ModelConfig(factory="mock"),
        cgvqm=CGVQMConfig(
            enabled=True,
            b_error_at=25.0,
            a_error_at=50.0,
            temporal_persistence_at=0.35,
            spatial_overlap_at=0.35,
        ),
        thresholds=ThresholdConfig(
            wrong_accept_at=0.45,
            severe_wrong_accept_at=0.70,
            solvable_accept_at=0.55,
        ),
    )


def _record(**updates):
    record = {
        "sample_id": "s1",
        "status": "accept",
        "validity_label": "accept",
        "in_scope_label": "accept",
        "mining_p_wrong": 0.80,
        "p_solvable": 0.80,
        "reasons": ["missing_part"],
        "regions": [
            {
                "p_wrong": 0.80,
                "reasons": ["missing_part"],
                "metrics": {"priority_weight": 1.0},
            }
        ],
    }
    record.update(updates)
    return record


def _evidence(**updates):
    evidence = {
        "error": 60.0,
        "temporal_persistence": 0.20,
        "spatial_overlap": 0.80,
    }
    evidence.update(updates)
    return evidence


def test_grade_a_requires_severe_traditional_and_deep_structure_evidence(tmp_path):
    decision = grade_candidate(_record(), _evidence(), _config(tmp_path))
    assert decision.grade == "A"
    assert decision.quality_gate["strong_structure"] is True
    assert decision.quality_gate["spatially_aligned"] is True


def test_confirmed_mild_error_is_b(tmp_path):
    decision = grade_candidate(
        _record(mining_p_wrong=0.55),
        _evidence(error=35.0),
        _config(tmp_path),
    )
    assert decision.grade == "B"


def test_traditional_and_cgvqm_conflicts_never_enter_training_folders(tmp_path):
    config = _config(tmp_path)
    low_error = grade_candidate(_record(), _evidence(error=10.0), config)
    displaced = grade_candidate(
        _record(),
        _evidence(spatial_overlap=0.05, temporal_persistence=0.10),
        config,
    )
    assert low_error.grade == "Review"
    assert displaced.grade == "Review"


def test_invalid_or_unsolvable_sample_is_rejected_before_cgvqm(tmp_path):
    config = _config(tmp_path)
    invalid = grade_candidate(
        _record(status="invalid", validity_label="reject"),
        _evidence(),
        config,
    )
    unsolvable = grade_candidate(
        _record(p_solvable=0.40),
        _evidence(),
        config,
    )
    assert invalid.grade == "Reject"
    assert unsolvable.grade == "Review"
