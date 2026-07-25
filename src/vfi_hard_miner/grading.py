"""Conservative A/B grading from traditional and CGVQM evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import AppConfig


TRAINING_GRADES = frozenset({"A", "B"})
STRUCTURAL_REASONS = frozenset(
    {
        "missing_part",
        "broken_structure",
        "ghosting",
        "edge_tearing",
        "endpoint_copy",
        "blend_mask_error",
    }
)


@dataclass(frozen=True, slots=True)
class GradeDecision:
    grade: str
    severity_score: float
    quality_gate: dict[str, Any]
    reason_confidence: dict[str, float]
    reasons: tuple[str, ...]


def _finite_probability(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(number, 0.0, 1.0))


def _reason_confidence(record: Mapping[str, Any]) -> dict[str, float]:
    confidence: dict[str, float] = {}
    for reason in record.get("reasons", ()):
        confidence[str(reason)] = max(confidence.get(str(reason), 0.0), 0.5)
    for region in record.get("regions", ()):
        if not isinstance(region, Mapping):
            continue
        score = _finite_probability(region.get("p_wrong", 0.0), name="region p_wrong")
        metrics = region.get("metrics", {})
        if isinstance(metrics, Mapping):
            score *= _finite_probability(
                metrics.get("priority_weight", 1.0),
                name="region priority_weight",
            )
        for reason in region.get("reasons", ()):
            key = str(reason)
            confidence[key] = max(confidence.get(key, 0.0), score)
    return {key: float(np.clip(value, 0.0, 1.0)) for key, value in confidence.items()}


def grade_candidate(
    record: Mapping[str, Any],
    cgvqm: Mapping[str, Any] | None,
    config: AppConfig,
) -> GradeDecision:
    """Grade one frozen sample without mutating its source record."""

    thresholds = config.thresholds
    status = str(record.get("status", "review"))
    validity = str(record.get("validity_label", "review"))
    in_scope = str(record.get("in_scope_label", "review"))
    wrong = _finite_probability(
        record.get("mining_p_wrong", record.get("p_wrong", 0.0)),
        name="mining_p_wrong",
    )
    solvable = _finite_probability(record.get("p_solvable", 0.0), name="p_solvable")
    reason_confidence = _reason_confidence(record)
    gate = {
        "validity": validity,
        "in_scope": in_scope,
        "traditional_status": status,
        "mining_p_wrong": wrong,
        "p_solvable": solvable,
        "cgvqm_required": bool(config.cgvqm.enabled),
        "cgvqm_confirmed": False,
    }

    blocking: list[str] = []
    if status in {"invalid", "reject"} or validity != "accept":
        blocking.append("input_not_valid")
    if status == "out_of_scope" or in_scope != "accept":
        blocking.append("not_in_scope")
    if solvable < thresholds.solvable_accept_at:
        blocking.append("solvability_not_accepted")
    if wrong < thresholds.wrong_accept_at:
        blocking.append("traditional_error_not_accepted")
    if not record.get("regions"):
        blocking.append("localized_evidence_missing")
    if blocking:
        gate["blocking_reasons"] = blocking
        return GradeDecision(
            "Reject" if status in {"invalid", "reject", "out_of_scope"} else "Review",
            wrong,
            gate,
            reason_confidence,
            tuple(blocking),
        )

    if not config.cgvqm.enabled:
        gate["cgvqm_confirmed"] = None
        return GradeDecision(
            "B",
            wrong,
            gate,
            reason_confidence,
            ("cgvqm_disabled_compatibility_grade",),
        )
    if cgvqm is None:
        return GradeDecision(
            "Review",
            wrong,
            gate,
            reason_confidence,
            ("cgvqm_evidence_missing",),
        )

    error = float(cgvqm.get("error", float("nan")))
    persistence = _finite_probability(
        cgvqm.get("temporal_persistence", 0.0),
        name="cgvqm temporal_persistence",
    )
    overlap = _finite_probability(
        cgvqm.get("spatial_overlap", 0.0),
        name="cgvqm spatial_overlap",
    )
    temporal_change = float(cgvqm.get("temporal_change", 0.0))
    if not np.isfinite(temporal_change):
        raise ValueError("cgvqm temporal_change must be finite")
    temporal_change = float(np.clip(temporal_change, 0.0, 100.0))
    if temporal_change >= config.cgvqm.flicker_change_at:
        reason_confidence["flicker"] = max(
            reason_confidence.get("flicker", 0.0),
            float(np.clip(temporal_change / 100.0, 0.0, 1.0)),
        )
    if not np.isfinite(error):
        raise ValueError("cgvqm error must be finite")
    error = float(np.clip(error, 0.0, 100.0))
    normalized_error = error / 100.0
    gate.update(
        {
            "cgvqm_error": error,
            "cgvqm_temporal_persistence": persistence,
            "cgvqm_temporal_change": temporal_change,
            "cgvqm_spatial_overlap": overlap,
            "cgvqm_confirmed": error >= config.cgvqm.b_error_at,
        }
    )
    severity = float(np.clip(0.45 * wrong + 0.55 * normalized_error, 0.0, 1.0))
    if error < config.cgvqm.b_error_at:
        return GradeDecision(
            "Review",
            severity,
            gate,
            reason_confidence,
            ("traditional_cgvqm_conflict",),
        )

    strong_structure = any(
        reason_confidence.get(reason, 0.0) >= thresholds.wrong_accept_at
        for reason in STRUCTURAL_REASONS
    )
    temporal_strong = (
        persistence >= config.cgvqm.temporal_persistence_at
        or temporal_change >= config.cgvqm.flicker_change_at
    )
    spatially_aligned = overlap >= config.cgvqm.spatial_overlap_at
    gate["strong_structure"] = strong_structure
    gate["strong_temporal_evidence"] = temporal_strong
    gate["spatially_aligned"] = spatially_aligned
    if not spatially_aligned and not temporal_strong:
        return GradeDecision(
            "Review",
            severity,
            gate,
            reason_confidence,
            ("traditional_cgvqm_location_conflict",),
        )
    if (
        error >= config.cgvqm.a_error_at
        and wrong >= thresholds.severe_wrong_accept_at
        and (strong_structure or temporal_strong)
    ):
        return GradeDecision(
            "A",
            severity,
            gate,
            reason_confidence,
            ("severe_cgvqm_and_structure",),
        )
    return GradeDecision(
        "B",
        severity,
        gate,
        reason_confidence,
        ("confirmed_mild_degradation",),
    )


def apply_grade(
    record: Mapping[str, Any],
    cgvqm: Mapping[str, Any] | None,
    config: AppConfig,
) -> dict[str, Any]:
    decision = grade_candidate(record, cgvqm, config)
    updated = dict(record)
    updated["grade"] = decision.grade
    updated["severity_score"] = decision.severity_score
    updated["quality_gate"] = decision.quality_gate
    updated["reason_confidence"] = decision.reason_confidence
    updated["grade_reasons"] = list(decision.reasons)
    updated["cgvqm"] = None if cgvqm is None else dict(cgvqm)
    return updated


__all__ = [
    "GradeDecision",
    "STRUCTURAL_REASONS",
    "TRAINING_GRADES",
    "apply_grade",
    "grade_candidate",
]
