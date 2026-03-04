"""
Scoring Pipeline — single source of truth for detection signal computation.

Centralises the score formula, unknown-score fallback, decision evaluation,
temporal smoothing, and capability version tagging that was previously
duplicated across ``detection_manager._processing_loop`` and
``analysis_service._build_detection_payload``.
"""

from __future__ import annotations

from dataclasses import dataclass

from detectors.interfaces.classification import DecisionState, compute_unknown_score
from detectors.services.bbox_quality_service import compute_bbox_quality
from detectors.services.capability_registry import CapabilityRegistry
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.temporal_decision_service import TemporalDecisionService


@dataclass
class ScoringResult:
    """All computed signals for a single detection."""

    score: float
    agreement_score: float
    bbox_quality: float
    unknown_score: float
    decision_state: DecisionState | None
    decision_reasons_json: str
    policy_version: str


def compute_detection_signals(
    *,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    od_conf: float,
    cls_conf: float,
    top_k_confidences: list[float] | None,
    decision_policy: DecisionPolicyService,
    temporal_service: TemporalDecisionService,
    capability_registry: CapabilityRegistry,
    species_key: str,
) -> ScoringResult:
    """
    Compute all detection quality signals in one place.

    This is the **single source of truth** for the composite score formula,
    agreement score, bbox quality, unknown score, decision evaluation,
    temporal smoothing, and capability version tag.

    When ``ENABLE_DECISION_POLICY`` is disabled in the capability registry,
    the decision evaluation and temporal smoothing steps are skipped.
    ``decision_state`` is set to ``None`` (legacy-compatible) and
    ``decision_reasons_json`` is ``"[]"``.  Signal computation (score,
    agreement, bbox_quality, unknown_score) remains active for diagnostics.

    Args:
        bbox:                Pixel coordinates ``(x1, y1, x2, y2)``.
        frame_shape:         Shape of the source frame ``(H, W, ...)``.
        od_conf:             Object-detection confidence.
        cls_conf:            Classification confidence (0.0 if no CLS ran).
        top_k_confidences:   Top-k class probabilities from classifier, or
                             ``None`` if classification did not run.
        decision_policy:     :class:`DecisionPolicyService` instance.
        temporal_service:    :class:`TemporalDecisionService` instance.
        capability_registry: :class:`CapabilityRegistry` instance.
        species_key:         Grouping key for temporal smoothing (species name
                             or ``"unknown"``).

    Returns:
        :class:`ScoringResult` with all computed values ready for persistence.
    """
    # --- Composite score & agreement ---
    if cls_conf > 0:
        score = 0.5 * od_conf + 0.5 * cls_conf
        agreement = min(od_conf, cls_conf)
    else:
        score = od_conf
        agreement = od_conf

    # --- BBox quality ---
    bbox_q = compute_bbox_quality(bbox, frame_shape)

    # --- Unknown score (P1-02 deterministic fallback) ---
    # No classification → compute_unknown_score([]) → 1.0 (max uncertainty).
    # Classification with no top-k → single-class fallback [cls_conf].
    if cls_conf > 0:
        top_k = top_k_confidences if top_k_confidences else [cls_conf]
        unknown_s = compute_unknown_score(top_k)
    else:
        unknown_s = compute_unknown_score([])

    # --- Capability version tag for persistence ---
    cap_tag = capability_registry.snapshot().version_tag()

    # --- Decision policy gate (T2: ENABLE_DECISION_POLICY) ---
    policy_enabled = capability_registry.is_enabled("decision_policy")

    if policy_enabled:
        # --- Decision policy evaluation ---
        decision_res = decision_policy.evaluate(
            bbox_quality=bbox_q,
            species_conf=cls_conf if cls_conf > 0 else None,
            unknown_score=unknown_s,
        )

        # --- Temporal smoothing (feature-flag gated) ---
        smoothed_state: DecisionState | None = temporal_service.smooth(
            species_key=species_key,
            raw_state=decision_res.decision_state,
        )

        reasons_json = decision_res.reasons_json
    else:
        # Legacy-conservative mode: no synthetic decision labels.
        smoothed_state = None
        reasons_json = "[]"

    return ScoringResult(
        score=score,
        agreement_score=agreement,
        bbox_quality=bbox_q,
        unknown_score=unknown_s,
        decision_state=smoothed_state,
        decision_reasons_json=reasons_json,
        policy_version=cap_tag,
    )
