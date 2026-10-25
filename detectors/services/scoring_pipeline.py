"""Compute capture decisions before persistence and channel output."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from detectors.interfaces.classification import DecisionState, compute_unknown_score
from detectors.od_classes import is_bird_od_class
from detectors.services.bbox_quality_service import compute_bbox_quality
from detectors.services.capability_registry import CapabilityRegistry
from detectors.services.decision_policy_service import DecisionPolicyService

DEFAULT_NON_BIRD_CONFIRM_THRESHOLD = 0.80


@dataclass
class ScoringResult:
    """Signals and the admission decision for one captured object."""

    score: float
    agreement_score: float
    bbox_quality: float
    unknown_score: float
    decision_state: DecisionState
    decision_level: str
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
    capability_registry: CapabilityRegistry,
    od_class_name: str | None = None,
    species_key: str | None = None,
    geography_json: str | None = None,
    decision_level: str | None = "species",
    non_bird_confirm_threshold: float = DEFAULT_NON_BIRD_CONFIRM_THRESHOLD,
    non_bird_confirm_threshold_fn: Callable[[str], float] | None = None,
) -> ScoringResult:
    """Apply signal thresholds and the shared admission checks once at capture."""
    bbox_q = compute_bbox_quality(bbox, frame_shape)
    cap_tag = capability_registry.snapshot().version_tag()
    non_bird = od_class_name is not None and not is_bird_od_class(od_class_name)
    if non_bird:
        score = agreement = od_conf
        unknown_s = 0.0
        floor = (
            float(non_bird_confirm_threshold_fn(od_class_name))
            if non_bird_confirm_threshold_fn is not None
            else non_bird_confirm_threshold
        )
        state = DecisionState.CONFIRMED if od_conf >= floor else DecisionState.UNCERTAIN
        reasons = []
    else:
        score = cls_conf if cls_conf > 0 else od_conf
        agreement = min(od_conf, cls_conf) if cls_conf > 0 else od_conf
        top_k = (top_k_confidences or [cls_conf]) if cls_conf > 0 else []
        unknown_s = compute_unknown_score(top_k)
        decision = decision_policy.evaluate(
            bbox_quality=bbox_q,
            species_conf=cls_conf if cls_conf > 0 else None,
            unknown_score=unknown_s,
        )
        state = decision.decision_state
        reasons = json.loads(decision.reasons_json)

    admission = decision_policy.admission_reasons(
        score=score,
        species_key=od_class_name if non_bird else species_key,
        geography_json=geography_json,
    )
    if decision_level != "species":
        admission.append("NO_SPECIES_LEVEL_DECISION")
    if admission and state == DecisionState.CONFIRMED:
        state = DecisionState.UNCERTAIN
    return ScoringResult(
        score=score,
        agreement_score=agreement,
        bbox_quality=bbox_q,
        unknown_score=unknown_s,
        decision_state=state,
        decision_level=decision_level or "species",
        decision_reasons_json=json.dumps([*reasons, *admission]),
        policy_version=cap_tag,
    )
