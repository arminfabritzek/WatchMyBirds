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
from detectors.od_classes import is_bird_od_class
from detectors.services.bbox_quality_service import compute_bbox_quality
from detectors.services.capability_registry import CapabilityRegistry
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.temporal_decision_service import TemporalDecisionService

# Default OD-confidence threshold above which a non-bird detection is
# considered CONFIRMED. Matches the app-wide SAVE_THRESHOLD default;
# call-sites override via ``non_bird_confirm_threshold`` when needed.
DEFAULT_NON_BIRD_CONFIRM_THRESHOLD = 0.65


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
    od_class_name: str | None = None,
    non_bird_confirm_threshold: float = DEFAULT_NON_BIRD_CONFIRM_THRESHOLD,
) -> ScoringResult:
    """
    Compute all detection quality signals in one place.

    This is the **single source of truth** for the composite score formula,
    agreement score, bbox quality, unknown score, decision evaluation,
    temporal smoothing, and capability version tag.

    Two routing tracks are supported:

    - **bird track** (``is_bird_od_class(od_class_name)`` is True; this is the
      default for the legacy call signature where ``od_class_name`` is not
      supplied): CLS may have produced a ``cls_conf`` and ``top_k_confidences``.
      Score, unknown-score, decision policy all run as they did before.

    - **non-bird track** (``od_class_name`` names a garden animal like
      ``squirrel``/``cat``/``marten_mustelid``/``hedgehog``): CLS is skipped by
      the caller, so ``cls_conf`` is 0 and ``top_k_confidences`` is None.
      Score = ``od_conf``; unknown_score = 0.0 (no uncertainty pollution from
      missing CLS); decision_state = CONFIRMED iff ``od_conf >=
      non_bird_confirm_threshold`` else UNCERTAIN. The decision policy is
      intentionally bypassed because its ``unknown_score >= 0.6`` branch would
      otherwise mark every non-bird as UNKNOWN and hide it from every surface.

    When ``ENABLE_DECISION_POLICY`` is disabled in the capability registry,
    the bird track's decision evaluation and temporal smoothing are skipped
    and ``decision_state`` is set to ``None`` (legacy-compatible). Non-bird
    CONFIRMED/UNCERTAIN routing still applies because it does not go through
    the capability gate.

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
        od_class_name:       OD class name from the detector output. When
                             omitted, the bird track runs (legacy behaviour).
        non_bird_confirm_threshold: Minimum OD confidence for non-bird
                             CONFIRMED state. Default matches the app-wide
                             ``SAVE_THRESHOLD``.

    Returns:
        :class:`ScoringResult` with all computed values ready for persistence.
    """
    # --- BBox quality (computed both tracks, useful for ranking) ---
    bbox_q = compute_bbox_quality(bbox, frame_shape)

    # --- Capability version tag for persistence ---
    cap_tag = capability_registry.snapshot().version_tag()

    non_bird = od_class_name is not None and not is_bird_od_class(od_class_name)

    if non_bird:
        # Non-bird track: OD confidence IS the species trust signal.
        # Bypass the decision policy entirely — its unknown-score branch
        # would otherwise mark every non-bird as UNKNOWN and the gallery
        # visibility SQL would hide it from every surface.
        score = od_conf
        agreement = od_conf
        unknown_s = 0.0
        if od_conf >= non_bird_confirm_threshold:
            decision_state: DecisionState | None = DecisionState.CONFIRMED
        else:
            decision_state = DecisionState.UNCERTAIN

        # Temporal smoothing still applies (it groups by species_key,
        # and a non-bird species is a valid key). The decision_policy
        # service is NOT consulted for non-bird detections.
        smoothed_state: DecisionState | None
        if capability_registry.is_enabled("decision_policy"):
            smoothed_state = temporal_service.smooth(
                species_key=species_key,
                raw_state=decision_state,
            )
        else:
            smoothed_state = None

        return ScoringResult(
            score=score,
            agreement_score=agreement,
            bbox_quality=bbox_q,
            unknown_score=unknown_s,
            decision_state=smoothed_state,
            decision_reasons_json="[]",
            policy_version=cap_tag,
        )

    # --- Bird track (legacy behaviour, with simplified score formula) ---
    # Score means "how trustworthy is the species identification":
    # CLS confidence when available, OD confidence as a fallback when CLS
    # never ran.
    if cls_conf > 0:
        score = cls_conf
        agreement = min(od_conf, cls_conf)
    else:
        score = od_conf
        agreement = od_conf

    # Unknown score fallback:
    # - No classification -> compute_unknown_score([]) -> 1.0 (max uncertainty)
    # - Classification with no top-k -> single-class fallback [cls_conf]
    if cls_conf > 0:
        top_k = top_k_confidences if top_k_confidences else [cls_conf]
        unknown_s = compute_unknown_score(top_k)
    else:
        unknown_s = compute_unknown_score([])

    policy_enabled = capability_registry.is_enabled("decision_policy")

    if policy_enabled:
        decision_res = decision_policy.evaluate(
            bbox_quality=bbox_q,
            species_conf=cls_conf if cls_conf > 0 else None,
            unknown_score=unknown_s,
        )
        smoothed_state = temporal_service.smooth(
            species_key=species_key,
            raw_state=decision_res.decision_state,
        )
        reasons_json = decision_res.reasons_json
    else:
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
