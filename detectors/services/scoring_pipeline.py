"""
Scoring Pipeline — single source of truth for detection signal computation.

Centralises the score formula, unknown-score fallback, decision evaluation,
and capability version tagging for the live detection pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from detectors.interfaces.classification import DecisionState, compute_unknown_score
from detectors.od_classes import is_bird_od_class
from detectors.services.bbox_quality_service import compute_bbox_quality
from detectors.services.capability_registry import CapabilityRegistry
from detectors.services.decision_policy_service import DecisionPolicyService

# Default OD-confidence threshold above which a non-bird detection is
# considered CONFIRMED. Higher than SAVE_THRESHOLD because non-bird OD
# classes (marten/cat/squirrel/hedgehog) ride OD confidence directly with
# no CLS sanity check, so they need a stricter floor than bird detections
# (which are gated on CLS confidence in a separate code path). Production
# call-sites override via ``non_bird_confirm_threshold`` from
# ``config["NON_BIRD_CONFIRM_THRESHOLD"]``.
DEFAULT_NON_BIRD_CONFIRM_THRESHOLD = 0.80


@dataclass
class ScoringResult:
    """All computed signals for a single detection."""

    score: float
    agreement_score: float
    bbox_quality: float
    unknown_score: float
    decision_state: DecisionState
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
    non_bird_confirm_threshold: float = DEFAULT_NON_BIRD_CONFIRM_THRESHOLD,
    non_bird_confirm_threshold_fn: Callable[[str], float] | None = None,
) -> ScoringResult:
    """
    Compute all detection quality signals in one place.

    This is the **single source of truth** for the composite score formula,
    agreement score, bbox quality, unknown score, decision evaluation,
    and capability version tag.

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

    Args:
        bbox:                Pixel coordinates ``(x1, y1, x2, y2)``.
        frame_shape:         Shape of the source frame ``(H, W, ...)``.
        od_conf:             Object-detection confidence.
        cls_conf:            Classification confidence (0.0 if no CLS ran).
        top_k_confidences:   Top-k class probabilities from classifier, or
                             ``None`` if classification did not run.
        decision_policy:     :class:`DecisionPolicyService` instance.
        capability_registry: :class:`CapabilityRegistry` instance.
        od_class_name:       OD class name from the detector output. When
                             omitted, the bird track runs (legacy behaviour).
        non_bird_confirm_threshold: Scalar fallback minimum OD confidence
                             for non-bird CONFIRMED state. Used when
                             ``non_bird_confirm_threshold_fn`` is None
                             (5-class models without a per-class block).
        non_bird_confirm_threshold_fn: Optional resolver
                             ``(od_class_name) -> float`` that returns
                             the per-class CONFIRMED floor. When set, it
                             takes precedence over the scalar. Used by
                             v2-coco-shaped models where each class has
                             its own calibrated operating point (person
                             0.30, marten 0.45, ...).

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
        # Per-class resolver wins over the scalar default when present.
        # The fn is expected to be total over class names (callers build
        # it with the scalar floor as the fallback for unlisted classes).
        if non_bird_confirm_threshold_fn is not None and od_class_name is not None:
            floor = float(non_bird_confirm_threshold_fn(od_class_name))
        else:
            floor = non_bird_confirm_threshold
        if od_conf >= floor:
            decision_state: DecisionState = DecisionState.CONFIRMED
        else:
            decision_state = DecisionState.UNCERTAIN

        return ScoringResult(
            score=score,
            agreement_score=agreement,
            bbox_quality=bbox_q,
            unknown_score=unknown_s,
            decision_state=decision_state,
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

    decision_res = decision_policy.evaluate(
        bbox_quality=bbox_q,
        species_conf=cls_conf if cls_conf > 0 else None,
        unknown_score=unknown_s,
    )

    return ScoringResult(
        score=score,
        agreement_score=agreement,
        bbox_quality=bbox_q,
        unknown_score=unknown_s,
        decision_state=decision_res.decision_state,
        decision_reasons_json=decision_res.reasons_json,
        policy_version=cap_tag,
    )
