"""Tests for the non-bird track of detectors.services.scoring_pipeline.

These cover:

- non-bird detections skip CLS yet survive the pipeline
- score = od_conf (no CLS mixing)
- unknown_score = 0.0 (not the killer 1.0 from compute_unknown_score([]))
- decision_state = CONFIRMED when od_conf >= non_bird_confirm_threshold
- decision_state = UNCERTAIN when below threshold
- bird track still behaves as before (regression guard)
"""

from __future__ import annotations

import pytest

from detectors.interfaces.classification import DecisionState
from detectors.services.capability_registry import build_default_registry
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.scoring_pipeline import compute_detection_signals
from detectors.services.temporal_decision_service import TemporalDecisionService


@pytest.fixture()
def hd_frame_shape() -> tuple[int, ...]:
    return (1080, 1920, 3)


@pytest.fixture()
def good_bbox() -> tuple[int, int, int, int]:
    return (400, 300, 700, 600)


@pytest.fixture()
def decision_policy() -> DecisionPolicyService:
    return DecisionPolicyService()


@pytest.fixture()
def temporal_service() -> TemporalDecisionService:
    return TemporalDecisionService()


@pytest.fixture()
def capability_registry():
    return build_default_registry()


# ---------------------------------------------------------------------------
# Non-bird track — high OD confidence
# ---------------------------------------------------------------------------


def test_non_bird_high_conf_confirmed(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """squirrel at od=0.90 with no CLS -> CONFIRMED, score=od, unknown=0."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.90,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="squirrel",
        od_class_name="squirrel",
        non_bird_confirm_threshold=0.65,
    )

    assert result.score == pytest.approx(0.90)
    assert result.agreement_score == pytest.approx(0.90)
    assert result.unknown_score == pytest.approx(0.0)  # NOT 1.0
    assert result.decision_state == DecisionState.CONFIRMED
    # Non-bird bypass does NOT consult the decision policy -> no reasons
    assert result.decision_reasons_json == "[]"


def test_non_bird_low_conf_uncertain(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """squirrel at od=0.30 below threshold -> UNCERTAIN."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.30,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="squirrel",
        od_class_name="squirrel",
        non_bird_confirm_threshold=0.65,
    )

    assert result.decision_state == DecisionState.UNCERTAIN
    assert result.score == pytest.approx(0.30)
    assert result.unknown_score == pytest.approx(0.0)


@pytest.mark.parametrize(
    "class_name",
    ["squirrel", "cat", "marten_mustelid", "hedgehog"],
)
def test_non_bird_all_garden_animals_confirmed(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    class_name,
):
    """All four non-bird classes follow the same non-bird pipeline."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.85,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key=class_name,
        od_class_name=class_name,
        non_bird_confirm_threshold=0.65,
    )
    assert result.decision_state == DecisionState.CONFIRMED


def test_non_bird_threshold_boundary(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """od_conf == non_bird_confirm_threshold -> CONFIRMED (>=)."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.65,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="squirrel",
        od_class_name="squirrel",
        non_bird_confirm_threshold=0.65,
    )
    assert result.decision_state == DecisionState.CONFIRMED


# ---------------------------------------------------------------------------
# Bird track — legacy behaviour (regression guard)
# ---------------------------------------------------------------------------


def test_bird_track_with_cls_still_uses_cls_conf(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """od_class_name='bird' with cls_conf -> score = cls_conf (bird track)."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.9,
        cls_conf=0.8,
        top_k_confidences=[0.8, 0.05, 0.03, 0.02, 0.01],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="Parus_major",
        od_class_name="bird",
        non_bird_confirm_threshold=0.65,
    )
    assert result.score == pytest.approx(0.8)
    assert result.agreement_score == pytest.approx(0.8)
    assert result.decision_state == DecisionState.CONFIRMED


def test_bird_track_without_cls_uses_od(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """od_class_name='bird' with cls_conf=0 -> score = od_conf fallback.

    But unknown_score = 1.0 because CLS never ran; decision policy then
    classifies this as UNKNOWN (not CONFIRMED). This mirrors the legacy
    behaviour before Phase 1 and is intentionally kept for birds with CLS
    failures. Non-bird gets a different treatment.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.9,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="unknown",
        od_class_name="bird",
        non_bird_confirm_threshold=0.65,
    )
    assert result.score == pytest.approx(0.9)
    assert result.agreement_score == pytest.approx(0.9)
    assert result.unknown_score == pytest.approx(1.0)
    # Bird without CLS -> decision policy kicks in with unknown_score=1.0
    # and marks UNKNOWN; that's fine — it keeps the crop for review.
    assert result.decision_state == DecisionState.UNKNOWN


def test_default_od_class_name_is_bird_track():
    """Legacy callers that don't pass od_class_name still get bird semantics.

    compute_detection_signals() without od_class_name defaults to the bird
    track, preserving backwards compatibility with older call-sites.
    """
    result = compute_detection_signals(
        bbox=(400, 300, 700, 600),
        frame_shape=(1080, 1920, 3),
        od_conf=0.9,
        cls_conf=0.8,
        top_k_confidences=[0.8, 0.05],
        decision_policy=DecisionPolicyService(),
        temporal_service=TemporalDecisionService(),
        capability_registry=build_default_registry(),
        species_key="Parus_major",
        # no od_class_name, no non_bird_confirm_threshold
    )
    assert result.score == pytest.approx(0.8)  # bird track: score = cls_conf


# ---------------------------------------------------------------------------
# Default-threshold sourcing — pinned at 0.80 by the night-FP suppression plan
# ---------------------------------------------------------------------------


def test_default_non_bird_threshold_is_80():
    """The default DEFAULT_NON_BIRD_CONFIRM_THRESHOLD is 0.80.

    Live RPi data on 2026-05-10 showed marten_mustelid never exceeds OD
    confidence 0.77 across 1760 detections in 7 days. A 0.80 default gates
    every observed marten while leaving bird detections (separate track)
    unaffected.
    """
    from detectors.services.scoring_pipeline import DEFAULT_NON_BIRD_CONFIRM_THRESHOLD

    assert DEFAULT_NON_BIRD_CONFIRM_THRESHOLD == pytest.approx(0.80)


def test_non_bird_marten_at_077_is_uncertain_with_default(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """marten_mustelid at the observed live max (0.77) -> UNCERTAIN by default.

    This is the key behaviour change: the same detection that would have
    been CONFIRMED under the previous 0.65 default is now correctly
    classified as UNCERTAIN, removing it from gallery / Stream / Telegram
    surfaces while keeping it in the DB for review.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.77,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
        # no explicit non_bird_confirm_threshold -> use the new 0.80 default
    )
    assert result.decision_state == DecisionState.UNCERTAIN


def test_non_bird_high_conf_cat_still_confirms_at_default(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """A genuine cat sighting (od=0.90) still CONFIRMS at the new default.

    Cats max out at 0.92 in the live data — they pass the 0.80 floor cleanly
    when the OD model is genuinely confident. Only the low-confidence,
    static-bbox night triggers fall through.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.90,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="cat",
        od_class_name="cat",
        # no explicit non_bird_confirm_threshold -> use the new 0.80 default
    )
    assert result.decision_state == DecisionState.CONFIRMED


def test_bird_track_at_077_unaffected_by_non_bird_default(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """Bird detections at OD 0.77 are unaffected by the non-bird default.

    Critical regression guard for long-sitter species: a Columba_palumbus
    or Garrulus_glandarius detection at the same OD confidence as a flagged
    marten must still flow through the bird track and be governed by CLS
    confidence, not by NON_BIRD_CONFIRM_THRESHOLD.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.77,
        cls_conf=0.85,
        top_k_confidences=[0.85, 0.05, 0.03, 0.02, 0.01],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="Columba_palumbus",
        od_class_name="bird",
        # no explicit non_bird_confirm_threshold -> default does not apply
    )
    # Bird track: score is driven by CLS, not OD. Decision policy uses
    # CLS confidence and bbox quality; with CLS=0.85 and a healthy bbox
    # this lands CONFIRMED regardless of OD being below 0.80.
    assert result.score == pytest.approx(0.85)
    assert result.decision_state == DecisionState.CONFIRMED
