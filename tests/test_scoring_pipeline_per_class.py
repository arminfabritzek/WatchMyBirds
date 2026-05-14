"""Per-class non-bird confirm-threshold tests for scoring_pipeline.

The non-bird CONFIRMED/UNCERTAIN gate accepts an optional
``non_bird_confirm_threshold_fn`` resolver. When set, it takes
precedence over the scalar fallback. Used by v2-coco's per-class
operating points (person 0.30, marten 0.45, hedgehog 0.35, squirrel/cat
0.70). 5-class models without a per-class block continue using the
scalar — verified by the existing test_scoring_pipeline_non_bird.py
suite, which we re-run as a regression guard alongside this file.
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


@pytest.fixture()
def v2_coco_resolver():
    """Per-class resolver matching v2-coco YAML thresholds.

    Unlisted classes fall back to 0.80 — same shape the live code path
    builds from detector.conf_per_class_name + config scalar fallback.
    """
    table = {
        "bird": 0.30,
        "squirrel": 0.70,
        "cat": 0.70,
        "marten_mustelid": 0.45,
        "hedgehog": 0.35,
        "person": 0.30,
    }
    return lambda name: table.get(name, 0.80)


# ---------------------------------------------------------------------------
# Per-class wins over scalar
# ---------------------------------------------------------------------------


def test_person_at_032_confirmed_with_per_class_resolver(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """Person at 0.32 passes its per-class floor (0.30); would FAIL the
    scalar 0.80 fallback."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.32,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="person",
        od_class_name="person",
        non_bird_confirm_threshold=0.80,  # ignored
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    assert result.decision_state == DecisionState.CONFIRMED


def test_person_at_028_uncertain_with_per_class_resolver(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """Person at 0.28 below its 0.30 floor -> UNCERTAIN."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.28,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="person",
        od_class_name="person",
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    assert result.decision_state == DecisionState.UNCERTAIN


def test_marten_at_050_confirmed_per_class(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """Marten at 0.50 passes its 0.45 floor — would FAIL the global 0.80."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.50,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    assert result.decision_state == DecisionState.CONFIRMED


def test_squirrel_at_060_uncertain_per_class(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """Squirrel at 0.60 below its 0.70 floor -> UNCERTAIN.

    This is *stricter* than the old global 0.80 — the per-class system
    can also tighten the gate, not just loosen it. Surfaces the value
    of model-owned thresholds: per-class can be high (squirrel) or low
    (person) on the same release.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.60,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="squirrel",
        od_class_name="squirrel",
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    assert result.decision_state == DecisionState.UNCERTAIN


# ---------------------------------------------------------------------------
# Fallback path — scalar still works when no resolver supplied
# ---------------------------------------------------------------------------


def test_scalar_path_still_works_when_fn_none(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """5-class regression: with non_bird_confirm_threshold_fn=None,
    falls back to the scalar. Marten at 0.40 with scalar 0.80 ->
    UNCERTAIN. Locks the fallback path in place."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.40,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
        non_bird_confirm_threshold=0.80,
        non_bird_confirm_threshold_fn=None,
    )
    assert result.decision_state == DecisionState.UNCERTAIN


def test_unlisted_class_falls_back_to_resolver_default(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """If a future model emits a class the resolver doesn't know about
    (e.g. 'fox'), the resolver's own fallback (0.80) applies."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.70,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="fox",
        od_class_name="fox",
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    # 0.70 < resolver's 0.80 fallback -> UNCERTAIN.
    assert result.decision_state == DecisionState.UNCERTAIN


# ---------------------------------------------------------------------------
# Bird track regression guard — per-class resolver MUST NOT touch birds
# ---------------------------------------------------------------------------


def test_bird_track_unaffected_by_per_class_resolver(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    v2_coco_resolver,
):
    """Long-sitter regression guard from the parked non-bird plan.

    Bird at OD 0.77 + CLS 0.85 must stay CONFIRMED — bird track uses
    CLS confidence, not OD, and never consults the non-bird resolver.
    """
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.77,
        cls_conf=0.85,
        top_k_confidences=[0.85, 0.05, 0.02, 0.01],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="Columba_palumbus",
        od_class_name="bird",
        non_bird_confirm_threshold_fn=v2_coco_resolver,
    )
    # Bird at high CLS -> CONFIRMED regardless of any non-bird resolver.
    assert result.decision_state == DecisionState.CONFIRMED
