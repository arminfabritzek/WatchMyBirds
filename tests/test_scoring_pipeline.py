"""
Tests for detectors.services.scoring_pipeline — the single source of truth
for composite score, agreement, bbox quality, unknown score, decision
evaluation, temporal smoothing, and version tagging.
"""

import pytest

from detectors.interfaces.classification import DecisionState
from detectors.services.bbox_quality_service import compute_bbox_quality
from detectors.services.capability_registry import build_default_registry
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.scoring_pipeline import ScoringResult, compute_detection_signals
from detectors.services.temporal_decision_service import TemporalDecisionService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hd_frame_shape() -> tuple[int, ...]:
    """Standard 1080p frame shape."""
    return (1080, 1920, 3)


@pytest.fixture()
def good_bbox() -> tuple[int, int, int, int]:
    """Well-centred, medium-sized bbox on an HD frame."""
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
# Happy path — classification available with full top-k
# ---------------------------------------------------------------------------


def test_happy_path_full_topk(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """With cls_conf > 0 and top-k available, the composite score formula
    ``0.5 * od + 0.5 * cls`` applies and agreement = min(od, cls)."""
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
    )

    assert isinstance(result, ScoringResult)
    assert result.score == pytest.approx(0.85)  # 0.5*0.9 + 0.5*0.8
    assert result.agreement_score == pytest.approx(0.8)  # min(0.9, 0.8)
    assert result.bbox_quality is not None
    assert result.unknown_score is not None
    assert result.decision_state == DecisionState.CONFIRMED
    assert result.policy_version  # non-empty version tag


# ---------------------------------------------------------------------------
# No classification — cls_conf == 0
# ---------------------------------------------------------------------------


def test_no_classification_fallback(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """When cls_conf == 0, score == od_conf, agreement == od_conf,
    and unknown_score == 1.0 (max uncertainty)."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.7,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="unknown",
    )

    assert result.score == pytest.approx(0.7)
    assert result.agreement_score == pytest.approx(0.7)
    assert result.unknown_score == pytest.approx(1.0)
    # No species conf → should NOT be confirmed
    assert result.decision_state != DecisionState.CONFIRMED


# ---------------------------------------------------------------------------
# Single-class fallback — top_k has 1 entry
# ---------------------------------------------------------------------------


def test_single_class_topk_fallback(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """When top_k_confidences is None but cls_conf > 0,
    the pipeline falls back to [cls_conf] as single-class top-k."""
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.9,
        cls_conf=0.85,
        top_k_confidences=None,  # no top-k provided
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="Parus_major",
    )

    assert result.score == pytest.approx(0.875)  # 0.5*0.9 + 0.5*0.85
    assert result.agreement_score == pytest.approx(0.85)
    # Single-class fallback: unknown_score = 1.0 - cls_conf
    assert result.unknown_score == pytest.approx(1.0 - 0.85)


# ---------------------------------------------------------------------------
# Score formula correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "od_conf, cls_conf, expected_score, expected_agreement",
    [
        (0.9, 0.8, 0.85, 0.8),
        (0.5, 0.5, 0.50, 0.5),
        (1.0, 0.6, 0.80, 0.6),
        (0.3, 0.9, 0.60, 0.3),
    ],
)
def test_score_formula(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
    od_conf,
    cls_conf,
    expected_score,
    expected_agreement,
):
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=od_conf,
        cls_conf=cls_conf,
        top_k_confidences=[cls_conf, 0.01],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="test_species",
    )

    assert result.score == pytest.approx(expected_score)
    assert result.agreement_score == pytest.approx(expected_agreement)


# ---------------------------------------------------------------------------
# BBox quality is computed correctly
# ---------------------------------------------------------------------------


def test_bbox_quality_matches_direct_call(
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    """The pipeline must produce the same bbox_quality as a direct call
    to compute_bbox_quality."""
    bbox = (100, 100, 500, 500)
    expected_q = compute_bbox_quality(bbox, hd_frame_shape)

    result = compute_detection_signals(
        bbox=bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.9,
        cls_conf=0.8,
        top_k_confidences=[0.8, 0.05],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="test",
    )

    assert result.bbox_quality == pytest.approx(expected_q)


# ---------------------------------------------------------------------------
# Version tag is non-empty
# ---------------------------------------------------------------------------


def test_version_tag_non_empty(
    good_bbox,
    hd_frame_shape,
    decision_policy,
    temporal_service,
    capability_registry,
):
    result = compute_detection_signals(
        bbox=good_bbox,
        frame_shape=hd_frame_shape,
        od_conf=0.9,
        cls_conf=0.8,
        top_k_confidences=[0.8, 0.05],
        decision_policy=decision_policy,
        temporal_service=temporal_service,
        capability_registry=capability_registry,
        species_key="test",
    )

    assert result.policy_version
    assert "+" in result.policy_version or result.policy_version != "none"
