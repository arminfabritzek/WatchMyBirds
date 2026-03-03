import json

import pytest

from detectors.interfaces.classification import DecisionState
from detectors.services.decision_policy_service import (
    REASON_HIGH_UNKNOWN_SCORE,
    REASON_LOW_BBOX_QUALITY,
    REASON_LOW_SPECIES_CONF,
    REASON_NO_SPECIES_CONF,
    DecisionPolicyService,
)


@pytest.fixture
def policy_engine():
    """Provides a deterministic Policy Engine for scenario validation."""
    config = {
        "BBOX_QUALITY_THRESHOLD": 0.5,
        "SPECIES_CONF_THRESHOLD": 0.7,
        "UNKNOWN_SCORE_THRESHOLD": 0.6,
    }
    return DecisionPolicyService(config=config)


def test_bbox_wrong_species_right(policy_engine):
    """
    Scenario 1: BBox is bad, but species model is highly confident.
    Expected: UNCERTAIN. We shouldn't blindly trust a bad crop.
    """
    # bbox_quality < 0.5 (bad), species_conf >= 0.7 (good), unknown < 0.6 (good)
    result = policy_engine.evaluate(
        bbox_quality=0.3, species_conf=0.85, unknown_score=0.2
    )

    assert result.decision_state == DecisionState.UNCERTAIN
    reasons = json.loads(result.reasons_json)
    assert REASON_LOW_BBOX_QUALITY in reasons
    assert REASON_LOW_SPECIES_CONF not in reasons
    assert REASON_HIGH_UNKNOWN_SCORE not in reasons


def test_bbox_wrong_species_wrong(policy_engine):
    """
    Scenario 2: BBox is bad AND species is badly recognized.
    Expected: REJECTED. It's likely junk/background.
    """
    # bbox_quality < 0.5 (bad), species_conf < 0.7 (bad), unknown < 0.6 (good)
    result = policy_engine.evaluate(
        bbox_quality=0.2, species_conf=0.4, unknown_score=0.3
    )

    assert result.decision_state == DecisionState.REJECTED
    reasons = json.loads(result.reasons_json)
    assert REASON_LOW_BBOX_QUALITY in reasons
    assert REASON_LOW_SPECIES_CONF in reasons
    assert REASON_HIGH_UNKNOWN_SCORE not in reasons


def test_bbox_right_species_wrong(policy_engine):
    """
    Scenario 3: Good bounding box, but species model fails/low confidence.
    Expected: UNCERTAIN. Might be a bird we don't know, or bad lighting.
    """
    # bbox_quality >= 0.5 (good), species_conf < 0.7 (bad), unknown < 0.6 (good)
    result = policy_engine.evaluate(
        bbox_quality=0.8, species_conf=0.5, unknown_score=0.4
    )

    assert result.decision_state == DecisionState.UNCERTAIN
    reasons = json.loads(result.reasons_json)
    assert REASON_LOW_BBOX_QUALITY not in reasons
    assert REASON_LOW_SPECIES_CONF in reasons
    assert REASON_HIGH_UNKNOWN_SCORE not in reasons


def test_unknown_species_bbox_wrong(policy_engine):
    """
    Scenario 4: High uncertainty/out-of-distribution score AND bad bbox.
    Expected: UNKNOWN. Uncertainty score takes precedence over rejection.
    """
    # bbox_quality < 0.5 (bad), species_conf < 0.7 (bad), unknown >= 0.6 (bad/high)
    result = policy_engine.evaluate(
        bbox_quality=0.3, species_conf=0.3, unknown_score=0.8
    )

    assert result.decision_state == DecisionState.UNKNOWN
    reasons = json.loads(result.reasons_json)
    assert REASON_LOW_BBOX_QUALITY in reasons
    assert REASON_LOW_SPECIES_CONF in reasons
    assert REASON_HIGH_UNKNOWN_SCORE in reasons


def test_unknown_species_bbox_right(policy_engine):
    """
    Scenario 5: Good bounding box, but very high uncertainty (OOD).
    Expected: UNKNOWN. Clearly a localized object, but model explicitly flags as unknown.
    """
    # bbox_quality >= 0.5 (good), species_conf < 0.7 (bad), unknown >= 0.6 (bad/high)
    result = policy_engine.evaluate(
        bbox_quality=0.9, species_conf=0.4, unknown_score=0.75
    )

    assert result.decision_state == DecisionState.UNKNOWN
    reasons = json.loads(result.reasons_json)
    assert REASON_LOW_BBOX_QUALITY not in reasons
    assert REASON_LOW_SPECIES_CONF in reasons
    assert REASON_HIGH_UNKNOWN_SCORE in reasons


# ---------------------------------------------------------------------------
# Regression: P0-Hotfix — species_conf=None must NEVER produce CONFIRMED
# ---------------------------------------------------------------------------


def test_all_signals_none_never_confirmed(policy_engine):
    """
    Regression: When all signals are None (e.g. classification never ran),
    the detection must NOT be marked CONFIRMED.
    """
    result = policy_engine.evaluate(
        bbox_quality=None, species_conf=None, unknown_score=None
    )

    assert result.decision_state != DecisionState.CONFIRMED, (
        "species_conf=None must not produce CONFIRMED"
    )
    assert result.decision_state == DecisionState.UNCERTAIN
    reasons = json.loads(result.reasons_json)
    assert REASON_NO_SPECIES_CONF in reasons


def test_good_bbox_no_classification_never_confirmed(policy_engine):
    """
    Regression: Good bbox but classification didn't run (species_conf=None).
    Must be UNCERTAIN, not CONFIRMED.
    """
    result = policy_engine.evaluate(
        bbox_quality=0.9, species_conf=None, unknown_score=0.1
    )

    assert result.decision_state != DecisionState.CONFIRMED, (
        "Missing species_conf with good bbox must not produce CONFIRMED"
    )
    assert result.decision_state == DecisionState.UNCERTAIN
    reasons = json.loads(result.reasons_json)
    assert REASON_NO_SPECIES_CONF in reasons
    assert REASON_LOW_BBOX_QUALITY not in reasons
