"""Tests for Unknown Score computation (P1-02)."""

import json

from detectors.interfaces.classification import (
    DecisionState,
    compute_unknown_score,
)
from detectors.services.decision_policy_service import DecisionPolicyService


def test_low_margin_increases_unknown_score():
    """
    When top-1 and top-2 are close (small margin), the model is confused.
    The unknown score should be high.
    """
    # Top-1: 0.30, Top-2: 0.28 → margin = 0.02, very uncertain
    top_k = [0.30, 0.28, 0.15, 0.12, 0.10]
    score = compute_unknown_score(top_k)

    assert score > 0.5, f"Low margin should yield high unknown score, got {score}"


def test_peaked_distribution_low_unknown_score():
    """
    When top-1 dominates (large margin, low entropy), the model is confident.
    The unknown score should be low.
    """
    # Top-1: 0.95, Top-2: 0.02 → margin = 0.93, very peaked
    top_k = [0.95, 0.02, 0.01, 0.01, 0.01]
    score = compute_unknown_score(top_k)

    assert score < 0.3, (
        f"Peaked distribution should yield low unknown score, got {score}"
    )


def test_empty_topk_returns_max_unknown():
    """
    When no top-k data is available, we should assume maximum uncertainty.
    """
    assert compute_unknown_score([]) == 1.0
    assert compute_unknown_score(None) == 1.0


def test_single_class_uses_inverse_confidence():
    """
    When only one class probability is available (no margin possible),
    fallback to inverse of top-1 confidence.
    """
    score_high = compute_unknown_score([0.95])
    score_low = compute_unknown_score([0.30])

    assert score_high < score_low, (
        f"High confidence should yield lower unknown score: {score_high} vs {score_low}"
    )
    assert score_high < 0.2


def test_score_is_bounded_zero_one():
    """Unknown score must always be in [0.0, 1.0]."""
    test_cases = [
        [1.0],
        [0.0],
        [0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.99, 0.005, 0.005],
    ]
    for top_k in test_cases:
        s = compute_unknown_score(top_k)
        assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for top_k {top_k}"


# ---------------------------------------------------------------------------
# Decision regression: high unknown overrides medium species conf
# ---------------------------------------------------------------------------


def test_high_unknown_overrides_medium_species_conf():
    """
    Even with decent species confidence (e.g. 0.65) and good bbox,
    a high unknown score should push the decision to UNKNOWN.
    """
    policy = DecisionPolicyService(
        config={
            "BBOX_QUALITY_THRESHOLD": 0.5,
            "SPECIES_CONF_THRESHOLD": 0.7,
            "UNKNOWN_SCORE_THRESHOLD": 0.6,
        }
    )
    result = policy.evaluate(
        bbox_quality=0.8,
        species_conf=0.65,  # below threshold but decent
        unknown_score=0.85,  # clearly high
    )

    assert result.decision_state == DecisionState.UNKNOWN
    reasons = json.loads(result.reasons_json)
    assert "HIGH_UNKNOWN_SCORE" in reasons


def test_missing_species_conf_remains_non_confirmed():
    """
    Regression: species_conf=None must never yield CONFIRMED,
    even with perfect bbox and low unknown score.
    """
    policy = DecisionPolicyService(
        config={
            "BBOX_QUALITY_THRESHOLD": 0.5,
            "SPECIES_CONF_THRESHOLD": 0.7,
            "UNKNOWN_SCORE_THRESHOLD": 0.6,
        }
    )
    result = policy.evaluate(
        bbox_quality=0.95,
        species_conf=None,
        unknown_score=0.1,
    )

    assert result.decision_state != DecisionState.CONFIRMED
    reasons = json.loads(result.reasons_json)
    assert "NO_SPECIES_CONF" in reasons
