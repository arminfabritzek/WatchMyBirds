"""Tests for Temporal Decision Service (P2-01)."""

import pytest

from detectors.interfaces.classification import DecisionState
from detectors.services.temporal_decision_service import TemporalDecisionService


@pytest.fixture
def enabled_service():
    """Service with temporal smoothing enabled, window=5."""
    return TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "true"},
        window_size=5,
    )


@pytest.fixture
def disabled_service():
    """Service with temporal smoothing disabled."""
    return TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "false"},
        window_size=5,
    )


def test_flicker_reduction(enabled_service):
    """
    A single UNCERTAIN frame amidst CONFIRMED ones should be
    smoothed out by majority vote.
    """
    svc = enabled_service
    key = "Parus_major"

    # Feed 4 CONFIRMED
    for _ in range(4):
        result = svc.smooth(key, DecisionState.CONFIRMED)

    # Now a single UNCERTAIN flicker
    result = svc.smooth(key, DecisionState.UNCERTAIN)

    # Window = [C, C, C, C, U] → majority = CONFIRMED
    assert result == DecisionState.CONFIRMED, (
        f"Single flicker should be smoothed out, got {result}"
    )


def test_sustained_change_propagates(enabled_service):
    """
    When the real state consistently changes, the smoothed state
    should eventually follow.
    """
    svc = enabled_service
    key = "Parus_major"

    # Fill window with CONFIRMED
    for _ in range(5):
        svc.smooth(key, DecisionState.CONFIRMED)

    # Now switch to UNCERTAIN consistently
    for _ in range(3):
        result = svc.smooth(key, DecisionState.UNCERTAIN)

    # Window = [C, C, U, U, U] → majority = UNCERTAIN
    assert result == DecisionState.UNCERTAIN


def test_disabled_returns_raw_state(disabled_service):
    """When disabled, smooth() must return the raw state unchanged."""
    svc = disabled_service

    # Even after feeding multiple states, disabled always returns raw
    for _ in range(3):
        svc.smooth("test_species", DecisionState.CONFIRMED)

    result = svc.smooth("test_species", DecisionState.UNCERTAIN)
    assert result == DecisionState.UNCERTAIN, "Disabled service must return raw state"


def test_separate_species_windows(enabled_service):
    """Each species should have its own independent window."""
    svc = enabled_service

    # Fill species A with CONFIRMED
    for _ in range(5):
        svc.smooth("species_a", DecisionState.CONFIRMED)

    # Fill species B with UNKNOWN
    for _ in range(5):
        svc.smooth("species_b", DecisionState.UNKNOWN)

    # Single flicker on A should stay CONFIRMED
    result_a = svc.smooth("species_a", DecisionState.UNKNOWN)
    assert result_a == DecisionState.CONFIRMED

    # Single flicker on B should stay UNKNOWN
    result_b = svc.smooth("species_b", DecisionState.CONFIRMED)
    assert result_b == DecisionState.UNKNOWN


def test_reset_clears_window(enabled_service):
    """Reset should clear the window, making the next call non-smoothed."""
    svc = enabled_service
    key = "test"

    for _ in range(5):
        svc.smooth(key, DecisionState.CONFIRMED)

    svc.reset(key)

    # After reset, first frame should be returned as-is
    result = svc.smooth(key, DecisionState.UNCERTAIN)
    assert result == DecisionState.UNCERTAIN


def test_fallback_without_track_context(enabled_service):
    """
    First frame for a new species (no history) should return
    the raw state since there's only one entry in the window.
    """
    svc = enabled_service
    result = svc.smooth("new_species", DecisionState.UNKNOWN)
    assert result == DecisionState.UNKNOWN
