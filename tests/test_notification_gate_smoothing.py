"""
P1-05 Notification gating regression tests.

Validates that the detection processing loop gates notifications on the
*smoothed* decision state (not raw), consistent with what gets persisted.

Cases:
1. Smoothing OFF  → old behavior preserved (smoothed == raw).
2. Smoothing ON + raw uncertain + smoothed confirmed → NOTIFIES.
3. Smoothing ON + raw confirmed + smoothed non-confirmed → does NOT notify.
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult, DecisionState
from detectors.interfaces.persistence import (
    DetectionPersistenceResult,
    ImagePersistenceResult,
)
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.temporal_decision_service import TemporalDecisionService

# ---------------------------------------------------------------------------
# Shared fixture — builds a DetectionManager with everything mocked except
# the decision policy + temporal smoothing (those are the subjects under test).
# ---------------------------------------------------------------------------


@pytest.fixture
def _base_manager(monkeypatch, tmp_path):
    """DetectionManager with all heavy IO/model deps mocked out."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    mgr = DetectionManager()

    mgr.persistence_service = MagicMock()
    mgr.notification_service = MagicMock()
    mgr.crop_service = MagicMock()
    mgr.classification_service = MagicMock()

    mgr.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    mgr.persistence_service.save_detection.return_value = DetectionPersistenceResult(
        success=True, thumbnail_path="thumb.webp"
    )
    mgr.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )

    # Deterministic policy service with known thresholds
    mgr.decision_policy_service = DecisionPolicyService(
        config={
            "BBOX_QUALITY_THRESHOLD": "0.40",
            "SPECIES_CONF_THRESHOLD": "0.70",
            "UNKNOWN_SCORE_THRESHOLD": "0.60",
        }
    )

    return mgr


def _make_job() -> dict:
    """Standard detection job with a single well-positioned bbox."""
    return {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "confidence": 0.90}
        ],
        "detection_time_ms": 50,
        "sleep_time_ms": 10,
    }


def _run_one_job(mgr: DetectionManager) -> None:
    """Feed one processing-loop iteration and block until it returns."""
    mgr.processing_queue.put(_make_job())

    call_count = 0

    def _stop_after_one():
        nonlocal call_count
        call_count += 1
        return call_count > 1

    mgr.stop_event.is_set = _stop_after_one
    mgr._processing_loop()


# =========================================================================
# Test 1: Smoothing OFF → old behavior preserved
# =========================================================================


def test_smoothing_off_confirmed_notifies(_base_manager):
    """
    When temporal smoothing is disabled, smoothed == raw.
    A confirmed detection MUST trigger a notification.
    """
    mgr = _base_manager

    # Smoothing OFF
    mgr.temporal_decision_service = TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "false"}
    )

    # High-confidence classification → raw state = CONFIRMED
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.85,
        model_id="test",
        top_k_confidences=[0.85, 0.05, 0.03, 0.02, 0.01],
    )
    mgr.notification_service.should_send.return_value = True

    _run_one_job(mgr)

    mgr.notification_service.queue_detection.assert_called_once()
    mgr.notification_service.send_summary.assert_called_once()


def test_smoothing_off_uncertain_does_not_notify(_base_manager):
    """
    When smoothing is off, a low-confidence detection (UNCERTAIN / UNKNOWN)
    must NOT trigger a notification — verifying old behavior is intact.
    """
    mgr = _base_manager

    mgr.temporal_decision_service = TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "false"}
    )

    # Low confidence → stays UNCERTAIN/UNKNOWN
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.40,
        model_id="test",
        top_k_confidences=[0.40, 0.30, 0.10, 0.10, 0.05],
    )

    _run_one_job(mgr)

    mgr.notification_service.queue_detection.assert_not_called()


# =========================================================================
# Test 2: Smoothing ON + raw uncertain + smoothed confirmed → NOTIFIES
# =========================================================================


def test_smoothing_on_raw_uncertain_smoothed_confirmed_notifies(_base_manager):
    """
    Temporal window is pre-filled with CONFIRMED history.
    Current frame is UNCERTAIN (raw), but the sliding window majority
    votes CONFIRMED (smoothed).  Notification MUST fire.
    """
    mgr = _base_manager

    svc = TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "true"},
        window_size=5,
    )
    # Pre-fill window with 4 × CONFIRMED so majority stays CONFIRMED
    for _ in range(4):
        svc.smooth("Parus_major", DecisionState.CONFIRMED)

    mgr.temporal_decision_service = svc

    # This frame: low CLS confidence → raw decision = UNCERTAIN/UNKNOWN,
    # but bbox is OK.  We need species_conf < threshold to get non-CONFIRMED.
    # Use species_conf=0.65 (< 0.70 threshold) → UNCERTAIN raw state.
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.65,
        model_id="test",
        top_k_confidences=[0.65, 0.15, 0.10, 0.05, 0.03],
    )
    mgr.notification_service.should_send.return_value = True

    _run_one_job(mgr)

    # Smoothed state should be CONFIRMED (4 CONFIRMED + 1 UNCERTAIN → majority = CONFIRMED).
    # Therefore notification fires.
    mgr.notification_service.queue_detection.assert_called_once()
    mgr.notification_service.send_summary.assert_called_once()


# =========================================================================
# Test 3: Smoothing ON + raw confirmed + smoothed non-confirmed → NO notify
# =========================================================================


def test_smoothing_on_raw_confirmed_smoothed_uncertain_no_notify(_base_manager):
    """
    Temporal window is pre-filled with UNCERTAIN history.
    Current frame is CONFIRMED (raw), but the sliding window majority
    votes UNCERTAIN (smoothed).  Notification must NOT fire.
    """
    mgr = _base_manager

    svc = TemporalDecisionService(
        config={"ENABLE_TEMPORAL_SMOOTHING": "true"},
        window_size=5,
    )
    # Pre-fill window with 4 × UNCERTAIN so majority stays UNCERTAIN
    for _ in range(4):
        svc.smooth("Parus_major", DecisionState.UNCERTAIN)

    mgr.temporal_decision_service = svc

    # This frame: high confidence → raw decision = CONFIRMED
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.85,
        model_id="test",
        top_k_confidences=[0.85, 0.05, 0.03, 0.02, 0.01],
    )

    _run_one_job(mgr)

    # Smoothed state = UNCERTAIN (4 UNCERTAIN + 1 CONFIRMED → majority = UNCERTAIN).
    # Therefore NO notification.
    mgr.notification_service.queue_detection.assert_not_called()
