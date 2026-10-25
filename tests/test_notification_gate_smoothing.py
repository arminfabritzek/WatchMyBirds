"""Notification and MQTT gating follow the persisted capture decision."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult
from detectors.interfaces.persistence import (
    DetectionPersistenceResult,
    ImagePersistenceResult,
)
from detectors.services.decision_policy_service import DecisionPolicyService

# ---------------------------------------------------------------------------
# Shared fixture — builds a DetectionManager with everything mocked except
# the decision policy (the subject under test).
# ---------------------------------------------------------------------------


@pytest.fixture
def _base_manager(monkeypatch, tmp_path):
    """DetectionManager with all heavy IO/model deps mocked out."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    mgr = DetectionManager()

    mgr.persistence_service = MagicMock()
    mgr.notification_service = MagicMock()
    mgr.mqtt_detection_service = MagicMock()
    mgr.crop_service = MagicMock()
    mgr.classification_service = MagicMock()

    mgr.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    mgr.persistence_service.save_detection.return_value = DetectionPersistenceResult(
        success=True,
        detection_id=42,
        thumbnail_path="thumb.webp",
        thumbnail_filename="20260919_114203_000000_crop_0.webp",
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
            "GALLERY_DISPLAY_THRESHOLD": "0.65",
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


def test_confirmed_notifies(_base_manager):
    """A confirmed detection triggers a notification."""
    mgr = _base_manager

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
    mgr.mqtt_detection_service.publish_detection.assert_called_once()


def test_uncertain_does_not_notify(_base_manager):
    """An uncertain or unknown detection must not trigger a notification."""
    mgr = _base_manager

    # Low confidence → stays UNCERTAIN/UNKNOWN
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.40,
        model_id="test",
        top_k_confidences=[0.40, 0.30, 0.10, 0.10, 0.05],
    )

    _run_one_job(mgr)

    mgr.notification_service.queue_detection.assert_not_called()
    mgr.mqtt_detection_service.publish_detection.assert_not_called()


def test_mqtt_failure_does_not_interrupt_detection_or_telegram(_base_manager):
    mgr = _base_manager
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.85,
        model_id="test",
        top_k_confidences=[0.85, 0.05, 0.03, 0.02, 0.01],
    )
    mgr.notification_service.should_send.return_value = True
    mgr.mqtt_detection_service.publish_detection.side_effect = RuntimeError(
        "broker unavailable"
    )

    _run_one_job(mgr)

    mgr.persistence_service.save_detection.assert_called_once()
    mgr.notification_service.send_summary.assert_called_once()


def test_mqtt_event_is_independent_of_telegram_mode(_base_manager):
    mgr = _base_manager
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.85,
        model_id="test",
        top_k_confidences=[0.85, 0.05, 0.03, 0.02, 0.01],
    )
    mgr.notification_service.should_send.return_value = False

    _run_one_job(mgr)

    mgr.notification_service.send_summary.assert_not_called()
    mgr.mqtt_detection_service.publish_detection.assert_called_once()
