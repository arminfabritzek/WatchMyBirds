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


@pytest.fixture
def mock_detection_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BBOX_QUALITY_THRESHOLD", "0.5")
    monkeypatch.setenv("SPECIES_CONF_THRESHOLD", "0.7")

    manager = DetectionManager()

    # Mock the internal services
    manager.persistence_service = MagicMock()
    manager.notification_service = MagicMock()
    manager.crop_service = MagicMock()
    manager.classification_service = MagicMock()

    # Ensure policy service is used but replace it with a controlled one
    manager.decision_policy_service = DecisionPolicyService()

    # Mock persistence results so it doesn't crash on return
    manager.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    manager.persistence_service.save_detection.return_value = (
        DetectionPersistenceResult(success=True, thumbnail_path="thumb.webp")
    )

    # Mock crop service to return a dummy image
    manager.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )

    return manager


def test_pipeline_persists_decision_fields(mock_detection_manager):
    # Setup high confidence classification to yield a CONFIRMED state
    mock_detection_manager.classification_service.classify.return_value = (
        ClassificationResult(
            class_name="parus_major",
            confidence=0.8,
            model_id="test_model",
            top_k_confidences=[0.8, 0.05, 0.03, 0.02, 0.01],
        )
    )

    # Enqueue a processing job
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "confidence": 0.9}
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }

    mock_detection_manager.processing_queue.put(job)

    def mock_is_set():
        if getattr(mock_is_set, "called", False):
            return True
        mock_is_set.called = True
        return False

    mock_detection_manager.stop_event.is_set = mock_is_set
    mock_detection_manager._processing_loop()

    # Assert save_detection was called with correct decision fields
    mock_detection_manager.persistence_service.save_detection.assert_called_once()

    call_args = mock_detection_manager.persistence_service.save_detection.call_args[1]
    det_data = call_args["detection"]

    assert det_data.decision_state == DecisionState.CONFIRMED
    assert det_data.decision_reasons == "[]"
    assert "decision_policy" in det_data.policy_version


def test_uncertain_decision_suppresses_species_notification(
    mock_detection_manager, monkeypatch
):
    # Enable notification mock
    mock_detection_manager.notification_service.should_send.return_value = True

    # Setup low confidence classification to yield an UNCERTAIN state
    mock_detection_manager.classification_service.classify.return_value = (
        ClassificationResult(
            class_name="parus_major",
            confidence=0.4,
            model_id="test_model",
            top_k_confidences=[0.4, 0.3, 0.1, 0.1, 0.05],
        )
    )

    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "confidence": 0.9}
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }

    mock_detection_manager.processing_queue.put(job)

    def mock_is_set():
        if getattr(mock_is_set, "called", False):
            return True
        mock_is_set.called = True
        return False

    mock_detection_manager.stop_event.is_set = mock_is_set
    mock_detection_manager._processing_loop()

    # Verify save_detection executed and noted the uncertainty
    call_args = mock_detection_manager.persistence_service.save_detection.call_args[1]
    det_data = call_args["detection"]
    assert det_data.decision_state == DecisionState.UNKNOWN
    assert "LOW_SPECIES_CONF" in det_data.decision_reasons
    assert "HIGH_UNKNOWN_SCORE" in det_data.decision_reasons

    # Verify notification service was NEVER called for queuing a species summary
    mock_detection_manager.notification_service.queue_detection.assert_not_called()
    mock_detection_manager.notification_service.send_summary.assert_not_called()


def test_policy_disabled_uses_conservative_notification_gate(monkeypatch, tmp_path):
    """When ENABLE_DECISION_POLICY=false, decision_state is None and notifications
    use the conservative legacy gate (cls_conf > 0 AND score >= SAVE_THRESHOLD)."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()

    # Inject a registry with decision_policy disabled
    from detectors.services.capability_registry import build_default_registry

    manager.capability_registry = build_default_registry(
        config={"ENABLE_DECISION_POLICY": "false"}
    )

    # Mock services
    manager.persistence_service = MagicMock()
    manager.notification_service = MagicMock()
    manager.crop_service = MagicMock()
    manager.classification_service = MagicMock()
    manager.notification_service.should_send.return_value = True

    manager.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    manager.persistence_service.save_detection.return_value = (
        DetectionPersistenceResult(success=True, thumbnail_path="thumb.webp")
    )
    manager.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )

    # High-confidence detection → should pass legacy gate
    manager.classification_service.classify.return_value = ClassificationResult(
        class_name="parus_major",
        confidence=0.8,
        model_id="test_model",
        top_k_confidences=[0.8, 0.05, 0.03, 0.02, 0.01],
    )

    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "confidence": 0.9}
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }

    manager.processing_queue.put(job)

    def mock_is_set():
        if getattr(mock_is_set, "called", False):
            return True
        mock_is_set.called = True
        return False

    manager.stop_event.is_set = mock_is_set
    manager._processing_loop()

    # Verify decision_state is None (policy off)
    call_args = manager.persistence_service.save_detection.call_args[1]
    det_data = call_args["detection"]
    assert det_data.decision_state is None
    assert det_data.decision_reasons == "[]"

    # Score=0.85 (0.5*0.9+0.5*0.8) >= SAVE_THRESHOLD(0.65) AND cls_conf=0.8 > 0
    # → conservative gate passes → notification should be sent
    manager.notification_service.queue_detection.assert_called_once()
    manager.notification_service.send_summary.assert_called_once()


def test_policy_disabled_suppresses_notification_below_threshold(monkeypatch, tmp_path):
    """When ENABLE_DECISION_POLICY=false and score < SAVE_THRESHOLD,
    the conservative legacy gate blocks notification."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()

    # Inject a registry with decision_policy disabled
    from detectors.services.capability_registry import build_default_registry

    manager.capability_registry = build_default_registry(
        config={"ENABLE_DECISION_POLICY": "false"}
    )

    manager.persistence_service = MagicMock()
    manager.notification_service = MagicMock()
    manager.crop_service = MagicMock()
    manager.classification_service = MagicMock()
    manager.notification_service.should_send.return_value = True

    manager.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    manager.persistence_service.save_detection.return_value = (
        DetectionPersistenceResult(success=True, thumbnail_path="thumb.webp")
    )
    manager.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )

    # Low-confidence detection → score below SAVE_THRESHOLD
    manager.classification_service.classify.return_value = ClassificationResult(
        class_name="parus_major",
        confidence=0.3,
        model_id="test_model",
        top_k_confidences=[0.3, 0.25, 0.2, 0.15, 0.05],
    )

    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 500, "y1": 300, "x2": 700, "y2": 500, "confidence": 0.5}
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }

    manager.processing_queue.put(job)

    def mock_is_set():
        if getattr(mock_is_set, "called", False):
            return True
        mock_is_set.called = True
        return False

    manager.stop_event.is_set = mock_is_set
    manager._processing_loop()

    # Score=0.4 (0.5*0.5+0.5*0.3) < SAVE_THRESHOLD(0.65)
    # → conservative gate blocks → no notification
    manager.notification_service.queue_detection.assert_not_called()
    manager.notification_service.send_summary.assert_not_called()
