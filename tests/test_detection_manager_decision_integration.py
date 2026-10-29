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
from detectors.services.capability_registry import build_default_registry
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


@pytest.mark.parametrize(
    "od_class, od_conf, cls_conf, expected_state",
    [
        ("bird", 0.9, 0.8, DecisionState.CONFIRMED),
        ("bird", 0.9, 0.4, DecisionState.UNCERTAIN),
        ("bird", 0.9, None, DecisionState.UNKNOWN),
        ("squirrel", 0.9, None, DecisionState.UNCERTAIN),
        ("squirrel", 0.7, None, DecisionState.UNCERTAIN),
    ],
)
def test_pipeline_persists_decision_fields(
    mock_detection_manager: DetectionManager,
    od_class: str,
    od_conf: float,
    cls_conf: float | None,
    expected_state: DecisionState,
) -> None:
    manager = mock_detection_manager
    manager.capability_registry = build_default_registry(
        config={"ENABLE_DECISION_POLICY": "false"}
    )
    manager.config = {**manager.config, "NON_BIRD_DROP_BELOW_CONFIRM": False}
    if cls_conf is None:
        manager.classification_service.classify.side_effect = RuntimeError(
            "Classification unavailable"
        )
    else:
        manager.classification_service.classify.return_value = ClassificationResult(
            class_name="parus_major",
            confidence=cls_conf,
            model_id="test_model",
            top_k_confidences=[cls_conf, 0.05, 0.03, 0.02, 0.01],
        )

    # Enqueue a processing job
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {
                "x1": 500,
                "y1": 300,
                "x2": 700,
                "y2": 500,
                "confidence": od_conf,
                "class_name": od_class,
            }
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

    assert det_data.decision_state == expected_state
    assert det_data.decision_state.value
    assert "decision_policy_v1" in det_data.policy_version.split("+")
    if od_class != "bird":
        manager.classification_service.classify.assert_not_called()


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


def test_processing_continues_after_failed_frame(mock_detection_manager) -> None:
    manager = mock_detection_manager
    import queue

    manager.processing_queue = queue.Queue()
    manager.processing_queue.put({"frame": "broken"})
    manager.processing_queue.put({"frame": "next"})
    processed = []

    def process(job: dict[str, object]) -> None:
        processed.append(job["frame"])
        if job["frame"] == "broken":
            raise RuntimeError("bad frame")
        manager.stop_event.set()

    manager._process_job = process
    manager._processing_loop()
    assert processed == ["broken", "next"]


def test_notification_database_error_does_not_stop_processing(
    mock_detection_manager, monkeypatch
) -> None:
    import queue
    import sqlite3
    from unittest.mock import patch

    from detectors.services.notification_service import NotificationService

    manager = mock_detection_manager
    manager.processing_queue = queue.Queue()
    manager.persistence_service.save_detection.return_value.detection_id = 1
    manager.classification_service.classify.return_value = ClassificationResult(
        class_name="Parus_major",
        confidence=0.95,
        model_id="test",
        top_k_confidences=[0.95],
    )
    service = NotificationService()
    service._config = {"TELEGRAM_ENABLED": True}
    monkeypatch.setattr(service, "should_send", lambda: False)
    monkeypatch.setattr(
        "detectors.services.notification_service.is_known_species",
        lambda *args, **kwargs: True,
    )
    manager.notification_service = service
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((200, 200, 3), dtype=np.uint8),
        "detection_info_list": [
            {
                "x1": 10,
                "y1": 10,
                "x2": 100,
                "y2": 100,
                "class_name": "bird",
                "confidence": 0.95,
            }
        ],
        "detection_time_ms": 10,
        "sleep_time_ms": 10,
    }
    manager.processing_queue.put(job)
    manager.processing_queue.put(job)
    manager._burst_admit = lambda **kwargs: True
    manager._same_bird_burst_admit = lambda *args, **kwargs: True
    states = iter((False, False, True))
    manager.stop_event.is_set = lambda: next(states)
    with patch(
        "detectors.services.notification_service.closing_connection",
        side_effect=sqlite3.OperationalError("database is locked"),
    ):
        manager._processing_loop()
    assert manager.persistence_service.save_detection.call_count == 2
    assert service.pending_count == 1
