from __future__ import annotations

import queue
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import config
from camera.video_capture import CapturedFrame, VideoCapture
from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult
from detectors.interfaces.detection import DetectionResult


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DetectionManager:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    settings = config._load_config()
    monkeypatch.setattr(config, "_CONFIG", settings)
    monkeypatch.setattr("detectors.detection_manager.config", settings)
    return DetectionManager()


def _set_frame(manager: DetectionManager, frame_id: int, age: float = 0) -> None:
    manager.latest_raw_frame = np.zeros((40, 80, 3), dtype=np.uint8)
    manager.latest_raw_frame_id = frame_id
    manager.latest_raw_timestamp = time.time() - age
    manager.latest_raw_monotonic = time.monotonic() - age


def test_a_frame_is_detected_only_once(manager: DetectionManager) -> None:
    _set_frame(manager, 1)

    assert manager._take_detection_frame() is not None
    assert manager._take_detection_frame() is None

    _set_frame(manager, 2)
    assert manager._take_detection_frame() is not None


def test_detection_owns_a_copy_and_preserves_receipt_time(
    manager: DetectionManager,
) -> None:
    _set_frame(manager, 1, age=1.0)
    sample = manager._take_detection_frame()

    assert sample is not None
    assert sample.received_at == manager.latest_raw_timestamp
    assert sample.received_monotonic == manager.latest_raw_monotonic
    manager.latest_raw_frame[:] = 255
    assert not sample.frame.any()


def test_stalled_stream_does_not_supply_old_frames(manager: DetectionManager) -> None:
    _set_frame(manager, 1, age=6.0)

    assert manager._take_detection_frame() is None


def test_clock_adjustment_does_not_expire_a_fresh_frame(
    manager: DetectionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_frame(manager, 1)
    monkeypatch.setattr("detectors.detection_manager.time.time", lambda: 999999999999)

    assert manager._take_detection_frame() is not None


def test_capture_keeps_metadata_and_legacy_array_api() -> None:
    capture = VideoCapture.__new__(VideoCapture)
    capture.frame_queue = queue.Queue()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    first = CapturedFrame(frame, 1, 100, 200)
    latest = CapturedFrame(frame, 2, 101, 201)
    capture.frame_queue.put(first)
    capture.frame_queue.put(latest)

    assert capture.get_frame_sample() is latest
    assert capture.get_frame_sample() is None
    capture.frame_queue.put(first)
    assert capture.get_frame() is frame


def test_processing_job_uses_frame_receipt_time(
    manager: DetectionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_frame(manager, 7, age=1.0)
    monkeypatch.setattr(manager, "_initialize_components", lambda: True)
    monkeypatch.setattr(manager, "_should_run_od_now", lambda: True)
    monkeypatch.setitem(manager.config, "MOTION_DETECTION_ENABLED", False)
    monkeypatch.setattr("detectors.detection_manager.time.sleep", lambda _: None)
    manager.auto_ptz_controller.handle_detections = Mock()
    manager.classification_service.classify = Mock()

    def detect(**kwargs: object) -> DetectionResult:
        manager.stop_event.set()
        return DetectionResult(
            detected=True, original_frame=kwargs["frame"], detections=[]
        )

    manager.detection_service.detect = detect
    manager._detection_loop()
    job = manager.processing_queue.get_nowait()

    assert job["frame_id"] == 7
    assert job["capture_time_precise"] == datetime.fromtimestamp(
        manager.latest_raw_timestamp
    )
    assert job["frame_received_monotonic"] == manager.latest_raw_monotonic
    manager.auto_ptz_controller.handle_detections.assert_called_once()
    manager.classification_service.classify.assert_not_called()


def test_reader_assigns_identity_and_receipt_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        VideoCapture, "_register_instance_for_shutdown", lambda self: None
    )
    monkeypatch.setattr(
        VideoCapture, "_prime_stream_settings_from_cache", lambda self: None
    )
    capture = VideoCapture("http://localhost/unused", auto_start=False)
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    calls = []

    def read(*, decode: bool) -> np.ndarray | None:
        calls.append(decode)
        if len(calls) == 1:
            return frame
        capture.stop_event.set()
        return None

    monkeypatch.setattr(capture, "_read_frame", read)
    received_after = time.time()
    monotonic_after = time.monotonic()
    capture._reader()
    sample = capture.get_frame_sample()

    assert sample is not None
    assert sample.frame_id > 0
    assert received_after <= sample.received_at <= time.time()
    assert monotonic_after <= sample.received_monotonic <= time.monotonic()
    assert sample.frame is frame


def test_frame_receipt_reaches_files_and_database(
    manager: DetectionManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_frame(manager, 9, age=1.0)
    captured_at = datetime.fromtimestamp(manager.latest_raw_timestamp)
    monkeypatch.setattr(manager, "_initialize_components", lambda: True)
    monkeypatch.setattr(manager, "_should_run_od_now", lambda: True)
    monkeypatch.setitem(manager.config, "MOTION_DETECTION_ENABLED", False)
    monkeypatch.setitem(manager.config, "SAVE_THRESHOLD_MODE", "manual")
    monkeypatch.setitem(manager.config, "SAVE_THRESHOLD", 0.65)
    monkeypatch.setattr("detectors.detection_manager.time.sleep", lambda _: None)
    manager.auto_ptz_controller.handle_detections = Mock()
    manager.notification_service = Mock()
    manager.mqtt_detection_service = Mock()
    monkeypatch.setattr("utils.live_event_bus.get_bus", Mock())
    manager.classification_service.classify = Mock(
        side_effect=[
            ClassificationResult(
                class_name="Parus_major",
                confidence=0.95,
                model_id="test-classifier",
                top_k_classes=["Parus_major", "Cyanistes_caeruleus"],
                top_k_confidences=[0.95, 0.05],
                raw_species_name="Parus_major",
            ),
            ClassificationResult(
                class_name="", confidence=0.9, decision_level="reject"
            ),
        ]
    )

    def detect(**kwargs: object) -> DetectionResult:
        manager.stop_event.set()
        return DetectionResult(
            detected=True,
            original_frame=kwargs["frame"],
            model_id="test-detector",
            detections=[
                {"x1": 5, "y1": 5, "x2": 30, "y2": 30, "confidence": 0.9},
                {"x1": 45, "y1": 5, "x2": 70, "y2": 30, "confidence": 0.9},
            ],
        )

    monkeypatch.setattr(manager.detection_service, "detect", detect)
    manager._detection_loop()
    manager.classification_service.classify.assert_not_called()
    manager.auto_ptz_controller.handle_detections.assert_called_once()
    manager._process_job(manager.processing_queue.get_nowait())

    image = manager.db_conn.execute("SELECT * FROM images").fetchone()
    assert image["timestamp"] == captured_at.strftime("%Y%m%d_%H%M%S_%f")
    assert image["detector_model_id"] == "test-detector"
    assert image["classifier_model_id"] == "test-classifier"
    assert (
        tmp_path / "originals" / captured_at.strftime("%Y-%m-%d") / image["filename"]
    ).is_file()
    detections = manager.db_conn.execute("SELECT * FROM detections").fetchall()
    assert len(detections) == 1
    assert detections[0]["image_filename"] == image["filename"]
    assert detections[0]["raw_species_name"] == "Parus_major"
    assert detections[0]["bbox_x"] == pytest.approx(5 / 80)
    assert not Path(detections[0]["thumbnail_path"]).is_absolute()
    assert manager.path_mgr.get_derivative_path(
        detections[0]["thumbnail_path"], "thumb"
    ).is_file()
    predictions = manager.db_conn.execute(
        "SELECT cls_class_name, cls_confidence FROM classifications ORDER BY rank"
    ).fetchall()
    assert [row["cls_class_name"] for row in predictions] == [
        "Parus_major",
        "Cyanistes_caeruleus",
    ]
    assert [row["cls_confidence"] for row in predictions] == pytest.approx([0.95, 0.05])
    assert len(manager._burst_timestamps) == 1
    assert len(manager._recent_admissions) == 1


def test_latest_processing_job_replaces_waiting_job(manager: DetectionManager) -> None:
    manager._enqueue_processing_job({"frame_id": 1})
    manager._enqueue_processing_job({"frame_id": 2})

    assert manager.processing_queue.get_nowait()["frame_id"] == 2
    assert manager.pipeline_metrics.drain()["counts"] == {
        "processing_enqueued": 2,
        "processing_queue_dropped": 1,
    }
