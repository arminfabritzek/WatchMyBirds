from __future__ import annotations

from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

from detectors.detection_manager import DetectionManager


def manager_with_capture() -> tuple[DetectionManager, SimpleNamespace]:
    manager = DetectionManager.__new__(DetectionManager)
    manager.video_source = "http://localhost/birds.mjpg"
    manager.config = {"VIDEO_SOURCE": manager.video_source}
    manager.motion_detector = None
    manager.frame_lock = Lock()
    manager.latest_raw_frame = object()
    manager.latest_raw_timestamp = 1
    capture = SimpleNamespace(stop_event=Event(), cap=Mock())
    manager.video_capture = capture
    return manager, capture


def test_location_update_preserves_active_capture() -> None:
    manager, capture = manager_with_capture()
    frame = manager.latest_raw_frame
    location = {"latitude": 0.0, "longitude": 0.0}

    manager.update_configuration(
        {
            "VIDEO_SOURCE": manager.video_source,
            "LOCATION_DATA": location,
            "EXIF_GPS_ENABLED": False,
        }
    )

    assert manager.video_capture is capture
    assert not capture.stop_event.is_set()
    capture.cap.release.assert_not_called()
    assert manager.latest_raw_frame is frame
    assert manager.latest_raw_timestamp == 1
    assert manager.location_config == location
    assert manager.config["LOCATION_DATA"] == location
    assert manager.exif_gps_enabled is False


def test_changed_source_stops_previous_capture() -> None:
    manager, capture = manager_with_capture()
    source = "http://localhost/other-birds.mjpg"

    manager.update_configuration({"VIDEO_SOURCE": source})

    assert manager.video_source == source
    assert manager.config["VIDEO_SOURCE"] == source
    assert capture.stop_event.is_set()
    capture.cap.release.assert_called_once_with()
    assert manager.video_capture is None
    assert manager.latest_raw_frame is None
    assert manager.latest_raw_timestamp == 0
