from types import SimpleNamespace

from utils.station_effort_monitor import StationEffortMonitor


class _DetectionService:
    def is_ready(self):
        return True


class _PtzController:
    def __init__(self, state):
        self.state = state

    def status(self):
        return {"state": self.state}


def test_snapshot_requires_fresh_stream_for_active_detector(monkeypatch):
    monkeypatch.setattr("utils.station_effort_monitor.time.time", lambda: 100.0)
    manager = SimpleNamespace(
        latest_raw_timestamp=80.0,
        video_capture=object(),
        detection_service=_DetectionService(),
        paused=False,
        get_od_status=lambda: {"od_active": True, "reason": "daytime"},
    )
    monitor = StationEffortMonitor(manager, interval_seconds=60)

    snapshot = monitor._snapshot(app_online=True)

    assert snapshot["camera_online"] is True
    assert snapshot["stream_online"] is False
    assert snapshot["detector_ready"] is True
    assert snapshot["detector_active"] is False


def test_snapshot_marks_complete_observation_path(monkeypatch):
    monkeypatch.setattr("utils.station_effort_monitor.time.time", lambda: 100.0)
    manager = SimpleNamespace(
        latest_raw_timestamp=99.0,
        video_capture=object(),
        detection_service=_DetectionService(),
        paused=False,
        get_od_status=lambda: {"od_active": True, "reason": "daytime"},
        auto_ptz_controller=_PtzController("tracking"),
    )

    snapshot = StationEffortMonitor(manager)._snapshot(app_online=True)

    assert snapshot["stream_online"] is True
    assert snapshot["detector_active"] is True
    assert snapshot["ptz_state"] == "tracking"
