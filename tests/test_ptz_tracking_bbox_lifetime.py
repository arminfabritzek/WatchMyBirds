"""Auto-PTZ controller target bbox + lifetime.

The winning detection's normalised bbox is threaded through the
controller and given a lifetime (held through lost_grace, cleared on
returning/overview/idle). It is surfaced as `last_bbox` in `status()`;
the tests below pin that contract.
"""

from core.ptz_tracking_core import AutoPtzController


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _camera(mode: str = "preset", acquire_frames: int = 1) -> dict:
    # Mirrors tests/test_ptz_tracking_core.py::_camera, trimmed to the
    # zones this module exercises (a centred bird lands in "center").
    return {
        "id": 0,
        "name": "Garden PTZ",
        "ip": "198.51.100.10",
        "ptz": {
            "enabled": True,
            "mode": mode,
            "overview_preset": "overview_token",
            "acquire_frames": acquire_frames,
            "lost_timeout_sec": 6.0,
            "command_cooldown_ms": 700,
            "deadband": 0.12,
            "max_speed": 0.35,
            "move_duration_ms": 250,
            "zones": [
                {
                    "name": "center",
                    "preset": "center_token",
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 1.0,
                },
            ],
        },
    }


def _make_idle_controller(mode: str = "preset", acquire_frames: int = 1):
    return AutoPtzController(
        camera_provider=lambda: _camera(mode=mode, acquire_frames=acquire_frames),
        command_runner=lambda _cmd: None,
        clock=FakeClock(),
        worker_enabled=False,
    )


def _make_tracking_controller_with_bbox():
    """An idle controller pumped one detection cycle so it is in a
    target-holding state and `_last_target_bbox` is set."""
    controller = _make_idle_controller(mode="preset", acquire_frames=1)
    # A bird centred in a 100x200 frame: bbox (50,20)-(150,80).
    det = {
        "x1": 50,
        "y1": 20,
        "x2": 150,
        "y2": 80,
        "confidence": 0.9,
        "class_name": "bird",
    }
    controller.handle_detections(frame_shape=(100, 200, 3), detections=[det])
    return controller


def test_select_target_returns_normalised_bbox():
    c = _make_idle_controller()
    frame_shape = (100, 200)  # h=100, w=200
    dets = [
        {
            "class_name": "bird",
            "x1": 50,
            "y1": 20,
            "x2": 150,
            "y2": 80,
            "confidence": 0.9,
        }
    ]
    target = c._select_target(frame_shape=frame_shape, detections=dets)
    assert target is not None
    cx, cy, conf, bbox = target  # 4-tuple: (cx, cy, conf, (x, y, w, h))
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6
    assert conf == 0.9
    x, y, w, h = bbox
    assert abs(x - 0.25) < 1e-6 and abs(y - 0.20) < 1e-6
    assert abs(w - 0.50) < 1e-6 and abs(h - 0.60) < 1e-6


def test_bbox_present_while_tracking_cleared_on_overview():
    c = _make_tracking_controller_with_bbox()
    assert c.status()["last_bbox"] is not None
    c.return_to_overview()
    assert c.status()["last_bbox"] is None


def test_bbox_held_through_lost_grace():
    c = _make_tracking_controller_with_bbox()
    c.handle_no_detection()  # within grace window → lost_grace, box persists
    assert c._state == "lost_grace"
    assert c.status()["last_bbox"] is not None


def test_status_includes_last_bbox_normalised():
    c = _make_tracking_controller_with_bbox()
    s = c.status()
    assert "last_bbox" in s
    assert s["last_bbox"] is not None
    x, y, w, h = s["last_bbox"]
    assert all(0.0 <= v <= 1.0 for v in (x, y, w, h))
