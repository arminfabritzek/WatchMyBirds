from core.ptz_tracking_core import AutoPtzController


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _camera(mode: str = "preset", acquire_frames: int = 2) -> dict:
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
                    "name": "left",
                    "preset": "left_token",
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.33,
                    "y_max": 1.0,
                },
                {
                    "name": "center",
                    "preset": "center_token",
                    "x_min": 0.33,
                    "y_min": 0.0,
                    "x_max": 0.67,
                    "y_max": 1.0,
                },
                {
                    "name": "right",
                    "preset": "right_token",
                    "x_min": 0.67,
                    "y_min": 0.0,
                    "x_max": 1.0,
                    "y_max": 1.0,
                },
            ],
        },
    }


def _detection(x1: int, x2: int) -> dict:
    return {
        "x1": x1,
        "y1": 40,
        "x2": x2,
        "y2": 60,
        "confidence": 0.9,
        "class_name": "bird",
    }


def test_preset_mode_queues_zone_preset_after_stable_acquisition():
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=2),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    assert commands == []

    clock.advance(0.8)
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )

    assert len(commands) == 1
    assert commands[0].action == "goto"
    assert commands[0].preset_token == "left_token"


def test_preset_mode_returns_to_overview_after_lost_timeout():
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    clock.advance(6.1)
    controller.handle_no_detection()

    assert [command.preset_token for command in commands] == [
        "left_token",
        "overview_token",
    ]


def test_hybrid_mode_queues_move_after_preset_and_cooldown():
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="hybrid", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(80, 96)]
    )
    clock.advance(0.8)
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(80, 96)]
    )

    assert commands[0].action == "goto"
    assert commands[0].preset_token == "right_token"
    assert commands[1].action == "move"
    assert commands[1].pan > 0
    assert commands[1].tilt == 0.0


def test_non_bird_detection_does_not_trigger_ptz_command():
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    detection = _detection(0, 20)
    detection["class_name"] = "cat"
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[detection])

    assert commands == []
    assert controller.status()["state"] == "idle"


def test_idle_no_detection_does_not_query_camera_provider():
    calls = 0

    def camera_provider() -> dict:
        nonlocal calls
        calls += 1
        return _camera(mode="preset", acquire_frames=1)

    controller = AutoPtzController(
        camera_provider=camera_provider,
        command_runner=lambda command: None,
        worker_enabled=False,
    )

    controller.handle_no_detection()

    assert calls == 0
    assert controller.status()["state"] == "idle"
    assert calls == 1


def test_preset_metadata_box_match_overrides_zone_fallback():
    """When preset_metadata boxes are placed they replace the 3-zone map.

    A bird detected inside a small box (e.g. on the right-most feeder)
    must trigger that box's preset, even though its center lies inside
    the legacy 'right' x-range too. Smaller boxes win on overlap.
    """
    clock = FakeClock()
    commands = []
    cam = _camera(mode="preset", acquire_frames=1)
    cam["ptz"]["preset_metadata"] = {
        # Big box covering most of the right half
        "wide_right": {
            "label": "wide",
            "center_x_pct": 0.75,
            "center_y_pct": 0.5,
            "box_w_pct": 0.40,
            "box_h_pct": 0.80,
        },
        # Small box on a single feeder
        "feeder_4": {
            "label": "4",
            "center_x_pct": 0.78,
            "center_y_pct": 0.45,
            "box_w_pct": 0.10,
            "box_h_pct": 0.15,
        },
        "overview_token": {
            "label": "home",
            "center_x_pct": 0.5,
            "center_y_pct": 0.5,
            "box_w_pct": 0.0,
            "box_h_pct": 0.0,
        },
    }
    controller = AutoPtzController(
        camera_provider=lambda: cam,
        command_runner=lambda c: commands.append(c),
        clock=clock,
        worker_enabled=False,
    )
    # Bird right where Feeder 4 is — smaller box wins.
    detection = {
        "x1": 76,
        "y1": 40,
        "x2": 80,
        "y2": 50,
        "confidence": 0.9,
        "class_name": "bird",
    }
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[detection])

    assert len(commands) == 1
    assert commands[0].preset_token == "feeder_4"


def test_box_change_resets_acquire_window():
    """A bird that hops to a new box must reacquire before the next goto.

    Without this guard a tracked bird flapping between two feeders would
    chain-trigger a goto on the first frame seen at the new box because
    the acquire counter would still be elevated from the previous target.
    """
    clock = FakeClock()
    commands = []
    cam = _camera(mode="preset", acquire_frames=2)
    cam["ptz"]["preset_metadata"] = {
        "feeder_left": {
            "label": "1",
            "center_x_pct": 0.20,
            "center_y_pct": 0.50,
            "box_w_pct": 0.20,
            "box_h_pct": 0.40,
        },
        "feeder_right": {
            "label": "4",
            "center_x_pct": 0.80,
            "center_y_pct": 0.50,
            "box_w_pct": 0.20,
            "box_h_pct": 0.40,
        },
    }
    controller = AutoPtzController(
        camera_provider=lambda: cam,
        command_runner=lambda c: commands.append(c),
        clock=clock,
        worker_enabled=False,
    )
    bird_left = {
        "x1": 18,
        "y1": 48,
        "x2": 22,
        "y2": 52,
        "confidence": 0.9,
        "class_name": "bird",
    }
    bird_right = {
        "x1": 78,
        "y1": 48,
        "x2": 82,
        "y2": 52,
        "confidence": 0.9,
        "class_name": "bird",
    }

    # Two frames in left box → goto fires.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[bird_left])
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[bird_left])
    assert len(commands) == 1
    assert commands[0].preset_token == "feeder_left"

    # First frame in right box must NOT goto yet — needs reacquire.
    clock.advance(5.0)  # past the 3 s cooldown
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[bird_right])
    assert len(commands) == 1, "single right-box frame must not trigger a goto"

    # Second confirming frame → goto right.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[bird_right])
    assert len(commands) == 2
    assert commands[1].preset_token == "feeder_right"


def test_preset_metadata_no_box_match_skips_goto():
    """Bird outside every placed box stays in 'acquiring' — no goto."""
    clock = FakeClock()
    commands = []
    cam = _camera(mode="preset", acquire_frames=1)
    cam["ptz"]["preset_metadata"] = {
        "feeder_left": {
            "label": "1",
            "center_x_pct": 0.15,
            "center_y_pct": 0.5,
            "box_w_pct": 0.10,
            "box_h_pct": 0.15,
        },
    }
    controller = AutoPtzController(
        camera_provider=lambda: cam,
        command_runner=lambda c: commands.append(c),
        clock=clock,
        worker_enabled=False,
    )
    detection = {
        "x1": 70,
        "y1": 40,
        "x2": 75,
        "y2": 50,  # bird far right, outside any box
        "confidence": 0.9,
        "class_name": "bird",
    }
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[detection])

    assert commands == []
    assert controller.status()["state"] == "acquiring"


def test_status_reports_configured_enabled_before_first_detection():
    """Fresh controller with enabled camera must report configured_enabled=true
    even though no detection frame has been processed yet — that is what the
    stream-page pill reads to decide whether to paint 'on' or 'off' on first
    page load after a service restart."""
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=list().append,
        clock=clock,
        worker_enabled=False,
    )

    status = controller.status()

    assert status["state"] == "idle"
    assert status["configured_enabled"] is True
    # Backwards-compat alias kept until callers migrate.
    assert status["enabled"] is True


def test_status_reports_configured_disabled_when_no_camera():
    controller = AutoPtzController(
        camera_provider=lambda: None,
        command_runner=list().append,
        clock=FakeClock(),
        worker_enabled=False,
    )

    status = controller.status()

    assert status["configured_enabled"] is False
    assert status["enabled"] is False
    assert status["camera_id"] is None


def test_status_reports_configured_disabled_when_camera_enabled_false():
    cam = _camera()
    cam["ptz"]["enabled"] = False
    controller = AutoPtzController(
        camera_provider=lambda: cam,
        command_runner=list().append,
        clock=FakeClock(),
        worker_enabled=False,
    )

    status = controller.status()

    assert status["configured_enabled"] is False
    assert status["enabled"] is False
    assert status["camera_id"] == 0
