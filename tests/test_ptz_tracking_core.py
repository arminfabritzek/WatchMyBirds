from core.ptz_tracking_core import AutoPtzController, PtzCommand


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


def _follow_detection(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Like _detection but with explicit y coords so bbox area is tunable.

    Follow mode reads the bbox area to drive zoom; the default _detection
    helper's fixed y1=40/y2=60 doesn't give enough range to test the
    zoom-in vs zoom-out branches in isolation.
    """
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "confidence": 0.9,
        "class_name": "bird",
    }


def test_follow_mode_steers_pan_tilt_toward_center():
    """A bird off-centre in follow mode triggers a continuous move
    (action='move', pan != 0) on the FIRST frame — no acquire window."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Bird in the top-right quadrant of a 100×100 frame:
    # bbox 70..90 horizontally, 10..30 vertically → centre (0.80, 0.20).
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.action == "move"
    # offset_x = +0.30 (right of centre) → pan should be positive.
    assert cmd.pan > 0
    # offset_y = -0.30 (above centre) → tilt should be positive (up).
    assert cmd.tilt > 0
    # Bbox area = 20×20 / (100×100) = 0.04, below the 0.18 target by
    # more than the 0.05 deadband → zoom IN (positive).
    assert cmd.zoom > 0
    assert controller.status()["state"] == "tracking"


def test_follow_mode_zoom_out_when_bird_too_big():
    """Bird covering most of the frame → bbox area > target → zoom out."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Centred big bird filling 60% of the frame area.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(20, 20, 100, 100)],
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.action == "move"
    # Bird is roughly centred → pan/tilt within deadband.
    assert abs(cmd.pan) <= 1e-6
    assert abs(cmd.tilt) <= 1e-6
    # Area = 80*80/10000 = 0.64, target=0.18 → zoom OUT (negative).
    assert cmd.zoom < 0


def test_follow_mode_no_move_when_centred_and_size_matches():
    """Bird centred AND at target size → no move command (within deadband).

    This is the steady-state — the controller should NOT chatter
    continuous moves when nothing needs adjusting."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Centred bbox roughly at the 0.18 target area:
    # 100*100*0.18 = 1800 → ~42×42 → bbox 29..71 in both axes.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(29, 29, 71, 71)],
    )

    assert commands == []
    # Still flagged as tracking — we saw the bird, we just don't need to move.
    assert controller.status()["state"] == "tracking"


def test_follow_mode_cooldown_blocks_back_to_back_moves():
    """Two detection frames within the cooldown → exactly one move enqueued.

    Cheap continuous-zoom cams can't queue back-to-back commands; the
    cooldown protects them just like in preset/hybrid mode."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(0.1)  # well below the 700ms cooldown
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )

    assert len(commands) == 1


def test_follow_mode_no_detection_stops_camera_immediately():
    """When follow-mode loses the bird mid-tracking, fire Stop() right
    away so the cam halts its in-flight continuous burst instead of
    completing it on the stale bbox target.

    Many cheap PTZ firmwares ignore the ONVIF duration_sec parameter
    and run each Continuous burst for ~800-1000ms. Without an explicit
    Stop() the cam keeps moving on the last bbox target for hundreds
    of milliseconds after the bird left — the operator sees this as
    'cam still following the old bbox'."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Track a bird first so state goes to "tracking".
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    assert controller.status()["state"] == "tracking"
    move_count_before = len([c for c in commands if c.action == "move"])

    # Next detection cycle: bird is gone.
    controller.handle_no_detection()

    # A Stop() command was enqueued on the tracking→lost_grace transition.
    stop_cmds = [c for c in commands if c.action == "stop"]
    assert len(stop_cmds) == 1, (
        f"expected exactly one Stop on no-detection, got: "
        f"{[(c.action, getattr(c, 'preset_token', '')) for c in commands]}"
    )
    # No new Move was issued.
    move_count_after = len([c for c in commands if c.action == "move"])
    assert move_count_after == move_count_before


def test_follow_mode_no_detection_only_stops_once_not_every_frame():
    """The Stop()-on-no-detection fires on the tracking→lost_grace
    transition only. Subsequent no-detection frames during lost_grace
    must NOT keep hammering Stop() at the cam — that would spam ONVIF
    and the cam log."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    # Three consecutive no-detection frames.
    controller.handle_no_detection()
    controller.handle_no_detection()
    controller.handle_no_detection()

    stop_cmds = [c for c in commands if c.action == "stop"]
    assert len(stop_cmds) == 1


def test_follow_mode_uses_low_p_gain_to_avoid_overshoot():
    """The follow-mode P-gain (0.8) is well below the hybrid-mode gain
    (2.0). Reason: cheap cams run each Continuous burst for ~800-1000ms
    regardless of duration_sec, so a 2.0 gain on the operator's cam
    over-corrects past the centre, the next detection sees the bird on
    the OTHER side, fires the opposite move, and the camera oscillates."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Bird at the extreme right edge: offset_x = +0.40, well above the
    # 0.12 deadband. With max_speed=0.35 and P-gain=0.8, the expected
    # pan command is 0.40 * 0.35 * 0.8 = 0.112. With the old P-gain of
    # 2.0 it would have been 0.280 — i.e. nearly the full max_speed for
    # what should be a moderate correction. The lower gain damps the
    # response so two cycles of motion are needed to centre instead of
    # one overshoot.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(85, 40, 95, 60)],
    )

    move_cmds = [c for c in commands if c.action == "move"]
    assert len(move_cmds) == 1
    pan = move_cmds[0].pan
    # P-gain * max_speed * 2.0 (old) would give ~0.28; with the 0.8
    # gain we expect ~0.11. Hard upper bound at 0.15 catches a
    # regression to the old gain without being brittle to small
    # bbox tweaks.
    assert 0 < pan < 0.15, f"expected damped pan correction, got {pan}"


def test_min_confidence_filters_weak_detections():
    """Detections under min_confidence are treated as 'no detection'.

    Without this, the cam keeps chasing phantom Bird boxes (leaves,
    shadows, low-confidence false positives between the detection
    floor and the save threshold). The cam would never reach the
    lost-timeout because each cycle still has 'something' it can
    fly toward."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    weak = _follow_detection(70, 10, 90, 30)
    weak["confidence"] = 0.35

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[weak],
        min_confidence=0.60,
    )

    # Weak detection rejected → no move enqueued.
    moves = [c for c in commands if c.action == "move"]
    assert moves == []


def test_min_confidence_accepts_strong_detections():
    """Detections at or above min_confidence still trigger moves —
    the gate must not break the happy path."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    strong = _follow_detection(70, 10, 90, 30)
    strong["confidence"] = 0.75

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[strong],
        min_confidence=0.60,
    )

    moves = [c for c in commands if c.action == "move"]
    assert len(moves) == 1


def test_min_confidence_mixed_keeps_only_strong():
    """A frame with one weak + one strong detection picks the strong
    one as the move target, not the strongest-of-all-weak."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Weak bird on the left, strong bird on the right.
    weak = _follow_detection(10, 40, 30, 60)
    weak["confidence"] = 0.30
    strong = _follow_detection(70, 40, 90, 60)
    strong["confidence"] = 0.80

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[weak, strong],
        min_confidence=0.60,
    )

    moves = [c for c in commands if c.action == "move"]
    assert len(moves) == 1
    # Strong bird is right-of-centre → pan should be positive.
    assert moves[0].pan > 0


def test_min_confidence_zero_disables_filter():
    """min_confidence=0 (the default) preserves the legacy behaviour
    where every bird-class detection is a candidate target."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    weak = _follow_detection(70, 10, 90, 30)
    weak["confidence"] = 0.10

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[weak],
        # min_confidence omitted → defaults to 0.0
    )

    moves = [c for c in commands if c.action == "move"]
    assert len(moves) == 1


def test_lost_detection_cooldown_blocks_new_moves_until_overview_arrives():
    """After handle_no_detection fires goto(overview), the cooldown
    must block fresh detection-driven moves for lost_timeout_sec
    seconds — long enough for the cam to actually reach the overview
    preset before another stray detection drags it back.

    Without this guard: handle_no_detection fires goto, then the
    detection loop's next frame ~2s later finds a Bird above the
    save threshold and yanks state back to 'tracking'. The cam never
    arrives at the overview, and the operator sees 'cam keeps moving
    even after the bird is gone' because the auto-return is being
    continuously overridden.
    """
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # 1. Track a bird.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    # 2. Bird disappears. Advance past lost_timeout_sec (6s) so the
    # next no-detection call fires goto(overview).
    clock.advance(7.0)
    controller.handle_no_detection()

    goto_cmds = [c for c in commands if c.action == "goto"]
    assert len(goto_cmds) == 1, "expected goto(overview) after lost_timeout"
    assert goto_cmds[0].preset_token == "overview_token"

    # 3. Stray high-confidence detection comes in 1 second later. The
    # cooldown should suppress it.
    clock.advance(1.0)
    stray = _follow_detection(40, 40, 60, 60)
    stray["confidence"] = 0.95
    moves_before = len([c for c in commands if c.action == "move"])
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[stray],
        min_confidence=0.60,
    )
    moves_after = len([c for c in commands if c.action == "move"])
    assert moves_after == moves_before, (
        f"cooldown should block new move, but {moves_after - moves_before} were fired"
    )


def test_lost_detection_cooldown_expires_after_lost_timeout():
    """Once the cooldown elapses, detections drive moves again."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(7.0)
    controller.handle_no_detection()

    # Advance past the cooldown window (lost_timeout_sec from the
    # goto-overview point).
    clock.advance(7.0)
    fresh = _follow_detection(40, 40, 60, 60)
    fresh["confidence"] = 0.95
    moves_before = len([c for c in commands if c.action == "move"])
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[fresh],
        min_confidence=0.60,
    )
    moves_after = len([c for c in commands if c.action == "move"])
    assert moves_after == moves_before + 1


def test_manual_drive_overrides_lost_detection_cooldown():
    """The cooldown blocks auto-PTZ, NOT manual joystick. Operator
    intent always wins."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Trigger an auto-return → arms the cooldown.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(7.0)
    controller.handle_no_detection()
    assert controller._lost_cooldown_until > 0

    # Manual drive must clear the cooldown.
    controller.notify_manual_drive()
    assert controller._lost_cooldown_until == 0


def test_follow_mode_lost_timeout_returns_to_overview():
    """No detection past lost_timeout_sec → goto(overview_preset).

    Follow mode reuses the existing handle_no_detection path. This test
    proves the path still works for a mode that never enqueued a preset
    goto during tracking (only continuous moves)."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # First, a real detection so _last_seen_mono is set.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(40, 40, 60, 60)],
    )
    # Then advance past lost_timeout and call no_detection.
    clock.advance(7.0)
    controller.handle_no_detection()

    goto_cmds = [c for c in commands if c.action == "goto"]
    assert len(goto_cmds) == 1
    assert goto_cmds[0].preset_token == "overview_token"


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


# ---------------------------------------------------------------------------
# snapshot_for_image_persistence — PTZ context for image rows
# ---------------------------------------------------------------------------


def test_snapshot_idle_returns_origin_none():
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "none"
    assert snap["ptz_state"] == "idle"
    assert snap["ptz_preset_token"] is None
    assert snap["ptz_zone"] is None
    assert snap["ptz_camera_id"] == 0
    assert snap["ptz_pan"] is None
    assert snap["ptz_tilt"] is None
    assert snap["ptz_zoom"] is None
    assert snap["ptz_position_at"] is None


def test_snapshot_tracking_returns_origin_preset_with_token_and_zone():
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    # _maybe_goto_zone transitions to "tracking" on the first acceptable frame
    # when acquire_frames == 1.
    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "preset"
    assert snap["ptz_state"] == "tracking"
    assert snap["ptz_preset_token"] == "left_token"
    assert snap["ptz_zone"] == "left"
    assert snap["ptz_camera_id"] == 0


def test_snapshot_overview_returns_origin_overview():
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    # Force the controller into "overview" state directly — equivalent to
    # the camera resting on the overview preset after a return cycle.
    controller._update_status(state="overview")

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "overview"
    assert snap["ptz_state"] == "overview"


def test_snapshot_returning_treated_as_overview():
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    controller._update_status(state="returning")

    snap = controller.snapshot_for_image_persistence()

    # Mid-fly back to overview is semantically "overview", not preset:
    # the frame is no longer a close-up, the camera is heading wide.
    assert snap["ptz_origin"] == "overview"


def test_snapshot_lost_grace_treated_as_preset():
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    # lost_grace means the bird just left but the camera is still at the
    # zone preset, waiting lost_timeout_sec before returning to overview.
    # Frames captured here are still close-ups.
    controller._update_status(state="lost_grace")

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "preset"


def test_snapshot_settling_treated_as_preset():
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    # settling = camera is mid-fly toward a zone preset triggered by an
    # external goto. Treat as preset so frames captured during the fly-in
    # are not undercounted in gallery bias.
    controller._update_status(state="settling")

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "preset"


def test_snapshot_with_no_camera_returns_none_camera_id():
    controller = AutoPtzController(
        camera_provider=lambda: None,
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "none"
    assert snap["ptz_camera_id"] is None


def test_empty_ptz_snapshot_has_full_column_keyset():
    from core.ptz_tracking_core import empty_ptz_snapshot

    snap = empty_ptz_snapshot()

    expected_keys = {
        "ptz_origin",
        "ptz_preset_token",
        "ptz_zone",
        "ptz_state",
        "ptz_camera_id",
        "ptz_pan",
        "ptz_tilt",
        "ptz_zoom",
        "ptz_position_at",
    }
    assert set(snap.keys()) == expected_keys
    # Empty snapshot is all-NULL: maps to "we do not know" in the DB.
    assert all(v is None for v in snap.values())


# ---------------------------------------------------------------------------
# notify_manual_drive — operator joystick from stream-page buttons
# ---------------------------------------------------------------------------


def test_notify_manual_drive_arms_grace_window():
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )

    controller.notify_manual_drive()

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_origin"] == "manual_drive"
    assert snap["ptz_state"] == "lost_grace"
    assert snap["ptz_zone"] == "manual_drive"
    # No preset token — operator is steering freely.
    assert snap["ptz_preset_token"] is None


def test_notify_manual_drive_refreshes_deadline_on_repeat():
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )

    controller.notify_manual_drive()
    # Without refresh, the camera would auto-return after manual_view_sec.
    # We advance most of that window, then send another heartbeat.
    clock.advance(10.0)
    controller.notify_manual_drive()

    # Deadline must now be 15s from the new clock value, not the original.
    status = controller.status()
    remaining = status["seconds_until_return"]
    assert remaining is not None
    assert 14.5 <= remaining <= 15.0, (
        f"deadline should refresh to ~15s, got {remaining}"
    )


def test_notify_manual_drive_noop_when_auto_disabled():
    def disabled_camera():
        cam = _camera()
        cam["ptz"]["enabled"] = False
        return cam

    controller = AutoPtzController(
        camera_provider=disabled_camera,
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )

    # Initial state should be idle.
    assert controller.snapshot_for_image_persistence()["ptz_state"] == "idle"

    controller.notify_manual_drive()

    # No grace, no state transition — auto is off, there's nothing to gate.
    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_state"] == "idle"
    assert snap["ptz_origin"] == "none"


def test_notify_manual_drive_noop_when_no_overview_preset():
    def no_overview_camera():
        cam = _camera()
        cam["ptz"]["overview_preset"] = ""
        return cam

    controller = AutoPtzController(
        camera_provider=no_overview_camera,
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )

    controller.notify_manual_drive()

    # Without an overview preset there is nowhere to return to, so the
    # manual-grace mechanism would be meaningless. Skip entirely.
    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_state"] == "idle"


def test_notify_manual_drive_overrides_auto_tracking_state():
    """Operator yanks the camera mid-auto-tracking — manual wins until released."""
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )

    # First, push the controller into auto-tracking.
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    assert controller.snapshot_for_image_persistence()["ptz_origin"] == "preset"

    # Operator grabs the joystick.
    controller.notify_manual_drive()

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_origin"] == "manual_drive"
    assert snap["ptz_zone"] == "manual_drive"


def test_manual_goto_blocks_detection_driven_counter_goto():
    """Bug regression: manual goto must seed the command cooldown.

    Reproduction: operator clicks preset 3. Backend fires the goto and
    parks in `settling`. While the cheap PTZ camera flies (no MoveStatus
    → 5 s sleep fallback), the detector keeps seeing the bird in the
    still-wide-angle frame and routes it to a neighbouring zone (2 or 4).
    Without the cooldown seed, _maybe_goto_zone fires a counter-goto on
    the very next frame — the camera lurches to the wrong preset before
    lost_grace eventually returns it home.

    Fix: notify_external_goto sets _last_command_mono so the detection
    path waits the full cooldown before it can issue another goto.
    """
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator clicks a non-overview preset (simulates the UI path).
    controller.notify_external_goto("right_token")

    # Immediately afterwards, a detection frame arrives that would
    # normally route to the "left" zone (bird at x_pct ≈ 0.1, which is
    # in the left-zone bounds 0.0–0.33).
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_detection(0, 20)],  # bird center near x=10/100 = 0.10
    )

    # With the fix in place, the detection-driven goto must be blocked
    # by the command cooldown that notify_external_goto seeded.
    # The only command in flight is the operator's own goto, which
    # bypasses the command_runner because it's enqueued via
    # _enqueue → camera client, not _maybe_goto_zone. Therefore
    # commands stays empty.
    assert commands == [], (
        f"detection-driven counter-goto leaked past manual: {commands}"
    )


def test_home_button_blocks_detection_driven_counter_goto():
    """Same race, but the trigger is the Home / return-to-overview button."""
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator clicks Home — the controller enqueues a goto to overview.
    assert controller.return_to_overview() is True
    # Drain the home command itself from the test runner.
    home_commands = list(commands)
    commands.clear()
    assert len(home_commands) == 1
    assert home_commands[0].preset_token == "overview_token"

    # Bird detection arrives while camera is still flying home.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_detection(0, 20)],
    )

    # No counter-goto fires.
    assert commands == []


def test_manual_joystick_drive_blocks_detection_driven_counter_goto():
    """Third leg of the cooldown-seeding race: manual joystick drive.

    Observed regression: in preset mode the camera springs to
    other presets in quick succession during a detection event,
    suspected to be a relic of a previous manual action. Commit 2e15f32
    closed this race for preset-click (notify_external_goto) and Home
    (return_to_overview) but missed the joystick path
    (notify_manual_drive) — because joystick uses continuous_move
    instead of goto, it was treated as "not a command". Physically
    though the camera moves just the same, the mid-flight frames feed
    the same false-zone routing, and the next detection-driven goto
    fires with no cooldown gate.
    """
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator nudges the joystick (one heartbeat — could be any direction).
    controller.notify_manual_drive()

    # Mid-flight, a detection lands in a zone that is NOT where the
    # joystick was taking the camera.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_detection(0, 20)],
    )

    assert commands == [], (
        f"detection-driven counter-goto leaked past manual joystick drive: {commands}"
    )


# ---------------------------------------------------------------------------
# Detection-driven settle window — cheap-PTZ movement time.
#
# Observed regression: in preset mode the camera flew to the correct
# preset on a bird detection, then 4 s later jumped to Preset 4 (where
# no bird was), then 4 s later home. The cooldown gate
# alone cannot prevent this because cheap cameras take 2–6 s to
# traverse — far longer than even the 10 s default — yet the controller
# was committing _state="tracking" the instant the goto was enqueued,
# and the very next mid-flight frame routed the bbox into a neighbour
# zone. The fix parks the controller in "settling" until the camera
# arrives (wait_until_idle / fallback), suppressing detection-driven
# gotos until the frames are honest again.
# ---------------------------------------------------------------------------


def test_detection_goto_parks_in_settling_state_until_resumed():
    """First detection-driven goto must leave the controller in 'settling'.

    In worker_enabled=False the synchronous fallback flips us straight
    back to 'tracking' (no real settle worker to call wait_until_idle),
    so this test exercises the production path by skipping that
    fast-track via a fresh controller plus a direct state assertion.
    """
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # First detection triggers a goto. In the synchronous test mode the
    # state ends up as 'tracking' (settling is auto-resumed). What we
    # actually want to verify is the gate logic: after pretending the
    # controller is mid-flight, the next detection must be suppressed.
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    assert len(commands) == 1, "first detection must issue exactly one goto"

    # Simulate the production state: settle worker has parked us in
    # 'settling' and a new detection arrives mid-flight in a different
    # zone. The bird's bbox is in the right-zone (x ≈ 0.88) — without
    # the gate the controller would issue a counter-goto to right_token.
    with controller._lock:
        controller._state = "settling"

    clock.advance(2.0)  # plenty of time has passed; cooldown irrelevant
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(80, 96)]
    )

    assert len(commands) == 1, (
        f"settling guard must block counter-goto from mid-flight frame: {commands}"
    )


def test_settle_window_refreshes_last_seen_so_lost_grace_does_not_fire():
    """During settling the bird is not lost — keep the lost-grace anchor live."""
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )

    # Drive controller into settling with a stale _last_seen_mono.
    with controller._lock:
        controller._state = "settling"
        controller._last_seen_mono = clock.now - 100.0  # ancient

    clock.advance(5.0)
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )

    assert controller._last_seen_mono == clock.now, (
        "settling guard must refresh _last_seen_mono so a settle in "
        "progress does not trip lost_grace mid-flight"
    )


def test_detection_settle_worker_not_spawned_when_worker_disabled():
    """Synchronous mode must not start a background settle worker.

    The worker would call ptz_core._client_for_camera() with no real
    camera registered, raise, hit the fallback time.sleep(5), and slow
    every test 5 s. The fix: skip the spawn when worker_enabled=False
    and flip _state straight to 'tracking'.
    """
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    import threading as _t

    before = {t.name for t in _t.enumerate()}
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    after = {t.name for t in _t.enumerate()}

    new_threads = after - before
    assert not any("auto-ptz-detection-settle" in n for n in new_threads), (
        f"settle worker leaked into synchronous test mode: {new_threads}"
    )
    assert controller.status()["state"] == "tracking", (
        "synchronous mode must reach 'tracking' immediately"
    )


def test_settle_then_resume_flips_state_back_to_tracking():
    """Worker behaviour: after the settle finishes, _state must become 'tracking'."""
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )
    with controller._lock:
        controller._state = "settling"

    # Stub out the ONVIF call so the worker doesn't try to reach a
    # camera; what we're testing is the post-settle state transition.
    class _StubClient:
        def wait_until_idle(self, *, max_wait_sec):  # noqa: ARG002
            return True

    from core import ptz_core

    original = ptz_core._client_for_camera
    ptz_core._client_for_camera = lambda _cam_id: _StubClient()
    try:
        controller._settle_then_resume_tracking(camera_id=0, settle_max_sec=1.0)
    finally:
        ptz_core._client_for_camera = original

    assert controller._state == "tracking", (
        f"settle worker must transition settling → tracking, got {controller._state}"
    )


def test_settle_resume_does_not_clobber_superseding_state():
    """If something else changed _state away from 'settling', leave it alone.

    A manual goto, an idle reset, or any other state change during the
    settle window means our resume should be a no-op — otherwise we'd
    drag the controller back to 'tracking' against the newer intent.
    """
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=lambda c: None,
        clock=clock,
        worker_enabled=False,
    )
    with controller._lock:
        controller._state = "idle"  # something else took over

    class _StubClient:
        def wait_until_idle(self, *, max_wait_sec):  # noqa: ARG002
            return True

    from core import ptz_core

    original = ptz_core._client_for_camera
    ptz_core._client_for_camera = lambda _cam_id: _StubClient()
    try:
        controller._settle_then_resume_tracking(camera_id=0, settle_max_sec=1.0)
    finally:
        ptz_core._client_for_camera = original

    assert controller._state == "idle", (
        "settle worker must not overwrite a superseding state"
    )


def _grid_camera(
    rows: int = 3,
    cols: int = 3,
    acquire_frames: int = 1,
    grid_acquire_frames: int | None = None,
    cooldown_ms: int = 100,
    cells: dict | None = None,
) -> dict:
    """A camera dict pre-configured for grid mode.

    Defaults give a 3×3 grid with every cell mapped to a preset token
    ``grid_token_r{row}_c{col}`` so the controller can issue a goto
    without bumping into the "cell not configured" branch.

    grid_acquire_frames mirrors acquire_frames when omitted so existing
    tests keep their single-frame trigger behaviour. New tests pass it
    explicitly when they want to exercise the grid-only hurdle.
    """
    if cells is None:
        cells = {
            f"r{r}_c{c}": f"grid_token_r{r}_c{c}"
            for r in range(rows)
            for c in range(cols)
        }
    if grid_acquire_frames is None:
        grid_acquire_frames = acquire_frames
    return {
        "id": 0,
        "name": "Garden PTZ",
        "ip": "198.51.100.10",
        "ptz": {
            "enabled": True,
            "mode": "grid",
            "overview_preset": "overview_token",
            "acquire_frames": acquire_frames,
            "grid_acquire_frames": grid_acquire_frames,
            "lost_timeout_sec": 6.0,
            "command_cooldown_ms": 10000,  # preset cooldown stays high
            "grid_command_cooldown_ms": cooldown_ms,
            "grid_shape": [rows, cols],
            "grid_cells": cells,
            "grid_hysteresis_margin": 0.05,
            "deadband": 0.12,
            "max_speed": 0.35,
            "move_duration_ms": 250,
            "zones": [],  # grid mode does not use legacy zones
        },
    }


def _grid_detection(
    x_pct: float, y_pct: float, frame_w: int = 100, frame_h: int = 100
) -> dict:
    """Build a detection whose center lands at (x_pct, y_pct) of the frame."""
    cx, cy = x_pct * frame_w, y_pct * frame_h
    return {
        "x1": cx - 5,
        "y1": cy - 5,
        "x2": cx + 5,
        "y2": cy + 5,
        "confidence": 0.9,
        "class_name": "bird",
    }


class TestGridMode:
    """Grid-mode dispatch path in handle_detections."""

    def test_grid_mode_routes_to_correct_cell(self):
        commands = []
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1),
            command_runner=commands.append,
            clock=FakeClock(),
            worker_enabled=False,
        )

        # Center of a 3×3 grid → cell (1, 1) → token grid_token_r1_c1.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )

        assert len(commands) == 1
        assert commands[0].preset_token == "grid_token_r1_c1"

    def test_grid_mode_top_left_cell(self):
        commands = []
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1),
            command_runner=commands.append,
            clock=FakeClock(),
            worker_enabled=False,
        )

        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.1, 0.1)],
        )

        assert commands[0].preset_token == "grid_token_r0_c0"

    def test_grid_mode_cooldown_blocks_rapid_adjacent_switch(self):
        commands = []
        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=2000),
            command_runner=commands.append,
            clock=clock,
            worker_enabled=False,
        )

        # First detection → cell (0, 0), goto fires.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.1, 0.1)],
        )
        # Decisive jump to a non-adjacent cell well outside hysteresis,
        # but within cooldown → no second goto.
        clock.advance(0.5)
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.9, 0.9)],
        )
        assert len(commands) == 1

        # After cooldown lapses → second goto fires.
        clock.advance(2.5)
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.9, 0.9)],
        )
        assert len(commands) == 2
        assert commands[1].preset_token == "grid_token_r2_c2"

    def test_grid_mode_hysteresis_suppresses_boundary_flap(self):
        commands = []
        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=500),
            command_runner=commands.append,
            clock=clock,
            worker_enabled=False,
        )

        # Lock onto cell (0, 0) first.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.1, 0.1)],
        )
        assert len(commands) == 1

        # Bird hops to x=0.34 — just over the 0.333 boundary, within
        # the 0.05 hysteresis margin. Advance well past the 500ms
        # cooldown so the only thing preventing a second goto is
        # hysteresis itself.
        clock.advance(2.0)
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.34, 0.1)],
        )
        assert len(commands) == 1  # still in cell (0, 0)

    def test_grid_mode_uncongifured_cell_does_not_crash(self):
        commands = []
        # Sparse grid: only (0, 0) is set.
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(
                acquire_frames=1,
                cells={"r0_c0": "grid_token_r0_c0"},
            ),
            command_runner=commands.append,
            clock=FakeClock(),
            worker_enabled=False,
        )

        # Bird in cell (1, 1) — not configured.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )
        assert len(commands) == 0

    def test_grid_mode_records_origin_preset_in_snapshot(self):
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1),
            command_runner=lambda c: None,
            clock=FakeClock(),
            worker_enabled=False,
        )

        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )

        snap = controller.snapshot_for_image_persistence()
        # Grid frames must inherit the preset-origin bias so the
        # gallery ranker promotes them just like preset-mode frames.
        assert snap["ptz_origin"] == "preset"
        assert snap["ptz_state"] == "tracking"

    def test_grid_mode_bird_hops_through_all_cells(self):
        """Detection that visits every cell in turn produces one goto per cell.

        Stresses the routing + cooldown interplay: each detection lands in
        a new cell, cooldown lapses between visits, every cell should
        trigger exactly one goto to its preset.

        Note: normalize_ptz_config clamps grid_command_cooldown_ms to a
        minimum of 500ms, so the smallest meaningful test cooldown is
        500ms — clock advance must exceed that between cells.
        """
        commands = []
        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=500),
            command_runner=commands.append,
            clock=clock,
            worker_enabled=False,
        )

        # Visit cell centers in row-major order. 3×3 → 9 cells.
        for r in range(3):
            for c in range(3):
                x_pct = (c + 0.5) / 3
                y_pct = (r + 0.5) / 3
                controller.handle_detections(
                    frame_shape=(100, 100, 3),
                    detections=[_grid_detection(x_pct, y_pct)],
                )
                clock.advance(0.6)  # > 0.5s cooldown

        # All 9 cells visited, all 9 should have triggered a goto.
        assert len(commands) == 9
        emitted_tokens = [cmd.preset_token for cmd in commands]
        expected = [f"grid_token_r{r}_c{c}" for r in range(3) for c in range(3)]
        assert emitted_tokens == expected

    def test_grid_mode_rapid_flap_blocked_by_cooldown(self):
        """Bird flaps fast between two non-adjacent cells — cooldown caps the gotos."""
        commands = []
        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=4000),
            command_runner=commands.append,
            clock=clock,
            worker_enabled=False,
        )

        # First detection in cell (0, 0) — goto fires.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.1, 0.1)],
        )
        assert len(commands) == 1

        # Now flap rapidly between (0, 0) and (2, 2) — 10 alternations
        # at 200ms intervals = 2s elapsed, less than the 4s cooldown.
        for i in range(10):
            clock.advance(0.2)
            x_pct = 0.9 if i % 2 == 0 else 0.1
            y_pct = 0.9 if i % 2 == 0 else 0.1
            controller.handle_detections(
                frame_shape=(100, 100, 3),
                detections=[_grid_detection(x_pct, y_pct)],
            )
        # Within the 4s window from the first goto, the cooldown must
        # block all further commands. Exactly one goto from the initial
        # detection, no more.
        assert len(commands) == 1

        # After cooldown lapses, a fresh detection in a new cell fires.
        clock.advance(2.5)  # total elapsed: 2 + 2.5 = 4.5s > 4s cooldown
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.9, 0.9)],
        )
        assert len(commands) == 2
        assert commands[1].preset_token == "grid_token_r2_c2"

    def test_grid_mode_lost_grace_returns_to_overview(self):
        """Bird leaves frame in grid mode → lost_grace → overview goto after timeout."""
        commands = []
        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=500),
            command_runner=commands.append,
            clock=clock,
            worker_enabled=False,
        )

        # Bird in cell (1, 1) → tracking goto.
        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )
        assert len(commands) == 1
        assert commands[0].preset_token == "grid_token_r1_c1"

        # Bird disappears. Within lost_timeout_sec, no return-to-overview yet.
        clock.advance(3.0)  # less than the 6s lost_timeout_sec default
        controller.handle_no_detection()
        assert len(commands) == 1  # still no return goto

        # Now past the timeout — handle_no_detection fires the overview goto.
        clock.advance(4.0)
        controller.handle_no_detection()
        assert len(commands) == 2
        assert commands[1].preset_token == "overview_token"

    def test_grid_and_manual_drive_lock_contention(self):
        """Interleaved manual-drive + grid auto-tracking calls don't deadlock.

        The controller's _lock is reentrant-free; the contract is that
        the two notify paths (manual_drive, handle_detections grid)
        each acquire the lock briefly and release. This test fires both
        from multiple threads simultaneously and asserts the suite
        completes without timing out and the final state is consistent.
        """
        import threading

        clock = FakeClock()
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=500),
            command_runner=lambda c: None,
            clock=clock,
            worker_enabled=False,
        )

        stop_event = threading.Event()
        errors: list[Exception] = []

        def manual_pump():
            try:
                while not stop_event.is_set():
                    controller.notify_manual_drive()
            except Exception as exc:  # noqa: BLE001 — capture for assertion
                errors.append(exc)

        def grid_pump():
            try:
                positions = [(0.5, 0.5), (0.1, 0.1), (0.9, 0.9), (0.1, 0.9)]
                idx = 0
                while not stop_event.is_set():
                    x_pct, y_pct = positions[idx % len(positions)]
                    controller.handle_detections(
                        frame_shape=(100, 100, 3),
                        detections=[_grid_detection(x_pct, y_pct)],
                    )
                    idx += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        manual_thread = threading.Thread(target=manual_pump, daemon=True)
        grid_thread = threading.Thread(target=grid_pump, daemon=True)
        manual_thread.start()
        grid_thread.start()

        # Run for 250ms — long enough to exercise heavy contention but
        # short enough to keep the test snappy. Both pumps call into
        # methods that acquire _lock; if there's a deadlock the threads
        # never exit and join() times out.
        import time as _time

        _time.sleep(0.25)
        stop_event.set()
        manual_thread.join(timeout=2.0)
        grid_thread.join(timeout=2.0)

        # If join timed out the threads are stuck — that's a deadlock.
        assert not manual_thread.is_alive(), "manual_pump did not exit (lock starved)"
        assert not grid_thread.is_alive(), "grid_pump did not exit (lock starved)"
        assert errors == [], f"unexpected exceptions: {errors}"

        # Final state must be one of the valid grid/manual states.
        # Both pumps set _last_command_mono so neither is starved.
        status = controller.status()
        assert status["state"] in {"tracking", "acquiring", "lost_grace"}


class TestGridAcquireFrames:
    """Grid dispatch reads grid_acquire_frames, not the preset acquire_frames."""

    def test_grid_triggers_on_first_frame_when_grid_acquire_is_one(self):
        commands = []
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(
                acquire_frames=5,
                grid_acquire_frames=1,
            ),
            command_runner=commands.append,
            clock=FakeClock(),
            worker_enabled=False,
        )

        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )

        assert len(commands) == 1
        assert commands[0].preset_token == "grid_token_r1_c1"

    def test_grid_waits_for_grid_acquire_frames_threshold(self):
        commands = []
        controller = AutoPtzController(
            camera_provider=lambda: _grid_camera(
                acquire_frames=1,
                grid_acquire_frames=3,
            ),
            command_runner=commands.append,
            clock=FakeClock(),
            worker_enabled=False,
        )

        for _ in range(2):
            controller.handle_detections(
                frame_shape=(100, 100, 3),
                detections=[_grid_detection(0.5, 0.5)],
            )
        assert commands == []

        controller.handle_detections(
            frame_shape=(100, 100, 3),
            detections=[_grid_detection(0.5, 0.5)],
        )
        assert len(commands) == 1
        assert commands[0].preset_token == "grid_token_r1_c1"


def test_lost_grace_without_manual_drive_still_maps_to_preset():
    """Auto-tracking lost_grace (zone-based) keeps origin='preset'."""
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    # Simulate the auto-tracking lost_grace path: state set to lost_grace
    # with a zone name, not "manual_drive".
    with controller._lock:
        controller._state = "lost_grace"
        controller._last_zone = "left"
        controller._last_preset = "left_token"

    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "preset"
    assert snap["ptz_zone"] == "left"


# ---------------------------------------------------------------------------
# Goto-failure rollback — cheap-PTZ camera semantics.
#
# Background: cheap ONVIF cameras may reject GotoPreset with "Preset token
# does not exist" (or eat the call silently), even when GetPresets lists
# the token. The controller commits _last_preset / _last_zone optimistically
# under _lock so the cooldown gate sees the in-flight target. If the goto
# is then rejected by the camera, that committed state lies — the next
# snapshot_for_image_persistence() would write ptz_preset_token=X into
# the DB even though the camera never moved. Rollback restores honest
# telemetry; the CAS check prevents a newer enqueue from being clobbered.
# ---------------------------------------------------------------------------


class _FailingRunner:
    """Command runner that raises on a configurable subset of gotos."""

    def __init__(self, fail_tokens: set[str] | None = None) -> None:
        self.fail_tokens = fail_tokens or set()
        self.attempts: list = []

    def __call__(self, command) -> None:
        self.attempts.append(command)
        if command.action == "goto" and command.preset_token in self.fail_tokens:
            raise RuntimeError(
                f"The requested preset token does not exist: {command.preset_token}"
            )


def test_goto_failure_rolls_back_last_preset_in_preset_mode():
    """Worker-side rejection of GotoPreset must undo the optimistic commit.

    Before the fix: snapshot_for_image_persistence would have returned
    ptz_preset_token='left_token' even though the camera never moved.
    """
    runner = _FailingRunner(fail_tokens={"left_token"})
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=runner,
        clock=FakeClock(),
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )

    # The runner was called (the command was issued) but the camera
    # rejected it, so the controller must have rolled back.
    assert len(runner.attempts) == 1
    assert runner.attempts[0].preset_token == "left_token"

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] is None, (
        "rollback must clear the committed preset token so the DB no "
        "longer claims the camera reached a preset it never reached"
    )
    assert snap["ptz_zone"] is None


def test_goto_failure_rolls_back_to_previous_successful_target():
    """First goto succeeds, second goto fails → snapshot shows the first."""
    runner = _FailingRunner(fail_tokens={"right_token"})
    clock = FakeClock()
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=runner,
        clock=clock,
        worker_enabled=False,
    )

    # First detection on the left — succeeds.
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    snap1 = controller.snapshot_for_image_persistence()
    assert snap1["ptz_preset_token"] == "left_token"

    # Bird moves to the right cell, cooldown elapsed — goto enqueued,
    # but the camera rejects right_token.
    clock.advance(2.0)
    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(80, 96)]
    )

    snap2 = controller.snapshot_for_image_persistence()
    assert snap2["ptz_preset_token"] == "left_token", (
        "after a failed goto, telemetry must reflect the last preset the "
        "camera actually reached, not the one it was asked to reach"
    )
    assert snap2["ptz_zone"] == "left"


def test_goto_failure_does_not_clobber_newer_committed_goto():
    """CAS check: if state has moved on, the stale failure must not undo it.

    Sequence:
      1. Issue goto A (this test fakes A as already-failed).
      2. Before the rollback fires, _last_preset has been overwritten by
         goto B (succeeded).
      3. Rollback for A runs — must not touch _last_preset because it no
         longer equals A's token.
    """
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=lambda c: None,
        clock=FakeClock(),
        worker_enabled=False,
    )
    # Seed state as if goto B already succeeded.
    with controller._lock:
        controller._last_preset = "right_token"
        controller._last_zone = "right"

    # Now process a stale failure for goto A (left_token). The rollback
    # would try to restore "" / "" — but the CAS check on _last_preset
    # must veto because _last_preset is now "right_token", not "left_token".
    from core.ptz_tracking_core import PtzCommand

    stale_command = PtzCommand(
        action="goto",
        camera_id=0,
        preset_token="left_token",
        rollback_preset="",
        rollback_zone="",
    )
    controller._on_command_failed(stale_command, RuntimeError("stale failure"))

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] == "right_token", (
        "the CAS check must protect a newer goto's committed state from "
        "being overwritten by a stale failure's rollback"
    )
    assert snap["ptz_zone"] == "right"


def test_goto_failure_in_grid_mode_rolls_back_cell_state():
    """Same rollback contract for grid-mode adjacent-cell switching.

    Production scenario: grid_r0_c1 → Preset006 enqueued, camera
    rejects, DB had been writing ptz_zone=grid_r0_c1 +
    ptz_preset_token=Preset006 anyway.
    """
    fail_token = "grid_token_r0_c1"
    runner = _FailingRunner(fail_tokens={fail_token})
    controller = AutoPtzController(
        camera_provider=lambda: _grid_camera(acquire_frames=1, cooldown_ms=100),
        command_runner=runner,
        clock=FakeClock(),
        worker_enabled=False,
    )

    # Detection center at (0.5, 0.16) → cell (0, 1) for a 3×3 grid.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_grid_detection(0.5, 0.16)],
    )

    assert len(runner.attempts) == 1
    assert runner.attempts[0].preset_token == fail_token

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] is None
    assert snap["ptz_zone"] is None


def test_successful_goto_leaves_last_preset_committed():
    """Sanity: when the runner does NOT raise, no rollback fires."""
    runner = _FailingRunner(fail_tokens=set())  # never fails
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=runner,
        clock=FakeClock(),
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )

    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] == "left_token"
    assert snap["ptz_zone"] == "left"


# ---------------------------------------------------------------------------
# Worker-thread retry — cheap-PTZ camera transient failures.
#
# Observed regression: every goto in a 3h+ Grid-Mode session was
# rejected with "Preset token does not exist", but a manual CLI test
# of the same token an hour later worked instantly. The rejection is
# transient (camera busy / mid-move / firmware quirk), not permanent.
# Retry converts most of these into eventual success.
#
# We exercise _run_with_retry directly on a worker_enabled=False
# controller so the test stays sync (no thread). The retry path itself
# does not depend on a running worker — only on the runner, the
# stop_event, and the queue, all of which are present either way.
# ---------------------------------------------------------------------------


class _FlakyRunner:
    """Goto fails the first N calls per token, then succeeds.

    Models the cheap-camera quirk where a goto is rejected if the camera
    is mid-busy, but accepted after a short wait.
    """

    def __init__(self, fail_first_n: dict[str, int] | None = None) -> None:
        self.fail_first_n = dict(fail_first_n or {})
        self.attempts: list = []

    def __call__(self, command) -> None:
        self.attempts.append(command)
        if command.action != "goto":
            return
        remaining = self.fail_first_n.get(command.preset_token, 0)
        if remaining > 0:
            self.fail_first_n[command.preset_token] = remaining - 1
            raise RuntimeError(
                f"The requested preset token does not exist: {command.preset_token}"
            )


def _make_retry_controller(runner):
    return AutoPtzController(
        camera_provider=lambda: _camera(mode="preset", acquire_frames=1),
        command_runner=runner,
        clock=FakeClock(),
        worker_enabled=False,  # we drive _run_with_retry by hand
    )


def _goto_command(preset: str = "left_token", *, prev_preset: str = "") -> PtzCommand:
    return PtzCommand(
        action="goto",
        camera_id=0,
        preset_token=preset,
        rollback_preset=prev_preset,
        rollback_zone="",
    )


def test_retry_recovers_when_camera_accepts_second_attempt(monkeypatch):
    """First goto rejected, second goto accepted → no rollback, no error logged as fatal."""
    # Make backoff trivially short for the test.
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 0.0)

    runner = _FlakyRunner(fail_first_n={"left_token": 1})  # fails once, then ok
    controller = _make_retry_controller(runner)
    # Pre-commit the optimistic state the way _maybe_goto_zone would.
    with controller._lock:
        controller._last_preset = "left_token"
        controller._last_zone = "left"

    controller._run_with_retry(_goto_command("left_token"))

    assert len(runner.attempts) == 2, "expected 2 attempts (1 fail + 1 success)"
    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] == "left_token", (
        "successful retry must leave the committed state intact — no rollback"
    )
    assert snap["ptz_zone"] == "left"


def test_retry_gives_up_after_three_total_attempts(monkeypatch):
    """All three attempts fail → rollback fires exactly as the no-retry path."""
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 0.0)

    runner = _FlakyRunner(fail_first_n={"left_token": 99})  # never recovers
    controller = _make_retry_controller(runner)
    with controller._lock:
        controller._last_preset = "left_token"
        controller._last_zone = "left"

    controller._run_with_retry(
        _goto_command("left_token", prev_preset="overview_token")
    )

    assert len(runner.attempts) == 3, "expected 3 total attempts before giving up"
    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] == "overview_token", (
        "after all retries exhausted, the CAS rollback must restore the "
        "previous successful preset, not leave the failed token in the DB"
    )


def test_retry_does_not_apply_to_move_commands(monkeypatch):
    """ContinuousMove is operator-joystick state — never replay a stale move.

    Replaying a 250ms-old move would jerk the camera against the
    operator's current heading.
    """
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 0.0)

    attempts = []

    def runner(command):
        attempts.append(command)
        raise RuntimeError("camera offline")

    controller = _make_retry_controller(runner)
    move_command = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_with_retry(move_command)

    assert len(attempts) == 1, "move commands must NOT be retried"


def test_retry_does_not_apply_to_stop_commands(monkeypatch):
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 0.0)

    attempts = []

    def runner(command):
        attempts.append(command)
        raise RuntimeError("camera offline")

    controller = _make_retry_controller(runner)
    controller._run_with_retry(PtzCommand(action="stop", camera_id=0))

    assert len(attempts) == 1, "stop commands must NOT be retried"


def test_retry_abandons_when_newer_command_queued(monkeypatch):
    """If a fresher goto lands on the queue during backoff, abandon the stale retry.

    The newer command supersedes the in-flight target; replaying the
    old goto would waste camera bandwidth on a stale destination.
    """
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 0.0)

    runner = _FlakyRunner(fail_first_n={"left_token": 99})
    controller = _make_retry_controller(runner)
    with controller._lock:
        controller._last_preset = "left_token"
        controller._last_zone = "left"
    # Put a newer command on the queue before retry kicks in.
    controller._queue.put_nowait(_goto_command("right_token"))

    controller._run_with_retry(
        _goto_command("left_token", prev_preset="overview_token")
    )

    assert len(runner.attempts) == 1, (
        "retry must abandon after the first failure when a fresher command "
        "is already queued"
    )
    snap = controller.snapshot_for_image_persistence()
    assert snap["ptz_preset_token"] == "overview_token", (
        "abandoning the stale retry must still trigger the CAS rollback"
    )


def test_retry_stops_on_shutdown(monkeypatch):
    """If stop_event fires during backoff, do not run more attempts."""
    # Real (non-zero) backoff so the test exercises the wait path.
    monkeypatch.setattr("core.ptz_tracking_core._GOTO_RETRY_BACKOFF_SEC", 5.0)

    runner = _FlakyRunner(fail_first_n={"left_token": 99})
    controller = _make_retry_controller(runner)
    with controller._lock:
        controller._last_preset = "left_token"
        controller._last_zone = "left"

    # Trigger stop concurrently with the retry.
    import threading

    def trip_stop():
        # tiny delay so the first attempt has time to fail
        import time as _t

        _t.sleep(0.05)
        controller._stop_event.set()

    t = threading.Thread(target=trip_stop)
    t.start()
    controller._run_with_retry(_goto_command("left_token"))
    t.join()

    assert len(runner.attempts) == 1, (
        "shutdown during backoff must abandon further attempts immediately, "
        "not wait the full backoff window"
    )


def _follow_camera_with_budget(budget_sec: float, *, move_duration_ms: int = 250) -> dict:
    """Follow-mode camera dict that carries the near-focus zoom budget.

    The budget field defaults to 0.0 (disabled) elsewhere; tests that
    care about the guard explicitly set it here.
    """
    cam = _camera(mode="follow", acquire_frames=1)
    cam["ptz"]["follow_zoom_max_burst_sec"] = float(budget_sec)
    cam["ptz"]["move_duration_ms"] = int(move_duration_ms)
    # Floor the cooldown so we can fire multiple zoom-in commands without
    # advancing the fake clock past 100ms each frame — the budget guard
    # is the only gate under test. The validator clamps below 100, so
    # we use the lowest valid value.
    cam["ptz"]["command_cooldown_ms"] = 100
    return cam


def test_follow_zoom_budget_blocks_zoom_in_after_exhaustion():
    """Once the operator's near-focus budget is spent on zoom-in bursts,
    further zoom-in commands are suppressed even when the bbox is still
    too small. Pan/tilt may still fire — only the zoom-in direction is
    capped. This protects the lens on cams without absolute zoom
    feedback (GetStatus stub)."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    # Budget = 0.5s; each burst charges move_duration_ms (0.25s). So
    # the third zoom-in attempt should see no zoom even though the
    # area is still well below target.
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(0.5),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    tiny_bbox = _follow_detection(70, 10, 90, 30)  # area = 4%, target 18%
    for _ in range(3):
        controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny_bbox])
        clock.advance(0.2)  # > 100ms cooldown floor

    zoom_in_cmds = [c for c in commands if c.action == "move" and c.zoom > 0]
    assert len(zoom_in_cmds) == 2, (
        f"expected exactly 2 zoom-in bursts (budget 0.5s / 0.25s each), "
        f"got {len(zoom_in_cmds)}"
    )
    # The third frame still issued a move (pan/tilt), but zoom must be 0.
    move_cmds = [c for c in commands if c.action == "move"]
    assert len(move_cmds) == 3
    assert move_cmds[2].zoom == 0.0
    assert move_cmds[2].pan > 0  # off-centre bird still gets steering


def test_follow_zoom_budget_does_not_block_zoom_out():
    """Zoom-out is the direction that *releases* the lens. Even after
    the budget is exhausted, a too-big bbox must still trigger zoom-out
    so the controller can recover from any over-zoom state."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(0.25),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Burn the budget with one zoom-in.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(0.2)

    # Now a too-big bbox arrives — zoom-out must still fire.
    big = _follow_detection(20, 20, 100, 100)  # area 64% >> 18% target
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[big])

    zoom_out_cmds = [c for c in commands if c.action == "move" and c.zoom < 0]
    assert len(zoom_out_cmds) == 1, "zoom-out must remain available after budget exhaustion"


def test_follow_zoom_budget_resets_after_return_to_overview():
    """The overview preset is the lens's only absolute reference. Every
    return-to-overview clears the zoom-in budget so the next bird gets
    the full allowance again."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(0.25),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    tiny = _follow_detection(70, 10, 90, 30)

    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    clock.advance(0.2)
    # Budget spent — second frame won't zoom in.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    assert commands[-1].zoom == 0.0

    # Operator triggers a return-to-overview.
    controller.return_to_overview()

    # Fresh bird arrives after the overview goto — zoom-in budget is back.
    clock.advance(1.0)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    fresh_zoom_in = [
        c for c in commands if c.action == "move" and c.zoom > 0
    ]
    assert len(fresh_zoom_in) >= 2, "return_to_overview must reset the zoom-in budget"


def test_follow_zoom_budget_zero_means_disabled():
    """Default value 0.0 must preserve legacy unbounded zoom-in behaviour
    so the field is fully opt-in for existing installs."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(0.0),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    tiny = _follow_detection(70, 10, 90, 30)
    for _ in range(5):
        controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
        clock.advance(0.2)

    zoom_in_cmds = [c for c in commands if c.action == "move" and c.zoom > 0]
    assert len(zoom_in_cmds) == 5, "budget=0 must impose no zoom-in cap"


def test_follow_zoom_locks_after_manual_joystick_drive():
    """After the operator touches the joystick, follow-mode must refuse
    to zoom-in until an overview goto re-establishes the wide-angle
    baseline. Without absolute zoom feedback we have no honest way to
    know where the lens is post-manual."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(2.0),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator nudges the joystick.
    controller.notify_manual_drive()
    clock.advance(1.0)

    # Bird arrives. Follow-mode should pan/tilt but NOT zoom in.
    tiny = _follow_detection(70, 10, 90, 30)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    move_cmds = [c for c in commands if c.action == "move"]
    assert len(move_cmds) == 1, "follow-mode still steers pan/tilt"
    assert move_cmds[0].zoom == 0.0, (
        "zoom-in must be locked after manual joystick drive — operator's "
        "lens position is unknown until an overview goto resets it"
    )
    assert move_cmds[0].pan > 0, "pan still fires (zoom is the only locked axis)"


def test_follow_zoom_locks_after_non_overview_preset_goto():
    """notify_external_goto with a non-overview token (e.g. operator
    clicked Preset005 in the UI) also leaves the lens at an unknown
    zoom level. Lock follow-mode zoom until overview is reached."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(2.0),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator clicks a non-overview preset in the UI.
    controller.notify_external_goto("Preset005")
    clock.advance(1.0)

    tiny = _follow_detection(70, 10, 90, 30)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    move_cmds = [c for c in commands if c.action == "move"]
    assert move_cmds
    assert move_cmds[0].zoom == 0.0, (
        "zoom-in must be locked after a non-overview preset goto"
    )


def test_follow_zoom_unlocks_after_overview_goto():
    """notify_external_goto to the overview_preset clears the lock —
    operator deliberately returned the lens to its known wide-angle."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    cam = _follow_camera_with_budget(2.0)
    controller = AutoPtzController(
        camera_provider=lambda: cam,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Lock via manual drive.
    controller.notify_manual_drive()
    clock.advance(1.0)
    tiny = _follow_detection(70, 10, 90, 30)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    assert commands[-1].zoom == 0.0, "lock active"

    # Operator clicks the overview preset in the UI.
    overview = cam["ptz"]["overview_preset"]
    controller.notify_external_goto(overview)
    clock.advance(1.0)

    # Fresh bird — follow-mode can zoom again now.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    later_zoom_in = [
        c for c in commands if c.action == "move" and c.zoom > 0
    ]
    assert later_zoom_in, (
        "overview goto must release the lock so follow-mode can zoom again"
    )


def test_follow_zoom_unlocks_after_lost_timeout_return():
    """The auto-return path (handle_no_detection after lost_timeout)
    issues a goto to the overview preset. That must also clear any
    manual-control lock so follow-mode can resume normally on the
    next bird."""
    clock = FakeClock()
    commands: list[PtzCommand] = []
    controller = AutoPtzController(
        camera_provider=lambda: _follow_camera_with_budget(2.0),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Set up a tracking state first so handle_no_detection can fire
    # the lost-timeout return.
    tiny = _follow_detection(70, 10, 90, 30)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    # Lock via manual drive mid-track.
    controller.notify_manual_drive()
    clock.advance(0.2)

    # Trigger lost-timeout. Default lost_timeout_sec is 6.0.
    clock.advance(20.0)
    controller.handle_no_detection()

    # Confirm auto-return goto fired.
    gotos = [c for c in commands if c.action == "goto"]
    assert gotos, "lost-timeout must have issued an overview goto"

    # Fresh bird later — zoom-in should be unlocked again.
    clock.advance(1.0)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    later_zoom_in = [
        c for c in commands if c.action == "move" and c.zoom > 0
    ]
    assert later_zoom_in, "lost-timeout overview return must clear the lock"


# ----------------------------------------------------------------------
# Auto-PTZ burst multipliers (mirror the joystick burst from stream.html).
#
# Velocity-ignorant cheap cams need pan/tilt corrections fired N times
# back-to-back per follow-mode tick. The controller's _run_command
# branches on action — move commands now go through _run_move_with_burst,
# which reads manual_pan_tilt_burst / manual_zoom_burst from per-cam
# config and replays the ContinuousMove that many times.
# ----------------------------------------------------------------------


def _burst_controller(burst_pan_tilt: int = 1, burst_zoom: int = 1):
    """Build a controller for direct _run_move_with_burst() calls.

    Uses worker_enabled=False so the worker queue can be touched
    without a real thread racing us, but DOES install the real
    _run_command (no command_runner override) — that's the path
    that contains the burst logic we're testing.
    """
    return AutoPtzController(
        camera_provider=lambda: _camera(mode="follow"),
        clock=FakeClock(),
        worker_enabled=False,
    )


def test_run_move_default_burst_one_fires_one_call(monkeypatch):
    """Default config (1/1) preserves legacy single-call behaviour."""
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(("move", camera_id, kw)),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {
            "manual_pan_tilt_burst": 1,
            "manual_zoom_burst": 1,
            "manual_move_duration_multiplier": 1.0,
        },
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1
    # Default multiplier 1.0 → duration unchanged.
    assert calls[0][2]["duration_ms"] == 250


def test_run_move_duration_multiplier_extends_each_call(monkeypatch):
    """manual_move_duration_multiplier=2.0 doubles each ContinuousMove's
    duration. Independent from burst — applies even at burst=1, where
    no burst-loop runs.
    """
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {
            "manual_pan_tilt_burst": 1,
            "manual_zoom_burst": 1,
            "manual_move_duration_multiplier": 2.0,
        },
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=300
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1
    assert calls[0]["duration_ms"] == 600


def test_run_move_burst_and_duration_combine(monkeypatch):
    """Burst and duration multipliers multiply, not max/min.

    Operator can use both knobs at the same time — burst=3, duration=2
    means 3 ContinuousMove calls of 2× duration each = effectively 6×
    the per-correction movement budget.
    """
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {
            "manual_pan_tilt_burst": 3,
            "manual_zoom_burst": 1,
            "manual_move_duration_multiplier": 2.0,
        },
    )
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: False
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=300
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 3
    for kw in calls:
        assert kw["duration_ms"] == 600, (
            "every burst-call uses the multiplied duration, not just the first"
        )


def test_run_move_pan_tilt_burst_fires_n_calls(monkeypatch):
    """manual_pan_tilt_burst=3 produces three ContinuousMove calls."""
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {"manual_pan_tilt_burst": 3, "manual_zoom_burst": 1},
    )
    # Make spacing-sleep instant so the test doesn't actually wait
    # ~800 ms per case. The wait() returns False when not signalled,
    # which is what we want for the "continue with next burst" path.
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: False
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 3
    # All three calls use the same pan/tilt/zoom magnitude.
    for kw in calls:
        assert kw["pan"] == 0.3
        assert kw["tilt"] == 0.0
        assert kw["zoom"] == 0.0


def test_run_move_pure_zoom_uses_zoom_burst(monkeypatch):
    """A zoom-only command picks manual_zoom_burst, not pan_tilt_burst.

    The follow controller issues pure pan/tilt moves OR pure zoom moves
    depending on which correction the bbox needs — mixed pan+zoom isn't
    produced by _follow_step. So a zoom-only command honouring zoom_burst
    and ignoring pan_tilt_burst is the correct axis classifier.
    """
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {"manual_pan_tilt_burst": 3, "manual_zoom_burst": 1},
    )
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: False
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.0, tilt=0.0, zoom=0.4, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1, "pure-zoom uses zoom_burst=1, not pan_tilt_burst=3"


def test_run_move_burst_aborts_when_stop_queued(monkeypatch):
    """A queued `stop` mid-burst preempts the rest of the burst.

    Operator-initiated stops and overview-returns must take effect
    quickly — finishing the burst would delay the response.
    """
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {"manual_pan_tilt_burst": 4, "manual_zoom_burst": 1},
    )
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: False
    )
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow"),
        clock=FakeClock(),
        worker_enabled=True,
    )
    controller._stop_event.set()  # prevent the worker thread from draining

    controller._queue.put_nowait(PtzCommand(action="stop", camera_id=0))
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1


def test_run_move_burst_does_not_abort_on_queued_follow_move(monkeypatch):
    """A queued follow `move` does NOT preempt the current burst.

    Critical for live behaviour: the detection loop enqueues a fresh
    follow-correction every ~250 ms, so the queue is almost always
    non-empty between burst iterations. If a queued `move` aborted the
    burst tail, every active burst would silently regress to burst=1
    (the regression that motivated this test). The queued move just
    runs next; it doesn't amputate the current one.
    """
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {"manual_pan_tilt_burst": 4, "manual_zoom_burst": 1},
    )
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: False
    )
    controller = AutoPtzController(
        camera_provider=lambda: _camera(mode="follow"),
        clock=FakeClock(),
        worker_enabled=True,
    )
    controller._stop_event.set()

    controller._queue.put_nowait(
        PtzCommand(action="move", camera_id=0, pan=0.2, duration_ms=250)
    )
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 4, (
        "burst with a queued follow-move must complete all 4 calls; "
        "aborting would silently regress to burst=1"
    )


def test_run_move_burst_aborts_on_shutdown(monkeypatch):
    """Shutdown event mid-burst stops further calls."""
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {"manual_pan_tilt_burst": 4, "manual_zoom_burst": 1},
    )
    # Event.wait returns True when the event is set → signals abort.
    monkeypatch.setattr(
        "threading.Event.wait", lambda self, timeout=None: True
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1, "shutdown signal must abort after first call"
