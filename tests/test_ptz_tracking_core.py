from core.ptz_tracking_core import AutoPtzController, PtzCommand


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _camera() -> dict:
    return {
        "id": 0,
        "name": "Garden PTZ",
        "ip": "198.51.100.10",
        "ptz": {
            "enabled": True,
            "overview_preset": "overview_token",
            "lost_timeout_sec": 6.0,
            "command_cooldown_ms": 700,
            "deadband": 0.12,
            "max_speed": 0.35,
            "move_duration_ms": 250,
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


def _follow_detection(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Like _detection but with explicit y coords so bbox area is tunable.

    Follow mode reads the bbox area to drive zoom; the default _detection
    helper's fixed y1=40/y2=60 doesn't give enough range to test the
    zoom-in vs zoom-out branches in isolation.
    """
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": 0.9,
        "class_name": "bird",
    }


def test_follow_mode_steers_pan_tilt_toward_center():
    """A bird off-centre in follow mode triggers a continuous move
    (action='move', pan != 0) on the FIRST frame — no acquire window."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
    assert cmd.zoom == 0.0
    # Tilt is active, so the tighter tilt safety cap applies (not the
    # 2000ms pan-only ceiling).
    assert cmd.duration_ms == 500
    assert controller.status()["state"] == "tracking"


def test_follow_mode_zoom_out_when_bird_too_big():
    """Bird covering most of the frame → bbox area > target → zoom out."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
    cooldown protects them from overlapping firmware commands."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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


def test_follow_mode_no_detection_holds_before_searching():
    """One missing low-FPS frame does not cancel the active acquisition."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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

    # The camera's bounded ContinuousMove stops itself. Do not enqueue a
    # competing Stop merely because one inference frame missed the bird.
    stop_cmds = [c for c in commands if c.action == "stop"]
    assert stop_cmds == []
    # No new Move was issued.
    move_count_after = len([c for c in commands if c.action == "move"])
    assert move_count_after == move_count_before


def test_follow_mode_no_detection_hold_does_not_spam_stop():
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
    assert stop_cmds == []


def test_follow_mode_lost_target_continues_along_recent_trajectory():
    """Two coherent target positions seed a short predictive search.

    The first missing frame should move toward the extrapolated position
    instead of stopping on the last observed bounding box. Directional search
    stays pan/tilt-only; a separate bounded zoom-out may widen the view later.
    """
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(25, 40, 45, 60)],
    )
    clock.advance(2.1)  # sample after the prior PTZ move has finished
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(45, 40, 65, 60)],
    )
    commands_before_lost = len(commands)

    clock.advance(0.1)
    controller.handle_no_detection()
    clock.advance(2.1)
    controller.handle_no_detection()

    assert len(commands) == commands_before_lost + 1
    search = commands[-1]
    assert search.action == "move"
    assert search.pan > 0.0
    assert search.tilt == 0.0
    assert search.zoom == 0.0
    status = controller.status()
    assert status["state"] == "lost_grace"
    assert status["prediction_active"] is True
    assert status["predicted_target_center"][0] > 0.55


def test_follow_mode_predictive_search_is_bounded_then_stops_once():
    """Lost pursuit emits at most four moves, then explicitly halts."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(25, 40, 45, 60)],
    )
    clock.advance(0.5)
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(45, 40, 65, 60)],
    )
    moves_before_lost = len([c for c in commands if c.action == "move"])

    clock.advance(0.1)
    controller.handle_no_detection()  # begin hold window
    clock.advance(2.1)
    controller.handle_no_detection()  # first predictive pulse
    for _ in range(3):
        clock.advance(0.8)
        controller.handle_no_detection()
    clock.advance(5.0)
    controller.handle_no_detection()  # search window expired → stop
    controller.handle_no_detection()  # stop remains edge-triggered

    lost_moves = len([c for c in commands if c.action == "move"]) - moves_before_lost
    assert lost_moves == 4
    assert len([c for c in commands if c.action == "stop"]) == 1
    assert controller.status()["prediction_active"] is False


def test_follow_mode_target_jump_does_not_continue_stale_trajectory():
    """A target switch drops velocity history rather than chasing the old bird."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(0, 40, 20, 60)],
    )
    clock.advance(0.5)
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(80, 40, 100, 60)],
    )
    commands_before_lost = len(commands)

    controller.handle_no_detection()
    clock.advance(2.1)
    controller.handle_no_detection()

    assert len(commands) == commands_before_lost + 1
    assert commands[-1].action == "stop"
    assert controller.status()["prediction_active"] is False


def test_follow_mode_uses_full_direction_with_bounded_duration():
    """Duration controls distance when camera velocity scaling is ineffective."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # The measured camera ignores velocity magnitude, so a large horizontal
    # error uses full direction and expresses distance as a bounded duration.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(85, 40, 95, 60)],
    )

    move_cmds = [c for c in commands if c.action == "move"]
    assert len(move_cmds) == 1
    assert move_cmds[0].pan == 0.35
    assert move_cmds[0].duration_ms == 2000


def test_follow_mode_uses_adaptive_duration_for_dominant_axis():
    """Large diagonal errors move both axes; duration is calibrated to
    whichever axis needs the longer command."""
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "deadband": 0.04,
            "follow_pan_rate_per_sec": 0.075,
            "follow_tilt_rate_per_sec": 0.15,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )

    command = commands[-1]
    assert command.pan == camera["ptz"]["max_speed"]
    assert command.tilt == camera["ptz"]["max_speed"]
    assert command.zoom == 0.0
    # Tilt is active, so the tighter tilt safety cap applies (not the
    # 2000ms pan-only ceiling).
    assert command.duration_ms == 500


def test_follow_mode_zooms_aggressively_only_after_centering():
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "deadband": 0.04,
            "follow_zoom_duration_ms": 500,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(45, 45, 55, 55)],
    )

    command = commands[-1]
    assert command.pan == 0.0
    assert command.tilt == 0.0
    assert command.zoom > 0.0
    assert command.duration_ms == 500


def test_follow_mode_caps_tilt_until_fresh_visual_feedback():
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "deadband": 0.04,
            "follow_tilt_rate_per_sec": 0.15,
            "follow_tilt_max_duration_ms": 500,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(45, 80, 55, 100)],
    )

    command = commands[-1]
    assert command.pan == 0.0
    assert command.tilt < 0.0
    assert command.duration_ms == 500


def test_follow_mode_single_observation_seeds_bounded_lost_search():
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "deadband": 0.04,
            "follow_lost_hold_sec": 2.0,
            "follow_search_sec": 8.0,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 40, 90, 60)],
    )
    moves_before_loss = len([c for c in commands if c.action == "move"])

    controller.handle_no_detection()
    clock.advance(2.1)
    controller.handle_no_detection()

    lost_moves = [c for c in commands if c.action == "move"][moves_before_loss:]
    assert len(lost_moves) == 1
    assert lost_moves[0].pan > 0.0
    assert lost_moves[0].zoom == 0.0


def test_follow_lost_search_zooms_out_once_to_widen_reacquisition_view():
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "follow_lost_hold_sec": 2.0,
            "follow_search_sec": 8.0,
            "follow_search_zoom_out_ms": 250,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 40, 90, 60)],
    )
    controller.handle_no_detection()

    clock.advance(5.0)
    controller.handle_no_detection()
    controller.handle_no_detection()

    zoom_out = [c for c in commands if c.action == "move" and c.zoom < 0]
    assert len(zoom_out) == 1
    assert zoom_out[0].duration_ms == 250


def test_follow_lost_timeout_starts_after_planned_tracking_move():
    clock = FakeClock()
    commands = []
    camera = _camera()
    camera["ptz"].update(
        {
            "deadband": 0.04,
            "lost_timeout_sec": 6.0,
            "follow_pan_rate_per_sec": 0.075,
        }
    )
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 40, 90, 60)],
    )
    assert commands[-1].duration_ms == 2000

    clock.advance(6.1)
    controller.handle_no_detection()
    assert not [command for command in commands if command.action == "goto"]

    clock.advance(2.0)
    controller.handle_no_detection()
    assert [command for command in commands if command.action == "goto"]


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
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
    clock.advance(9.0)
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
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(9.0)
    controller.handle_no_detection()

    # Advance past the cooldown window (lost_timeout_sec from the
    # goto-overview point).
    clock.advance(9.0)
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
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Trigger an auto-return → arms the cooldown.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_follow_detection(70, 10, 90, 30)],
    )
    clock.advance(9.0)
    controller.handle_no_detection()
    assert controller._lost_cooldown_until > 0

    # Manual drive must clear the cooldown.
    controller.notify_manual_drive()
    assert controller._lost_cooldown_until == 0


def test_follow_mode_lost_timeout_returns_to_overview():
    """No detection past lost_timeout_sec → goto(overview_preset).

    Auto-follow only enqueues a preset goto for the overview return;
    normal tracking uses continuous moves."""
    clock = FakeClock()
    commands = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
        camera_provider=lambda: _camera(),
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
        return _camera()

    controller = AutoPtzController(
        camera_provider=camera_provider,
        command_runner=lambda command: None,
        worker_enabled=False,
    )

    controller.handle_no_detection()

    assert calls == 0
    assert controller.status()["state"] == "idle"
    assert calls == 1


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


def test_snapshot_tracking_uses_legacy_ptz_origin_without_preset_token():
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    controller.handle_detections(
        frame_shape=(100, 100, 3), detections=[_detection(0, 20)]
    )
    snap = controller.snapshot_for_image_persistence()

    assert snap["ptz_origin"] == "preset"
    assert snap["ptz_state"] == "tracking"
    assert snap["ptz_preset_token"] is None
    assert snap["ptz_zone"] == "follow"
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
        camera_provider=lambda: _camera(),
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
    """Manual preset movement owns the camera until its settle phase ends."""
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )

    # Operator clicks a non-overview preset (simulates the UI path).
    controller.notify_external_goto("right_token")

    # A detection frame arrives while the camera is still moving.
    controller.handle_detections(
        frame_shape=(100, 100, 3),
        detections=[_detection(0, 20)],  # bird center near x=10/100 = 0.10
    )

    # The manual settle gate prevents auto-follow from fighting it.
    assert commands == [], (
        f"detection-driven counter-goto leaked past manual: {commands}"
    )


def test_home_button_blocks_detection_driven_counter_goto():
    """Same race, but the trigger is the Home / return-to-overview button."""
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
    """Manual joystick control temporarily suppresses auto-follow."""
    clock = FakeClock()
    commands: list = []
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
# Worker-thread retry — cheap-PTZ camera transient failures.
#
# Cheap cameras can transiently reject a valid overview goto while
# busy or mid-move. Retry converts most of these into eventual success.
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
        camera_provider=lambda: _camera(),
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


def _follow_camera_with_budget(
    budget_sec: float, *, move_duration_ms: int = 250
) -> dict:
    """Follow-mode camera dict that carries the near-focus zoom budget.

    The budget field defaults to 0.0 (disabled) elsewhere; tests that
    care about the guard explicitly set it here.
    """
    cam = _camera()
    cam["ptz"]["follow_zoom_max_burst_sec"] = float(budget_sec)
    cam["ptz"]["move_duration_ms"] = int(move_duration_ms)
    cam["ptz"]["follow_zoom_duration_ms"] = int(move_duration_ms)
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

    tiny_bbox = _follow_detection(40, 40, 60, 60)  # area = 4%, target 18%
    for _ in range(3):
        controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny_bbox])
        clock.advance(0.2)  # > 100ms cooldown floor

    zoom_in_cmds = [c for c in commands if c.action == "move" and c.zoom > 0]
    assert len(zoom_in_cmds) == 2, (
        f"expected exactly 2 zoom-in bursts (budget 0.5s / 0.25s each), "
        f"got {len(zoom_in_cmds)}"
    )
    # The third frame emits no command because the only needed action is blocked.
    move_cmds = [c for c in commands if c.action == "move"]
    assert len(move_cmds) == 2


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
        detections=[_follow_detection(40, 40, 60, 60)],
    )
    clock.advance(0.2)

    # Now a too-big bbox arrives — zoom-out must still fire.
    big = _follow_detection(20, 20, 100, 100)  # area 64% >> 18% target
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[big])

    zoom_out_cmds = [c for c in commands if c.action == "move" and c.zoom < 0]
    assert len(zoom_out_cmds) == 1, (
        "zoom-out must remain available after budget exhaustion"
    )


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

    tiny = _follow_detection(40, 40, 60, 60)

    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    clock.advance(0.2)
    # Budget spent — second frame won't zoom in.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    assert len([c for c in commands if c.zoom > 0]) == 1

    # Operator triggers a return-to-overview.
    controller.return_to_overview()

    # Fresh bird arrives after the overview goto — zoom-in budget is back.
    clock.advance(1.0)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    fresh_zoom_in = [c for c in commands if c.action == "move" and c.zoom > 0]
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

    tiny = _follow_detection(40, 40, 60, 60)
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

    # A centred bird needs only zoom; while locked, no command is needed.
    tiny = _follow_detection(40, 40, 60, 60)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    move_cmds = [c for c in commands if c.action == "move"]
    assert move_cmds == []


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

    tiny = _follow_detection(40, 40, 60, 60)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])

    move_cmds = [c for c in commands if c.action == "move"]
    assert move_cmds == []


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
    tiny = _follow_detection(40, 40, 60, 60)
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    assert not [c for c in commands if c.action == "move"], "lock active"

    # Operator clicks the overview preset in the UI.
    overview = cam["ptz"]["overview_preset"]
    controller.notify_external_goto(overview)
    clock.advance(1.0)

    # Fresh bird — follow-mode can zoom again now.
    controller.handle_detections(frame_shape=(100, 100, 3), detections=[tiny])
    later_zoom_in = [c for c in commands if c.action == "move" and c.zoom > 0]
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
    tiny = _follow_detection(40, 40, 60, 60)
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
    later_zoom_in = [c for c in commands if c.action == "move" and c.zoom > 0]
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
        camera_provider=lambda: _camera(),
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


def test_calibrated_follow_move_ignores_legacy_manual_tuning(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "core.ptz_core.continuous_move",
        lambda camera_id, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "core.ptz_core.get_ptz_config",
        lambda cid: {
            "manual_pan_tilt_burst": 6,
            "manual_zoom_burst": 6,
            "manual_move_duration_multiplier": 5.0,
        },
    )
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move",
        camera_id=0,
        pan=0.35,
        duration_ms=900,
        use_manual_tuning=False,
    )

    controller._run_move_with_burst(cmd)

    assert len(calls) == 1
    assert calls[0]["duration_ms"] == 900


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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: False)
    controller = AutoPtzController(
        camera_provider=lambda: _camera(),
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
    monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: True)
    controller = _burst_controller()
    cmd = PtzCommand(
        action="move", camera_id=0, pan=0.3, tilt=0.0, zoom=0.0, duration_ms=250
    )
    controller._run_move_with_burst(cmd)
    assert len(calls) == 1, "shutdown signal must abort after first call"
