"""Tests for the in-UI PTZ empirical probe backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core import ptz_core, ptz_empirical_probe
from core.ptz_empirical_probe import ProbeSession, StepResult, cases_for

# ---------------------------------------------------------------------------
# Test fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_probe_state(monkeypatch):
    """Each test starts with no in-flight sessions and no controller.

    Also shortens the movestatus poll window to keep the full-walkthrough
    test fast — real wizard runs use the 3s default."""
    ptz_empirical_probe.reset_for_tests()
    ptz_core.clear_capabilities_cache()
    monkeypatch.setattr(ptz_empirical_probe, "_MOVESTATUS_POLL_WINDOW_SEC", 0.05)
    yield
    ptz_empirical_probe.reset_for_tests()
    ptz_core.clear_capabilities_cache()


def _fake_pm_for(tmp_path: Path) -> MagicMock:
    """PathManager mock that resolves the cap-dir under tmp_path."""
    fake_pm = MagicMock()

    def _path_for(cam_id: int) -> Path:
        cap_dir = tmp_path / "ptz_capabilities"
        cap_dir.mkdir(parents=True, exist_ok=True)
        return cap_dir / f"cam{int(cam_id)}.yaml"

    fake_pm.get_ptz_capabilities_path.side_effect = _path_for
    return fake_pm


def _stub_cap_data(
    *,
    ip: str = "10.0.0.42",
    continuous_pt: bool = True,
    continuous_zoom: bool = True,
    relative_pt: bool = False,
    absolute_pt: bool = False,
) -> dict[str, Any]:
    """Shape probe_capabilities() returns — just enough for cases_for + start."""
    return {
        "camera_id": 0,
        "ip": ip,
        "declared": {
            "continuous_pan_tilt": continuous_pt,
            "continuous_zoom": continuous_zoom,
            "relative_pan_tilt": relative_pt,
            "relative_zoom": relative_pt,
            "absolute_pan_tilt": absolute_pt,
            "absolute_zoom": absolute_pt,
        },
        "empirical": None,
    }


def _stub_ptz_config(overview: str = "Preset001") -> dict[str, Any]:
    """ptz_core.get_ptz_config() return shape — only the bits start() reads."""
    return {"overview_preset": overview, "enabled": True}


# ---------------------------------------------------------------------------
# cases_for tests
# ---------------------------------------------------------------------------


def test_cases_for_continuous_only_cam():
    """The operator's actual cam shape: continuous declared, relative/
    absolute not declared. Wizard walks only continuous + the always-on
    movestatus poll. Without an overview preset, preset/multi-goto/speed
    blocks are also skipped — they need a home to fly to."""
    cases = cases_for(
        {"continuous_pan_tilt": True, "continuous_zoom": True}
    )
    kinds = [c["kind"] for c in cases]
    assert "relative" not in kinds
    assert "absolute" not in kinds
    assert "continuous" in kinds
    # movestatus_poll always runs (even when MoveStatus declared false,
    # confirming the stub is itself useful evidence).
    assert kinds[-1] == "movestatus"


def test_cases_for_continuous_with_overview_inserts_homecome_and_preset_block():
    """With an overview_preset the wizard inserts a home-return after
    the continuous block AND adds the preset/multi-goto/speed blocks."""
    cases = cases_for(
        {"continuous_pan_tilt": True, "continuous_zoom": True},
        overview_preset="Preset001",
        probe_slot="20",
    )
    kinds = [c["kind"] for c in cases]
    assert "continuous" in kinds
    assert "homecome" in kinds  # inter-block home-return
    assert "preset_set" in kinds  # SetPreset on probe slot
    assert "preset_goto" in kinds  # GotoPreset round-trip + speed cases
    assert "multi_goto" in kinds  # 3-back-to-back gotos
    assert kinds[-1] == "movestatus"
    # The c_pan_long stop-timing case is now part of the continuous block.
    assert any(c["id"] == "c_pan_long" for c in cases)


def test_cases_for_full_caps_cam():
    """A cam declaring everything gets the full ladder PLUS the preset
    block (when overview_preset + probe_slot are provided)."""
    cases = cases_for(
        {
            "continuous_pan_tilt": True,
            "continuous_zoom": True,
            "relative_pan_tilt": True,
            "relative_zoom": True,
            "absolute_pan_tilt": True,
            "absolute_zoom": True,
        },
        overview_preset="Preset001",
        probe_slot="20",
    )
    kinds = {c["kind"] for c in cases}
    assert kinds >= {
        "continuous", "relative", "absolute",
        "homecome", "preset_set", "preset_goto", "multi_goto",
        "movestatus",
    }
    ids = {c["id"] for c in cases}
    # CLI-parity additions are all present.
    assert {"c_pan_long", "r_pan_right_tiny_repeat",
            "r_tilt_up_small", "r_tilt_down_small",
            "r_zoom_in_small", "r_zoom_out_small",
            "a_left_half", "a_zoom_quarter", "a_zoom_zero"} <= ids


def test_cases_for_no_probe_slot_skips_set_preset():
    """When the operator skips the slot picker, the preset block runs
    GotoPreset on overview only — no SetPreset is issued."""
    cases = cases_for(
        {"continuous_pan_tilt": True},
        overview_preset="Preset001",
        probe_slot=None,
    )
    kinds = [c["kind"] for c in cases]
    assert "preset_set" not in kinds  # no SetPreset case
    assert "preset_goto" in kinds      # but GotoPreset verification stays
    # Multi-goto and speed cases still run, all using overview as target.
    assert "multi_goto" in kinds


def test_cases_for_empty_declared_returns_only_movestatus():
    """Defensive: even when no caps are declared, we still poll
    MoveStatus — the absence of movement evidence is itself a finding.
    Preset block also skipped when no overview is configured."""
    cases = cases_for({})
    assert len(cases) == 1
    assert cases[0]["kind"] == "movestatus"


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------


def test_start_session_refuses_without_overview_preset(tmp_path, monkeypatch):
    """start_session must refuse if Auto-PTZ has no overview preset
    configured — the probe needs a safe return point between tests."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(
        ptz_core, "get_ptz_config", lambda _cid: {"overview_preset": ""}
    )

    with pytest.raises(ValueError, match="overview preset"):
        ptz_empirical_probe.start_session(0)


def test_start_session_pauses_auto_ptz_controller(tmp_path, monkeypatch):
    """Successful start_session must pause the registered controller.
    This is the core safety guarantee — wizard owns the cam exclusively."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    session = ptz_empirical_probe.start_session(0)

    controller.pause_for_external.assert_called_once_with("empirical probe")
    assert session.camera_id == 0
    assert session.current_index == 0
    assert len(session.cases) > 0  # continuous + movestatus at minimum


def test_start_session_is_idempotent_for_in_flight_session(tmp_path, monkeypatch):
    """Calling start twice returns the same session — the second call
    must not reset operator progress or re-pause unnecessarily."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    s1 = ptz_empirical_probe.start_session(0)
    # Advance past the first step so we can detect a reset.
    s1.current_index = 3
    s2 = ptz_empirical_probe.start_session(0)

    assert s1 is s2  # same in-memory object
    assert s2.current_index == 3  # not reset
    controller.pause_for_external.assert_called_once()  # only the FIRST start paused


def test_abort_session_resumes_controller_and_clears_state(tmp_path, monkeypatch):
    """Abort: emergency stop the cam, resume Auto-PTZ, drop the session.
    Required for the wizard's close button and Ctrl-C-equivalent paths."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        aborted = ptz_empirical_probe.abort_session(0)

    assert aborted is True
    # Emergency stop fired against pan_tilt AND zoom.
    fake_client.stop.assert_called_once_with(pan_tilt=True, zoom=True)
    controller.resume_from_external.assert_called_once()
    # Session is gone from memory and from disk.
    assert ptz_empirical_probe.get_session(0) is None
    session_file = tmp_path / "ptz_capabilities" / "cam0.session.yaml"
    assert not session_file.exists()


def test_abort_session_when_no_session_returns_false():
    """abort_session is safe to call when nothing is in flight."""
    assert ptz_empirical_probe.abort_session(0) is False


def test_full_walkthrough_writes_canonical_cache(tmp_path, monkeypatch):
    """End-to-end happy path: start → execute → feedback through all
    steps → finalize writes the canonical cache file that the Settings
    UI reads. Same file format as scripts/ptz_probe."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core,
        "probe_capabilities",
        lambda _cid, **_kw: _stub_cap_data(ip="192.168.1.100"),
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        # Walk through every step, alternating yes/no so the rollup
        # has something interesting to compute. Free-form mode: addressed
        # by step_id, no auto-advance, no auto-finalize.
        total = session.total_steps()
        for i, case in enumerate(list(session.cases)):
            ptz_empirical_probe.execute_step(0, case["id"])
            fb = "yes" if i % 2 == 0 else "no"
            ptz_empirical_probe.record_feedback(0, case["id"], fb)
        # Operator explicitly finalises when satisfied with verdicts.
        result = ptz_empirical_probe.finalize_session(0)
        assert result["cache_path"]
        assert result["verdict_count"] == total

    # The canonical cache file exists and matches the WMB format.
    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    assert cache_file.exists()
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert payload["camera_id"] == 0
    assert payload["connection"]["ip"] == "192.168.1.100"
    # The empirical block carries every rollup the wizard computes —
    # the original four plus the preset/multi-goto/speed additions.
    # follow_zoom_max_burst_sec is optional: only present when the
    # operator actually rated the near-focus step yes/skip, so we use
    # a subset check rather than equality.
    required_keys = {
        "continuous_works",
        "relative_works",
        "absolute_works",
        "movestatus_transitions",
        "preset_set_works",
        "preset_goto_works",
        "multi_goto_works",
        "goto_speed_scales",
    }
    assert required_keys.issubset(payload["empirical"].keys())
    # Tri-state semantics: each rollup is True / False / None. None
    # means "no step of this kind was rated yes-or-no" (operator
    # skipped or never ran them), distinct from False = "rated and
    # broken". Tests alternate yes/no across all cases, so blocks
    # whose every case got a verdict are True or False, but blocks
    # with no cases in the freshly-built case list collapse to None.
    for k in required_keys:
        assert payload["empirical"][k] in (True, False, None), (
            f"empirical[{k}] = {payload['empirical'][k]!r} "
            "must be True, False, or None (tri-state)"
        )
    # recommended_strategy is one of the known values, including the
    # new "unknown" surfaced when nothing empirically worked.
    assert payload["recommended_strategy"] in {
        "absolute",
        "relative",
        "continuous_pulse",
        "presets_only",
        "unknown",
    }
    # Per-step operator verdicts are preserved for later debugging.
    assert "operator_verdicts" in payload
    assert len(payload["operator_verdicts"]) == total

    # Controller is resumed after finalize.
    controller.resume_from_external.assert_called_once()


def test_session_persists_to_disk_and_resumes_after_restart(
    tmp_path, monkeypatch
):
    """Browser refresh / app restart simulation: session is written to
    disk after every step; get_session() finds it again from an empty
    in-memory dict by reading the disk state."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        first_id = session.cases[0]["id"]
        ptz_empirical_probe.execute_step(0, first_id)
        ptz_empirical_probe.record_feedback(
            0, first_id, "yes", comment="worked great"
        )

    # Simulate app restart: drop in-memory state but keep disk.
    ptz_empirical_probe.reset_for_tests()
    ptz_empirical_probe.register_auto_ptz_controller(controller)
    # Clear the pre-restart pause call so we can assert what happens
    # post-restart from a clean slate.
    controller.reset_mock()

    # get_session is pure now — must NOT re-pause from a polling call
    # (F2: avoid surprising side-effects from /status GET requests).
    peek = ptz_empirical_probe.get_session(0)
    assert peek is None  # not yet claimed from disk
    controller.pause_for_external.assert_not_called()

    # claim_session is the explicit entry point that resurrects from
    # disk AND re-pauses Auto-PTZ. /start uses this path.
    recovered = ptz_empirical_probe.claim_session(0)
    assert recovered is not None
    assert recovered.current_index == 1  # advanced past step 0
    assert recovered.results[recovered.cases[0]["id"]].feedback == "yes"
    assert recovered.results[recovered.cases[0]["id"]].comment == "worked great"
    # Resume re-paused the controller exactly once.
    controller.pause_for_external.assert_called_with("empirical probe")


def test_pause_makes_handle_detections_noop(monkeypatch):
    """AutoPtzController.handle_detections must be a no-op while paused.
    This is the safety guarantee that prevents the controller from
    issuing detection-driven moves during the probe."""
    from core.ptz_tracking_core import AutoPtzController

    # Build a controller with no worker thread so we can inspect state.
    controller = AutoPtzController(
        camera_provider=lambda: None,  # would cause _set_idle without pause
        worker_enabled=False,
    )

    # Pause first.
    assert controller.pause_for_external("empirical probe") is True
    paused, reason = controller.is_paused()
    assert paused is True
    assert reason == "empirical probe"

    # handle_detections must NOT call _set_idle while paused (would
    # happen otherwise because camera_provider returns None).
    with patch.object(controller, "_set_idle") as mock_set_idle:
        controller.handle_detections(
            frame_shape=(1080, 1920, 3), detections=[]
        )
        mock_set_idle.assert_not_called()

    # Resume restores normal behaviour.
    assert controller.resume_from_external() is True
    paused, _ = controller.is_paused()
    assert paused is False

    # After resume, handle_detections proceeds and reaches _set_idle
    # (because camera_provider is still None).
    with patch.object(controller, "_set_idle") as mock_set_idle:
        controller.handle_detections(
            frame_shape=(1080, 1920, 3), detections=[]
        )
        mock_set_idle.assert_called_once()


def test_pause_for_external_is_idempotent_for_same_owner():
    """Re-pause from the same owner is a no-op success. One resume
    releases the lock regardless of how many same-owner pauses ran."""
    from core.ptz_tracking_core import AutoPtzController

    controller = AutoPtzController(worker_enabled=False)

    assert controller.pause_for_external("empirical probe") is True
    assert controller.pause_for_external("empirical probe") is False

    _, reason = controller.is_paused()
    assert reason == "empirical probe"

    assert controller.resume_from_external() is True
    assert controller.resume_from_external() is False


def test_pause_for_external_refuses_foreign_owner():
    """A second pause with a DIFFERENT reason must raise instead of
    silently overwriting — otherwise the first owner's resume would
    release a lock they no longer own (F5 in the audit)."""
    from core.ptz_tracking_core import AutoPtzController

    controller = AutoPtzController(worker_enabled=False)

    assert controller.pause_for_external("empirical probe") is True
    with pytest.raises(RuntimeError, match="already paused"):
        controller.pause_for_external("firmware update")

    # First owner's lock is intact — error did not corrupt state.
    _, reason = controller.is_paused()
    assert reason == "empirical probe"
    controller.resume_from_external()


def test_pause_state_surfaces_in_controller_status():
    """status() exposes external_pause_reason so the UI banner can show
    'Auto-PTZ paused — <reason>' even when the operator is on a screen
    other than the wizard."""
    from core.ptz_tracking_core import AutoPtzController

    controller = AutoPtzController(
        camera_provider=lambda: None, worker_enabled=False
    )

    status = controller.status()
    assert status["external_pause_reason"] == ""

    controller.pause_for_external("empirical probe")
    status = controller.status()
    assert status["external_pause_reason"] == "empirical probe"


def test_execute_relative_step_calls_ptz_client_relative_move(tmp_path, monkeypatch):
    """execute_current_step drives the real PtzClient.relative_move
    with the case's inputs. The operator's cam will likely return
    success here even when the move empirically fails
    (cheap-cam endless-pan); the operator catches that via ✗ feedback."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core,
        "probe_capabilities",
        lambda _cid, **_kw: _stub_cap_data(relative_pt=True),
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        first_rel = next(c for c in session.cases if c["kind"] == "relative")
        result = ptz_empirical_probe.execute_step(0, first_rel["id"])

    assert result["kind"] == "relative"
    assert result["onvif_error"] == ""  # no NotImplementedError anymore
    fake_client.relative_move.assert_called_once()
    kwargs = fake_client.relative_move.call_args.kwargs
    # First relative case is r_pan_right_tiny: pan=+0.02, speed=0.5
    assert kwargs["pan"] == 0.02
    assert kwargs["speed"] == 0.5


def test_execute_absolute_step_calls_ptz_client_absolute_move(tmp_path, monkeypatch):
    """Absolute moves are dangerous on cheap cams (some go to endstop).
    The wizard issues the move; the operator confirms or denies via ✗."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core,
        "probe_capabilities",
        lambda _cid, **_kw: _stub_cap_data(absolute_pt=True),
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        first_abs = next(c for c in session.cases if c["kind"] == "absolute")
        result = ptz_empirical_probe.execute_step(0, first_abs["id"])

    assert result["kind"] == "absolute"
    assert result["onvif_error"] == ""
    fake_client.absolute_move.assert_called_once()


def _start_with_overview(tmp_path, monkeypatch, *, probe_slot="20"):
    """Helper: stub PathManager + probe_capabilities + ptz_config so a
    session can start with overview_preset='Preset001' (so preset block
    is included). Returns the started session + fake PtzClient."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())
    fake_client = MagicMock()
    patcher = patch.object(ptz_core, "_client_for_camera", return_value=fake_client)
    patcher.start()
    monkeypatch.setattr(
        ptz_empirical_probe, "_build_client", lambda cid: fake_client
    )
    session = ptz_empirical_probe.start_session(0, probe_slot=probe_slot)
    return session, fake_client, patcher


def test_execute_preset_set_calls_ptz_client_set_preset(tmp_path, monkeypatch):
    """preset_set kind drives PtzClient.set_preset with the operator-picked slot."""
    session, fake_client, patcher = _start_with_overview(
        tmp_path, monkeypatch, probe_slot="20"
    )
    try:
        case = next(c for c in session.cases if c["kind"] == "preset_set")
        result = ptz_empirical_probe.execute_step(0, case["id"])
        assert result["kind"] == "preset_set"
        assert result["onvif_error"] == ""
        fake_client.set_preset.assert_called_once()
        kwargs = fake_client.set_preset.call_args.kwargs
        assert kwargs["preset_token"] == "20"
    finally:
        patcher.stop()


def test_execute_preset_goto_calls_ptz_client_goto_preset(tmp_path, monkeypatch):
    """preset_goto kind drives PtzClient.goto_preset against the case's token."""
    session, fake_client, patcher = _start_with_overview(
        tmp_path, monkeypatch, probe_slot="20"
    )
    try:
        case = next(c for c in session.cases if c["kind"] == "preset_goto")
        result = ptz_empirical_probe.execute_step(0, case["id"])
        assert result["kind"] == "preset_goto"
        assert result["onvif_error"] == ""
        fake_client.goto_preset.assert_called()
    finally:
        patcher.stop()


def test_execute_preset_goto_with_speed_passes_scalar(tmp_path, monkeypatch):
    """The goto-speed cases must forward speed=0.2/0.5/0.8 to goto_preset."""
    session, fake_client, patcher = _start_with_overview(
        tmp_path, monkeypatch, probe_slot="20"
    )
    try:
        speed_cases = [
            c for c in session.cases
            if c["id"].startswith("p_goto_speed_")
        ]
        assert len(speed_cases) == 3  # slow / mid / fast triplet
        expected_speeds = []
        for case in speed_cases:
            fake_client.reset_mock()
            ptz_empirical_probe.execute_step(0, case["id"])
            # The terminal goto_preset call carries the speed; earlier
            # calls in the same step are the prep_goto.
            calls = fake_client.goto_preset.call_args_list
            assert calls, "expected at least one goto_preset call"
            terminal = calls[-1].kwargs
            expected_speeds.append(terminal["speed"])
        assert expected_speeds == [0.2, 0.5, 0.8]
    finally:
        patcher.stop()


def test_execute_multi_goto_fires_three_back_to_back(tmp_path, monkeypatch):
    """multi_goto kind must call goto_preset once per token in the sequence."""
    session, fake_client, patcher = _start_with_overview(
        tmp_path, monkeypatch, probe_slot="20"
    )
    try:
        # Monkeypatch sleep so the test doesn't actually wait 4s × 3.
        monkeypatch.setattr(ptz_empirical_probe.time, "sleep", lambda _s: None)
        case = next(c for c in session.cases if c["kind"] == "multi_goto")
        result = ptz_empirical_probe.execute_step(0, case["id"])
        assert result["kind"] == "multi_goto"
        assert result["onvif_error"] == ""
        # 3 gotos in the sequence overview→probe_slot→overview.
        assert fake_client.goto_preset.call_count == 3
        tokens = [c.args[0] for c in fake_client.goto_preset.call_args_list]
        assert tokens == ["Preset001", "20", "Preset001"]
    finally:
        patcher.stop()


def test_execute_homecome_calls_goto_overview(tmp_path, monkeypatch):
    """homecome kind drives goto_preset against the overview token only."""
    session, fake_client, patcher = _start_with_overview(
        tmp_path, monkeypatch, probe_slot="20"
    )
    try:
        case = next(c for c in session.cases if c["kind"] == "homecome")
        result = ptz_empirical_probe.execute_step(0, case["id"])
        assert result["kind"] == "homecome"
        assert result["onvif_error"] == ""
        fake_client.goto_preset.assert_called()
        first_call = fake_client.goto_preset.call_args_list[0]
        assert first_call.args[0] == "Preset001"
    finally:
        patcher.stop()


def test_start_session_refuses_when_probe_slot_equals_overview(tmp_path, monkeypatch):
    """Operator must not be allowed to overwrite the overview preset."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    with pytest.raises(ValueError, match="same as the overview"):
        ptz_empirical_probe.start_session(0, probe_slot="Preset001")


def test_start_session_canonicalizes_numeric_probe_slot(tmp_path, monkeypatch):
    """Operator types '20' but cam returns 'Preset020' — the slot must
    resolve to the canonical token so the overview-collision check
    catches '20' == 'Preset020' on a cam where both point at the same
    physical slot. This is the bug the operator hit in the 2026-05-19
    probe run: probe_slot='20' + overview='Preset020' → silent same-slot
    multi-goto."""
    from camera.ptz_client import PtzPreset

    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    # Overview is Preset020 — same physical slot as the operator's "20" pick.
    monkeypatch.setattr(
        ptz_core,
        "get_ptz_config",
        lambda _cid: {"overview_preset": "Preset020", "enabled": True},
    )
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    fake_client.list_presets.return_value = [
        PtzPreset(token="Preset020", name="Overview"),
        PtzPreset(token="Preset011", name="Feeder Left"),
    ]
    monkeypatch.setattr(
        ptz_empirical_probe, "_build_client", lambda _cid: fake_client
    )

    # Bare "20" canonicalizes to "Preset020" which equals overview →
    # rejected, even though the literal strings differ.
    with pytest.raises(ValueError, match="same as the overview"):
        ptz_empirical_probe.start_session(0, probe_slot="20")


def test_start_session_canonicalizes_to_non_colliding_slot(tmp_path, monkeypatch):
    """When the operator picks an integer that resolves to a different
    existing token, the session is saved with the canonical token —
    so the preset block tests against a real slot the cam already knows."""
    from camera.ptz_client import PtzPreset

    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    fake_client.list_presets.return_value = [
        PtzPreset(token="Preset020", name="Overview"),
        PtzPreset(token="Preset011", name="Feeder Left"),
    ]
    monkeypatch.setattr(
        ptz_empirical_probe, "_build_client", lambda _cid: fake_client
    )

    # Operator picks slot 11, overview is Preset020 (= Preset001 from stub).
    # Note _stub_ptz_config defaults overview to "Preset001" not "Preset020",
    # so slot 11 → "Preset011" is non-colliding regardless.
    session = ptz_empirical_probe.start_session(0, probe_slot="11")
    assert session.probe_slot == "Preset011"


def test_record_feedback_returns_409_while_execute_in_flight(tmp_path, monkeypatch):
    """The execute-in-flight gate prevents a Skip-during-multi-goto
    from getting silently overwritten when the 12s execute finally
    returns. Backend raises ExecuteInFlightError → route layer surfaces
    as HTTP 409 → wizard JS retries."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    monkeypatch.setattr(
        ptz_empirical_probe, "_build_client", lambda _cid: fake_client
    )

    session = ptz_empirical_probe.start_session(0)
    first_id = session.cases[0]["id"]

    # Manually mark execute-in-flight (no need to actually run a step).
    with ptz_empirical_probe._session_lock:
        ptz_empirical_probe._executing.add(0)

    with pytest.raises(ptz_empirical_probe.ExecuteInFlightError):
        ptz_empirical_probe.record_feedback(0, first_id, "skip")

    # After the execute completes (we clear the flag), feedback works.
    with ptz_empirical_probe._session_lock:
        ptz_empirical_probe._executing.discard(0)
    result = ptz_empirical_probe.record_feedback(0, first_id, "skip")
    assert result["verdict_count"] == 1


def test_execute_clears_in_flight_flag_on_success_and_error(tmp_path, monkeypatch):
    """The execute-in-flight flag must be cleared whether the step
    succeeds or raises — otherwise the whole session is wedged."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    monkeypatch.setattr(
        ptz_empirical_probe, "_build_client", lambda _cid: fake_client
    )

    session = ptz_empirical_probe.start_session(0)
    first_id = session.cases[0]["id"]
    ptz_empirical_probe.execute_step(0, first_id)
    assert 0 not in ptz_empirical_probe._executing

    # Now make the cam raise on the next execute and confirm the flag
    # is still cleared.
    fake_client.continuous_move.side_effect = RuntimeError("simulated ONVIF error")
    ptz_empirical_probe.execute_step(0, first_id)
    assert 0 not in ptz_empirical_probe._executing


def test_execute_movestatus_step_uses_poll_move_status(tmp_path, monkeypatch):
    """Movestatus uses the structured PtzClient.poll_move_status now —
    the result holds StatusSample dicts not just idle-booleans. This is
    the data the empirical probe writes to disk."""
    from camera.ptz_client import StatusSample

    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    fake_client.poll_move_status.return_value = [
        StatusSample(
            pan=0.0, tilt=0.0, zoom=0.0,
            move_status_pan_tilt="IDLE", move_status_zoom="IDLE",
            utc_time="t", error=None,
        ),
        StatusSample(
            pan=0.0, tilt=0.0, zoom=0.0,
            move_status_pan_tilt="MOVING", move_status_zoom="IDLE",
            utc_time="t", error=None,
        ),
    ]
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        ms = next(c for c in session.cases if c["kind"] == "movestatus")
        result = ptz_empirical_probe.execute_step(0, ms["id"])

    assert result["kind"] == "movestatus"
    fake_client.poll_move_status.assert_called_once()
    # Samples are serialised as plain dicts so YAML can handle them.
    assert len(result["poll_samples"]) == 2
    assert result["poll_samples"][1]["move_status_pan_tilt"] == "MOVING"


def test_start_session_rolls_back_pause_when_persist_fails(tmp_path, monkeypatch):
    """F1: a successful pause followed by a persistence failure must
    resume the controller, otherwise Auto-PTZ stays locked until app
    restart with no session to clean up.
    """
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    # Force the in-memory store to blow up after pause succeeds.
    def _explode(_session):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(ptz_empirical_probe, "_persist_session", _explode)

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        ptz_empirical_probe.start_session(0)

    controller.pause_for_external.assert_called_once_with("empirical probe")
    # The rollback path resumed the controller — Auto-PTZ is NOT stuck.
    controller.resume_from_external.assert_called_once()
    # No session is left dangling in memory.
    assert ptz_empirical_probe.get_session(0) is None


def test_finalize_surfaces_cache_write_failure(tmp_path, monkeypatch):
    """F7: when the canonical cache write fails (permissions, disk full),
    record_feedback's final return value carries the error so the wizard
    can show a warning instead of a silent 'success'.
    """
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))

    # PathManager that always raises on the empirical-cache path —
    # simulates a permissions error on /opt/app/data/ptz_capabilities.
    bad_pm = MagicMock()
    bad_pm.get_ptz_capabilities_path.side_effect = PermissionError(
        "Permission denied: /opt/app/data/ptz_capabilities/cam0.yaml"
    )

    # The session machinery itself needs a working PathManager for the
    # session.yaml side; only the finalize step's path lookup fails.
    real_pm = _fake_pm_for(tmp_path)
    pm_calls = {"n": 0}

    def _pm_router():
        # Return the bad PM only when _finalize_session calls it; the
        # session-persist path is hit during execute/feedback first.
        pm_calls["n"] += 1
        return bad_pm if pm_calls["n"] > 1 else real_pm

    monkeypatch.setattr(ptz_empirical_probe, "get_path_manager", _pm_router)
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        for case in list(session.cases):
            ptz_empirical_probe.execute_step(0, case["id"])
            ptz_empirical_probe.record_feedback(0, case["id"], "yes")
        # Free-form: operator explicitly finalises. The cache-write
        # failure surfaces in the finalize_session return.
        result = ptz_empirical_probe.finalize_session(0)

    # Path is empty (write failed), error carries the reason.
    assert result["cache_path"] == ""
    assert "cache_error" in result
    assert "Permission denied" in result["cache_error"]


def test_step_result_dataclass_roundtrips_through_yaml(tmp_path, monkeypatch):
    """Persistence sanity: a session with non-trivial results survives a
    write+read cycle. Catches dataclass-serialisation breakage."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )

    session = ProbeSession(
        camera_id=5,
        camera_ip="10.0.0.42",
        cases=[
            {"id": "x_test", "kind": "continuous", "inputs": {"pan": 0.1},
             "description": "d", "expectation": "e", "purpose": "p"}
        ],
        current_index=1,
        started_at=12345.6,
        finished_at=12400.0,
    )
    session.results["x_test"] = StepResult(
        step_id="x_test",
        feedback="yes",
        comment="all good",
        executed_at=12350.0,
        poll_samples=[{"idle": True, "at": 1.5}],
    )

    ptz_empirical_probe._persist_session(session)
    loaded = ptz_empirical_probe._load_session_from_disk(5)
    assert loaded is not None
    assert loaded.camera_id == 5
    assert loaded.camera_ip == "10.0.0.42"
    assert loaded.current_index == 1
    assert loaded.results["x_test"].feedback == "yes"
    assert loaded.results["x_test"].comment == "all good"
    assert loaded.results["x_test"].poll_samples == [{"idle": True, "at": 1.5}]


# ---------------------------------------------------------------------------
# Near-focus zoom budget tests (read-side of probe-UI plan)
# ---------------------------------------------------------------------------


def test_near_focus_case_skipped_when_continuous_zoom_undeclared():
    """Cams without declared continuous zoom can't have a budget — the
    near_focus step must not appear at all."""
    cases = cases_for({"continuous_pan_tilt": True, "continuous_zoom": False})
    kinds = [c["kind"] for c in cases]
    assert "near_focus" not in kinds


def test_near_focus_case_present_when_continuous_zoom_declared():
    """When continuous_zoom is declared, exactly one near_focus case is
    inserted, with a homecome after it for lens recovery."""
    cases = cases_for(
        {"continuous_pan_tilt": True, "continuous_zoom": True},
        overview_preset="Preset001",
    )
    nf = [c for c in cases if c["kind"] == "near_focus"]
    assert len(nf) == 1
    assert nf[0]["id"] == "nf_zoom_budget"
    # A homecome must follow the near-focus case so the lens recovers
    # to wide-angle after STOP / Done, regardless of which the operator
    # picks. The case immediately after near_focus in the list is the
    # homecome.
    idx = cases.index(nf[0])
    assert cases[idx + 1]["kind"] == "homecome"


def test_near_focus_execute_appends_burst_samples(tmp_path, monkeypatch):
    """Each execute_step call on the near_focus step appends one
    burst-duration sample to the StepResult.poll_samples list. This is
    the substrate finalize sums into follow_zoom_max_burst_sec."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())

    controller = MagicMock()
    ptz_empirical_probe.register_auto_ptz_controller(controller)

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.execute_step(0, "nf_zoom_budget")

    result = session.results.get("nf_zoom_budget")
    assert result is not None
    assert len(result.poll_samples) == 3
    for s in result.poll_samples:
        assert s["burst_sec"] == pytest.approx(0.25)


def test_near_focus_finalize_with_stop_excludes_last_burst(
    tmp_path, monkeypatch
):
    """STOP (blurry, feedback='yes') means the last burst was over the
    limit. The recorded budget is the sum of all-but-the-last bursts."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        # 4 zoom-in bursts at 0.25s each = 1.0s total. STOP after the
        # 4th means the budget is 0.75s (3 sharp bursts).
        for _ in range(4):
            ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.record_feedback(0, "nf_zoom_budget", "yes")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    assert cache_file.exists()
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert payload["empirical"]["follow_zoom_max_burst_sec"] == pytest.approx(0.75)


def test_near_focus_finalize_with_done_records_full_sum(
    tmp_path, monkeypatch
):
    """Done (feedback='skip') means no blur was seen — record the full
    elapsed time as the budget."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        for _ in range(3):
            ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.record_feedback(0, "nf_zoom_budget", "skip")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert payload["empirical"]["follow_zoom_max_burst_sec"] == pytest.approx(0.75)


def test_near_focus_finalize_without_verdict_omits_field(
    tmp_path, monkeypatch
):
    """Operator didn't rate the near-focus step — the field is absent
    from the cache YAML so the runtime keeps its 0.0 default."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        # Operator ran bursts but never clicked STOP or Done.
        for _ in range(2):
            ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert "follow_zoom_max_burst_sec" not in payload["empirical"]


# ---------------------------------------------------------------------------
# Cache-hydrate tests (cache-hydrate: pre-seed verdicts from previous probe)
# ---------------------------------------------------------------------------


def _write_prior_cache(
    tmp_path: Path, cam_id: int, verdicts: dict[str, dict[str, str]]
) -> Path:
    """Write a minimal cam<id>.yaml carrying just operator_verdicts so
    the hydrate path has a file to read. Other fields are stubbed —
    the hydrate helper only reads operator_verdicts."""
    cap_dir = tmp_path / "ptz_capabilities"
    cap_dir.mkdir(parents=True, exist_ok=True)
    path = cap_dir / f"cam{cam_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "camera_id": cam_id,
                "probed_at": "20260520_000000",
                "connection": {"ip": "10.0.0.42"},
                "empirical": {"continuous_works": True},
                "recommended_strategy": "continuous_pulse",
                "operator_verdicts": verdicts,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_hydrate_seeds_matching_verdicts_from_prior_cache(tmp_path, monkeypatch):
    """Re-opening the wizard after a prior probe pre-loads each step's
    feedback, so the operator sees their earlier yes/no/skip badges
    instead of starting with everything 'not tested'."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    _write_prior_cache(
        tmp_path,
        0,
        {
            "c_pan_right_slow": {"feedback": "yes", "comment": "worked"},
            "c_tilt_up_slow": {"feedback": "no", "comment": "stuck"},
            "movestatus_poll": {"feedback": "skip", "comment": ""},
        },
    )

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)

    assert session.results["c_pan_right_slow"].feedback == "yes"
    assert session.results["c_pan_right_slow"].comment == "worked"
    assert session.results["c_tilt_up_slow"].feedback == "no"
    assert session.results["movestatus_poll"].feedback == "skip"
    # verdict_count surfaces in /status — wizard uses it for the
    # "N of M rated" header. Must reflect hydrated count, not zero.
    assert session.current_index == 3


def test_hydrate_drops_step_ids_not_in_current_case_list(tmp_path, monkeypatch):
    """Firmware drift: a previously rated step disappears from the new
    case list (e.g. declared caps changed). Drop those silently, hydrate
    only matching IDs — never crash."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core,
        "probe_capabilities",
        # New session is continuous-only; the prior YAML had relative
        # verdicts that no longer have matching cases.
        lambda _cid, **_kw: _stub_cap_data(relative_pt=False),
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    _write_prior_cache(
        tmp_path,
        0,
        {
            "c_pan_right_slow": {"feedback": "yes", "comment": ""},
            "r_pan_right_tiny": {"feedback": "no", "comment": "ghost step"},
            "totally_made_up": {"feedback": "yes", "comment": ""},
        },
    )

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)

    # The continuous-block step is hydrated.
    assert session.results["c_pan_right_slow"].feedback == "yes"
    # The relative-block step IS NOT — it's not in the current case list.
    assert "r_pan_right_tiny" not in session.results
    # The fabricated step ID is silently ignored.
    assert "totally_made_up" not in session.results


def test_hydrate_is_noop_when_no_prior_cache(tmp_path, monkeypatch):
    """Fresh install — no cam<id>.yaml exists. start_session must not
    raise; session.results is empty as before this slice."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)

    assert session.results == {}
    assert session.current_index == 0


def test_finalize_after_only_near_focus_preserves_old_verdicts(
    tmp_path, monkeypatch
):
    """The operator flow this slice exists for: re-open the wizard,
    touch ONLY the near-focus step (rate it 'skip'), finish. The new
    cache YAML must still carry the previously rated continuous/move-
    status verdicts — they shouldn't get replaced by 'pending'."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    _write_prior_cache(
        tmp_path,
        0,
        {
            "c_pan_right_slow": {"feedback": "yes", "comment": "ok"},
            "c_zoom_in_slow": {"feedback": "yes", "comment": ""},
            "movestatus_poll": {"feedback": "no", "comment": "stub"},
        },
    )

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        # Operator only touches the near-focus step.
        for _ in range(2):
            ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.record_feedback(0, "nf_zoom_budget", "skip")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    verdicts = payload["operator_verdicts"]
    # Old verdicts survived the round-trip.
    assert verdicts["c_pan_right_slow"]["feedback"] == "yes"
    assert verdicts["c_zoom_in_slow"]["feedback"] == "yes"
    assert verdicts["movestatus_poll"]["feedback"] == "no"
    # The near-focus verdict is now recorded.
    assert verdicts["nf_zoom_budget"]["feedback"] == "skip"
    # And the budget was computed from the bursts.
    assert payload["empirical"]["follow_zoom_max_burst_sec"] == pytest.approx(0.5)
    # Empirical rollups reflect the (preserved) prior verdicts: at
    # least one continuous case rated 'yes' → continuous_works=true.
    assert payload["empirical"]["continuous_works"] is True


def test_hydrate_handles_malformed_yaml_gracefully(tmp_path, monkeypatch):
    """Corrupted cache file must not block the wizard. Log + skip."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    cap_dir = tmp_path / "ptz_capabilities"
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "cam0.yaml").write_text(
        "not: valid: yaml: [unclosed", encoding="utf-8"
    )

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        # Must not raise — wizard renders a clean empty session.
        session = ptz_empirical_probe.start_session(0)

    assert session.results == {}


# ---------------------------------------------------------------------------
# Tri-state rollup: "not tested" must NEVER masquerade as "broken"
# ---------------------------------------------------------------------------


def test_finalize_with_no_verdicts_writes_null_empirical(tmp_path, monkeypatch):
    """The whole point of the tri-state refactor: an empty wizard
    session (operator clicked Finish without rating anything) must
    write null/None for every rollup, never False. False would imply
    we tested and the cam failed — a lie when the operator never
    actually exercised the move."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        # No execute, no record_feedback — straight to finalize.
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    # Every rollup must be None — the cam has no empirical evidence
    # for or against any capability.
    assert payload["empirical"]["continuous_works"] is None
    assert payload["empirical"]["relative_works"] is None
    assert payload["empirical"]["absolute_works"] is None
    assert payload["empirical"]["movestatus_transitions"] is None
    # Strategy must be "unknown" — no rated case yes, no rated case no,
    # so the app has no honest basis to recommend anything.
    assert payload["recommended_strategy"] == "unknown"


def test_finalize_with_one_continuous_yes_does_not_falsify_other_blocks(
    tmp_path, monkeypatch
):
    """Operator rates ONE continuous-block case yes, leaves the rest
    untouched. continuous_works → True (we have positive evidence).
    relative/absolute/movestatus → None (no rated case at all). The
    bug this guards against: the old rollup returned False for blocks
    where every case was pending — a Settings pill would then render
    red ⚠ as if the cam refused, when in fact the operator just hadn't
    tested it yet."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        ptz_empirical_probe.execute_step(0, "c_pan_right_slow")
        ptz_empirical_probe.record_feedback(0, "c_pan_right_slow", "yes")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert payload["empirical"]["continuous_works"] is True
    # Other blocks: not rated → None, not False.
    assert payload["empirical"]["relative_works"] is None
    assert payload["empirical"]["absolute_works"] is None
    assert payload["empirical"]["movestatus_transitions"] is None
    # Strategy must reflect what was actually demonstrated.
    assert payload["recommended_strategy"] == "continuous_pulse"


def test_finalize_with_rated_no_and_others_pending_keeps_pending_as_none(
    tmp_path, monkeypatch
):
    """Operator rates one relative case 'no' (Translation truly broken),
    leaves continuous untouched. relative_works → False (we have
    negative evidence for relative). continuous_works → None.
    Strategy must reflect the mixed reality: there IS rated-no
    evidence for relative, NO evidence for continuous, so we can't
    fall back to continuous_pulse — we drop to presets_only because
    SOMETHING is broken even if not everything was tested."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core,
        "probe_capabilities",
        lambda _cid, **_kw: _stub_cap_data(relative_pt=True),
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        ptz_empirical_probe.execute_step(0, "r_pan_right_tiny")
        ptz_empirical_probe.record_feedback(0, "r_pan_right_tiny", "no")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    assert payload["empirical"]["relative_works"] is False
    assert payload["empirical"]["continuous_works"] is None
    assert payload["empirical"]["absolute_works"] is None


def test_finalize_skip_counts_as_not_tested_not_yes(tmp_path, monkeypatch):
    """Skip is a deliberate non-decision, not positive evidence.
    Rating a step "skip" carries no information about whether the
    cam can do the move. The block-level rollup must treat skip
    the same as pending. (Near-focus is a documented exception,
    handled separately by the near-focus-specific finalize path.)"""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        ptz_empirical_probe.start_session(0)
        ptz_empirical_probe.execute_step(0, "c_pan_right_slow")
        ptz_empirical_probe.record_feedback(0, "c_pan_right_slow", "skip")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    # The only continuous step rated was skip → no rated case in the
    # block → continuous_works must be None, not False.
    assert payload["empirical"]["continuous_works"] is None


def test_near_focus_burst_after_hydrate_writes_budget(tmp_path, monkeypatch):
    """Regression for the read-side interaction bug observed in
    production: operator re-opened the wizard, hydrate seeded the
    prior near_focus 'yes' verdict, then operator clicked 'Zoom in
    once' (twice) + STOP. Finalize should write a fresh
    follow_zoom_max_burst_sec — but the original execute_step gated on
    `existing.feedback == PENDING`, so the hydrated 'yes' blocked the
    write and the budget was lost.

    Fix: near_focus has its own branch that resets feedback to PENDING
    on the first burst after a hydrated verdict, so the new bursts
    accumulate cleanly.
    """
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    monkeypatch.setattr(
        ptz_empirical_probe, "get_path_manager", lambda: _fake_pm_for(tmp_path)
    )
    monkeypatch.setattr(
        ptz_core, "probe_capabilities", lambda _cid, **_kw: _stub_cap_data()
    )
    monkeypatch.setattr(ptz_core, "get_ptz_config", lambda _cid: _stub_ptz_config())
    ptz_empirical_probe.register_auto_ptz_controller(MagicMock())

    # Pre-existing cache YAML with a 'yes' verdict on the near_focus
    # step from a prior session, no burst samples persisted (matches
    # the real wizard YAML schema — operator_verdicts has only
    # feedback+comment, never poll_samples).
    _write_prior_cache(
        tmp_path,
        0,
        {"nf_zoom_budget": {"feedback": "yes", "comment": ""}},
    )

    fake_client = MagicMock()
    with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        session = ptz_empirical_probe.start_session(0)
        # Hydrate seeded the prior 'yes'.
        assert session.results["nf_zoom_budget"].feedback == "yes"
        assert session.results["nf_zoom_budget"].poll_samples == []

        # Operator now clicks "Zoom in once" twice on a re-opened wizard.
        ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        ptz_empirical_probe.execute_step(0, "nf_zoom_budget")
        # Then STOP — blurry (last burst caused the blur).
        ptz_empirical_probe.record_feedback(0, "nf_zoom_budget", "yes")
        ptz_empirical_probe.finalize_session(0)

    cache_file = tmp_path / "ptz_capabilities" / "cam0.yaml"
    payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
    # Budget = sum of all-but-last bursts. 2 bursts at 0.25s → first
    # one sharp = 0.25s, second one blurry = excluded. Budget = 0.25s.
    assert payload["empirical"]["follow_zoom_max_burst_sec"] == pytest.approx(0.25)
    assert payload["operator_verdicts"]["nf_zoom_budget"]["feedback"] == "yes"


def test_loader_preserves_null_empirical_values(tmp_path, monkeypatch):
    """_load_empirical_from_disk must round-trip None correctly.
    Previously, the loader coerced everything through bool() which
    silently turned None into False — the exact misrepresentation
    this refactor exists to fix."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_prior_cache(tmp_path, 0, {})  # no verdicts at all
    cap_dir = tmp_path / "ptz_capabilities"
    # Write an empirical block where everything is explicitly null
    # (as the new finalize would produce).
    target = cap_dir / "cam0.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "camera_id": 0,
                "probed_at": "20260520_120000",
                "connection": {"ip": "10.0.0.42"},
                "empirical": {
                    "continuous_works": None,
                    "relative_works": None,
                    "absolute_works": None,
                    "movestatus_transitions": None,
                },
                "recommended_strategy": "unknown",
            }
        ),
        encoding="utf-8",
    )

    result = ptz_core._load_empirical_from_disk(0)

    assert result is not None
    assert result["continuous_works"] is None
    assert result["relative_works"] is None
    assert result["absolute_works"] is None
    assert result["movestatus_transitions"] is None
    assert result["recommended_strategy"] == "unknown"
