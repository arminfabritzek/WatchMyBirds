"""Tests for the OD night-pause gate on DetectionManager.

These tests stub utils.sun_times.is_daytime to keep them independent
of the wall clock and astral's refraction model. The gate itself is
the contract: master-switch ON → always run; master-switch OFF + no
location → run (defensive); master-switch OFF + location + night →
pause.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from detectors.detection_manager import DetectionManager


@pytest.fixture
def manager(monkeypatch, tmp_path):
    """A bare DetectionManager rooted in a tmp OUTPUT_DIR."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    return DetectionManager()


def test_master_switch_on_means_always_run(manager):
    """DAY_AND_NIGHT_CAPTURE=True → gate is a no-op."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = True
    assert manager._should_run_od_now() is True


def test_master_switch_off_without_location_defaults_to_run(manager):
    """No LOCATION_DATA → gate refuses to pause (defensive)."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = None
    assert manager._should_run_od_now() is True


def test_master_switch_off_with_zero_location_defaults_to_run(manager):
    """LOCATION_DATA=(0,0) is treated as 'not configured'."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 0, "longitude": 0}
    assert manager._should_run_od_now() is True


def test_night_with_location_pauses(monkeypatch, manager):
    """Master switch off, location set, is_daytime returns False → pause."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}

    future = datetime.now(tz=UTC) + timedelta(hours=6)
    monkeypatch.setattr(
        "utils.sun_times.is_daytime",
        lambda *a, **kw: (False, future),
    )
    assert manager._should_run_od_now() is False


def test_day_with_location_runs(monkeypatch, manager):
    """Master switch off, location set, is_daytime True → run."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}

    future = datetime.now(tz=UTC) + timedelta(hours=6)
    monkeypatch.setattr(
        "utils.sun_times.is_daytime",
        lambda *a, **kw: (True, future),
    )
    assert manager._should_run_od_now() is True


def test_cache_avoids_repeated_calls_within_ttl(monkeypatch, manager):
    """Two calls within TTL → is_daytime called exactly once."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}
    manager._daytime_ttl = 300

    calls = {"n": 0}

    def fake_is_daytime(*a, **kw):
        calls["n"] += 1
        return False, datetime.now(tz=UTC) + timedelta(hours=6)

    monkeypatch.setattr("utils.sun_times.is_daytime", fake_is_daytime)

    assert manager._should_run_od_now() is False
    assert manager._should_run_od_now() is False
    assert manager._should_run_od_now() is False
    assert calls["n"] == 1


def test_cache_refreshes_after_ttl(monkeypatch, manager):
    """After TTL expires, is_daytime is consulted again."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}
    manager._daytime_ttl = 0  # force refresh on every call

    calls = {"n": 0}

    def fake_is_daytime(*a, **kw):
        calls["n"] += 1
        return False, datetime.now(tz=UTC) + timedelta(hours=6)

    monkeypatch.setattr("utils.sun_times.is_daytime", fake_is_daytime)

    manager._should_run_od_now()
    manager._should_run_od_now()
    manager._should_run_od_now()
    assert calls["n"] == 3


def test_transition_log_fires_once(monkeypatch, manager, caplog):
    """Day → night transition emits exactly one info log line."""
    import logging

    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}
    manager._daytime_ttl = 0  # always refresh

    # First call: seed cache as daytime.
    monkeypatch.setattr(
        "utils.sun_times.is_daytime",
        lambda *a, **kw: (True, datetime.now(tz=UTC) + timedelta(hours=6)),
    )
    manager._should_run_od_now()

    # Second call: flip to night → one transition log.
    caplog.set_level(logging.INFO, logger="detectors.detection_manager")
    monkeypatch.setattr(
        "utils.sun_times.is_daytime",
        lambda *a, **kw: (False, datetime.now(tz=UTC) + timedelta(hours=6)),
    )
    manager._should_run_od_now()

    transition_logs = [
        rec for rec in caplog.records if "transition" in rec.message
    ]
    assert len(transition_logs) == 1
    assert "daytime → night" in transition_logs[0].message


def test_get_od_status_master_switch_on(manager):
    """Master switch ON → reason='master-switch-on'."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = True
    s = manager.get_od_status()
    assert s["od_active"] is True
    assert s["reason"] == "master-switch-on"


def test_get_od_status_no_location(manager):
    """Master switch OFF + no location → reason='no-location'."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = None
    s = manager.get_od_status()
    assert s["od_active"] is True
    assert s["reason"] == "no-location"


def test_get_od_status_night(monkeypatch, manager):
    """Master switch OFF + location + night → reason='night-paused'."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}
    manager._daytime_ttl = 0

    future = datetime.now(tz=UTC) + timedelta(hours=6)
    monkeypatch.setattr(
        "utils.sun_times.is_daytime",
        lambda *a, **kw: (False, future),
    )
    s = manager.get_od_status()
    assert s["od_active"] is False
    assert s["reason"] == "night-paused"
    assert s["next_transition_utc"] is not None
    assert s["lat"] == 52.52
    assert s["lon"] == 13.40


def test_location_string_format_is_accepted(manager):
    """Legacy 'lat,lon' string format must still resolve."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = "52.52,13.40"
    lat, lon = manager._resolve_location()
    assert lat == pytest.approx(52.52)
    assert lon == pytest.approx(13.40)


def test_location_dict_format_is_accepted(manager):
    """Standard dict format is the primary representation."""
    manager.config["DAY_AND_NIGHT_CAPTURE"] = False
    manager.config["LOCATION_DATA"] = {"latitude": 52.52, "longitude": 13.40}
    lat, lon = manager._resolve_location()
    assert lat == pytest.approx(52.52)
    assert lon == pytest.approx(13.40)
