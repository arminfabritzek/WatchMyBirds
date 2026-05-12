"""Smoke tests for web.services.aesthetic_tag_scheduler.

The scheduler is a thin wrapper around scripts.aesthetic_tag_nightly
that fires once per day at the configured time. We verify:

1. _parse_time accepts valid HH:MM and rejects garbage gracefully.
2. _should_run respects the duplicate-send guard within the same day.
3. start_aesthetic_tag_scheduler returns None when AESTHETIC_TAG_ENABLED
   is False (zero-touch opt-out).
4. start_aesthetic_tag_scheduler returns a Thread when enabled and
   dependencies are present.
5. The thread is a daemon (won't block app shutdown).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web.services import aesthetic_tag_scheduler as ats  # noqa: E402


def test_parse_time_valid():
    assert ats._parse_time("02:10") == (2, 10)
    assert ats._parse_time("00:00") == (0, 0)
    assert ats._parse_time("23:59") == (23, 59)


def test_parse_time_invalid_falls_back():
    # All garbage input must return the fallback (default 02:10) without raising.
    assert ats._parse_time("not-a-time") == (2, 10)
    assert ats._parse_time("25:99") == (2, 10)
    assert ats._parse_time("") == (2, 10)
    # Custom fallback is honoured.
    assert ats._parse_time("invalid", fallback=(3, 30)) == (3, 30)


def test_should_run_only_at_configured_minute():
    """_should_run returns True only when both hour AND minute match
    AND no run has been recorded today."""
    from datetime import datetime

    # Reset module-level guard before each test.
    ats._last_run_date = None

    # Mock datetime.now() to a known time.
    with patch("web.services.aesthetic_tag_scheduler.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 2, 2, 10)
        # Pass through real `date` for `now.date()`.
        mock_dt.side_effect = datetime
        assert ats._should_run(2, 10) is True

        # Wrong minute
        mock_dt.now.return_value = datetime(2026, 5, 2, 2, 11)
        assert ats._should_run(2, 10) is False

        # Wrong hour
        mock_dt.now.return_value = datetime(2026, 5, 2, 3, 10)
        assert ats._should_run(2, 10) is False


def test_should_run_duplicate_guard():
    """After _mark_run_today, _should_run is False for the rest of the day."""
    from datetime import datetime

    ats._last_run_date = None

    with patch("web.services.aesthetic_tag_scheduler.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 2, 2, 10)
        mock_dt.side_effect = datetime
        # _mark_run_today must read the same (mocked) clock as _should_run,
        # so call it inside the patch context.
        ats._mark_run_today()
        # Even at the right time, the guard prevents another fire today.
        assert ats._should_run(2, 10) is False


def test_disabled_returns_none():
    """When AESTHETIC_TAG_ENABLED is False, no thread starts."""
    ats._last_run_date = None
    with patch("config.get_config", return_value={"AESTHETIC_TAG_ENABLED": False}):
        result = ats.start_aesthetic_tag_scheduler()
    assert result is None


def test_missing_deps_returns_none():
    """If torch/open_clip cannot be imported, scheduler stays idle."""
    ats._last_run_date = None
    with patch("config.get_config", return_value={"AESTHETIC_TAG_ENABLED": True}), \
         patch.object(ats, "_check_dependencies_available", return_value=False):
        result = ats.start_aesthetic_tag_scheduler()
    assert result is None


def test_enabled_returns_daemon_thread():
    """When enabled and deps present, a daemon thread is started."""
    ats._last_run_date = None
    with patch("config.get_config", return_value={
        "AESTHETIC_TAG_ENABLED": True,
        "AESTHETIC_TAG_TIME": "02:10",
    }), patch.object(ats, "_check_dependencies_available", return_value=True):
        result = ats.start_aesthetic_tag_scheduler(check_interval=60)

    assert isinstance(result, threading.Thread)
    assert result.daemon, "scheduler thread must be a daemon so app shutdown is clean"
    assert result.is_alive()
    assert result.name == "AestheticTagScheduler"


def test_check_dependencies_available_returns_bool():
    """The dependency probe always returns bool, never raises."""
    rv = ats._check_dependencies_available()
    assert isinstance(rv, bool)


def _capture_argv():
    """Patch main_with_args to capture the argv the scheduler passes."""
    captured = {"argv": None}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    return captured, fake_main


def _patch_lease_passthrough():
    """A lease stub whose acquire() yields without altering state."""
    class _Stub:
        def acquire(self, *_args, **_kwargs):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield self
            return cm()

    return _Stub()


def test_run_now_passes_per_species_cap_to_worker():
    """run_now() with an explicit cap forwards --per-species-cap=N
    into the worker argv. The pre-Telegram bridge depends on this
    plumbing — the cap is meaningless if it never reaches the CLI."""
    ats._last_run_date = None
    ats._run_mutex = __import__("threading").Lock()  # fresh mutex per test
    captured, fake_main = _capture_argv()

    with patch("config.get_config", return_value={
        "AESTHETIC_TAG_ENABLED": True,
        "AESTHETIC_BRIDGE_PER_SPECIES_CAP": 8,  # default; explicit cap=5 should override
    }), patch.object(ats, "_check_dependencies_available", return_value=True), \
         patch("scripts.aesthetic_tag_nightly.main_with_args", fake_main), \
         patch("web.services.compute_lease_service.get_compute_lease_service",
               return_value=_patch_lease_passthrough()):
        ok = ats.run_now(
            "test bridge",
            since="2026-05-11T00:00:00+00:00",
            throttle_ms=100,
            per_species_cap=5,
        )

    assert ok is True
    assert captured["argv"] is not None
    assert "--per-species-cap" in captured["argv"]
    cap_idx = captured["argv"].index("--per-species-cap")
    assert captured["argv"][cap_idx + 1] == "5"


def test_run_now_falls_back_to_config_default_for_cap():
    """When per_species_cap is unset, run_now reads
    AESTHETIC_BRIDGE_PER_SPECIES_CAP from config. The Telegram
    blueprint always calls run_now(today_only=True) without the cap
    kwarg, so the config-driven default is the only knob the operator
    can turn without code changes."""
    ats._last_run_date = None
    ats._run_mutex = __import__("threading").Lock()
    captured, fake_main = _capture_argv()

    with patch("config.get_config", return_value={
        "AESTHETIC_TAG_ENABLED": True,
        "AESTHETIC_BRIDGE_PER_SPECIES_CAP": 12,
    }), patch.object(ats, "_check_dependencies_available", return_value=True), \
         patch("scripts.aesthetic_tag_nightly.main_with_args", fake_main), \
         patch("web.services.compute_lease_service.get_compute_lease_service",
               return_value=_patch_lease_passthrough()):
        ats.run_now("test bridge", today_only=True)

    assert captured["argv"] is not None
    cap_idx = captured["argv"].index("--per-species-cap")
    assert captured["argv"][cap_idx + 1] == "12"


def test_run_now_zero_cap_disables_the_flag():
    """per_species_cap=0 (or config value 0) means 'no cap'. The
    --per-species-cap flag must NOT be passed in that case so the
    worker reverts to the full-backfill behaviour."""
    ats._last_run_date = None
    ats._run_mutex = __import__("threading").Lock()
    captured, fake_main = _capture_argv()

    with patch("config.get_config", return_value={
        "AESTHETIC_TAG_ENABLED": True,
        "AESTHETIC_BRIDGE_PER_SPECIES_CAP": 0,
    }), patch.object(ats, "_check_dependencies_available", return_value=True), \
         patch("scripts.aesthetic_tag_nightly.main_with_args", fake_main), \
         patch("web.services.compute_lease_service.get_compute_lease_service",
               return_value=_patch_lease_passthrough()):
        ats.run_now("test bridge", today_only=True)

    assert captured["argv"] is not None
    assert "--per-species-cap" not in captured["argv"], (
        f"cap=0 must omit the flag, got argv={captured['argv']!r}"
    )


def test_run_tagger_nightly_path_omits_per_species_cap():
    """The daily loop calls _run_tagger(reason) with no per_species_cap
    kwarg. The nightly run must score every detection of the previous
    day; passing a cap would silently leave species under-scored."""
    captured, fake_main = _capture_argv()

    with patch("scripts.aesthetic_tag_nightly.main_with_args", fake_main), \
         patch("web.services.compute_lease_service.get_compute_lease_service",
               return_value=_patch_lease_passthrough()):
        rc = ats._run_tagger("daily @ 02:10")

    assert rc == 0
    assert "--per-species-cap" not in (captured["argv"] or []), (
        "nightly _run_tagger must never pass --per-species-cap; "
        f"got argv={captured['argv']!r}"
    )
