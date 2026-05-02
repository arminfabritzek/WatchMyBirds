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
    from datetime import date, datetime

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
    ats._mark_run_today()

    with patch("web.services.aesthetic_tag_scheduler.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 2, 2, 10)
        mock_dt.side_effect = datetime
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
