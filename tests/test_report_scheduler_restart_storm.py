"""Restart-storm regression for the Telegram report scheduler.

Before the fix, ``_last_interval_send_ts`` initialised to ``0.0`` (the
Unix epoch) caused ``_should_send_interval`` to return True on the
very first tick after process start, which then chain-fired the
aesthetic-tagger bridge run and the Telegram report. On a Pi with
hundreds of new detections waiting to be scored, this looked like a
full-app freeze right after every restart.

The fix anchors the guard to ``time.time()`` on the first call, so the
first interval-tick after restart happens ``interval_hours`` hours
later, matching the wall-clock cadence the operator expects.
"""

from __future__ import annotations

import time

from web.services import report_scheduler as rs


def _reset_module_state():
    rs._last_interval_send_ts = 0.0
    rs._interval_guard_initialised = False


def test_first_tick_after_restart_does_not_fire_immediately():
    """No restart storm: first call after process start returns False."""
    _reset_module_state()
    assert rs._should_send_interval(1) is False


def test_subsequent_tick_within_interval_does_not_fire():
    _reset_module_state()
    # First call initialises the guard; second call within the interval
    # must remain quiet.
    rs._should_send_interval(1)
    assert rs._should_send_interval(1) is False


def test_tick_after_interval_elapsed_fires():
    _reset_module_state()
    # Manually rewind the guard so the interval has technically elapsed.
    rs._should_send_interval(1)  # initialises to now
    rs._last_interval_send_ts = time.time() - 3601  # > 1h ago
    assert rs._should_send_interval(1) is True


def test_mark_sent_advances_guard():
    _reset_module_state()
    rs._should_send_interval(1)  # initialise
    before = rs._last_interval_send_ts
    time.sleep(0.01)
    rs._mark_sent_interval()
    assert rs._last_interval_send_ts > before


def test_interval_clamped_to_minimum_one_hour():
    """Operator can set interval_hours=0 by mistake; guard treats it as 1."""
    _reset_module_state()
    rs._should_send_interval(0)  # initialise
    # Less than an hour ago: must NOT fire even with interval_hours=0.
    rs._last_interval_send_ts = time.time() - 60
    assert rs._should_send_interval(0) is False


def test_guard_initialisation_is_idempotent():
    """Calling _should_send_interval many times in a row at process start
    must not flap — the guard is only initialised once."""
    _reset_module_state()
    for _ in range(5):
        assert rs._should_send_interval(1) is False
    # And the anchor timestamp has not drifted forward on each call.
    first_ts = rs._last_interval_send_ts
    for _ in range(5):
        rs._should_send_interval(1)
    assert rs._last_interval_send_ts == first_ts
