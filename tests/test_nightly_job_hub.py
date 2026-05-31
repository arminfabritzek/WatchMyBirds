"""Tests for web.services.nightly_job_hub lifecycle.

Strategy: register fake JobBase subclasses that signal progress
through threading.Events, then assert the hub's lock/stop/status
contract. No real I/O.
"""

from __future__ import annotations

import threading
import time

import pytest

from web.services import nightly_job_hub
from web.services.nightly_job_hub import JobBase


@pytest.fixture(autouse=True)
def clean_registry():
    """Wipe the registry between tests so they don't pollute each other."""
    # nightly_job_hub uses module-level state; reset before & after.
    nightly_job_hub._registry.clear()  # type: ignore[attr-defined]
    nightly_job_hub._last_fire_date.clear()  # type: ignore[attr-defined]
    yield
    nightly_job_hub._registry.clear()  # type: ignore[attr-defined]
    nightly_job_hub._last_fire_date.clear()  # type: ignore[attr-defined]


class _FakeJob(JobBase):
    """A controllable job for testing the hub."""

    def __init__(self, slug: str = "fake", run_seconds: float = 0.1):
        self._name = slug
        self._run_seconds = run_seconds
        self.started_event = threading.Event()
        self.finished_event = threading.Event()
        self.received_reason: str | None = None
        self._exit_code = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"Fake {self._name}"

    def run(self, stop_event: threading.Event, reason: str) -> int:
        self.received_reason = reason
        self.started_event.set()
        # Sleep in small chunks so the stop_event can interrupt.
        deadline = time.monotonic() + self._run_seconds
        while time.monotonic() < deadline:
            if stop_event.is_set():
                break
            time.sleep(0.01)
        self.finished_event.set()
        return self._exit_code


def test_unknown_job_returns_status():
    result = nightly_job_hub.run_now("does_not_exist")
    assert result["status"] == "unknown_job"


def test_register_and_list_returns_job():
    j = _FakeJob("foo")
    nightly_job_hub.register_job(j)
    jobs = nightly_job_hub.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["name"] == "foo"
    assert jobs[0]["running"] is False


def test_run_now_starts_job_and_finishes(monkeypatch):
    j = _FakeJob("foo", run_seconds=0.05)
    nightly_job_hub.register_job(j)

    result = nightly_job_hub.run_now("foo", reason="unit-test")
    assert result["status"] == "started"
    assert j.started_event.wait(timeout=1.0)
    assert j.finished_event.wait(timeout=1.0)
    assert j.received_reason == "unit-test"

    # Wait for thread cleanup.
    time.sleep(0.05)
    status = nightly_job_hub.status("foo")
    assert status is not None
    assert status["running"] is False
    assert status["last_rc"] == 0
    assert status["last_reason"] == "unit-test"


def test_second_run_now_while_running_returns_already_running():
    j = _FakeJob("foo", run_seconds=1.0)
    nightly_job_hub.register_job(j)
    nightly_job_hub.run_now("foo")
    assert j.started_event.wait(timeout=1.0)

    result = nightly_job_hub.run_now("foo")
    assert result["status"] == "already_running"

    # Cleanup
    nightly_job_hub.stop("foo")
    assert j.finished_event.wait(timeout=2.0)


def test_stop_signals_running_job():
    j = _FakeJob("foo", run_seconds=10.0)
    nightly_job_hub.register_job(j)
    nightly_job_hub.run_now("foo")
    assert j.started_event.wait(timeout=1.0)

    result = nightly_job_hub.stop("foo")
    assert result["status"] == "stop_requested"
    assert j.finished_event.wait(timeout=1.0)


def test_stop_when_not_running_returns_not_running():
    j = _FakeJob("foo")
    nightly_job_hub.register_job(j)
    result = nightly_job_hub.stop("foo")
    assert result["status"] == "not_running"


def test_stop_unknown_returns_unknown_job():
    result = nightly_job_hub.stop("missing")
    assert result["status"] == "unknown_job"


def test_two_different_jobs_can_run_in_parallel():
    """The hub uses per-job locks, not a global one. Two distinct
    jobs may run concurrently."""
    a = _FakeJob("a", run_seconds=0.3)
    b = _FakeJob("b", run_seconds=0.3)
    nightly_job_hub.register_job(a)
    nightly_job_hub.register_job(b)

    nightly_job_hub.run_now("a")
    nightly_job_hub.run_now("b")
    assert a.started_event.wait(timeout=1.0)
    assert b.started_event.wait(timeout=1.0)

    # Both should be running at once.
    statuses = {s["name"]: s for s in nightly_job_hub.list_jobs()}
    assert statuses["a"]["running"] is True
    assert statuses["b"]["running"] is True

    a.finished_event.wait(timeout=2.0)
    b.finished_event.wait(timeout=2.0)


def test_progress_updates_visible_to_callers():
    """A job writing to update_progress shows up in list_jobs()."""
    update_seen = threading.Event()

    class _ProgressJob(JobBase):
        @property
        def name(self) -> str:
            return "prog"

        @property
        def display_name(self) -> str:
            return "Prog"

        def run(self, stop_event, reason):
            nightly_job_hub.update_progress(self.name, {"done": 42, "total": 100})
            update_seen.set()
            # hold the lock long enough for the test to read state
            stop_event.wait(timeout=0.3)
            return 0

    nightly_job_hub.register_job(_ProgressJob())
    nightly_job_hub.run_now("prog")
    assert update_seen.wait(timeout=1.0)

    status = nightly_job_hub.status("prog")
    assert status is not None
    assert status["progress"]["done"] == 42
    assert status["progress"]["total"] == 100

    nightly_job_hub.stop("prog")


def test_should_run_in_daily_loop_false_skips_fire():
    """A job that returns False from should_run_in_daily_loop is
    not fired by the daily loop, even at night."""

    class _OptOutJob(_FakeJob):
        def should_run_in_daily_loop(self) -> bool:
            return False

    j = _OptOutJob("opt_out", run_seconds=0.1)
    nightly_job_hub.register_job(j)

    nightly_job_hub._maybe_fire_due_jobs()  # type: ignore[attr-defined]
    # The fake should NOT have started.
    assert not j.started_event.is_set()
    # But the date guard IS marked, so next minute won't retry.
    from datetime import datetime, timezone
    today = datetime.now(tz=timezone.utc).date().isoformat()
    assert nightly_job_hub._last_fire_date.get("opt_out") == today  # type: ignore[attr-defined]


def test_daily_loop_fires_each_job_once_per_day():
    """Two ticks of the daily fire on the same day → one start per job."""
    j = _FakeJob("daily", run_seconds=0.05)
    nightly_job_hub.register_job(j)

    # First fire — triggers a start.
    nightly_job_hub._maybe_fire_due_jobs()  # type: ignore[attr-defined]
    assert j.started_event.wait(timeout=1.0)
    j.finished_event.wait(timeout=1.0)

    # Second fire same day — no restart.
    j.started_event.clear()
    j.finished_event.clear()
    nightly_job_hub._maybe_fire_due_jobs()  # type: ignore[attr-defined]
    time.sleep(0.1)
    assert not j.started_event.is_set()
