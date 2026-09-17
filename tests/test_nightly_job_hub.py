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
def clean_registry(monkeypatch, tmp_path):
    """Wipe the registry between tests so they don't pollute each other."""
    monkeypatch.setattr(
        nightly_job_hub.nightly_job_state,
        "get_config",
        lambda: {"OUTPUT_DIR": str(tmp_path)},
    )
    # nightly_job_hub uses module-level state; reset before & after.
    nightly_job_hub._registry.clear()  # type: ignore[attr-defined]
    nightly_job_hub._last_fire_date.clear()  # type: ignore[attr-defined]
    yield
    for runtime in nightly_job_hub._registry.values():
        runtime.stop_event.set()
        if runtime.thread is not None:
            runtime.thread.join(timeout=2)
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


def test_crashed_job_exposes_error_status():
    class _CrashingJob(_FakeJob):
        def run(self, stop_event, reason):
            raise RuntimeError("retention test failure")

    nightly_job_hub.register_job(_CrashingJob("crash"))
    nightly_job_hub.run_now("crash")
    nightly_job_hub._registry["crash"].thread.join(timeout=1)

    status = nightly_job_hub.status("crash")
    assert status["last_rc"] == 1
    assert status["last_error"] == "RuntimeError: retention test failure"


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
    # No date is recorded: a live config change may make it eligible later.
    assert "opt_out" not in nightly_job_hub._last_fire_date  # type: ignore[attr-defined]


def test_only_independent_job_fires_without_night_pause():
    class _IndependentJob(_FakeJob):
        @property
        def requires_night_pause(self) -> bool:
            return False

    night_job = _FakeJob("night", run_seconds=0.05)
    independent = _IndependentJob("independent", run_seconds=0.05)
    nightly_job_hub.register_job(night_job)
    nightly_job_hub.register_job(independent)

    nightly_job_hub._maybe_fire_due_jobs(is_night=False)

    assert independent.started_event.wait(timeout=1)
    assert not night_job.started_event.is_set()


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


@pytest.mark.parametrize("name", ["aesthetic_tagger", "sharpness", "retention"])
@pytest.mark.parametrize("rc", [0, 1])
def test_completed_run_survives_restart(name: str, rc: int) -> None:
    job = _FakeJob(name, run_seconds=0)
    job._exit_code = rc
    nightly_job_hub.register_job(job)
    nightly_job_hub.run_now(name)
    nightly_job_hub._registry[name].thread.join(timeout=2)
    before = nightly_job_hub.status(name)
    nightly_job_hub.unregister_job(name)
    nightly_job_hub.register_job(_FakeJob(name))
    assert nightly_job_hub.status(name) == before
    assert before["last_started_at"]
    assert before["last_finished_at"]
    assert before["last_result"] == ("succeeded" if rc == 0 else "failed")


def test_start_is_persisted_and_unfinished_run_restores_as_interrupted() -> None:
    job = _FakeJob(run_seconds=10)
    nightly_job_hub.register_job(job)
    nightly_job_hub.run_now(job.name)
    assert job.started_event.wait(timeout=1)
    saved = nightly_job_hub.nightly_job_state.load_status(job.name)
    assert saved["last_started_at"]
    assert saved["last_finished_at"] is None
    restored = nightly_job_hub._JobRuntime(_FakeJob())
    assert restored.last_result == "interrupted"
    assert restored.last_finished_at is None


def test_daily_marker_survives_manual_run_and_restart() -> None:
    job = _FakeJob(run_seconds=0)
    nightly_job_hub.register_job(job)
    nightly_job_hub._maybe_fire_due_jobs()
    nightly_job_hub._registry[job.name].thread.join(timeout=2)
    nightly_job_hub.run_now(job.name)
    nightly_job_hub._registry[job.name].thread.join(timeout=2)
    nightly_job_hub._registry.clear()
    nightly_job_hub._last_fire_date.clear()
    replacement = _FakeJob()
    nightly_job_hub.register_job(replacement)
    nightly_job_hub._maybe_fire_due_jobs()
    assert not replacement.started_event.is_set()


@pytest.mark.parametrize("confirmed", [False, True])
def test_stop_result_distinguishes_request_from_confirmed_stop(confirmed: bool) -> None:
    class StopJob(_FakeJob):
        def run(self, stop_event: threading.Event, reason: str) -> int:
            self.started_event.set()
            assert stop_event.wait(timeout=2)
            nightly_job_hub.update_progress(self.name, {"stopped": confirmed})
            return 0

    job = StopJob()
    nightly_job_hub.register_job(job)
    nightly_job_hub.run_now(job.name)
    assert job.started_event.wait(timeout=1)
    nightly_job_hub.stop(job.name)
    nightly_job_hub._registry[job.name].thread.join(timeout=2)
    expected = "stopped" if confirmed else "finished_after_stop_request"
    nightly_job_hub.unregister_job(job.name)
    nightly_job_hub.register_job(_FakeJob())
    assert nightly_job_hub.status(job.name)["last_result"] == expected


def test_persistence_failure_does_not_lock_out_future_runs(monkeypatch) -> None:
    def fail_save(*args: object) -> None:
        raise OSError("Disk full")

    monkeypatch.setattr(nightly_job_hub.nightly_job_state, "save_status", fail_save)
    nightly_job_hub.register_job(_FakeJob(run_seconds=0))
    for _ in range(2):
        assert nightly_job_hub.run_now("fake")["status"] == "started"
        nightly_job_hub._registry["fake"].thread.join(timeout=2)
        assert nightly_job_hub.status("fake")["last_result"] == "succeeded"
