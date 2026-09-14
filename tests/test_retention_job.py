"""Storage-retention integration with the nightly job hub."""

import threading
from datetime import UTC, datetime

import pytest

from web.services import nightly_job_hub
from web.services.nightly_jobs import retention_job
from web.services.nightly_jobs.retention_job import RetentionJob


@pytest.fixture(autouse=True)
def clean_hub(monkeypatch, tmp_path):
    import config

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    config._CONFIG = None
    nightly_job_hub.stop_daily_loop()
    nightly_job_hub._registry.clear()
    nightly_job_hub._last_fire_date.clear()
    yield
    nightly_job_hub.stop_daily_loop()
    if nightly_job_hub._daily_loop_thread is not None:
        nightly_job_hub._daily_loop_thread.join(timeout=1)
    nightly_job_hub._registry.clear()
    nightly_job_hub._last_fire_date.clear()
    config._CONFIG = None


def test_daily_loop_requires_explicit_auto_opt_in(monkeypatch):
    monkeypatch.setattr(
        retention_job,
        "get_config",
        lambda: {"RETENTION_POSTURE": "conservative", "RETENTION_AUTO_ENABLED": False},
    )
    job = RetentionJob()
    nightly_job_hub.register_job(job)

    nightly_job_hub._maybe_fire_due_jobs(is_night=False)

    assert nightly_job_hub.status("retention")["last_started_at"] is None


def test_daily_loop_does_not_run_when_posture_is_off(monkeypatch):
    monkeypatch.setattr(
        retention_job,
        "get_config",
        lambda: {"RETENTION_POSTURE": "off", "RETENTION_AUTO_ENABLED": True},
    )
    nightly_job_hub.register_job(RetentionJob())

    nightly_job_hub._maybe_fire_due_jobs(is_night=False)

    assert nightly_job_hub.status("retention")["last_started_at"] is None


def test_daily_loop_runs_enabled_retention_and_keeps_result(monkeypatch):
    monkeypatch.setattr(
        retention_job,
        "get_config",
        lambda: {"RETENTION_POSTURE": "conservative", "RETENTION_AUTO_ENABLED": True},
    )
    finished = threading.Event()

    def fake_run(*, stop_requested, progress_callback):
        result = {
            "deleted": 2,
            "freed_bytes": 4096,
            "missing": 0,
            "errors": 0,
            "protected": {"favorite": 1},
            "processed": 3,
            "total": 3,
            "stopped": False,
        }
        progress_callback(result)
        finished.set()
        return result

    monkeypatch.setattr(retention_job.retention_core, "run", fake_run)
    nightly_job_hub.register_job(RetentionJob())

    daily_loop = nightly_job_hub.start_daily_loop(lambda: False, check_interval_s=0.01)
    assert finished.wait(timeout=1)
    nightly_job_hub.stop_daily_loop()
    daily_loop.join(timeout=1)
    nightly_job_hub._registry["retention"].thread.join(timeout=1)

    status = nightly_job_hub.status("retention")
    assert status["last_reason"] == "nightly auto"
    assert status["last_rc"] == 0
    assert status["progress"]["freed_bytes"] == 4096
    assert status["progress"]["protected"] == {"favorite": 1}


def test_retention_file_errors_set_failed_job_status(monkeypatch):
    monkeypatch.setattr(
        retention_job.retention_core,
        "run",
        lambda **kwargs: {
            "deleted": 0,
            "freed_bytes": 0,
            "missing": 0,
            "errors": 1,
            "protected": {},
            "processed": 1,
            "total": 1,
            "stopped": False,
        },
    )
    nightly_job_hub.register_job(RetentionJob())

    nightly_job_hub.run_now("retention")
    nightly_job_hub._registry["retention"].thread.join(timeout=1)

    status = nightly_job_hub.status("retention")
    assert status["last_rc"] == 1
    assert status["progress"]["errors"] == 1


def test_last_result_is_restored_after_registry_restart(monkeypatch):
    result = {
        "deleted": 1,
        "freed_bytes": 2048,
        "missing": 0,
        "errors": 0,
        "protected": {"favorite": 2},
        "processed": 3,
        "total": 3,
        "stopped": False,
    }
    monkeypatch.setattr(retention_job.retention_core, "run", lambda **kwargs: result)
    nightly_job_hub.register_job(RetentionJob())
    nightly_job_hub.run_now("retention", reason="persistence test")
    nightly_job_hub._registry["retention"].thread.join(timeout=1)

    nightly_job_hub._registry.clear()
    nightly_job_hub.register_job(RetentionJob())

    restored = nightly_job_hub.status("retention")
    assert restored["last_reason"] == "persistence test"
    assert restored["last_rc"] == 0
    assert restored["progress"] == result


def test_restored_automatic_run_is_not_repeated_same_day(monkeypatch):
    retention_job.retention_core.save_last_run_status(
        {
            "last_started_at": datetime.now(tz=UTC).isoformat(),
            "last_finished_at": datetime.now(tz=UTC).isoformat(),
            "last_reason": "nightly auto",
            "last_rc": 0,
            "last_error": None,
            "last_daily_fire_date": datetime.now(tz=UTC).date().isoformat(),
            "progress": {"deleted": 0},
        }
    )
    monkeypatch.setattr(
        retention_job,
        "get_config",
        lambda: {"RETENTION_POSTURE": "conservative", "RETENTION_AUTO_ENABLED": True},
    )
    called = threading.Event()

    def unexpected_run(**kwargs):
        called.set()
        return {"errors": 0}

    monkeypatch.setattr(retention_job.retention_core, "run", unexpected_run)
    nightly_job_hub.register_job(RetentionJob())

    nightly_job_hub._maybe_fire_due_jobs(is_night=False)

    assert not called.is_set()
