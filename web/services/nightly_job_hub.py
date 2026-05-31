"""Nightly Job Hub — generic registry for night-only batch work.

This is the second iteration of the nightly-scheduler pattern. The
first was ``aesthetic_tag_scheduler.py``, which baked the CLIP-tagger
lifecycle into the scheduler. That design works for one job; for two
or more it duplicates: lock, run_now, stop_event, status snapshot,
"is it night?" gate.

The hub generalises that pattern. Each job is a small subclass of
``JobBase`` that knows how to:
  * report its ``name`` (unique slug) and ``display_name`` (UI label),
  * answer ``should_run_in_daily_loop()`` (most return True; aesthetic
    declines if its dependencies aren't installed),
  * execute ``run(stop_event, reason)`` to a non-negative integer
    exit code (0 = success).

The hub owns the rest:
  * per-job mutex (two runs of the same job cannot overlap),
  * stop_event wiring (UI Stop button → event set → job exits at next
    inner-loop checkpoint),
  * status snapshot for the polling UI endpoint,
  * the daily wake-up loop that fires registered jobs once per night.

Crucially: **different jobs can run in parallel**. The hub does NOT
hold a global lock. Each job has its own lock; the operator can
trigger sharpness manually while the aesthetic-tagger is mid-run.
This is by design — the two are I/O and CPU bound on different
resources, and the operator wants to be able to "click and see
results" without waiting for an unrelated job to finish.

The aesthetic-tagger keeps its own ``run_now`` (used by the Telegram
report bridge with very specific arguments — since, throttle,
per-species-cap). The hub registers a thin adapter that calls
``aesthetic_tag_scheduler.run_now`` with defaults; advanced callers
still go through the scheduler module directly.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job interface
# ---------------------------------------------------------------------------


class JobBase(ABC):
    """Abstract base for a hub-registered nightly job.

    Subclasses implement ``name``, ``display_name``, and ``run()``.
    The hub provides lifecycle (lock, stop_event, status), so jobs
    must not allocate threading primitives themselves.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique slug, e.g. ``aesthetic_tagger`` or ``sharpness``.

        Used in API routes (``/api/v1/nightly/<name>/run_now``) and
        in log lines. Must be URL-safe.
        """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label for the Settings UI."""

    def should_run_in_daily_loop(self) -> bool:
        """Whether the daily loop should fire this job at all.

        Default True. Override to short-circuit when optional
        dependencies are missing (e.g. aesthetic-tagger declines if
        ``open_clip`` isn't installed).
        """
        return True

    @abstractmethod
    def run(self, stop_event: threading.Event, reason: str) -> int:
        """Execute the job. Return 0 on success, non-zero on failure.

        The job must poll ``stop_event.is_set()`` periodically and
        exit promptly when set. Partial progress is allowed (it's
        the job's responsibility to leave the DB in a consistent
        state on abort).
        """


# ---------------------------------------------------------------------------
# Hub registry + lifecycle
# ---------------------------------------------------------------------------


class _JobRuntime:
    """Per-job runtime state: lock, current thread, stop_event, last-run info."""

    def __init__(self, job: JobBase) -> None:
        self.job = job
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_started_at: datetime | None = None
        self.last_finished_at: datetime | None = None
        self.last_reason: str | None = None
        self.last_rc: int | None = None
        # Optional progress hook the job can write to; the UI polls
        # this to render "142/8910 crops".
        self.progress: dict[str, Any] = {}


_registry: dict[str, _JobRuntime] = {}
_registry_lock = threading.Lock()


def register_job(job: JobBase) -> None:
    """Register a job with the hub. Safe to call multiple times."""
    with _registry_lock:
        if job.name in _registry:
            logger.debug("nightly_job_hub: job %r already registered", job.name)
            return
        _registry[job.name] = _JobRuntime(job)
        logger.info("nightly_job_hub: registered job %r", job.name)


def unregister_job(name: str) -> None:
    """Remove a job from the registry (tests use this for isolation)."""
    with _registry_lock:
        _registry.pop(name, None)


def list_jobs() -> list[dict[str, Any]]:
    """Snapshot of every registered job's lifecycle state. For the UI poller."""
    with _registry_lock:
        items = list(_registry.values())

    out = []
    for rt in items:
        running = rt.thread is not None and rt.thread.is_alive()
        out.append(
            {
                "name": rt.job.name,
                "display_name": rt.job.display_name,
                "running": running,
                "stop_requested": rt.stop_event.is_set() and running,
                "last_started_at": rt.last_started_at.isoformat()
                if rt.last_started_at
                else None,
                "last_finished_at": rt.last_finished_at.isoformat()
                if rt.last_finished_at
                else None,
                "last_reason": rt.last_reason,
                "last_rc": rt.last_rc,
                "progress": dict(rt.progress) if running else {},
            }
        )
    return out


def run_now(name: str, reason: str = "manual trigger") -> dict[str, Any]:
    """Start the named job synchronously in a background thread.

    Returns a status dict:
      * ``{"status": "started", ...}`` — job thread started.
      * ``{"status": "already_running", ...}`` — lock held, no-op.
      * ``{"status": "unknown_job", ...}`` — name not in registry.

    The call returns immediately; the job runs in the background.
    Use ``list_jobs()`` or ``status(name)`` to poll progress.
    """
    with _registry_lock:
        rt = _registry.get(name)
    if rt is None:
        return {"status": "unknown_job", "name": name}

    if not rt.lock.acquire(blocking=False):
        return {"status": "already_running", "name": name}

    # Lock acquired. Spin up the worker thread; the thread releases
    # the lock on exit.
    rt.stop_event.clear()
    rt.last_started_at = datetime.now(tz=UTC)
    rt.last_finished_at = None
    rt.last_reason = reason
    rt.last_rc = None
    rt.progress = {}

    def _worker():
        try:
            logger.info(
                "nightly_job_hub: starting %r (reason=%s)", rt.job.name, reason
            )
            rc = rt.job.run(rt.stop_event, reason)
            rt.last_rc = int(rc)
            logger.info(
                "nightly_job_hub: %r finished rc=%s", rt.job.name, rt.last_rc
            )
        except Exception:
            rt.last_rc = 1
            logger.exception("nightly_job_hub: %r crashed", rt.job.name)
        finally:
            rt.last_finished_at = datetime.now(tz=UTC)
            rt.lock.release()

    rt.thread = threading.Thread(
        target=_worker,
        name=f"NightlyJob-{rt.job.name}",
        daemon=True,
    )
    rt.thread.start()
    return {"status": "started", "name": name, "reason": reason}


def stop(name: str) -> dict[str, Any]:
    """Request the named job to stop cleanly.

    Sets the stop_event. The job's inner loop must check
    ``stop_event.is_set()`` between work units and exit when set.
    Returns one of:
      * ``{"status": "stop_requested", ...}`` — event set, job exits soon.
      * ``{"status": "not_running", ...}`` — no active job to stop.
      * ``{"status": "unknown_job", ...}`` — name not in registry.
    """
    with _registry_lock:
        rt = _registry.get(name)
    if rt is None:
        return {"status": "unknown_job", "name": name}
    if rt.thread is None or not rt.thread.is_alive():
        return {"status": "not_running", "name": name}
    rt.stop_event.set()
    return {"status": "stop_requested", "name": name}


def status(name: str) -> dict[str, Any] | None:
    """Snapshot of one job's state. None if unknown."""
    for entry in list_jobs():
        if entry["name"] == name:
            return entry
    return None


def update_progress(name: str, progress: dict[str, Any]) -> None:
    """Job-side helper to write into its progress dict. UI polls it."""
    with _registry_lock:
        rt = _registry.get(name)
    if rt is None:
        return
    rt.progress = dict(progress)


# ---------------------------------------------------------------------------
# Daily wake-up loop
# ---------------------------------------------------------------------------
#
# The hub does NOT decide what "night" means — it just polls a
# callback. The detection_manager's is_daytime() is the source of
# truth. When that flips to night, every registered job that
# returns ``should_run_in_daily_loop() == True`` fires once per
# night.
#
# Duplicate-protection is date-based: each job tracks its last fire
# date (UTC). A second night-trigger on the same date skips.


_daily_loop_thread: threading.Thread | None = None
_daily_loop_stop = threading.Event()
_last_fire_date: dict[str, str] = {}  # job_name → ISO date string


def start_daily_loop(
    is_night_callback: Callable[[], bool],
    check_interval_s: int = 60,
) -> threading.Thread:
    """Start the hub's background daily-fire thread.

    Args:
        is_night_callback: Called every ``check_interval_s`` seconds.
            Returns True if it's currently night (i.e. OD is paused).
            When True, every registered job is fired exactly once per
            UTC date. Typically wired to the detection_manager's
            ``_should_run_od_now()`` (inverted).
        check_interval_s: Poll interval. Default 60 s — at 1-minute
            granularity, jobs fire within a minute of dusk-plus-offset.
    """
    global _daily_loop_thread

    if _daily_loop_thread is not None and _daily_loop_thread.is_alive():
        logger.debug("nightly_job_hub: daily loop already running")
        return _daily_loop_thread

    _daily_loop_stop.clear()

    def _loop():
        logger.info(
            "nightly_job_hub: daily loop started (check every %ds)",
            check_interval_s,
        )
        while not _daily_loop_stop.is_set():
            try:
                if is_night_callback():
                    _maybe_fire_due_jobs()
            except Exception:
                logger.exception("nightly_job_hub: daily loop error")
            _daily_loop_stop.wait(check_interval_s)

    _daily_loop_thread = threading.Thread(
        target=_loop,
        name="NightlyJobHubDailyLoop",
        daemon=True,
    )
    _daily_loop_thread.start()
    return _daily_loop_thread


def stop_daily_loop() -> None:
    """Stop the daily loop (for tests + clean shutdown)."""
    _daily_loop_stop.set()


def _maybe_fire_due_jobs() -> None:
    """For each registered job, fire if not yet fired today."""
    today_iso = datetime.now(tz=UTC).date().isoformat()
    with _registry_lock:
        items = list(_registry.values())

    for rt in items:
        if _last_fire_date.get(rt.job.name) == today_iso:
            continue
        if not rt.job.should_run_in_daily_loop():
            _last_fire_date[rt.job.name] = today_iso
            continue
        # Mark first; even if run_now skips due to a manual run
        # already in progress, we don't want to retry every minute.
        _last_fire_date[rt.job.name] = today_iso
        result = run_now(rt.job.name, reason="nightly auto")
        logger.info(
            "nightly_job_hub: daily fire of %r → %s",
            rt.job.name,
            result.get("status"),
        )
