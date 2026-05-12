"""
In-App scheduler for the nightly aesthetic auto-tagger.

Runs as a daemon thread inside the main application process and fires
``scripts.aesthetic_tag_nightly.main_with_args(...)`` once per day at
the configured time.

Why a thread instead of systemd:
- Same code on Pi and Docker. Docker has no systemd; both deployments
  now share one scheduling mechanism.
- Zero-touch deploy: pip install requirements + requirements-aesthetic
  is enough, no separate venv, no systemctl enable.

Duplicate-send protection is minute-grained (guards against restart
storms near the scheduled minute), mirroring report_scheduler.py.

If the optional ``open_clip_torch`` / ``torch`` packages are missing
(slim image variant), the scheduler logs a warning and stays idle
instead of crashing. This is intentional: a small Pi or a stripped
Docker image without the aesthetic stack should still boot the app.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import UTC, date, datetime

# Redirect HF cache to a writable location BEFORE huggingface_hub gets
# imported transitively (which happens inside _check_dependencies_available
# the first time `import open_clip` runs). huggingface_hub.constants reads
# HF_HOME / XDG_CACHE_HOME at module-import time and freezes the resolved
# cache path into module globals — setting HF_HOME afterwards is a no-op.
# Container deploys set XDG_CACHE_HOME=/tmp/fontconfig (fontconfig workaround)
# which the runtime user can't write to; HF would derive its cache there
# and fail with EACCES. Honour an explicit HF_HOME if the operator set
# one; otherwise root the cache inside OUTPUT_DIR so it sits on the
# mounted volume and survives container rebuilds (~600MB of weights).
if not os.environ.get("HF_HOME"):
    _output_dir = os.environ.get("OUTPUT_DIR", "/opt/app/data/output")
    os.environ["HF_HOME"] = os.path.join(_output_dir, "huggingface")

logger = logging.getLogger(__name__)

# Daily-mode guard: tracks the last date a tag-run finished so we don't
# re-fire after restart near the scheduled minute.
_last_run_date: date | None = None
_lock = threading.Lock()
# Run-mutex: prevents the daily loop and an external bridge call (e.g.
# the Telegram pre-run from report_scheduler) from triggering the
# tagger in parallel. CLIP loads ~700 MB and is single-threaded
# anyway; concurrent runs would just thrash.
_run_mutex = threading.Lock()


def _should_run(scheduled_hour: int, scheduled_minute: int) -> bool:
    """True when the current minute matches the configured time and no
    tagger run has finished today yet."""
    global _last_run_date
    now = datetime.now()
    if now.hour != scheduled_hour or now.minute != scheduled_minute:
        return False
    with _lock:
        if _last_run_date == now.date():
            return False
        return True


def _mark_run_today() -> None:
    """Mark today as 'tagger ran' for the duplicate guard.

    Uses ``datetime.now().date()`` rather than ``date.today()`` so the
    duplicate guard reads a single clock source — matching ``_should_run``
    above. The two are functionally identical in steady state, but the
    single-source form also closes a midnight-boundary race where
    ``_should_run`` and ``_mark_run_today`` could disagree on "today" if
    one call fell on either side of midnight.
    """
    global _last_run_date
    with _lock:
        _last_run_date = datetime.now().date()


def _parse_time(
    time_str: str, *, fallback: tuple[int, int] = (2, 10)
) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute). Falls back to fallback."""
    try:
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except (ValueError, IndexError):
        # Malformed HH:MM; fall through to the warning + fallback below.
        pass
    logger.warning(
        "Invalid AESTHETIC_TAG_TIME '%s', falling back to %02d:%02d",
        time_str,
        fallback[0],
        fallback[1],
    )
    return fallback


def _today_midnight_utc_iso() -> str:
    """ISO timestamp for today's local midnight, in UTC.

    The aesthetic tagger's ``--since`` is treated as a UTC SQL bound. To
    score "everything detected today (operator's wall clock)" we anchor
    on the operator's local midnight and convert to UTC. The host
    timezone is whatever ``datetime.now()`` reports — same convention
    the rest of the report stack uses.
    """
    local_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # naive local → aware UTC via timestamp() round-trip (handles DST).
    return datetime.fromtimestamp(local_midnight.timestamp(), tz=UTC).isoformat()


_LEASE_HOLDER = "aesthetic_tagger"


def _apply_cpu_friendliness_env() -> None:
    """Read the AESTHETIC_TAGGER_* config keys and surface them as env
    vars the worker honours.

    The worker is invoked in-process via ``main_with_args`` and reads
    ``os.environ`` for niceness, torch-thread-cap, and throttle. Setting
    them here (and only here) means config changes via Settings → Apply
    take effect on the next bridge / daily run with no extra plumbing.
    """
    try:
        from config import get_config

        cfg = get_config()
    except Exception:
        return

    nice = cfg.get("AESTHETIC_TAGGER_NICE")
    if nice is not None:
        os.environ["WMB_AESTHETIC_NICE"] = str(int(nice))

    threads = cfg.get("AESTHETIC_TAGGER_TORCH_THREADS")
    if threads is not None:
        os.environ["WMB_AESTHETIC_TORCH_THREADS"] = str(int(threads))


def _run_tagger(
    reason: str,
    *,
    since: str | None = None,
    throttle_ms: int | None = None,
    per_species_cap: int | None = None,
) -> int:
    """Fire scripts.aesthetic_tag_nightly.main_with_args() in this process.

    Returns the worker's exit code (0 == success). Wrapped so both the
    daily loop and the external ``run_now`` bridge share one code path.

    ``since`` overrides the worker's default --since (yesterday 00:00 UTC).
    For the Telegram-pre-run bridge we pass today's local midnight in UTC
    so only today's unscored detections get processed (faster than a full
    24h sweep).

    ``throttle_ms`` controls the per-inference sleep. The default for the
    nightly run is 0 (camera is idle at 02:10, no live detector to
    starve), but the bridge passes 100 by default so the daytime
    detector keeps its frame budget while CLIP runs on the same CPU.

    ``per_species_cap`` is the bridge-only knob that caps the run at N
    detections per CLS species, ranked by detector score / bbox quality
    / created_at. The nightly loop never sets this — it must still
    score every detection of the previous day for completeness.

    Acquires the shared compute lease with ``pause_detection=False`` so
    Companion inference and the tagger cannot run concurrently. The
    tagger does not pause OD on its own — the existing throttle
    behaviour stays in place.

    CPU-friendliness knobs (``AESTHETIC_TAGGER_NICE``,
    ``AESTHETIC_TAGGER_TORCH_THREADS``) are read live and exported into
    the worker's environment so the live OD pipeline gets priority
    while CLIP runs on the same Pi cores.
    """
    try:
        from scripts.aesthetic_tag_nightly import main_with_args
    except ImportError as exc:
        logger.error(
            "Aesthetic scheduler: cannot import worker (%s); skipping run.",
            exc,
        )
        return 1

    argv: list[str] = []
    if since is not None:
        argv = ["--since", since]
    if throttle_ms is not None and throttle_ms >= 0:
        argv += ["--throttle-ms", str(throttle_ms)]
    if per_species_cap is not None and per_species_cap > 0:
        argv += ["--per-species-cap", str(per_species_cap)]

    _apply_cpu_friendliness_env()

    logger.info("Aesthetic tagger firing (%s, argv=%s)...", reason, argv)
    from web.services.compute_lease_service import (
        LeaseBusy,
        get_compute_lease_service,
    )

    lease = get_compute_lease_service()
    if lease is None:
        # Lease not initialised (early boot, slim test harness). Fall
        # back to the old direct-call behaviour. This keeps the tagger
        # functional even when the WMB Flask app is not the host.
        return _invoke_tagger(reason, main_with_args, argv)

    try:
        with lease.acquire(
            _LEASE_HOLDER,
            pause_detection=False,
            reason=f"aesthetic tagger: {reason}",
        ):
            return _invoke_tagger(reason, main_with_args, argv)
    except LeaseBusy as exc:
        logger.info(
            "Aesthetic tagger skipped (%s): compute lease busy with %r.",
            reason,
            exc.current_holder,
        )
        return 1


def _invoke_tagger(reason, main_with_args, argv: list[str]) -> int:
    try:
        rc = main_with_args(argv)
        if rc == 0:
            logger.info("Aesthetic tagger finished successfully (%s).", reason)
        else:
            logger.warning(
                "Aesthetic tagger returned non-zero exit (%s, rc=%d).",
                reason,
                rc,
            )
        return rc
    except Exception as exc:
        logger.error(
            "Aesthetic tagger failed (%s): %s",
            reason,
            exc,
            exc_info=True,
        )
        return 1


def run_now(
    reason: str,
    *,
    since: str | None = None,
    today_only: bool = False,
    throttle_ms: int | None = 100,
    per_species_cap: int | None = None,
) -> bool:
    """Synchronously trigger the tagger from outside the daily loop.

    Used by the report scheduler to "bridge" the gap between the nightly
    tagger run (02:10) and the Telegram report send (e.g. 21:00) — by
    that time today's detections still have no aesthetic_score, so the
    report falls back to detector-confidence ranking. Calling this just
    before the report send ensures today's detections are scored too.

    Args:
        reason: Human-readable trigger description for the log
            ("pre-telegram bridge", "manual API trigger", etc.).
        since: Explicit ``--since`` ISO timestamp. Mutually exclusive
            with ``today_only``.
        today_only: When True, run with ``--since`` set to today's local
            midnight in UTC. Convenience for the pre-Telegram bridge
            so callers don't have to compute the timestamp.
        throttle_ms: Per-inference sleep in milliseconds. Default 100 ms
            for bridge runs because they fire while the live detector is
            running on the same CPU; nightly runs (which call
            ``_run_tagger`` directly) leave it at 0. Pass ``None`` here
            to defer to the worker's env / CLI default (also 0).
        per_species_cap: Cap the bridge at N detections per CLS species.
            ``None`` (default) falls back to config key
            ``AESTHETIC_BRIDGE_PER_SPECIES_CAP``; that key defaults to 8
            so the bridge stays bounded even on busy days. Pass ``0`` to
            disable the cap and score every unscored detection (the
            old pre-cap behaviour).

    Returns:
        True on success (rc == 0), False on failure or when the
        scheduler is disabled / dependencies missing / another run
        is already in progress.
    """
    try:
        from config import get_config

        config = get_config()
    except Exception as exc:
        logger.warning("Aesthetic run_now: cannot load config (%s); skipping.", exc)
        return False

    if not bool(config.get("AESTHETIC_TAG_ENABLED", True)):
        logger.info("Aesthetic run_now: scheduler disabled via config; skipping.")
        return False

    if not _check_dependencies_available():
        return False

    if since is not None and today_only:
        raise ValueError("run_now: pass since OR today_only, not both")
    if today_only:
        since = _today_midnight_utc_iso()

    if per_species_cap is None:
        try:
            per_species_cap = int(config.get("AESTHETIC_BRIDGE_PER_SPECIES_CAP", 8))
        except (TypeError, ValueError):
            per_species_cap = 8
    if per_species_cap <= 0:
        # Explicit opt-out from the cap (e.g. operator wants the full
        # backfill); pass None to _run_tagger so no --per-species-cap
        # flag reaches the worker.
        per_species_cap = None

    if not _run_mutex.acquire(blocking=False):
        logger.info(
            "Aesthetic run_now (%s): another run in progress; skipping.", reason
        )
        return False
    try:
        rc = _run_tagger(
            reason,
            since=since,
            throttle_ms=throttle_ms,
            per_species_cap=per_species_cap,
        )
        # Mark today as run too: the daily loop's duplicate guard should
        # not fire a second run a few hours later when we already
        # bridged this date.
        _mark_run_today()
        return rc == 0
    finally:
        _run_mutex.release()


def _check_dependencies_available() -> bool:
    """Verify that torch + open_clip_torch are importable.

    Slim image variants without the aesthetic stack should boot the
    app; they just don't run the tagger.
    """
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError as exc:
        logger.warning(
            "Aesthetic tagger dependencies not installed (%s); "
            "scheduler will stay idle. Install requirements-aesthetic.txt to enable.",
            exc,
        )
        return False


def start_aesthetic_tag_scheduler(check_interval: int = 30):
    """
    Start the background scheduler thread.

    Args:
        check_interval: Seconds between time checks. Default 30s keeps us
                        from missing the configured minute.

    Returns:
        The daemon Thread, or None if dependencies are missing or
        the scheduler is disabled by config.
    """
    try:
        from config import get_config

        config = get_config()
    except Exception as exc:
        logger.warning(
            "Aesthetic scheduler: cannot load config (%s); not starting.", exc
        )
        return None

    enabled = bool(config.get("AESTHETIC_TAG_ENABLED", True))
    if not enabled:
        logger.info("Aesthetic tag scheduler disabled via config; not starting.")
        return None

    if not _check_dependencies_available():
        return None

    time_str = str(config.get("AESTHETIC_TAG_TIME", "02:10")).strip()
    scheduled_hour, scheduled_minute = _parse_time(time_str)

    def _loop():
        logger.info(
            "Aesthetic tag scheduler started; daily run at %02d:%02d.",
            scheduled_hour,
            scheduled_minute,
        )
        while True:
            try:
                if _should_run(scheduled_hour, scheduled_minute):
                    # Acquire the run-mutex so this loop and an external
                    # ``run_now`` bridge call can never overlap.
                    if _run_mutex.acquire(blocking=False):
                        try:
                            _run_tagger(
                                f"daily @ {scheduled_hour:02d}:{scheduled_minute:02d}",
                            )
                        finally:
                            _run_mutex.release()
                    # Mark as run regardless of success: the duplicate
                    # guard prevents restart-storm resends; a real
                    # failure should not retry every 30s for the rest
                    # of the minute window.
                    _mark_run_today()
            except Exception as exc:
                logger.error(
                    "Aesthetic tag scheduler error: %s",
                    exc,
                    exc_info=True,
                )
            time.sleep(check_interval)

    t = threading.Thread(target=_loop, name="AestheticTagScheduler", daemon=True)
    t.start()
    return t
