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
- See agent_handoff/workflow/plans/2026-05-02_HANDOFF_aesthetic-tagger-in-app-scheduler.md

Duplicate-send protection is minute-grained (guards against restart
storms near the scheduled minute), mirroring report_scheduler.py.

If the optional ``open_clip_torch`` / ``torch`` packages are missing
(slim image variant), the scheduler logs a warning and stays idle
instead of crashing. This is intentional: a small Pi or a stripped
Docker image without the aesthetic stack should still boot the app.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)

# Daily-mode guard: tracks the last date a tag-run finished so we don't
# re-fire after restart near the scheduled minute.
_last_run_date: date | None = None
_lock = threading.Lock()


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
    """Mark today as 'tagger ran' for the duplicate guard."""
    global _last_run_date
    with _lock:
        _last_run_date = date.today()


def _parse_time(time_str: str, *, fallback: tuple[int, int] = (2, 10)) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute). Falls back to fallback."""
    try:
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except (ValueError, IndexError):
        pass
    logger.warning(
        "Invalid AESTHETIC_TAG_TIME '%s', falling back to %02d:%02d",
        time_str, fallback[0], fallback[1],
    )
    return fallback


def _check_dependencies_available() -> bool:
    """Verify that torch + open_clip_torch are importable.

    Slim image variants without the aesthetic stack should boot the
    app; they just don't run the tagger.
    """
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401
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
        logger.warning("Aesthetic scheduler: cannot load config (%s); not starting.", exc)
        return None

    enabled = bool(config.get("AESTHETIC_TAG_ENABLED", True))
    if not enabled:
        logger.info("Aesthetic tag scheduler disabled via config; not starting.")
        return None

    if not _check_dependencies_available():
        return None

    time_str = str(config.get("AESTHETIC_TAG_TIME", "02:10")).strip()
    scheduled_hour, scheduled_minute = _parse_time(time_str)

    def _run_tagger(reason: str) -> None:
        """Fire scripts.aesthetic_tag_nightly.main_with_args() in this process."""
        try:
            from scripts.aesthetic_tag_nightly import main_with_args
        except ImportError as exc:
            logger.error(
                "Aesthetic scheduler: cannot import worker (%s); skipping run.",
                exc,
            )
            return

        logger.info("Aesthetic tagger firing (%s)...", reason)
        try:
            # main_with_args returns an int exit code (0 == success). We let
            # the worker decide its own --since default (last-day window).
            rc = main_with_args([])
            if rc == 0:
                logger.info("Aesthetic tagger finished successfully (%s).", reason)
            else:
                logger.warning(
                    "Aesthetic tagger returned non-zero exit (%s, rc=%d).",
                    reason, rc,
                )
        except Exception as exc:
            logger.error(
                "Aesthetic tagger failed (%s): %s", reason, exc, exc_info=True,
            )

    def _loop():
        logger.info(
            "Aesthetic tag scheduler started; daily run at %02d:%02d.",
            scheduled_hour, scheduled_minute,
        )
        while True:
            try:
                if _should_run(scheduled_hour, scheduled_minute):
                    _run_tagger(
                        f"daily @ {scheduled_hour:02d}:{scheduled_minute:02d}",
                    )
                    # Mark as run regardless of success: the duplicate guard
                    # prevents restart-storm resends; a real failure should
                    # not retry every 30s for the rest of the minute window.
                    _mark_run_today()
            except Exception as exc:
                logger.error(
                    "Aesthetic tag scheduler error: %s", exc, exc_info=True,
                )
            time.sleep(check_interval)

    t = threading.Thread(target=_loop, name="AestheticTagScheduler", daemon=True)
    t.start()
    return t
