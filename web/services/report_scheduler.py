"""
In-App Scheduler for the Daily Evening Report.

Runs as a daemon thread inside the main application process.
Triggers utils.daily_report.main() once per day at the configured time.

Configuration keys (via settings.yaml / env / defaults):
    TELEGRAM_REPORT_TIME   – HH:MM string, default "21:00"
    TELEGRAM_ENABLED       – must be True for the report to send
"""

import logging
import threading
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)

# In-memory guard: tracks the last date a report was sent
# to prevent duplicate reports on restart or config reload.
_last_report_date: date | None = None
_lock = threading.Lock()


def _should_send_now(report_hour: int, report_minute: int) -> bool:
    """
    Returns True if the current time matches the configured report time
    AND no report has been sent today yet.
    """
    global _last_report_date
    now = datetime.now()

    if now.hour != report_hour or now.minute != report_minute:
        return False

    with _lock:
        if _last_report_date == now.date():
            return False
        return True


def _mark_sent():
    """Mark today as 'report sent'."""
    global _last_report_date
    with _lock:
        _last_report_date = date.today()


def _parse_report_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute). Falls back to (21, 0)."""
    try:
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except (ValueError, IndexError):
        pass
    logger.warning("Invalid TELEGRAM_REPORT_TIME '%s', falling back to 21:00", time_str)
    return 21, 0


def start_report_scheduler(check_interval: int = 30, detection_manager=None):
    """
    Start the background scheduler thread.

    Args:
        check_interval: Seconds between time checks (default 30s).
                        Kept short so we don't miss the minute window.
        detection_manager: Optional DetectionManager instance.  When set,
                           its ``get_ingest_health_snapshot`` method is
                           forwarded to the report so the status section
                           reflects real ingest state instead of a
                           hard-coded "running normally".
    """

    def _loop():
        logger.info("Daily report scheduler started.")
        while True:
            try:
                from config import get_config

                config = get_config()

                # Skip entirely if Telegram is disabled
                if not config.get("TELEGRAM_ENABLED", False):
                    time.sleep(check_interval)
                    continue

                time_str = config.get("TELEGRAM_REPORT_TIME", "21:00")
                report_hour, report_minute = _parse_report_time(time_str)

                if _should_send_now(report_hour, report_minute):
                    logger.info(
                        "Report time reached (%02d:%02d). Generating evening report...",
                        report_hour,
                        report_minute,
                    )
                    try:
                        from utils.daily_report import main as run_report

                        # Build ingest-health provider (fail-safe handled
                        # inside run_report itself).
                        health_provider = None
                        if detection_manager is not None:
                            health_provider = getattr(
                                detection_manager,
                                "get_ingest_health_snapshot",
                                None,
                            )

                        run_report(ingest_health_provider=health_provider)
                        _mark_sent()
                        logger.info("Evening report sent successfully.")
                    except Exception as e:
                        logger.error("Evening report failed: %s", e, exc_info=True)
                        # Still mark as sent to avoid spamming on persistent errors
                        _mark_sent()

            except Exception as e:
                logger.error("Report scheduler error: %s", e, exc_info=True)

            time.sleep(check_interval)

    t = threading.Thread(target=_loop, name="DailyReportScheduler", daemon=True)
    t.start()
    return t
