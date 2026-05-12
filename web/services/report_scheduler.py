"""
In-App Scheduler for Telegram Reports.

Runs as a daemon thread inside the main application process and fires
``utils.daily_report.main()`` based on the configured Telegram mode:

    TELEGRAM_MODE == "daily"    -> once per day at TELEGRAM_REPORT_TIME.
    TELEGRAM_MODE == "interval" -> every TELEGRAM_REPORT_INTERVAL_HOURS hours.
    TELEGRAM_MODE == "off"/"live" -> scheduler stays idle.

Duplicate-send protection is minute-grained for the daily mode (guards
against restart storms near the scheduled minute) and interval-grained for
the hourly mode (last-sent timestamp must be at least ``interval_hours * 3600``
seconds old).
"""

import logging
import threading
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)

# Daily-mode guard: tracks the last date a report was sent so we don't
# double-fire after restart near the scheduled minute.
_last_report_date: date | None = None
# Interval-mode guard: wall-clock timestamp of the last successful send.
# Initialised lazily on first scheduler tick to the process start time so
# an app restart does NOT trigger an immediate bridge+report+tagger storm
# (the previous 0.0 sentinel always satisfied "now - 0 >= interval_hours
# * 3600"). The first interval-tick after restart now fires interval_hours
# after boot, matching wall-clock cadence regardless of restarts.
_last_interval_send_ts: float = 0.0
_interval_guard_initialised: bool = False
_lock = threading.Lock()


def _should_send_daily(report_hour: int, report_minute: int) -> bool:
    """True when the current minute matches the configured time and no
    report has been sent today yet."""
    global _last_report_date
    now = datetime.now()

    if now.hour != report_hour or now.minute != report_minute:
        return False

    with _lock:
        if _last_report_date == now.date():
            return False
        return True


def _should_send_interval(interval_hours: int) -> bool:
    """True when enough wall-clock time has passed since the last send.

    On first call after process start, anchors the guard to ``now()`` so
    a restart does not retrigger an immediate send. From there on the
    guard works on real elapsed wall-clock between successful sends.
    """
    global _last_interval_send_ts, _interval_guard_initialised
    now_ts = time.time()
    with _lock:
        if not _interval_guard_initialised:
            _last_interval_send_ts = now_ts
            _interval_guard_initialised = True
            return False
        if (now_ts - _last_interval_send_ts) >= max(1, interval_hours) * 3600:
            return True
        return False


def _mark_sent_daily():
    """Mark today as 'report sent' for the daily-mode guard."""
    global _last_report_date
    with _lock:
        _last_report_date = date.today()


def _mark_sent_interval():
    """Record the timestamp for the interval-mode guard."""
    global _last_interval_send_ts
    with _lock:
        _last_interval_send_ts = time.time()


def _parse_report_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute). Falls back to (21, 0)."""
    try:
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except (ValueError, IndexError):
        # Malformed HH:MM; fall through to the warning + fallback below.
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

    def _run_report(reason: str) -> None:
        """Fire utils.daily_report.main() with the optional health provider.

        Pre-step (the "bridge"): synchronously trigger the aesthetic
        tagger for today's local-day window first, so the Telegram
        report has aesthetic_score available for detections that
        happened since the nightly 02:10 run. Without this bridge,
        anything detected between 02:10 and the report-send time falls
        back to detector-confidence ranking, which picks "biggest /
        most-confident detection" instead of "prettiest photo".
        """
        # Bridge: tag today's unscored detections before composing the
        # report. Best-effort — if the tagger is disabled, deps are
        # missing, or another run is already in progress, run_now()
        # returns False and we just send the report unchanged.
        try:
            from web.services.aesthetic_tag_scheduler import run_now as tag_now

            tag_now(f"pre-telegram bridge ({reason})", today_only=True)
        except Exception as exc:
            # Never let the bridge block the report — log and proceed.
            logger.warning("Aesthetic pre-run bridge failed (%s): %s", reason, exc)

        try:
            from utils.daily_report import main as run_report

            health_provider = None
            if detection_manager is not None:
                health_provider = getattr(
                    detection_manager,
                    "get_ingest_health_snapshot",
                    None,
                )

            logger.info("Report scheduler firing (%s)...", reason)
            run_report(ingest_health_provider=health_provider)
            logger.info("Report sent successfully (%s).", reason)
        except Exception as e:
            logger.error("Report failed (%s): %s", reason, e, exc_info=True)

    def _loop():
        logger.info("Telegram report scheduler started.")
        while True:
            try:
                from config import get_config

                config = get_config()
                mode = str(config.get("TELEGRAM_MODE", "off") or "off").strip().lower()

                if mode == "daily":
                    time_str = str(config.get("TELEGRAM_REPORT_TIME", "") or "").strip()
                    if time_str:
                        report_hour, report_minute = _parse_report_time(time_str)
                        if _should_send_daily(report_hour, report_minute):
                            _run_report(
                                f"daily @ {report_hour:02d}:{report_minute:02d}"
                            )
                            # Mark as sent regardless of success: the duplicate
                            # guard prevents restart-storm resends; a real
                            # failure should not retry every 30s for the rest
                            # of the minute window.
                            _mark_sent_daily()

                elif mode == "interval":
                    try:
                        interval_hours = int(
                            float(config.get("TELEGRAM_REPORT_INTERVAL_HOURS", 1))
                        )
                    except Exception:
                        interval_hours = 1
                    interval_hours = max(1, min(24, interval_hours))

                    if _should_send_interval(interval_hours):
                        _run_report(f"interval every {interval_hours}h")
                        _mark_sent_interval()

                # mode == "off" or "live" -> scheduler idle.

            except Exception as e:
                logger.error("Report scheduler error: %s", e, exc_info=True)

            time.sleep(check_interval)

    t = threading.Thread(target=_loop, name="DailyReportScheduler", daemon=True)
    t.start()
    return t
