"""
Health Core - System Health Business Logic.

Aggregates system vitals, database status, and storage metrics.
"""

import datetime
import logging
import shutil
import sqlite3
import time
from typing import Any

import psutil

from config import get_config
from utils import system_monitor
from utils.db import closing_connection

logger = logging.getLogger(__name__)

# How many times to re-attempt the health probe when SQLite reports
# `database is locked`. The 15 s busy_timeout in get_connection()
# already waits in the kernel for the writer to release the lock; this
# retry is the secondary safety net for the rare case the timeout is
# itself exhausted (e.g. when the aesthetic-tagger bridge holds a long
# write transaction under peak detector load).
_HEALTH_CHECK_LOCK_RETRIES = 1
# Sleep before retrying — short enough that the health endpoint stays
# responsive, long enough that whatever was holding the lock has had a
# chance to commit.
_HEALTH_CHECK_LOCK_RETRY_SLEEP_SEC = 0.25


def get_system_health() -> dict[str, Any]:
    """
    Collects a comprehensive system health snapshot.

    Returns:
        Dictionary with status of: app, database, disk, and system vitals.
    """
    monitor_vitals = _collect_monitor_vitals()
    db_status = _check_database()
    disk_status = _check_disk_space()

    overall = "ok"
    if not db_status["connected"]:
        overall = "error"
    elif disk_status["percent"] > 90:
        overall = "warning"

    return {
        "status": overall,
        "timestamp": datetime.datetime.now().isoformat(),
        "database": db_status,
        "disk": disk_status,
        "system": monitor_vitals,
    }


def _collect_monitor_vitals() -> dict[str, Any]:
    """Collects OS-level vitals using system_monitor helpers."""
    try:
        # Re-use the lightweight logic from SystemMonitor
        is_rpi = system_monitor.is_raspberry_pi()
        throttled = system_monitor.parse_throttled(
            system_monitor.run_vcgencmd("get_throttled")
        )

        vitals = {
            "cpu_temp_c": system_monitor.get_cpu_temp(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
            "throttled": throttled,
            "is_rpi": is_rpi,
        }

        if is_rpi:
            vitals["core_voltage"] = system_monitor.get_core_voltage()

        return vitals
    except Exception as e:
        logger.error(f"Failed to collect system vitals: {e}")
        return {"error": str(e)}


def _check_database() -> dict[str, Any]:
    """Checks database connectivity and stats.

    Retries once on transient `database is locked` errors before
    declaring the DB unhealthy. Lock-errors during the retry are
    logged at WARNING level (not ERROR) because they're a known race
    when the detector, aesthetic-tagger bridge, and health-check all
    touch the DB at once. Every other sqlite/Python failure stays at
    ERROR — those are real problems.
    """
    status = {"connected": False, "latency_ms": None, "last_detection": None}
    attempts = 1 + _HEALTH_CHECK_LOCK_RETRIES
    last_lock_error: str | None = None

    for attempt in range(1, attempts + 1):
        try:
            start = datetime.datetime.now()
            with closing_connection() as conn:
                conn.execute("SELECT 1")

                cursor = conn.execute(
                    "SELECT created_at FROM detections ORDER BY detection_id DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row:
                    status["last_detection"] = row[0]

            latency = (datetime.datetime.now() - start).total_seconds() * 1000
            status["connected"] = True
            status["latency_ms"] = round(latency, 2)
            return status

        except sqlite3.OperationalError as e:
            msg = str(e)
            if "database is locked" in msg.lower():
                last_lock_error = msg
                if attempt < attempts:
                    # Will retry — silent, let it through.
                    time.sleep(_HEALTH_CHECK_LOCK_RETRY_SLEEP_SEC)
                    continue
                # Final attempt also locked — log at WARNING (not ERROR)
                # because the detector + bridge race is a known transient.
                # The reading layer can degrade to "stale snapshot"
                # without the operator panicking.
                logger.warning(
                    "Database health check still locked after %d attempts: %s",
                    attempts, msg,
                )
                status["error"] = msg
                status["transient_lock"] = True
                return status
            # Other OperationalError — real problem.
            logger.error("Database health check failed: %s", e)
            status["error"] = msg
            return status

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            status["error"] = str(e)
            return status

    # Loop exited without return (shouldn't happen given the structure
    # above, but keeps the type checker happy).
    if last_lock_error:
        status["error"] = last_lock_error
        status["transient_lock"] = True
    return status


def _check_disk_space() -> dict[str, Any]:
    """Checks disk space usage for the output directory."""
    config = get_config()
    output_dir = config.get("OUTPUT_DIR", ".")

    try:
        total, used, free = shutil.disk_usage(output_dir)
        percent = (used / total) * 100 if total > 0 else 0

        return {
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent": round(percent, 1),
            "path": output_dir,
        }
    except Exception as e:
        logger.error(f"Disk check failed: {e}")
        return {"error": str(e), "percent": 0}
