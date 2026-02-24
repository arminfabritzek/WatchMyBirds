"""
Health Core - System Health Business Logic.

Aggregates system vitals, database status, and storage metrics.
"""

import datetime
import logging
import shutil
from typing import Any

import psutil

from config import get_config
from utils import system_monitor
from utils.db import closing_connection

logger = logging.getLogger(__name__)


def get_system_health() -> dict[str, Any]:
    """
    Collects a comprehensive system health snapshot.

    Returns:
        Dictionary with status of: app, database, disk, and system vitals.
    """
    monitor_vitals = _collect_monitor_vitals()
    db_status = _check_database()
    disk_status = _check_disk_space()

    # Determine overall status
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
    """Checks database connectivity and stats."""
    status = {"connected": False, "latency_ms": None, "last_detection": None}
    try:
        start = datetime.datetime.now()
        with closing_connection() as conn:
            # 1. Connectivity Check
            conn.execute("SELECT 1")

            # 2. Last Detection Check
            cursor = conn.execute(
                "SELECT created_at FROM detections ORDER BY detection_id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                status["last_detection"] = row[0]

        latency = (datetime.datetime.now() - start).total_seconds() * 1000
        status["connected"] = True
        status["latency_ms"] = round(latency, 2)

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        status["error"] = str(e)

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
