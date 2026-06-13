import os
import platform
import subprocess
from pathlib import Path

from flask import Blueprint, jsonify, render_template

from config import get_config
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.power_actions import (
    POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
    get_power_action_success_message,
    is_power_management_available,
    schedule_power_action,
)
from web.security import error_response as _error_response
from web.services import backup_restore_service

logger = get_logger(__name__)
config = get_config()

system_bp = Blueprint("system", __name__)

_detection_manager = None


def init_system_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


@system_bp.route("/logs")
@login_required
def logs_route():
    from collections import deque

    def _tail_text(path: Path, max_lines: int) -> str:
        if not path.exists():
            return f"File not found: {path}"
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return "".join(reversed(deque(f, maxlen=max_lines)))
        except Exception as e:
            return f"Error reading {path.name}: {e}"

    logs_dir = Path(config["OUTPUT_DIR"]) / "logs"
    app_log_path = logs_dir / "app.log"
    vitals_path = logs_dir / "vital_signs.csv"

    return render_template(
        "logs.html",
        current_path="/logs",
        app_logs=_tail_text(app_log_path, 300),
        vitals_logs=_tail_text(vitals_path, 240),
        app_log_path=str(app_log_path),
        vitals_log_path=str(vitals_path),
    )


@system_bp.route("/api/system/versions", methods=["GET"])
@login_required
def system_versions_route():
    from utils.deploy_info import read_build_metadata

    meta = read_build_metadata()

    data = {
        "app_version": meta["app_version"],
        "git_commit": meta["git_commit"],
        "build_date": meta["build_date"],
        "deploy_type": meta["deploy_type"],
        "kernel": "Unknown",
        "os": "Unknown",
        "bootloader": "Unknown",
    }

    try:
        data["kernel"] = platform.release()
    except OSError:
        pass

    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        data["os"] = line.split("=")[1].strip().strip('"')
                        break
    except OSError:
        pass

    try:
        import shutil

        if shutil.which("rpi-eeprom-update"):
            res = subprocess.run(
                ["rpi-eeprom-update"], capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "CURRENT:" in line:
                        parts = line.split("CURRENT:", 1)
                        if len(parts) > 1:
                            data["bootloader"] = parts[1].strip()
                            break
    except (OSError, subprocess.SubprocessError):
        pass

    return jsonify(data)


@system_bp.route("/api/system/shutdown", methods=["POST"])
@login_required
def shutdown_route():
    try:
        logger.warning("System shutdown initiated via Web UI.")

        if not is_power_management_available():
            logger.warning(
                "Shutdown ignored: systemd not available (likely container)."
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                    }
                ),
                400,
            )
        schedule_power_action("shutdown", logger)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": get_power_action_success_message("shutdown"),
                }
            ),
            200,
        )
    except Exception as exc:
        return _error_response("Error initiating shutdown", exc)


@system_bp.route("/api/system/restart", methods=["POST"])
@login_required
def restart_route():
    try:
        logger.warning("System restart initiated via Web UI.")

        if not is_power_management_available():
            logger.warning(
                "Restart ignored: systemd not available (likely container)."
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                    }
                ),
                400,
            )
        schedule_power_action("restart", logger)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": get_power_action_success_message("restart"),
                }
            ),
            200,
        )
    except Exception as exc:
        return _error_response("Error initiating restart", exc)


@system_bp.route("/api/system/stats", methods=["GET"])
@login_required
def system_stats_route():
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        disk = None
        try:
            output_dir = config.get("OUTPUT_DIR", "./data/output")
            disk_usage = psutil.disk_usage(output_dir)
            disk = {
                "total_gb": round(disk_usage.total / (1024**3), 1),
                "used_gb": round(disk_usage.used / (1024**3), 1),
                "free_gb": round(disk_usage.free / (1024**3), 1),
                "percent": disk_usage.percent,
            }
        except (OSError, ValueError):
            pass

        temp = None
        try:
            import subprocess as _sp

            result = _sp.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace("temp=", "").replace("'C", ""))
        except (OSError, ValueError, ImportError):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for _name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            break
            except (AttributeError, OSError):
                pass

        response = {"status": "success", "cpu": cpu_percent, "ram": mem.percent}
        if temp is not None:
            response["temp"] = temp
        if disk is not None:
            response["disk"] = disk
        return jsonify(response)
    except Exception as exc:
        return _error_response("Error fetching system stats", exc)


@system_bp.route("/api/status", methods=["GET"])
@login_required
def api_status():
    try:
        output_dir = config.get("OUTPUT_DIR", "./data/output")

        response = {
            "detection_paused": _detection_manager.paused,
            "detection_running": not _detection_manager.paused,
            "restart_required": backup_restore_service.is_restart_required(output_dir),
        }

        try:
            from core.analysis_queue import analysis_queue
            from web.services.analysis_service import count_deep_scan_candidates

            response["deep_scan_active"] = _detection_manager.is_deep_scan_active()
            response["deep_scan_queue_pending"] = analysis_queue.pending_count()
            response["deep_scan_candidates_remaining"] = count_deep_scan_candidates()
        except Exception as ds_err:
            logger.debug(f"Could not compute deep scan status: {ds_err}")

        try:
            response["decision_state_counts"] = dict(
                _detection_manager.decision_state_counts
            )
        except (AttributeError, TypeError):
            pass

        return jsonify(response)
    except Exception as exc:
        logger.error("Status API error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "Status read failed"}), 500


@system_bp.route("/api/detection/pause", methods=["POST"])
@login_required
def detection_pause():
    try:
        if _detection_manager.paused:
            return jsonify(
                {
                    "status": "paused",
                    "message": "Detection was already paused",
                }
            )

        _detection_manager.paused = True
        logger.info("Detection paused via API")

        return jsonify(
            {
                "status": "success",
                "message": "Detection paused",
            }
        )

    except Exception as exc:
        logger.error("Detection pause error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "Detection pause failed"}), 500


@system_bp.route("/api/detection/resume", methods=["POST"])
@login_required
def detection_resume():
    try:
        if backup_restore_service.is_restore_active():
            return (
                jsonify(
                    {"error": "Cannot resume detection during restore operation"}
                ),
                409,
            )

        if not _detection_manager.paused:
            return jsonify(
                {
                    "status": "running",
                    "message": "Detection was already running",
                }
            )

        _detection_manager.paused = False
        logger.info("Detection resumed via API")

        return jsonify(
            {
                "status": "success",
                "message": "Detection resumed",
            }
        )

    except Exception as exc:
        logger.error("Detection resume error [%s]", type(exc).__name__, exc_info=True)
        return jsonify({"error": "Detection resume failed"}), 500


@system_bp.route("/healthz", methods=["GET"])
def healthz():
    return "ok\n", 200, {"Content-Type": "text/plain; charset=utf-8"}
