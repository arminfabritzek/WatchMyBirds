"""
API v1 Blueprint.

This blueprint provides versioned API endpoints under /api/v1/*.
It is a 1:1 mirror of the existing /api/* routes - no changes to behavior or response format.

Purpose: Enable API versioning without breaking existing clients.
"""

from flask import Blueprint, jsonify, request

from config import get_config
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import (
    backup_restore_service,
    db_service,
    onvif_service,
)

logger = get_logger(__name__)
config = get_config()

# Create Blueprint
api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


# =============================================================================
# Status & Control
# =============================================================================


@api_v1.route("/status", methods=["GET"])
@login_required
def status():
    """
    Returns system status including detection state.
    Mirror of: GET /api/status
    """
    # Note: detection_manager is injected via init_api_v1()
    try:
        output_dir = config.get("OUTPUT_DIR", "./data/output")
        dm = api_v1.detection_manager

        return jsonify(
            {
                "detection_paused": dm.paused,
                "detection_running": not dm.paused,
                "restart_required": backup_restore_service.is_restart_required(
                    output_dir
                ),
            }
        )
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({"error": str(e)}), 500


@api_v1.route("/detection/pause", methods=["POST"])
@login_required
def detection_pause():
    """
    Pauses the detection loop.
    Mirror of: POST /api/detection/pause
    """
    try:
        dm = api_v1.detection_manager

        if dm.paused:
            return jsonify(
                {
                    "status": "paused",
                    "message": "Detection was already paused",
                }
            )

        dm.paused = True
        logger.info("Detection paused via API v1")

        return jsonify(
            {
                "status": "success",
                "message": "Detection paused",
            }
        )
    except Exception as e:
        logger.error(f"Detection pause error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/detection/resume", methods=["POST"])
@login_required
def detection_resume():
    """
    Resumes the detection loop.
    Mirror of: POST /api/detection/resume
    """
    try:
        dm = api_v1.detection_manager

        if not dm.paused:
            return jsonify(
                {
                    "status": "running",
                    "message": "Detection was already running",
                }
            )

        dm.paused = False
        logger.info("Detection resumed via API v1")

        return jsonify(
            {
                "status": "success",
                "message": "Detection resumed",
            }
        )
    except Exception as e:
        logger.error(f"Detection resume error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Settings
# =============================================================================


@api_v1.route("/settings", methods=["GET"])
@login_required
def settings_get():
    """
    Returns current application settings.
    Mirror of: GET /api/settings
    """
    from config import get_settings_payload

    return jsonify(get_settings_payload())


@api_v1.route("/settings", methods=["POST"])
@login_required
def settings_post():
    """
    Updates application settings.
    Mirror of: POST /api/settings
    """
    from config import update_runtime_settings, validate_runtime_updates

    try:
        data = request.get_json() or {}
        valid, errors = validate_runtime_updates(data)

        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        if valid:
            update_runtime_settings(valid)
            dm = api_v1.detection_manager
            dm.update_configuration(valid)

        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# ONVIF Camera Discovery
# =============================================================================


@api_v1.route("/onvif/discover", methods=["GET"])
@login_required
def onvif_discover():
    """
    Scans network for ONVIF cameras.
    Mirror of: GET /api/onvif/discover
    """
    try:
        cameras = onvif_service.discover_cameras(fast=False)

        if not cameras:
            return jsonify({"status": "success", "cameras": []})

        return jsonify({"status": "success", "cameras": cameras})
    except Exception as e:
        logger.error(f"ONVIF Discovery error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/onvif/get_stream_uri", methods=["POST"])
@login_required
def onvif_get_stream_uri():
    """
    Retrieves RTSP stream URI for a camera.
    Mirror of: POST /api/onvif/get_stream_uri
    """
    try:
        data = request.get_json() or {}
        ip = data.get("ip")
        port = int(data.get("port", 80))
        user = data.get("username", "")
        password = data.get("password", "")

        if not ip:
            return jsonify({"status": "error", "message": "IP is required"}), 400

        uri = onvif_service.get_stream_uri(ip, port, user, password)

        if uri:
            return jsonify({"status": "success", "uri": uri})
        else:
            return jsonify(
                {"status": "error", "message": "Could not retrieve URI"}
            ), 404
    except Exception as e:
        logger.error(f"ONVIF Stream URI error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Camera Management
# =============================================================================


@api_v1.route("/cameras", methods=["GET"])
@login_required
def cameras_list():
    """
    Lists all saved cameras.
    Mirror of: GET /api/cameras
    """
    try:
        cameras = onvif_service.get_saved_cameras()
        return jsonify({"status": "success", "cameras": cameras})
    except Exception as e:
        logger.error(f"Camera list error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras", methods=["POST"])
@login_required
def cameras_add():
    """
    Adds a new camera.
    Mirror of: POST /api/cameras
    """
    try:
        data = request.get_json() or {}
        ip = data.get("ip")
        port = int(data.get("port", 80))
        username = data.get("username", "")
        password = data.get("password", "")
        name = data.get("name", "")

        if not ip:
            return jsonify({"status": "error", "message": "IP is required"}), 400

        camera = onvif_service.add_camera(
            ip=ip, port=port, username=username, password=password, name=name
        )

        return jsonify({"status": "success", "camera": camera})
    except Exception as e:
        logger.error(f"Camera add error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>", methods=["PUT"])
@login_required
def cameras_update(camera_id: int):
    """
    Updates an existing camera.
    Mirror of: PUT /api/cameras/<camera_id>
    """
    try:
        data = request.get_json() or {}
        onvif_service.update_camera(camera_id, data)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Camera update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>", methods=["DELETE"])
@login_required
def cameras_delete(camera_id: int):
    """
    Deletes a camera.
    Mirror of: DELETE /api/cameras/<camera_id>
    """
    try:
        onvif_service.delete_camera(camera_id)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Camera delete error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>/test", methods=["POST"])
@login_required
def cameras_test(camera_id: int):
    """
    Tests camera connection.
    Mirror of: POST /api/cameras/<camera_id>/test
    """
    try:
        success = onvif_service.test_camera(camera_id)
        if success:
            return jsonify(
                {"status": "success", "message": "Camera connection successful"}
            )
        else:
            return jsonify(
                {"status": "error", "message": "Camera connection failed"}
            ), 500
    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>/use", methods=["POST"])
@login_required
def cameras_use(camera_id: int):
    """
    Sets camera as active video source.
    Mirror of: POST /api/cameras/<camera_id>/use
    """
    try:
        uri = onvif_service.get_camera_uri(camera_id)
        if not uri:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        dm = api_v1.detection_manager
        dm.update_configuration({"VIDEO_SOURCE": uri})

        return jsonify({"status": "success", "message": "Video source updated"})
    except Exception as e:
        logger.error(f"Camera use error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Analytics
# =============================================================================


@api_v1.route("/analytics/summary", methods=["GET"])
@login_required
def analytics_summary():
    """
    Returns detection analytics summary.
    Mirror of: GET /api/analytics/summary (via add_url_rule)
    """
    with db_service.get_connection() as conn:
        summary = db_service.fetch_analytics_summary(conn)
    return jsonify(summary)


# =============================================================================
# System
# =============================================================================


@api_v1.route("/system/stats", methods=["GET"])
@login_required
def system_stats():
    """
    Returns system resource statistics.
    Mirror of: GET /api/system/stats
    """
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        # Disk usage
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
        except Exception:
            pass

        # Temperature
        temp = None
        try:
            import subprocess

            result = subprocess.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace("temp=", "").replace("'C", ""))
        except Exception:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for _name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            break
            except Exception:
                pass

        response = {"status": "success", "cpu": cpu_percent, "ram": mem.percent}
        if temp is not None:
            response["temp"] = temp
        if disk is not None:
            response["disk"] = disk

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/vitals", methods=["GET"])
@login_required
def system_vitals():
    """
    Returns system vitals from SystemMonitor.

    Provides hardware metrics collected by SystemMonitor including:
    - ts: ISO timestamp
    - cpu_percent: CPU usage percentage
    - ram_percent: RAM usage percentage
    - cpu_temp_c: CPU temperature in Celsius
    - throttled: RPi throttling flags (if applicable)
    - core_voltage: RPi core voltage (if applicable)

    If SystemMonitor is not running, returns a fallback response.
    """
    try:
        # Get system_monitor from blueprint (injected via init_api_v1)
        system_monitor = getattr(api_v1, "system_monitor", None)

        if system_monitor is None:
            # Fallback: return basic stats without monitor
            from datetime import datetime

            import psutil

            return jsonify(
                {
                    "status": "success",
                    "monitor_active": False,
                    "vitals": {
                        "ts": datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "ram_percent": psutil.virtual_memory().percent,
                        "cpu_temp_c": None,
                        "throttled": None,
                    },
                }
            )

        vitals = system_monitor.get_current_vitals()

        return jsonify(
            {
                "status": "success",
                "monitor_active": True,
                "vitals": vitals,
            }
        )
    except Exception as e:
        logger.error(f"System vitals error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/versions", methods=["GET"])
@login_required
def system_versions():
    """
    Returns software version information.
    Mirror of: GET /api/system/versions
    """
    try:
        import sys

        import cv2

        # Read app version from file
        app_version = "unknown"
        try:
            import os

            version_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "version.txt",
            )
            if os.path.exists(version_file):
                with open(version_file) as f:
                    app_version = f.read().strip()
        except Exception:
            pass

        return jsonify(
            {
                "status": "success",
                "app_version": app_version,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "opencv_version": cv2.__version__,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/shutdown", methods=["POST"])
@login_required
def system_shutdown():
    """
    Initiates system shutdown.
    Mirror of: POST /api/system/shutdown
    """
    import subprocess
    import threading

    try:

        def delayed_shutdown():
            import time

            time.sleep(2)
            try:
                subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            except Exception as e:
                logger.error(f"Shutdown command failed: {e}")

        t = threading.Thread(target=delayed_shutdown)
        t.start()

        return jsonify(
            {"status": "success", "message": "System is shutting down..."}
        ), 200
    except Exception as e:
        logger.error(f"Error initiating shutdown: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/restart", methods=["POST"])
@login_required
def system_restart():
    """
    Initiates system restart.
    Mirror of: POST /api/system/restart
    """
    import subprocess
    import threading

    try:

        def delayed_restart():
            import time

            time.sleep(2)
            try:
                subprocess.run(["sudo", "reboot"], check=False)
            except Exception as e:
                logger.error(f"Restart command failed: {e}")

        t = threading.Thread(target=delayed_restart)
        t.start()

        return jsonify({"status": "success", "message": "System is restarting..."}), 200
    except Exception as e:
        logger.error(f"Error initiating restart: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Blueprint Initialization
# =============================================================================


def init_api_v1(app, detection_manager, system_monitor=None):
    """
    Initialize the API v1 blueprint and register it with the app.

    Args:
        app: Flask application instance
        detection_manager: DetectionManager instance for detection control
        system_monitor: Optional SystemMonitor instance for vitals API
    """
    # Store detection_manager reference on blueprint for route access
    api_v1.detection_manager = detection_manager

    # Store system_monitor reference for vitals API (optional)
    api_v1.system_monitor = system_monitor

    # Register blueprint
    app.register_blueprint(api_v1)

    logger.info("API v1 blueprint registered at /api/v1")
