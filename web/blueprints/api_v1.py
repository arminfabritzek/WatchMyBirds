"""
API v1 Blueprint.

This blueprint provides versioned API endpoints under /api/v1/*.
It is a 1:1 mirror of the existing /api/* routes - no changes to behavior or response format.

Purpose: Enable API versioning without breaking existing clients.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import get_config
from logging_config import get_logger
from utils.settings import mask_rtsp_url, unmask_rtsp_url
from web.blueprints.auth import login_required
from web.power_actions import (
    POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
    get_power_action_success_message,
    is_power_management_available,
    schedule_power_action,
)
from web.services import (
    backup_restore_service,
    db_service,
    onvif_service,
)

logger = get_logger(__name__)
config = get_config()

# Create Blueprint
api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def _read_file_tail(path: Path, max_lines: int = 200) -> dict:
    """Read the last lines of a text file safely for diagnostics endpoints."""
    result = {
        "path": str(path),
        "exists": path.exists(),
        "tail_text": "",
        "line_count": 0,
        "error": "",
    }
    if not path.exists():
        return result

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            tail_lines = list(deque(f, maxlen=max_lines))
        text = "".join(tail_lines)
        result["tail_text"] = text
        result["line_count"] = len(text.splitlines())
    except Exception as e:
        result["error"] = str(e)

    return result


def _detect_runtime_environment() -> str:
    """Detect whether the app runs on host or in a container runtime."""
    if Path("/.dockerenv").exists():
        return "docker"

    try:
        cgroup_text = Path("/proc/1/cgroup").read_text(
            encoding="utf-8", errors="ignore"
        )
        lowered = cgroup_text.lower()
        if "kubepods" in lowered:
            return "kubernetes"
        if "docker" in lowered:
            return "docker"
        if "containerd" in lowered:
            return "containerd"
    except Exception:
        pass

    return "host"


def _run_command_safe(
    cmd: list[str],
    timeout_sec: float = 2.5,
    max_output_chars: int = 12000,
    expected_permission_error: bool = False,
) -> dict:
    """Run a diagnostic command with availability checks and strict timeout."""
    binary = cmd[0] if cmd else ""
    if not binary:
        return {
            "available": False,
            "ok": False,
            "returncode": -1,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": "empty command",
        }

    if shutil.which(binary) is None:
        return {
            "available": False,
            "ok": False,
            "returncode": 127,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": f"{binary} not available",
        }

    permission_error_markers = (
        "insufficient permissions",
        "not seeing messages from other users",
        "no journal files were opened due to insufficient permissions",
        "permission denied",
    )

    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
        combined_output = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        if stderr_text and stderr_text not in combined_output:
            combined_output = (
                f"{combined_output}\n{stderr_text}".strip()
                if combined_output
                else stderr_text
            )

        truncated = False
        if len(combined_output) > max_output_chars:
            combined_output = combined_output[:max_output_chars] + "\n... (truncated)"
            truncated = True

        normalized = combined_output.lower()
        permission_limited = expected_permission_error and any(
            marker in normalized for marker in permission_error_markers
        )

        return {
            "available": True,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "timed_out": False,
            "truncated": truncated,
            "output": combined_output,
            "error": "" if completed.returncode == 0 else stderr_text,
            "expected_permission_error": permission_limited,
        }
    except subprocess.TimeoutExpired as e:
        timeout_output = ""
        if e.stdout:
            timeout_output += e.stdout
        if e.stderr:
            timeout_output += f"\n{e.stderr}" if timeout_output else e.stderr
        timeout_output = timeout_output.strip()

        return {
            "available": True,
            "ok": False,
            "returncode": -1,
            "timed_out": True,
            "truncated": False,
            "output": timeout_output,
            "error": f"timeout after {timeout_sec:.1f}s",
            "expected_permission_error": False,
        }
    except Exception as e:
        return {
            "available": True,
            "ok": False,
            "returncode": -1,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": str(e),
            "expected_permission_error": False,
        }


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


@api_v1.route("/species/thumbnails", methods=["GET"])
@login_required
def get_species_thumbnails():
    """
    Returns a mapping of species names to their latest thumbnail URL.

    Uses gallery_service.get_captured_detections() following established patterns.
    Returns thumbnails keyed by: scientific name (both formats) and German name.
    """
    # Load common names for German mapping
    import json
    import os

    from web.services import gallery_service

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    common_names_file = os.path.join(project_root, "assets", "common_names_DE.json")
    common_names = {}
    try:
        with open(common_names_file, encoding="utf-8") as f:
            common_names = json.load(f)
    except Exception:
        pass

    mapping = {}
    try:
        # Use existing gallery service - same pattern as species_route
        all_detections = gallery_service.get_captured_detections()

        # Build mapping: best thumbnail per species (later entries overwrite)
        for det in all_detections:
            species_lat = det.get("cls_class_name")
            thumb_virt = det.get("thumbnail_path_virtual")

            if not species_lat or not thumb_virt:
                continue

            url = f"/uploads/derivatives/thumbs/{thumb_virt}"

            # Key by scientific (underscore), scientific (spaces), German
            mapping[species_lat] = url
            mapping[species_lat.replace("_", " ")] = url
            german = common_names.get(species_lat)
            if german:
                mapping[german] = url

    except Exception as e:
        logger.error(f"Failed to fetch species thumbnails: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    # Fallback: Wikipedia thumbnails from DB cache (fast, no HTTP calls)
    missing_species = []
    try:
        conn = db_service.get_connection()
        try:
            # 1. Read all cached Wikipedia thumbnails
            cached = conn.execute(
                "SELECT scientific_name, image_url FROM species_meta "
                "WHERE image_url IS NOT NULL AND image_url != ''"
            ).fetchall()

            for row in cached:
                sci = row[0]
                url = row[1]
                sci_u = sci.replace(" ", "_")
                if sci not in mapping and sci_u not in mapping:
                    mapping[sci] = url
                    mapping[sci_u] = url
                    german = common_names.get(sci_u) or common_names.get(sci)
                    if german:
                        mapping[german] = url

        finally:
            conn.close()
    except Exception as cache_err:
        logger.debug(f"Wikipedia cache read failed: {cache_err}")

    # 3. Trigger background fetch for missing species (non-blocking)
    if missing_species:
        import threading

        def _prime_wiki_cache(species_list):
            from utils.wikipedia import get_cached_species_thumbnail

            for sci in species_list:
                try:
                    get_cached_species_thumbnail(sci)
                except Exception:
                    pass
            logger.info(f"Wiki cache primed for {len(species_list)} species")

        t = threading.Thread(
            target=_prime_wiki_cache, args=(missing_species,), daemon=True
        )
        t.start()
        logger.info(f"Background wiki fetch started for {len(missing_species)} species")

    return jsonify({"status": "success", "thumbnails": mapping})


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

    payload = get_settings_payload()
    if "VIDEO_SOURCE" in payload and isinstance(payload["VIDEO_SOURCE"], dict):
        payload["VIDEO_SOURCE"]["value"] = mask_rtsp_url(
            payload["VIDEO_SOURCE"]["value"]
        )
    if "CAMERA_URL" in payload and isinstance(payload["CAMERA_URL"], dict):
        payload["CAMERA_URL"]["value"] = mask_rtsp_url(payload["CAMERA_URL"]["value"])

    return jsonify(payload)


@api_v1.route("/settings", methods=["POST"])
@login_required
def settings_post():
    """
    Updates application settings.
    Mirror of: POST /api/settings
    """
    from config import (
        ensure_go2rtc_stream_synced,
        get_config,
        resolve_effective_sources,
        update_runtime_settings,
        validate_runtime_updates,
    )

    try:
        data = request.get_json() or {}

        # Security: Unmask RTSP password if placeholder is present
        current_config = get_config()
        if "VIDEO_SOURCE" in data:
            original_url = current_config.get("VIDEO_SOURCE")
            data["VIDEO_SOURCE"] = unmask_rtsp_url(data["VIDEO_SOURCE"], original_url)
        if "CAMERA_URL" in data:
            original_cam = current_config.get("CAMERA_URL", "")
            data["CAMERA_URL"] = unmask_rtsp_url(data["CAMERA_URL"], original_cam)

        valid, errors = validate_runtime_updates(data)

        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        if valid:
            update_runtime_settings(valid)

            # --- Pre-sync go2rtc before resolving stream sources ---
            cfg = get_config()
            ensure_go2rtc_stream_synced(cfg)

            # --- Resolve effective sources after settings change ---
            resolved = resolve_effective_sources(cfg)
            cfg["VIDEO_SOURCE"] = resolved["video_source"]

            logger.info(
                "STREAM_SOURCE stream_mode=%s video_source=%s reason=%s",
                resolved["effective_mode"],
                resolved["video_source"][:40]
                if resolved["video_source"]
                else "(empty)",
                resolved["reason"],
            )

            dm = api_v1.detection_manager
            dm.update_configuration({"VIDEO_SOURCE": resolved["video_source"]})

        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Telegram Report  (Job-Status Flow)
# =============================================================================


# In-memory job registry  {job_id: {status, message, created_at}}
# Auto-evicts entries older than _REPORT_JOB_TTL seconds.
_report_jobs: dict[str, dict] = {}
_report_jobs_lock = threading.Lock()
_REPORT_JOB_TTL = 600  # 10 min


def _evict_stale_report_jobs() -> None:
    """Remove jobs older than TTL.  Called under lock."""
    cutoff = time.time() - _REPORT_JOB_TTL
    stale = [jid for jid, j in _report_jobs.items() if j["created_at"] < cutoff]
    for jid in stale:
        del _report_jobs[jid]


@api_v1.route("/telegram/send-report", methods=["POST"])
@login_required
def telegram_send_report():
    """
    Starts an on-demand daily report as a background job.

    Returns ``job_id`` immediately.  Poll status via
    ``GET /api/v1/telegram/send-report/<job_id>/status``.
    """
    cfg = get_config()
    bot_token = str(cfg.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    chat_id = str(cfg.get("TELEGRAM_CHAT_ID", "") or "").strip()

    if not bot_token or not chat_id:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Telegram credentials missing. Set Bot Token and Chat ID first.",
                }
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]

    with _report_jobs_lock:
        _evict_stale_report_jobs()
        _report_jobs[job_id] = {
            "status": "pending",
            "message": "Job queued.",
            "created_at": time.time(),
        }

    def _run(jid: str) -> None:
        # Mark running
        with _report_jobs_lock:
            if jid in _report_jobs:
                _report_jobs[jid]["status"] = "running"
                _report_jobs[jid]["message"] = "Report is being generated…"
        logger.info("Telegram report job %s started.", jid)

        try:
            from utils.daily_report import main as run_report

            # Provide ingest health for truthful status rendering
            health_provider = None
            dm = getattr(api_v1, "detection_manager", None)
            if dm is not None:
                health_provider = getattr(dm, "get_ingest_health_snapshot", None)

            run_report(ingest_health_provider=health_provider)

            with _report_jobs_lock:
                if jid in _report_jobs:
                    _report_jobs[jid]["status"] = "success"
                    _report_jobs[jid]["message"] = "Report sent successfully."
            logger.info("Telegram report job %s completed.", jid)

        except Exception as exc:
            error_msg = str(exc) or "Unknown error"
            with _report_jobs_lock:
                if jid in _report_jobs:
                    _report_jobs[jid]["status"] = "error"
                    _report_jobs[jid]["message"] = error_msg
            logger.error("Telegram report job %s failed: %s", jid, exc, exc_info=True)

    t = threading.Thread(
        target=_run, args=(job_id,), name=f"TgReport-{job_id}", daemon=True
    )
    t.start()

    return jsonify(
        {
            "status": "accepted",
            "job_id": job_id,
            "message": "Report job started.",
        }
    )


@api_v1.route("/telegram/send-report/<job_id>/status", methods=["GET"])
@login_required
def telegram_report_status(job_id: str):
    """
    Poll the status of a report job.

    Response shape::

        {
            "job_id":  "abc123",
            "status":  "pending" | "running" | "success" | "error",
            "message": "…"
        }
    """
    with _report_jobs_lock:
        job = _report_jobs.get(job_id)

    if not job:
        return jsonify(
            {
                "job_id": job_id,
                "status": "error",
                "message": "Job not found (expired or invalid ID).",
            }
        ), 404

    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "message": job["message"],
        }
    )


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

        camera = onvif_service.save_camera(
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
        onvif_service.update_camera(
            camera_id=camera_id,
            ip=data.get("ip"),
            port=int(data["port"]) if data.get("port") else None,
            username=data.get("username"),
            password=data.get("password"),
            name=data.get("name"),
        )
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

    Updates CAMERA_URL (user-facing) and resolves effective VIDEO_SOURCE
    through the central resolver.
    """
    try:
        from config import (
            ensure_go2rtc_stream_synced,
            get_config,
            resolve_effective_sources,
            update_runtime_settings,
        )

        uri = onvif_service.get_camera_uri(camera_id)
        if not uri:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        # Set CAMERA_URL (not VIDEO_SOURCE directly)
        update_runtime_settings({"CAMERA_URL": uri})

        # --- Pre-sync go2rtc before resolving ---
        cfg = get_config()
        ensure_go2rtc_stream_synced(cfg)

        # Resolve and apply
        resolved = resolve_effective_sources(cfg)
        cfg["VIDEO_SOURCE"] = resolved["video_source"]

        logger.info(
            "cameras_use camera_id=%s stream_mode=%s video_source=%s",
            camera_id,
            resolved["effective_mode"],
            resolved["video_source"][:40] if resolved["video_source"] else "(empty)",
        )

        dm = api_v1.detection_manager
        dm.update_configuration({"VIDEO_SOURCE": resolved["video_source"]})

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
    conn = db_service.get_connection()
    try:
        summary = db_service.fetch_analytics_summary(conn)
    finally:
        conn.close()
    return jsonify(summary)


# =============================================================================
# Weather
# =============================================================================


@api_v1.route("/weather/now", methods=["GET"])
def weather_now():
    """
    Returns the current cached weather data.
    No login required - weather is public information.
    """
    from web.services.weather_service import get_current_weather

    try:
        weather = get_current_weather()
        if weather.get("timestamp") is None:
            return jsonify(
                {
                    "status": "pending",
                    "message": "Weather data not yet available. First fetch in progress.",
                    "weather": weather,
                }
            )
        return jsonify({"status": "success", "weather": weather})
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/weather/history", methods=["GET"])
def weather_history():
    """
    Returns weather history for the last N hours (default 24).
    Query param: ?hours=24
    """
    from web.services.weather_service import get_weather_history

    try:
        hours = request.args.get("hours", 24, type=int)
        hours = max(1, min(168, hours))  # Clamp 1h - 7d
        history = get_weather_history(hours=hours)
        return jsonify({"status": "success", "hours": hours, "data": history})
    except Exception as e:
        logger.error(f"Weather history API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# System
# =============================================================================


@api_v1.route("/health", methods=["GET"])
@login_required
def system_health():
    """
    Returns comprehensive system health status.

    Includes:
    - Overall status (ok/error/warning)
    - Database connectivity and latency
    - Disk space usage
    - OS vital signs (CPU/RAM/Temp/Throttling)
    """
    from web.services import health_service

    try:
        health = health_service.get_system_health()
        status_code = 200
        if health.get("status") == "error":
            status_code = 503

        return jsonify(health), status_code
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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


@api_v1.route("/system/diagnostics", methods=["GET"])
@login_required
def system_diagnostics():
    """
    Returns an extended diagnostics snapshot for admin log view.

    Includes:
    - Runtime/process metadata
    - Current monitor vitals (if available)
    - app.log / vital_signs.csv / fd_leak_dump tails
    - Safe command probes (systemctl/journalctl/docker) with timeout
    """
    try:
        import psutil

        output_dir = Path(config.get("OUTPUT_DIR", "./data/output"))
        log_dir = output_dir / "logs"

        app_lines = max(50, min(int(request.args.get("app_lines", 300)), 2000))
        vitals_lines = max(30, min(int(request.args.get("vitals_lines", 240)), 2000))
        fd_dump_lines = max(20, min(int(request.args.get("fd_lines", 300)), 4000))

        app_log_tail = _read_file_tail(log_dir / "app.log", max_lines=app_lines)
        vitals_tail = _read_file_tail(
            log_dir / "vital_signs.csv", max_lines=vitals_lines
        )
        fd_dump_tail = _read_file_tail(
            log_dir / "fd_leak_dump.txt", max_lines=fd_dump_lines
        )
        fd_dump_present = (
            fd_dump_tail["exists"]
            and bool(fd_dump_tail["tail_text"].strip())
            and "fd_leak_dump_not_present" not in fd_dump_tail["tail_text"].lower()
        )

        monitor = getattr(api_v1, "system_monitor", None)
        monitor_active = monitor is not None
        if monitor_active:
            try:
                vitals = monitor.get_current_vitals()
            except Exception:
                vitals = {}
        else:
            vitals = {}

        proc = psutil.Process()
        process_rss_mb = 0.0
        process_threads = 0
        process_fds = -1
        with proc.oneshot():
            process_rss_mb = proc.memory_info().rss / (1024 * 1024)
            process_threads = proc.num_threads()
            try:
                process_fds = proc.num_fds()
            except Exception:
                process_fds = -1

        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(output_dir))

        load_avg = None
        if hasattr(os, "getloadavg"):
            try:
                load_avg = os.getloadavg()
            except Exception:
                load_avg = None

        runtime = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "environment": _detect_runtime_environment(),
            "generated_at": datetime.now().isoformat(),
        }

        boot_time_iso = None
        try:
            boot_time_iso = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        except Exception:
            boot_time_iso = None

        system = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": vm.percent,
            "ram_total_mb": round(vm.total / (1024 * 1024), 1),
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "disk_used_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "load_avg": list(load_avg) if load_avg else None,
            "boot_time_iso": boot_time_iso,
        }

        process = {
            "rss_mb": round(process_rss_mb, 1),
            "threads": process_threads,
            "fds": process_fds,
        }

        commands = {
            "systemctl_app": _run_command_safe(
                [
                    "systemctl",
                    "show",
                    "app",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "NRestarts",
                ],
                timeout_sec=2.5,
            ),
            "journal_app_tail": _run_command_safe(
                ["journalctl", "-u", "app", "-n", "80", "--no-pager"],
                timeout_sec=2.5,
                expected_permission_error=True,
            ),
            "docker_ps": _run_command_safe(
                [
                    "docker",
                    "ps",
                    "--format",
                    "table {{.Names}}\t{{.Status}}\t{{.Image}}",
                ],
                timeout_sec=2.5,
            ),
            "docker_stats": _run_command_safe(
                ["docker", "stats", "--no-stream", "--all"],
                timeout_sec=2.5,
            ),
        }

        return jsonify(
            {
                "status": "success",
                "runtime": runtime,
                "monitor_active": monitor_active,
                "vitals": vitals,
                "system": system,
                "process": process,
                "files": {
                    "app_log": app_log_tail,
                    "vitals_csv": vitals_tail,
                    "fd_leak_dump": {
                        **fd_dump_tail,
                        "present": fd_dump_present,
                    },
                },
                "commands": commands,
            }
        )
    except Exception as e:
        logger.error(f"System diagnostics error: {e}")
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

        return jsonify(
            {
                "status": "success",
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
    try:
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
    try:
        if not is_power_management_available():
            logger.warning("Restart ignored: systemd not available (likely container).")
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
    except Exception as e:
        logger.error(f"Error initiating restart: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/public/go2rtc/health", methods=["GET"])
def go2rtc_health_public():
    """
    Public same-origin health endpoint for frontend go2rtc checks.

    Avoids browser CORS issues when the app UI (port 8050) probes go2rtc
    directly on port 1984.

    Returns diagnostic ``detail`` when unhealthy so the root cause
    (timeout, DNS, connection refused …) is visible without shell access.
    """
    try:
        import urllib.request

        from config import get_config

        cfg = get_config()
        api_base = str(cfg.get("GO2RTC_API_BASE", "http://127.0.0.1:1984") or "")
        probe_url = f"{api_base.rstrip('/')}/api/streams"
        detail = None

        try:
            req = urllib.request.Request(probe_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                healthy = resp.status == 200
        except Exception as exc:
            healthy = False
            detail = str(exc)

        result = {
            "status": "success",
            "healthy": healthy,
            "api_base": api_base,
        }
        if detail:
            result["detail"] = detail
        return jsonify(result)
    except Exception as e:
        logger.error(f"go2rtc health API error: {e}")
        return jsonify({"status": "error", "healthy": False, "message": str(e)}), 500


@api_v1.route("/public/bbox-heatmap", methods=["GET"])
def bbox_heatmap_public():
    """
    Disabled in the public backport build.
    """
    return (
        jsonify(
            {
                "status": "error",
                "message": "This feature is not available in the public backport build.",
            }
        ),
        404,
    )


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
