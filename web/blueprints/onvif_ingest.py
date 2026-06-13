import os
import threading

from flask import Blueprint, jsonify, request

from config import get_config
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.security import error_response as _error_response
from web.services import onvif_service

logger = get_logger(__name__)
config = get_config()

onvif_ingest_bp = Blueprint("onvif_ingest", __name__)

_detection_manager = None


def init_onvif_ingest_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


@onvif_ingest_bp.route("/api/ingest/start", methods=["POST"])
@login_required
def start_ingest_endpoint():
    try:
        env_path = config.get("INGEST_DIR")
        cwd_path = os.path.abspath(os.path.join(os.getcwd(), "ingest"))

        logger.info(
            f"Ingest Request: CWD={os.getcwd()}, Configured: {env_path}, Local: {cwd_path}"
        )

        if os.path.exists(env_path):
            ingest_path = env_path
            logger.info(f"Using configured ingest path: {ingest_path}")
        elif os.path.exists(cwd_path):
            ingest_path = cwd_path
            logger.info(
                f"Configured path not found. Using local CWD fallback: {ingest_path}"
            )
        else:
            ingest_path = env_path
            logger.warning(
                f"No valid ingest dir found. Falling back to configured: {ingest_path}"
            )

        def run_ingest():
            _detection_manager.start_user_ingest(ingest_path)

        t = threading.Thread(target=run_ingest)
        t.start()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "User Ingest started. Stream will pause.",
                }
            ),
            200,
        )
    except Exception as exc:
        return _error_response("Error starting ingest", exc)


@onvif_ingest_bp.route("/api/onvif/discover", methods=["GET"])
@login_required
def onvif_discover_route():
    try:
        cameras = onvif_service.discover_cameras(fast=False)
        if not cameras:
            return jsonify({"status": "success", "cameras": []})
        return jsonify({"status": "success", "cameras": cameras})
    except Exception as exc:
        return _error_response("ONVIF Discovery route failed", exc)


@onvif_ingest_bp.route("/api/onvif/get_stream_uri", methods=["POST"])
@login_required
def onvif_get_stream_uri_route():
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
    except Exception as exc:
        return _error_response("ONVIF Stream URI route failed", exc)
