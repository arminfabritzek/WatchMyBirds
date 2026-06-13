from flask import Blueprint, jsonify, request

from config import (
    ensure_go2rtc_stream_synced,
    get_config,
    update_runtime_settings,
)
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.security import error_response as _error_response
from web.security import safe_log_value as _slv
from web.services import onvif_service

logger = get_logger(__name__)

cameras_bp = Blueprint("cameras", __name__)

_detection_manager = None


def init_cameras_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


@cameras_bp.route("/api/cameras", methods=["GET"])
@login_required
def cameras_list_route():
    try:
        cameras = onvif_service.get_saved_cameras()
        return jsonify({"status": "success", "cameras": cameras})
    except Exception as exc:
        return _error_response("Camera list failed", exc)


@cameras_bp.route("/api/cameras", methods=["POST"])
@login_required
def cameras_add_route():
    try:
        data = request.get_json() or {}
        ip = data.get("ip")
        port = int(data.get("port", 80))
        username = data.get("username", "")
        password = data.get("password", "")
        name = data.get("name", "")

        if not ip:
            return jsonify({"status": "error", "message": "IP is required"}), 400

        result = onvif_service.save_camera(
            ip=ip,
            port=port,
            username=username,
            password=password,
            name=name,
        )
        return jsonify({"status": "success", "camera": result})
    except ValueError as exc:
        logger.info("Camera add rejected [%s]", type(exc).__name__, exc_info=True)
        return jsonify(
            {"status": "error", "message": "Invalid camera parameters"}
        ), 400
    except Exception as exc:
        return _error_response("Camera add failed", exc)


@cameras_bp.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
@login_required
def cameras_delete_route(camera_id: int):
    try:
        if onvif_service.delete_camera(camera_id):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
    except Exception as exc:
        return _error_response("Camera delete failed", exc)


@cameras_bp.route("/api/cameras/<int:camera_id>", methods=["PUT"])
@login_required
def cameras_update_route(camera_id: int):
    try:
        data = request.get_json() or {}
        if onvif_service.update_camera(
            camera_id,
            ip=data.get("ip"),
            port=data.get("port"),
            username=data.get("username"),
            password=data.get("password"),
            name=data.get("name"),
        ):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
    except Exception as exc:
        return _error_response("Camera update failed", exc)


@cameras_bp.route("/api/cameras/<int:camera_id>/test", methods=["POST"])
@login_required
def cameras_test_route(camera_id: int):
    try:
        cam = onvif_service.get_camera(camera_id, include_password=True)

        if not cam:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        details = onvif_service.get_device_info(
            ip=cam["ip"],
            port=cam.get("port", 80),
            username=cam.get("username", ""),
            password=cam.get("password", ""),
        )

        if details:
            has_ptz = bool(details.get("has_ptz", False))

            onvif_service.update_test_result(
                camera_id,
                success=True,
                manufacturer=details.get("manufacturer", ""),
                model=details.get("model", ""),
                has_ptz=has_ptz,
            )
            return jsonify(
                {
                    "status": "success",
                    "details": {
                        "manufacturer": details.get("manufacturer"),
                        "model": details.get("model"),
                        "firmware": details.get("firmware"),
                        "has_ptz": has_ptz,
                    },
                }
            )
        else:
            onvif_service.update_test_result(camera_id, success=False)
            return jsonify({"status": "error", "message": "Connection failed."}), 400
    except Exception as exc:
        return _error_response("Camera test failed", exc)


@cameras_bp.route("/api/cameras/<int:camera_id>/use", methods=["POST"])
@login_required
def cameras_use_route(camera_id: int):
    try:
        from config import resolve_effective_sources

        cam = onvif_service.get_camera(camera_id, include_password=True)

        if not cam:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        logger.info(
            "Activating camera %s: %s @ %s:%s",
            _slv(camera_id),
            _slv(cam.get("name")),
            _slv(cam.get("ip")),
            _slv(cam.get("port", 80)),
        )

        if not cam.get("username") or not cam.get("password"):
            logger.warning("Camera %s has no credentials stored", _slv(camera_id))
            return jsonify(
                {
                    "status": "error",
                    "message": "No credentials stored for this camera. Please edit the camera and add username/password.",
                }
            ), 400

        try:
            uri = onvif_service.get_stream_uri(
                camera_ip=cam["ip"],
                port=cam.get("port", 80),
                username=cam.get("username", ""),
                password=cam.get("password", ""),
            )
            logger.info("Retrieved RTSP URI for camera %s", _slv(camera_id))
        except Exception as exc:
            logger.error(
                "Failed to get stream URI for camera %s [%s]",
                _slv(camera_id),
                type(exc).__name__,
                exc_info=True,
            )
            return jsonify(
                {"status": "error", "message": "ONVIF connection failed."}
            ), 400

        if not uri:
            return jsonify(
                {
                    "status": "error",
                    "message": "Could not retrieve stream URI from camera",
                }
            ), 400

        logger.info(f"Setting CAMERA_URL to: {uri[:50]}...")
        update_runtime_settings({"CAMERA_URL": uri})

        cfg = get_config()
        ensure_go2rtc_stream_synced(cfg)
        resolved = resolve_effective_sources(cfg)
        cfg["VIDEO_SOURCE"] = resolved["video_source"]

        _detection_manager.update_configuration(
            {"VIDEO_SOURCE": resolved["video_source"]}
        )

        logger.info(
            "Camera %s activated: mode=%s video_source=%s",
            _slv(camera_id),
            _slv(resolved["effective_mode"]),
            _slv(resolved["video_source"][:40]),
        )

        return jsonify(
            {
                "status": "success",
                "message": f"Camera '{cam.get('name', 'Camera')}' is now active",
                "uri_set": True,
            }
        )

    except Exception as exc:
        logger.error("Camera use failed [%s]", type(exc).__name__, exc_info=True)
        return jsonify(
            {"status": "error", "message": "Camera activation failed"}
        ), 500
