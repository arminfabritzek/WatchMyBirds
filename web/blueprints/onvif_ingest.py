from flask import Blueprint, jsonify, request

from web.blueprints.auth import login_required
from web.security import error_response as _error_response
from web.services import onvif_service

onvif_ingest_bp = Blueprint("onvif_ingest", __name__)


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
