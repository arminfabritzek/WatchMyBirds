"""Retention blueprint — dry-run preview + manual "Clean up now".

Routes parse the request, call the retention service, and return JSON
(S-01 thinness). The run route is login-protected like the trash
hard-delete routes.
"""

from flask import Blueprint, current_app, jsonify, request
from flask.blueprints import BlueprintSetupState

from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import retention_service

logger = get_logger(__name__)

retention_bp = Blueprint("retention", __name__)


@retention_bp.record_once
def init_retention_preview(state: BlueprintSetupState) -> None:
    state.app.extensions["retention_preview"] = retention_service.create_preview()


@retention_bp.route("/api/v1/retention/preview", methods=["GET"])
@login_required
def retention_preview():
    result = retention_service.preview(
        current_app.extensions["retention_preview"],
        refresh=request.args.get("refresh") == "1",
    )
    return jsonify(result), 202 if result["status"] == "running" else 200


@retention_bp.route("/api/v1/retention/run", methods=["POST"])
@login_required
def retention_run():
    result = retention_service.start_run()
    logger.info(
        "retention run request: status=%s reason=%s",
        result.get("status"),
        result.get("reason"),
    )
    code = 200 if result.get("status") != "unknown_job" else 503
    return jsonify(result), code
