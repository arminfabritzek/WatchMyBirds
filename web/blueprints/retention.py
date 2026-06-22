"""Retention blueprint — dry-run preview + manual "Clean up now".

Routes parse the request, call the retention service, and return JSON
(S-01 thinness). The run route is login-protected like the trash
hard-delete routes.
"""

from flask import Blueprint, jsonify

from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import retention_service

logger = get_logger(__name__)

retention_bp = Blueprint("retention", __name__)


@retention_bp.route("/api/v1/retention/preview", methods=["GET"])
@login_required
def retention_preview():
    return jsonify(retention_service.preview())


@retention_bp.route("/api/v1/retention/run", methods=["POST"])
@login_required
def retention_run():
    result = retention_service.run()
    logger.info(
        "retention run: deleted=%s freed_bytes=%s missing=%s errors=%s",
        result.get("deleted"),
        result.get("freed_bytes"),
        result.get("missing"),
        result.get("errors"),
    )
    return jsonify(result)
