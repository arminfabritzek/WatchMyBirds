"""Authenticated storage diagnostics; scanning is always an explicit background job."""

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from flask.blueprints import BlueprintSetupState

from web.blueprints.auth import login_required
from web.services import storage_service

storage_bp = Blueprint("storage", __name__)


@storage_bp.record_once
def init_storage(state: BlueprintSetupState) -> None:
    state.app.extensions["storage_accounting"] = storage_service.create()


@storage_bp.get("/storage")
@login_required
def storage_page() -> str:
    return render_template(
        "storage.html",
        current_path="/settings",
        deletion_state=storage_service.deletion_state(),
    )


@storage_bp.get("/api/v1/storage/status")
@login_required
def storage_status() -> Response:
    return jsonify(current_app.extensions["storage_accounting"].status())


@storage_bp.post("/api/v1/storage/scan")
@login_required
def storage_scan() -> tuple[Response, int]:
    if (
        request.args
        or request.content_length
        or request.headers.get("Transfer-Encoding")
    ):
        return jsonify(error="This scan accepts no parameters"), 400
    started = current_app.extensions["storage_accounting"].start()
    return jsonify(started=started), 202 if started else 200
