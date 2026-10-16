"""Current station report page and its background-build status."""

from threading import Lock

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    make_response,
    render_template,
    request,
)
from flask.blueprints import BlueprintSetupState

from web.services import station_service

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.record
def _initialize_report_lock(state: BlueprintSetupState) -> None:
    state.app.extensions["station_report_lock"] = Lock()


def _report_status(
    *, opening: bool = False, refresh: bool = False
) -> tuple[dict, dict[str, str]]:
    try:
        selection = station_service.validate_selection(
            request.args.get("start", ""),
            request.args.get("end", ""),
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    with current_app.extensions["station_report_lock"]:
        cache = current_app.extensions.get("station_report")
        if cache is None:
            cache = station_service.create_report_cache()
            current_app.extensions["station_report"] = cache
    return cache.status(selection, opening=opening, refresh=refresh), selection


@analytics_bp.route("/analytics", methods=["GET"])
def analytics_page() -> Response:
    from web import view_helpers

    status, selection = _report_status(opening=request.args.get("content") != "1")
    report = status["report"]
    highlights = (
        view_helpers.build_station_views(report["highlights"]) if report else []
    )
    response = make_response(
        render_template(
            "station_report.html",
            status=status,
            report=report,
            highlights=highlights,
            common_name_for_species=view_helpers.get_common_name,
            selection=selection,
            current_path="/analytics",
        )
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@analytics_bp.route("/api/station-report", methods=["GET", "POST"])
def station_report_status() -> Response:
    status, _ = _report_status(refresh=request.method == "POST")
    response = jsonify(state=status["state"], revision=status["revision"])
    response.headers["Cache-Control"] = "no-store"
    return response
