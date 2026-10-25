"""
Moderation Blueprint.

Handles all bulk moderation routes:
- POST /api/moderation/resolve-selection  — Resolve selection to concrete IDs
- POST /api/moderation/bulk/download      — ZIP the originals behind a selection
"""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import (
    db_service,
    image_download_service,
)
from web.services.filter_service import FilterContext, resolve_filtered_ids

logger = get_logger(__name__)

moderation_bp = Blueprint("moderation", __name__)


def _parse_iso_date(raw_value: str, field_name: str) -> str:
    """Validate a YYYY-MM-DD date string and return it unchanged."""
    if not raw_value:
        raise ValueError(f"{field_name} required")

    try:
        datetime.strptime(raw_value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}") from exc

    return raw_value


# ---------------------------------------------------------------------------
# POST /api/moderation/resolve-selection
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/resolve-selection", methods=["POST"])
@login_required
def resolve_selection() -> tuple:
    """Resolve a selection to concrete detection IDs / filenames.

    Accepts:
        { mode: "explicit" | "all_filtered" | "date_range",
          ids: [int, ...],              // only for mode=explicit
          filter_context: { ... },      // only for mode=all_filtered
          from_date: "YYYY-MM-DD",      // only for mode=date_range
          to_date: "YYYY-MM-DD"         // only for mode=date_range
        }

    Returns:
        { status, detection_ids, image_filenames, total_count }
    """
    data = request.get_json() or {}
    mode = data.get("mode", "explicit")

    if mode == "explicit":
        ids = data.get("ids", [])
        filenames = data.get("filenames", [])
        return jsonify(
            {
                "status": "success",
                "detection_ids": ids,
                "image_filenames": filenames,
                "total_count": len(ids) + len(filenames),
            }
        )

    if mode == "all_filtered":
        raw_ctx = data.get("filter_context")
        if not raw_ctx or not isinstance(raw_ctx, dict):
            return jsonify(
                {"status": "error", "message": "filter_context required"}
            ), 400

        try:
            ctx = FilterContext.from_dict(raw_ctx)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Invalid filter_context [%s]", type(exc).__name__, exc_info=True
            )
            return jsonify(
                {"status": "error", "message": "Invalid filter_context"}
            ), 400

        result = resolve_filtered_ids(ctx)
        return jsonify(
            {
                "status": "success",
                "detection_ids": result.detection_ids,
                "image_filenames": result.image_filenames,
                "total_count": result.total_count,
            }
        )

    if mode == "date_range":
        try:
            from_date = _parse_iso_date(data.get("from_date"), "from_date")
            to_date = _parse_iso_date(data.get("to_date"), "to_date")
        except ValueError as exc:
            # ValueError messages from _parse_iso_date are authored-here
            # constants, not arbitrary exc text — surface a generic
            # public message anyway to keep the contract uniform.
            logger.info(
                "moderation/resolve-selection date_range rejected [%s]",
                type(exc).__name__,
                exc_info=True,
            )
            return jsonify(
                {"status": "error", "message": "Invalid from_date or to_date"}
            ), 400

        if from_date > to_date:
            return jsonify(
                {
                    "status": "error",
                    "message": "from_date must be on or before to_date",
                }
            ), 400

        with db_service.closing_connection() as conn:
            selection = db_service.fetch_trash_candidate_selection_in_date_range(
                conn, from_date, to_date
            )

        return jsonify(
            {
                "status": "success",
                "detection_ids": selection.get("detection_ids", []),
                "image_filenames": selection.get("image_filenames", []),
                "total_count": len(selection.get("detection_ids", []))
                + int(selection.get("orphan_count", 0) or 0),
                "detection_count": len(selection.get("detection_ids", [])),
                "image_count": selection.get("image_count", 0),
                "orphan_count": int(selection.get("orphan_count", 0) or 0),
            }
        )

    return jsonify({"status": "error", "message": f"Unknown mode: {mode}"}), 400


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# POST /api/moderation/bulk/download
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/bulk/download", methods=["POST"])
@login_required
def bulk_download():
    """Return a ZIP of the originals behind the given detections.

    Date-independent, unlike the Gallery edit page's download: the Species
    view selects across days, so the archive is named for the request date
    rather than the frames' date.

    Accepts:
        { detection_ids: [int, ...] }
    """
    data = request.get_json() or {}
    raw_ids = data.get("detection_ids") or data.get("ids") or []

    try:
        detection_ids = [int(i) for i in raw_ids]
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid detection ids"}), 400

    if not detection_ids:
        return jsonify({"status": "error", "message": "No targets provided"}), 400

    archive = image_download_service.build_download_archive(detection_ids)
    if archive is None:
        return jsonify({"status": "error", "message": "No downloadable images"}), 404

    stamp = datetime.now().strftime("%Y%m%d")
    return send_file(
        archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"watchmybirds_selection_{stamp}.zip",
    )
