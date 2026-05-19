"""
Unclear Blueprint.

Surface for classifier-rejected, smoother-confirmed detections —
the "shadow pile" that is otherwise invisible to Gallery / Review /
Trash. Per-day grouping with two bulk actions:

- **Confirm Day** — promote every reject detection of that day to a
  species-confirmed Gallery row, using the classifier's own
  ``raw_species_name`` as the manual override. Items move to Gallery.
- **Discard Day** — soft-reject every reject detection of that day.
  Items move to Trash for the regular hard-delete pipeline.

Routes:
- GET  /unclear                       — page view
- POST /api/unclear/confirm-day       — bulk confirm one day
- POST /api/unclear/discard-day       — bulk discard (soft-reject) one day
"""

from flask import Blueprint, jsonify, render_template, request

from config import get_config
from core import gallery_core as gallery_service
from logging_config import get_logger
from utils.species_names import load_common_names
from web.blueprints.auth import login_required
from web.security import safe_log_value as _slv
from web.services import db_service

logger = get_logger(__name__)

unclear_bp = Blueprint("unclear", __name__)


@unclear_bp.route("/unclear", methods=["GET"])
@login_required
def unclear_page():
    """Unclear page — per-day cards for classifier-rejected detections."""
    cfg = get_config()
    locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    names = load_common_names(locale)

    conn = db_service.get_connection()
    try:
        days = db_service.fetch_unclear_days(conn, sample_limit=9)
        total = db_service.fetch_unclear_total(conn)
    finally:
        conn.close()

    # Decorate each day's species breakdown + samples with the locale
    # common name. Keeps the template free of name-lookup logic.
    for day in days:
        for entry in day["species_breakdown"]:
            sci = entry["raw_species_name"]
            entry["common_name"] = names.get(sci, sci.replace("_", " "))
        for sample in day["samples"]:
            sci = sample["raw_species_name"]
            sample["common_name"] = names.get(sci, sci.replace("_", " "))
            sample["display_path"] = (
                f"/uploads/derivatives/thumbs/{sample['thumbnail_path_virtual']}"
            )

    return render_template(
        "unclear.html",
        days=days,
        total_count=total,
    )


@unclear_bp.route("/api/unclear/confirm-day", methods=["POST"])
@login_required
def unclear_confirm_day():
    """Bulk confirm every Unclear detection of one ISO day.

    Expects JSON: ``{"day": "YYYY-MM-DD"}``

    Each detection has its classifier suggestion (``raw_species_name``)
    promoted to ``manual_species_override`` and the visibility flags
    flipped to ``decision_state='confirmed' / decision_level='species'``
    so the Gallery filter starts including them.
    """
    data = request.get_json() or {}
    day = data.get("day")
    if not day:
        return jsonify({"status": "error", "message": "day required"}), 400

    conn = db_service.get_connection()
    try:
        detection_ids = db_service.fetch_unclear_detection_ids_for_day(conn, day)
        if not detection_ids:
            return jsonify(
                {"status": "success", "confirmed": 0, "day": day}
            )

        confirmed = db_service.confirm_unclear_detections(
            conn,
            detection_ids,
            source="manual_bulk_confirm_day",
        )
    finally:
        conn.close()

    gallery_service.invalidate_cache()
    logger.info(
        "Unclear: confirmed %s detections for day %s",
        _slv(confirmed),
        _slv(day),
    )
    return jsonify({"status": "success", "confirmed": confirmed, "day": day})


@unclear_bp.route("/api/unclear/discard-day", methods=["POST"])
@login_required
def unclear_discard_day():
    """Bulk discard (soft-reject) every Unclear detection of one ISO day.

    Expects JSON: ``{"day": "YYYY-MM-DD"}``

    Items move to Trash for the regular Empty-Trash hard-delete
    pipeline. Files stay on disk until purge — matches the established
    soft-then-hard pattern in Gallery / Review.
    """
    data = request.get_json() or {}
    day = data.get("day")
    if not day:
        return jsonify({"status": "error", "message": "day required"}), 400

    conn = db_service.get_connection()
    try:
        detection_ids = db_service.fetch_unclear_detection_ids_for_day(conn, day)
        if not detection_ids:
            return jsonify(
                {"status": "success", "discarded": 0, "day": day}
            )

        db_service.reject_detections(conn, detection_ids)
    finally:
        conn.close()

    gallery_service.invalidate_cache()
    logger.info(
        "Unclear: discarded %s detections for day %s",
        _slv(len(detection_ids)),
        _slv(day),
    )
    return jsonify(
        {
            "status": "success",
            "discarded": len(detection_ids),
            "day": day,
        }
    )
