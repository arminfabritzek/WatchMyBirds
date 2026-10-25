"""
Trash Blueprint.

Handles all trash-related routes:
- GET /trash - Trash page view
- POST /api/trash/restore - Restore trashed items
- POST /api/trash/purge - Permanently delete specific items
- POST /api/trash/empty - Empty entire trash
- POST /api/trash/move - Move selected objects to trash
- GET /api/species-list - List known species
"""

import math
from datetime import UTC, datetime

from flask import Blueprint, jsonify, render_template, request

from config import get_config
from core import gallery_core as gallery_service
from core.species_colours import assign_species_colours as _assign_species_colours
from logging_config import get_logger
from utils.review_metadata import format_review_timestamp
from utils.species_names import (
    UNKNOWN_SPECIES_KEY,
    build_species_picker_entries,
    load_common_names,
)
from web.blueprints.auth import login_required
from web.security import error_response
from web.security import safe_log_value as _slv
from web.services import (
    cache_service,
    db_service,
    detections_service,
)

logger = get_logger(__name__)

# Blueprint with url_prefix for API routes
trash_bp = Blueprint("trash", __name__)

# Image width for template (matches web_interface.py)
IMAGE_WIDTH = 450


@trash_bp.route("/api/trash/move", methods=["POST"])
@cache_service.invalidates("analytics.")
@login_required
def move_to_trash():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(status="error", message="invalid request"), 400
    try:
        with db_service.closing_connection() as conn:
            result = detections_service.move_to_trash(conn, data)
    except ValueError as exc:
        return jsonify(status="error", message=str(exc)), 409
    gallery_service.invalidate_cache()
    return jsonify(status="success", **result)


@trash_bp.route("/trash", methods=["GET"])
@login_required
def trash_page():
    """Trash page showing rejected detections (no_bird images are preserved as hard-negatives, not shown here)."""
    page = request.args.get("page", 1, type=int)
    limit = 50
    cfg = get_config()
    locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    names = load_common_names(locale)

    conn = db_service.get_connection()
    try:
        items, total_count = db_service.fetch_trash_items(conn, page=page, limit=limit)
    finally:
        conn.close()

    processed_items = []
    for item in items:
        ts = item.get("image_timestamp", "")
        trash_type = item.get("trash_type", "detection")

        # Handle display path based on trash type
        if trash_type == "detection":
            # Detection: use thumbnail or optimized image
            full_path = item.get("relative_path") or item.get("image_optimized", "")
            thumb_virtual = item.get("thumbnail_path_virtual")

            if thumb_virtual:
                display_path = f"/uploads/derivatives/thumbs/{thumb_virtual}"
            else:
                display_path = f"/uploads/derivatives/optimized/{full_path}"

            species_key = item.get("species_key") or UNKNOWN_SPECIES_KEY
            common_name = names.get(species_key, species_key.replace("_", " "))
        else:
            # Image (no_bird): use on-demand review thumbnail
            filename = item.get("filename", "")
            display_path = f"/api/review-thumb/{filename}"
            common_name = "No Bird"  # Label for no-bird images

        formatted_time = format_review_timestamp(ts) or "Unknown"

        processed_items.append(
            {
                "trash_type": trash_type,
                "item_id": item.get(
                    "item_id"
                ),  # Unified ID (detection_id str or filename)
                "detection_id": item.get("detection_id"),  # Only for detections
                "filename": item.get("filename"),  # For images
                "species_key": item.get("species_key"),
                "display_path": display_path,
                "common_name": common_name,
                "formatted_time": formatted_time,
            }
        )

    # Stamp species_colour so the trash template can apply the Wong
    # palette token to each tile.
    _trash_colour_map = _assign_species_colours(
        [item.get("species_key") or "" for item in processed_items]
    )
    for item in processed_items:
        item["species_colour"] = _trash_colour_map.get(
            item.get("species_key") or "", None
        )

    total_pages = math.ceil(total_count / limit) if limit > 0 else 1

    return render_template(
        "trash.html",
        items=processed_items,
        page=page,
        total_pages=total_pages,
        total_items=total_count,
        image_width=IMAGE_WIDTH,
    )


@trash_bp.route("/api/trash/restore", methods=["POST"])
@cache_service.invalidates("analytics.")
@login_required
def trash_restore():
    """
    Restores trashed items back to their original state.
    Accepts: { detection_ids: [...], image_filenames: [...] }
    OR legacy: { ids: [...] } (treated as detection_ids)
    """
    try:
        data = request.get_json() or {}

        # Support both new format and legacy format
        detection_ids = data.get("detection_ids", data.get("ids", []))
        if data.get("image_filenames"):
            return jsonify(status="error", message="Select trashed objects"), 400

        restored_count = 0

        conn = db_service.get_connection()
        try:
            # Restore detections
            if detection_ids:
                db_service.restore_detections(conn, detection_ids)
                restored_count += len(detection_ids)

        finally:
            conn.close()

        return jsonify({"status": "success", "result": {"restored": restored_count}})
    except Exception as exc:
        return error_response("Error restoring trash", exc)


@trash_bp.route("/api/trash/purge", methods=["POST"])
@cache_service.invalidates("analytics.")
@login_required
def trash_purge():
    """
    Permanently deletes trashed items.
    Accepts: { detection_ids: [...], image_filenames: [...] }
    OR legacy: { ids: [...] } (treated as detection_ids)
    """
    try:
        data = request.get_json() or {}

        # Support both new format and legacy format
        detection_ids = data.get("detection_ids", data.get("ids", []))
        if data.get("image_filenames"):
            return jsonify(status="error", message="Select trashed objects"), 400

        det_deleted = 0
        img_deleted = 0
        files_deleted = 0

        conn = db_service.get_connection()
        try:
            # Purge detections
            if detection_ids:
                result = detections_service.hard_delete_detections_with_conn(
                    conn, detection_ids=detection_ids
                )
                det_deleted = result.get("rows_deleted", 0)
                files_deleted = result.get("files_deleted", 0)

        finally:
            conn.close()

        logger.info(
            f"Trash purge: {det_deleted} detections, {img_deleted} images, {files_deleted} files deleted"
        )
        return jsonify(
            {
                "status": "success",
                "result": {
                    "purged": True,
                    "rows_deleted": det_deleted + img_deleted,
                    "det_deleted": det_deleted,
                    "img_deleted": img_deleted,
                    "files_deleted": files_deleted,
                },
            }
        )
    except Exception as exc:
        return error_response("Error purging trash", exc)


@trash_bp.route("/api/trash/empty", methods=["POST"])
@cache_service.invalidates("analytics.")
@login_required
def trash_empty():
    """Delete objects deliberately moved to trash after answer migration."""
    try:
        det_deleted = 0
        files_deleted = 0

        conn = db_service.get_connection()
        try:
            result = detections_service.hard_delete_detections_with_conn(
                conn, before_date="2099-12-31"
            )
            det_deleted = result.get("rows_deleted", 0)
            files_deleted = result.get("files_deleted", 0)
        finally:
            conn.close()

        logger.info(
            f"Trash emptied: {det_deleted} detections, {files_deleted} files deleted "
            "(no_bird images preserved as hard-negatives)"
        )
        return jsonify(
            {
                "status": "success",
                "result": {
                    "purged": True,
                    "rows_deleted": det_deleted,
                    "det_deleted": det_deleted,
                    "img_deleted": 0,
                    "files_deleted": files_deleted,
                },
            }
        )
    except Exception as exc:
        return error_response("Error emptying trash", exc)


@trash_bp.route("/api/species-list", methods=["GET"])
def species_list():
    """Returns the list of known species for relabeling (locale-aware)."""
    try:
        cfg = get_config()
        locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
        detection_id = request.args.get("detection_id", type=int)
        conn = db_service.get_connection()
        try:
            species = build_species_picker_entries(
                conn,
                locale=locale,
                detection_id=detection_id,
            )
        finally:
            conn.close()
        return jsonify({"status": "success", "species": species})
    except Exception as exc:
        return error_response("Failed to load species list", exc)


@trash_bp.route("/api/detections/favorite", methods=["POST"])
@login_required
def toggle_favorite():
    """
    Toggle favorite status for a detection (❤️ on/off).
    Accepts: { detection_id: int }
    Returns the new is_favorite state.
    """
    data = request.get_json() or {}
    detection_id = data.get("detection_id")

    if not detection_id:
        return jsonify({"error": "detection_id required"}), 400

    conn = db_service.get_connection()
    try:
        # Serialize the read-toggle-write sequence with retention's final
        # protection-check/delete transaction. Whichever request obtains the
        # write lock first completes its decision before the other proceeds.
        conn.execute("BEGIN IMMEDIATE")
        # Read current state
        row = conn.execute(
            "SELECT COALESCE(is_favorite, 0) as is_favorite FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()

        if not row:
            return jsonify({"error": "Detection not found"}), 404

        new_state = 0 if row["is_favorite"] else 1
        # Stamp rating_source='manual' so the legacy backfill (which migrates
        # rating_source='auto' rows into is_gallery_eligible) can never
        # mistake a manual heart-click for a stale auto-tag on the next app
        # start. is_favorite is the gold-label column and the backfill must
        # leave manual rows alone.
        action_at = datetime.now(UTC).isoformat()
        conn.execute(
            """
            UPDATE detections
            SET is_favorite = ?,
                rating_source = 'manual',
                species_updated_at = ?
            WHERE detection_id = ?
            """,
            (new_state, action_at, detection_id),
        )
        conn.commit()
        gallery_service.invalidate_cache()

        logger.info(
            "Detection %s favorite=%s",
            _slv(detection_id),
            "on" if new_state else "off",
        )
        return jsonify({"status": "success", "is_favorite": bool(new_state)})
    except Exception as exc:
        return error_response(
            f"Error toggling favorite for detection {_slv(detection_id)}", exc
        )
    finally:
        conn.close()
