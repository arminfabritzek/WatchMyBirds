"""
Trash Blueprint.

Handles all trash-related routes:
- GET /trash - Trash page view
- POST /api/trash/restore - Restore trashed items
- POST /api/trash/purge - Permanently delete specific items
- POST /api/trash/empty - Empty entire trash
- POST /api/detections/reject - Reject detections (move to trash)
"""

import math
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import db_service, detections_service

logger = get_logger(__name__)

# Blueprint with url_prefix for API routes
trash_bp = Blueprint("trash", __name__)

# Image width for template (matches web_interface.py)
IMAGE_WIDTH = 450


@trash_bp.route("/trash", methods=["GET"])
@login_required
def trash_page():
    """Trash page showing rejected detections and no_bird images."""
    page = request.args.get("page", 1, type=int)
    limit = 50

    with db_service.get_connection() as conn:
        items, total_count = db_service.fetch_trash_items(conn, page=page, limit=limit)

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

            common_name = (
                item.get("cls_class_name") or item.get("od_class_name") or "Unknown"
            )
        else:
            # Image (no_bird): use on-demand review thumbnail
            filename = item.get("filename", "")
            display_path = f"/api/review-thumb/{filename}"
            common_name = "No Bird"  # Label for no-bird images

        # Format timestamp
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_time = ts if ts else "Unknown"

        processed_items.append(
            {
                "trash_type": trash_type,
                "item_id": item.get(
                    "item_id"
                ),  # Unified ID (detection_id str or filename)
                "detection_id": item.get("detection_id"),  # Only for detections
                "filename": item.get("filename"),  # For images
                "display_path": display_path,
                "common_name": common_name,
                "formatted_time": formatted_time,
            }
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
        image_filenames = data.get("image_filenames", [])

        restored_count = 0

        with db_service.get_connection() as conn:
            # Restore detections
            if detection_ids:
                db_service.restore_detections(conn, detection_ids)
                restored_count += len(detection_ids)

            # Restore no_bird images (back to untagged)
            if image_filenames:
                restored_count += db_service.restore_no_bird_images(
                    conn, image_filenames
                )

        return jsonify({"status": "success", "result": {"restored": restored_count}})
    except Exception as e:
        logger.error(f"Error restoring trash: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@trash_bp.route("/api/trash/purge", methods=["POST"])
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
        image_filenames = data.get("image_filenames", [])

        det_deleted = 0
        img_deleted = 0
        files_deleted = 0

        with db_service.get_connection() as conn:
            # Purge detections
            if detection_ids:
                result = detections_service.hard_delete_detections_with_conn(
                    conn, detection_ids=detection_ids
                )
                det_deleted = result.get("rows_deleted", 0)
                files_deleted = result.get("files_deleted", 0)

            # Purge no_bird images (with full file cleanup)
            if image_filenames:
                img_result = detections_service.hard_delete_images_with_conn(
                    conn, filenames=image_filenames
                )
                img_deleted = img_result.get("rows_deleted", 0)
                files_deleted += img_result.get("files_deleted", 0)

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
    except Exception as e:
        logger.error(f"Error purging trash: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@trash_bp.route("/api/trash/empty", methods=["POST"])
@login_required
def trash_empty():
    """Empties entire trash (detections + no_bird images)."""
    try:
        det_deleted = 0
        img_deleted = 0
        files_deleted = 0

        with db_service.get_connection() as conn:
            # Empty rejected detections
            result = detections_service.hard_delete_detections_with_conn(
                conn, before_date="2099-12-31"
            )
            det_deleted = result.get("rows_deleted", 0)
            files_deleted = result.get("files_deleted", 0)

            # Empty no_bird images (with full file cleanup)
            img_result = detections_service.hard_delete_images_with_conn(
                conn, delete_all=True
            )
            img_deleted = img_result.get("rows_deleted", 0)
            files_deleted += img_result.get("files_deleted", 0)

        logger.info(
            f"Trash emptied: {det_deleted} detections, {img_deleted} images, {files_deleted} files deleted"
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
    except Exception as e:
        logger.error(f"Error emptying trash: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@trash_bp.route("/api/detections/reject", methods=["POST"])
@login_required
def reject_detection():
    """Rejects detections (moves them to trash)."""
    data = request.get_json() or {}
    ids = data.get("ids", [])
    if not ids:
        return jsonify({"error": "No IDs provided"}), 400
    with db_service.get_connection() as conn:
        db_service.reject_detections(conn, ids)
    return jsonify({"status": "success"})
