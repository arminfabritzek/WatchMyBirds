"""
Trash Blueprint.

Handles all trash-related routes:
- GET /trash - Trash page view
- POST /api/trash/restore - Restore trashed items
- POST /api/trash/purge - Permanently delete specific items
- POST /api/trash/empty - Empty entire trash
- POST /api/detections/reject - Reject detections (move to trash)
- POST /api/detections/relabel - Relabel a detection's species
- POST /api/detections/rate - Set manual rating for a detection
- GET /api/species-list - List known species
"""

import math

from flask import Blueprint, jsonify, render_template, request

from config import get_config
from logging_config import get_logger
from utils.review_metadata import (
    REVIEW_STATUS_CONFIRMED_BIRD,
    format_review_timestamp,
)
from core import gallery_core as gallery_service
from core.species_colours import assign_species_colours as _assign_species_colours
from utils.species_names import (
    UNKNOWN_SPECIES_KEY,
    build_species_picker_entries,
    load_common_names,
)
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

        conn = db_service.get_connection()
        try:
            # Restore detections
            if detection_ids:
                db_service.restore_detections(conn, detection_ids)
                restored_count += len(detection_ids)

            # Restore no_bird images (back to untagged)
            if image_filenames:
                restored_count += db_service.restore_no_bird_images(
                    conn, image_filenames
                )
        finally:
            conn.close()

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

        conn = db_service.get_connection()
        try:
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

        conn = db_service.get_connection()
        try:
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
        finally:
            conn.close()

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
    conn = db_service.get_connection()
    try:
        db_service.reject_detections(conn, ids)
    finally:
        conn.close()
    return jsonify({"status": "success"})


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
    except Exception as e:
        logger.error(f"Failed to load species list: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@trash_bp.route("/api/detections/relabel", methods=["POST"])
@login_required
def relabel_detection():
    """
    Relabels a detection to a different species.
    Accepts: { detection_id: int, species: "Scientific_name" }
    Updates the final/manual species override on the detection only.
    """
    data = request.get_json() or {}
    detection_id = data.get("detection_id")
    new_species = data.get("species")

    if not detection_id or not new_species:
        return jsonify({"error": "detection_id and species required"}), 400

    conn = db_service.get_connection()
    try:
        cfg = get_config()
        locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
        allowed_species = {
            entry["scientific"]
            for entry in build_species_picker_entries(conn, locale=locale)
        }
        if new_species not in allowed_species:
            return jsonify({"error": "unknown species"}), 400

        db_service.apply_species_override(conn, detection_id, new_species, "manual")

        # Mark this detection as human-confirmed
        conn.execute(
            """
            UPDATE detections
            SET decision_state = 'confirmed'
            WHERE detection_id = ?
              AND COALESCE(status, 'active') = 'active'
            """,
            (detection_id,),
        )

        # If all active detections on the same image are now resolved,
        # promote the image review_status to confirmed_bird
        filename = conn.execute(
            "SELECT image_filename FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
        if filename:
            filename = filename[0]
            unresolved = conn.execute(
                """
                SELECT COUNT(*)
                FROM detections d
                WHERE d.image_filename = ?
                  AND COALESCE(d.status, 'active') = 'active'
                  AND COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
                """,
                (filename,),
            ).fetchone()[0]
            if unresolved == 0:
                db_service.update_review_status(
                    conn, [filename], REVIEW_STATUS_CONFIRMED_BIRD
                )
        conn.commit()

        gallery_service.invalidate_cache()
        logger.info(f"Detection {detection_id} relabeled to {new_species}")
        return jsonify({"status": "success", "new_species": new_species})
    except Exception as e:
        logger.error(f"Error relabeling detection {detection_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


def compute_auto_rating(od_confidence, cls_confidence, bbox_w, bbox_h):
    """
    Compute automatic detection quality rating (1-5).
    5 = Audio+Visual match (gold), 4 = excellent, 3 = good, 2 = uncertain, 1 = poor.

    NOTE: Legacy function kept for backward compatibility. The UI now uses
    the simpler is_favorite toggle for cover image selection.
    """
    od_conf = od_confidence or 0
    cls_conf = cls_confidence or 0
    bbox_area = (bbox_w or 0) * (bbox_h or 0)

    # Visual quality score
    visual_score = od_conf * 0.4 + cls_conf * 0.6

    # BBox size bonus/penalty
    if bbox_area > 0.05:
        visual_score += 0.1
    elif bbox_area < 0.005:
        visual_score -= 0.15

    # Map to 1-4
    if visual_score >= 0.65:
        return 4
    elif visual_score >= 0.45:
        return 3
    elif visual_score >= 0.25:
        return 2
    else:
        return 1


@trash_bp.route("/api/detections/rate", methods=["POST"])
@login_required
def rate_detection():
    """
    Set a manual rating for a detection.
    Accepts: { detection_id: int, rating: int (1-5) }

    NOTE: Legacy endpoint kept for backward compatibility.
    The UI now uses /api/detections/favorite instead.
    """
    data = request.get_json() or {}
    detection_id = data.get("detection_id")
    rating = data.get("rating")

    if not detection_id or rating is None:
        return jsonify({"error": "detection_id and rating required"}), 400

    rating = int(rating)
    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be 1-5"}), 400

    conn = db_service.get_connection()
    try:
        conn.execute(
            "UPDATE detections SET rating = ?, rating_source = 'manual' WHERE detection_id = ?",
            (rating, detection_id),
        )
        conn.commit()
        # Invalidate gallery cache so rating-based sorting updates immediately
        try:
            from web.web_interface import _cached_images

            _cached_images["images"] = None
        except Exception:
            pass
        logger.info(f"Detection {detection_id} rated {rating}/5 (manual)")
        return jsonify({"status": "success", "rating": rating})
    except Exception as e:
        logger.error(f"Error rating detection {detection_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


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
        # Read current state
        row = conn.execute(
            "SELECT COALESCE(is_favorite, 0) as is_favorite FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()

        if not row:
            return jsonify({"error": "Detection not found"}), 404

        new_state = 0 if row["is_favorite"] else 1
        conn.execute(
            "UPDATE detections SET is_favorite = ? WHERE detection_id = ?",
            (new_state, detection_id),
        )
        conn.commit()

        # Invalidate gallery cache
        try:
            from web.web_interface import _cached_images

            _cached_images["images"] = None
        except Exception:
            pass

        logger.info(f"Detection {detection_id} favorite={'on' if new_state else 'off'}")
        return jsonify({"status": "success", "is_favorite": bool(new_state)})
    except Exception as e:
        logger.error(f"Error toggling favorite for detection {detection_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()
