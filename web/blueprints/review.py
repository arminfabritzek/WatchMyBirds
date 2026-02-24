"""
Review Blueprint.

Handles review queue routes:
- GET /admin/review - Review queue page (was orphans)
- GET /api/review-thumb/<filename> - On-demand thumbnail
- POST /api/review/decision - Review decisions (confirm/no_bird/skip)
"""

from datetime import datetime

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from config import get_config
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import db_service, detections_service, gallery_service

logger = get_logger(__name__)
config = get_config()

review_bp = Blueprint("review", __name__)


@review_bp.route("/admin/review", methods=["GET"])
@login_required
def review_page():
    """
    Review Queue: Images needing user decision.
    Shows orphans (no detections) AND low-confidence detections.
    Sorted oldest first.
    """
    output_dir = config.get("OUTPUT_DIR", "output")
    gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

    rows = detections_service.get_review_queue_images(output_dir, gallery_threshold)

    orphans = []
    for row in rows:
        filename = row["filename"]
        timestamp = row["timestamp"] or ""
        review_reason = row["review_reason"]  # 'orphan' or 'low_score'
        max_score = row["max_score"]

        # Format date/time from timestamp (YYYYMMDD_HHMMSS)
        formatted_date = ""
        if len(timestamp) >= 15:
            try:
                dt = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_date = timestamp[:15]

        # Resolve paths using service
        original_path = gallery_service.get_image_paths(output_dir, filename)[
            "original"
        ]

        # Get file size (original)
        file_size = 0
        if original_path.exists():
            file_size = original_path.stat().st_size

        # Format file size
        if file_size >= 1024 * 1024:
            file_size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size >= 1024:
            file_size_str = f"{file_size / 1024:.1f} KB"
        else:
            file_size_str = f"{file_size} B"

        # Use on-demand orphan thumbnail endpoint
        thumb_url = f"/api/review-thumb/{filename}"

        # Construct Full URL for Lightbox
        full_url = ""
        if len(timestamp) >= 8:
            date_folder_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
            full_url = f"/uploads/originals/{date_folder_str}/{filename}"

        # Badge label for review reason
        if review_reason == "orphan":
            reason_label = "No Detection"
        else:
            score_pct = round((max_score or 0) * 100)
            reason_label = f"Low Score ({score_pct}%)"

        orphans.append(
            {
                "filename": filename,
                "timestamp": timestamp,
                "formatted_date": formatted_date,
                "file_size": file_size,
                "file_size_str": file_size_str,
                "thumb_url": thumb_url,
                "full_url": full_url,
                "review_reason": review_reason,
                "reason_label": reason_label,
                "max_score": max_score,
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_w": row["bbox_w"],
                "bbox_h": row["bbox_h"],
            }
        )

    return render_template(
        "orphans.html", orphans=orphans, current_path="/admin/review"
    )


@review_bp.route("/api/review-thumb/<filename>", methods=["GET"])
@login_required
def review_thumb(filename):
    """On-demand thumbnail generation for orphan images."""
    output_dir = config.get("OUTPUT_DIR", "output")
    paths = gallery_service.get_image_paths(output_dir, filename)

    original_path = paths["original"]
    preview_path = paths["preview"]

    # If preview already cached, serve it
    if preview_path.exists():
        return send_file(str(preview_path), mimetype="image/webp")

    # Original must exist to generate preview
    if not original_path.exists():
        abort(404)

    # Generate preview thumbnail via service
    success = gallery_service.generate_preview_thumbnail(
        original_path, preview_path, size=256
    )

    if success and preview_path.exists():
        return send_file(str(preview_path), mimetype="image/webp")
    else:
        abort(500)


@review_bp.route("/api/review/decision", methods=["POST"])
@login_required
def review_decision():
    """
    API endpoint for Review Queue decisions.
    POST /api/review/decision
    Payload: { filenames: [...], action: "confirm" | "no_bird" | "skip" }

    - confirm -> review_status = 'confirmed_bird'
    - no_bird -> review_status = 'no_bird' (soft-trash, no file deletion)
    - skip -> no change

    Only updates images with review_status = 'untagged' (no way back).
    """
    try:
        data = request.get_json() or {}
        filenames = data.get("filenames", [])
        action = data.get("action", "")

        if not filenames:
            return (
                jsonify({"status": "error", "message": "No filenames provided"}),
                400,
            )

        if action not in ("confirm", "no_bird", "skip"):
            return (
                jsonify({"status": "error", "message": f"Invalid action: {action}"}),
                400,
            )

        # Skip action: no database change
        if action == "skip":
            return jsonify({"status": "success", "updated": 0, "action": "skip"})

        # Map action to review_status
        status_map = {"confirm": "confirmed_bird", "no_bird": "no_bird"}
        new_status = status_map[action]

        conn = db_service.get_connection()
        try:
            if action == "confirm":
                # Confirming "Bird Present" requires an existing detection.
                # Otherwise the image becomes non-eligible for Deep Scan and can get stuck.
                placeholders = ",".join("?" for _ in filenames)
                rows = conn.execute(
                    f"""
                    SELECT
                        i.filename,
                        EXISTS(
                            SELECT 1
                            FROM detections d
                            WHERE d.image_filename = i.filename
                        ) AS has_detections
                    FROM images i
                    WHERE i.filename IN ({placeholders})
                    """,
                    filenames,
                ).fetchall()

                missing = [row["filename"] for row in rows if not row["has_detections"]]
                if missing:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Cannot confirm Bird Present for items without detections. Use Deep Scan first.",
                                "filenames": missing,
                            }
                        ),
                        409,
                    )

            updated = db_service.update_review_status(conn, filenames, new_status)
        finally:
            conn.close()

        logger.info(f"Review decision: {action} -> {updated} images updated")
        return jsonify({"status": "success", "updated": updated, "action": action})

    except Exception as e:
        logger.error(f"Error in review decision: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/analyze/<filename>", methods=["POST"])
@login_required
def analyze_review_item(filename):
    """
    Triggers a deep analysis for the given file.
    Query params:
      force=1  — bypass no-hit DB exclusion (re-scan already-scanned images)
    """
    try:
        from web.services.analysis_service import (
            check_deep_analysis_eligibility,
            submit_analysis_job,
        )

        force = request.args.get("force", "0") in ("1", "true", "yes")

        is_eligible, reason = check_deep_analysis_eligibility(filename, force=force)
        if not is_eligible:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": reason or "Image is not eligible for deep analysis.",
                    }
                ),
                409,
            )

        if submit_analysis_job(filename, force=force):
            return jsonify({"status": "success", "message": "Deep analysis queued"})

        return jsonify({"status": "error", "message": "Failed to queue analysis"}), 500

    except Exception as e:
        logger.error(f"Error triggering analysis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
