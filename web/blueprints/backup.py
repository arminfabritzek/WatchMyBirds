"""
Backup Blueprint.

Handles backup and restore routes:
- GET /backup - Backup page
- GET /api/backup/stats - Backup statistics
- POST /api/backup/create - Create and stream backup archive
- GET /restore - Restore page
- POST /api/restore/upload - Upload backup archive
- POST /api/restore/analyze - Analyze uploaded archive
- POST /api/restore/apply - Apply restore
- GET /api/restore/status - Restore progress
- POST /api/restore/cleanup - Cleanup temp files
"""

import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import backup_restore_service, path_service

logger = get_logger(__name__)

backup_bp = Blueprint("backup", __name__)

# Detection manager reference - will be set by init function
_detection_manager = None

# Global restore progress tracking
_restore_progress = {"active": False, "progress": None}


def init_backup_bp(detection_manager):
    """Initialize backup blueprint with detection manager reference."""
    global _detection_manager
    _detection_manager = detection_manager


@backup_bp.route("/backup")
@login_required
def backup_page():
    return render_template("backup.html")


@backup_bp.route("/api/backup/stats", methods=["GET"])
@login_required
def backup_stats():
    """Returns statistics about data available for backup."""
    try:
        stats = backup_restore_service.get_backup_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Backup stats error: {e}")
        return jsonify({"error": str(e)}), 500


@backup_bp.route("/api/backup/create", methods=["POST"])
@login_required
def backup_create():
    """
    Create and stream backup archive.
    Policy: Detection is automatically paused during backup.
    """
    try:
        # Parse options
        data = request.get_json(silent=True) or {}
        include_db = data.get("include_db", True)
        include_originals = data.get("include_originals", True)
        include_derivatives = data.get("include_derivatives", False)
        include_settings = data.get("include_settings", True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"watchmybirds_backup_{timestamp}.tar.gz"

        def generate_with_pause():
            """Generator that auto-pauses detection during backup streaming."""
            was_paused = _detection_manager.paused

            try:
                if not was_paused:
                    logger.info("Backup: Pausing detection for backup...")
                    _detection_manager.paused = True
                    time.sleep(1)

                yield from backup_restore_service.stream_backup(
                    include_db=include_db,
                    include_originals=include_originals,
                    include_derivatives=include_derivatives,
                    include_settings=include_settings,
                )
            finally:
                if not was_paused:
                    logger.info("Backup: Resuming detection after backup.")
                    _detection_manager.paused = False

        response = Response(
            generate_with_pause(),
            mimetype="application/gzip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache",
            },
        )
        return response

    except Exception as e:
        logger.error(f"Backup create error: {e}")
        return jsonify({"error": str(e)}), 500


@backup_bp.route("/restore")
@login_required
def restore_page():
    """Serves the restore/import page."""
    return render_template("restore.html")


@backup_bp.route("/api/restore/upload", methods=["POST"])
@login_required
def restore_upload():
    """
    Uploads a backup archive for restore.
    Validates extension, magic header, and max size.
    """
    MAX_ARCHIVE_SIZE_BYTES = backup_restore_service.MAX_ARCHIVE_SIZE_BYTES

    try:
        if backup_restore_service.is_restore_active():
            return (
                jsonify({"error": "Another restore operation is in progress"}),
                409,
            )

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        if not filename.endswith(".tar.gz") and not filename.endswith(".tgz"):
            return (
                jsonify({"error": "Invalid file type. Only .tar.gz archives allowed."}),
                400,
            )

        pm = path_service.get_path_manager()
        upload_path = pm.get_restore_upload_path(filename)

        total_size = 0
        chunk_size = 8192

        with open(upload_path, "wb") as f:
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_ARCHIVE_SIZE_BYTES:
                    f.close()
                    upload_path.unlink(missing_ok=True)
                    return (
                        jsonify(
                            {
                                "error": f"File too large. Maximum size: "
                                f"{MAX_ARCHIVE_SIZE_BYTES // (1024**3)} GB"
                            }
                        ),
                        413,
                    )
                f.write(chunk)

        with open(upload_path, "rb") as f:
            magic = f.read(2)
            if magic != b"\x1f\x8b":
                upload_path.unlink(missing_ok=True)
                return (
                    jsonify(
                        {"error": "Invalid archive format (not a valid gzip file)"}
                    ),
                    400,
                )

        logger.info(f"Restore: Uploaded archive {filename} ({total_size} bytes)")

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "path": str(upload_path),
                "size_bytes": total_size,
            }
        )

    except Exception as e:
        logger.error(f"Restore upload error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@backup_bp.route("/api/restore/analyze", methods=["POST"])
@login_required
def restore_analyze():
    """
    Analyzes an uploaded backup archive without extracting.
    """
    try:
        data = request.get_json(silent=True) or {}
        archive_path = data.get("path")

        if not archive_path:
            return jsonify({"error": "No archive path provided"}), 400

        path = Path(archive_path)
        if not path.exists():
            return jsonify({"error": "Archive file not found"}), 404

        analysis = backup_restore_service.analyze_archive(path)

        return jsonify(
            {
                "status": "success",
                "analysis": analysis,
            }
        )

    except Exception as e:
        logger.error(f"Restore analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@backup_bp.route("/api/restore/apply", methods=["POST"])
@login_required
def restore_apply():
    """
    Starts the restore process in background.
    """
    global _restore_progress

    try:
        if backup_restore_service.is_restore_active() or _restore_progress["active"]:
            return (
                jsonify({"error": "Another restore operation is in progress"}),
                409,
            )

        was_paused = _detection_manager.paused
        if not was_paused:
            logger.info("Restore: Auto-pausing detection")
            _detection_manager.paused = True

        data = request.get_json(silent=True) or {}
        archive_path = data.get("path")

        if not archive_path:
            if not was_paused:
                _detection_manager.paused = False
            return jsonify({"error": "No archive path provided"}), 400

        path = Path(archive_path)
        if not path.exists():
            if not was_paused:
                _detection_manager.paused = False
            return jsonify({"error": "Archive file not found"}), 404

        analysis = backup_restore_service.analyze_archive(path)
        if analysis["blockers"]:
            if not was_paused:
                _detection_manager.paused = False
            return (
                jsonify(
                    {
                        "error": "Archive has blockers",
                        "blockers": analysis["blockers"],
                    }
                ),
                400,
            )

        include_db = data.get("include_db", True)
        include_originals = data.get("include_originals", True)
        include_derivatives = data.get("include_derivatives", False)
        include_settings = data.get("include_settings", False)
        db_strategy = data.get("db_strategy", "merge")

        if db_strategy not in ("merge", "replace"):
            if not was_paused:
                _detection_manager.paused = False
            return jsonify({"error": "Invalid db_strategy"}), 400

        def run_restore():
            global _restore_progress
            _restore_progress = {"active": True, "progress": None}

            try:
                for progress in backup_restore_service.restore_from_archive(
                    path,
                    include_db=include_db,
                    include_originals=include_originals,
                    include_derivatives=include_derivatives,
                    include_settings=include_settings,
                    db_strategy=db_strategy,
                ):
                    _restore_progress["progress"] = progress

                    if progress.get("completed"):
                        break

            except Exception as e:
                logger.error(f"Restore thread error: {e}", exc_info=True)
                _restore_progress["progress"] = {
                    "stage": "error",
                    "progress": 0,
                    "total": 1,
                    "message": str(e),
                    "completed": True,
                    "error": str(e),
                }
            finally:
                _restore_progress["active"] = False
                if not was_paused:
                    _detection_manager.paused = False
                    logger.info("Restore: Detection resumed after restore")

                try:
                    if path.exists():
                        path.unlink()
                        logger.info(f"Restore: Deleted uploaded archive {path}")
                except Exception as cleanup_err:
                    logger.warning(
                        f"Restore: Could not delete archive {path}: {cleanup_err}"
                    )

        thread = threading.Thread(target=run_restore, daemon=True)
        thread.start()

        logger.info("Restore: Started restore process in background")

        return jsonify(
            {
                "status": "started",
                "message": "Restore process started",
                "detection_auto_paused": not was_paused,
            }
        )

    except Exception as e:
        if "was_paused" in locals() and not was_paused:
            _detection_manager.paused = False
        logger.error(f"Restore apply error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@backup_bp.route("/api/restore/status", methods=["GET"])
@login_required
def restore_status():
    """Returns the current status of the restore operation."""
    return jsonify(
        {
            "active": _restore_progress["active"],
            "progress": _restore_progress.get("progress"),
        }
    )


@backup_bp.route("/api/restore/cleanup", methods=["POST"])
@login_required
def restore_cleanup():
    """Cleans up the restore temp directory."""
    try:
        backup_restore_service.cleanup_temp_files()
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Restore cleanup error: {e}")
        return jsonify({"error": str(e)}), 500
