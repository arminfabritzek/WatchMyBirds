"""
Inbox Blueprint.

Handles inbox routes for web upload with explicit processing:
- GET /inbox - Inbox page
- POST /api/inbox - File upload
- GET /api/inbox/status - Processing status
- POST /api/inbox/process - Start processing
"""

import os
import threading
import time
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import get_config
from logging_config import get_logger
from web.blueprints.auth import login_required
from web.services import backup_restore_service, ingest_service, path_service

logger = get_logger(__name__)
config = get_config()

inbox_bp = Blueprint("inbox", __name__)

# Thread state for inbox processing (module-level)
_inbox_processing = {"active": False, "lock": threading.Lock()}

# Detection manager reference - will be set by init function
_detection_manager = None


def init_inbox_bp(detection_manager):
    """Initialize inbox blueprint with detection manager reference."""
    global _detection_manager
    _detection_manager = detection_manager


@inbox_bp.route("/inbox")
@login_required
def inbox_page():
    return render_template("inbox.html")


@inbox_bp.route("/api/inbox", methods=["POST"])
@login_required
def inbox_upload():
    """
    Handle file uploads to inbox/pending.
    Max 20 files, max 50MB each.
    """
    try:
        # Block during restore
        if backup_restore_service.is_restore_active():
            return (
                jsonify({"error": "Upload blocked during restore operation"}),
                409,
            )

        if "files[]" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files[]")
        if len(files) > 100:
            return jsonify({"error": "Maximum 100 files allowed per upload"}), 400

        ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

        uploaded = []
        errors = []
        skipped = []  # Track duplicates

        # Get path manager
        output_dir = config.get("OUTPUT_DIR", "./data/output")
        path_mgr = path_service.get_path_manager(output_dir)
        pending_dir = path_mgr.get_inbox_pending_dir()

        for f in files:
            if not f or not f.filename:
                continue

            # Extension check
            ext = os.path.splitext(f.filename.lower())[1]
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{f.filename}: Invalid format (JPG/PNG only)")
                continue

            # Size check
            f.seek(0, 2)  # Seek to end
            size = f.tell()
            f.seek(0)  # Reset

            if size > MAX_FILE_SIZE:
                errors.append(f"{f.filename}: File too large (max 50 MB)")
                continue

            # Safe filename
            safe_name = secure_filename(f.filename)
            if not safe_name:
                safe_name = f"upload_{int(time.time() * 1000)}{ext}"

            # SKIP duplicates instead of renaming
            target_path = pending_dir / safe_name
            if target_path.exists():
                skipped.append(safe_name)
                continue

            try:
                f.save(str(target_path))
                uploaded.append(safe_name)
            except Exception as e:
                errors.append(f"{f.filename}: Save failed ({e})")

        return (
            jsonify(
                {
                    "uploaded": uploaded,
                    "skipped": skipped,
                    "skipped_count": len(skipped),
                    "errors": errors,
                    "pending_count": len(list(pending_dir.iterdir())),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Inbox upload error: {e}")
        return jsonify({"error": str(e)}), 500


@inbox_bp.route("/api/inbox/status", methods=["GET"])
@login_required
def inbox_status():
    """
    Returns inbox status:
    - pending_count: files in pending/
    - processing: bool from thread state
    - processed_today: files in processed/YYYYMMDD/
    - skipped_today: files in skipped/YYYYMMDD/
    - error_count: files in error/
    - detection_running: whether detection is active
    """
    try:
        today = datetime.now().strftime("%Y%m%d")

        # Get path manager
        output_dir = config.get("OUTPUT_DIR", "./data/output")
        path_mgr = path_service.get_path_manager(output_dir)

        pending_dir = path_mgr.inbox_pending_dir
        processed_dir = path_mgr.inbox_dir / "processed" / today
        skipped_dir = path_mgr.inbox_dir / "skipped" / today
        error_dir = path_mgr.inbox_error_dir

        pending_count = len(list(pending_dir.iterdir())) if pending_dir.exists() else 0
        processed_today = (
            len(list(processed_dir.iterdir())) if processed_dir.exists() else 0
        )
        skipped_today = len(list(skipped_dir.iterdir())) if skipped_dir.exists() else 0
        error_count = len(list(error_dir.iterdir())) if error_dir.exists() else 0

        # Detection state check
        detection_running = (
            not _detection_manager.paused if _detection_manager else False
        )

        return jsonify(
            {
                "pending_count": pending_count,
                "processing": _inbox_processing["active"],
                "processed_today": processed_today,
                "skipped_today": skipped_today,
                "error_count": error_count,
                "detection_running": detection_running,
            }
        )
    except Exception as e:
        logger.error(f"Inbox status error: {e}")
        return jsonify({"error": str(e)}), 500


@inbox_bp.route("/api/inbox/process", methods=["POST"])
@login_required
def inbox_process():
    """
    Start processing of inbox/pending files.
    Policy: Detection MUST be stopped (paused) before processing.
    Returns 409 if detection running or processing already active.
    """
    try:
        # Block during restore
        if backup_restore_service.is_restore_active():
            return (
                jsonify({"error": "Inbox processing blocked during restore operation"}),
                409,
            )

        # Check 1: Already processing?
        with _inbox_processing["lock"]:
            if _inbox_processing["active"]:
                return jsonify({"error": "Processing already in progress"}), 409

        # Get path manager
        output_dir = config.get("OUTPUT_DIR", "./data/output")
        path_mgr = path_service.get_path_manager(output_dir)

        # Take snapshot of pending files
        pending_dir = path_mgr.get_inbox_pending_dir()
        snapshot = list(pending_dir.iterdir())
        file_count = len([f for f in snapshot if f.is_file()])

        if file_count == 0:
            return (
                jsonify({"message": "No files to process", "count": 0}),
                200,
            )

        # Start background processing
        def run_inbox_ingest():
            try:
                with _inbox_processing["lock"]:
                    _inbox_processing["active"] = True

                ingest_service.process_inbox(
                    str(pending_dir), [str(f) for f in snapshot if f.is_file()]
                )

            except Exception as e:
                logger.error(f"Inbox ingest error: {e}", exc_info=True)
            finally:
                with _inbox_processing["lock"]:
                    _inbox_processing["active"] = False

        t = threading.Thread(target=run_inbox_ingest, daemon=True)
        t.start()

        return (
            jsonify(
                {
                    "message": f"Processing {file_count} files started",
                    "count": file_count,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Inbox process error: {e}")
        return jsonify({"error": str(e)}), 500
