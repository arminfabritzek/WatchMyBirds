# ------------------------------------------------------------------------------
# web_interface.py
# ------------------------------------------------------------------------------

import json
import logging
import math
import os
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

from config import (
    get_config,
    get_settings_payload,
    update_runtime_settings,
    validate_runtime_updates,
)
from utils.db import (
    fetch_all_detection_times,
    # Analytics functions
    fetch_analytics_summary,
    fetch_daily_covers,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    fetch_review_queue_count,
    fetch_species_timestamps,
    fetch_trash_count,
    fetch_trash_items,
    get_connection,
    reject_detections,
    restore_detections,
    restore_no_bird_images,
    # delete_no_bird_images removed - use file_gc.hard_delete_images instead
    update_downloaded_timestamp,
    update_review_status,
)
from utils.file_gc import hard_delete_detections
from utils.path_manager import get_path_manager


config = get_config()
db_conn = get_connection()

_CACHE_TIMEOUT = 60  # Set cache timeout in seconds
_cached_images = {"images": None, "timestamp": 0}
# In-Memory Caches
_species_summary_cache = {"timestamp": 0, "payload": None}


def create_web_interface(detection_manager):
    """
    Creates and returns a Flask web server for the project.

    Args:
        detection_manager: The DetectionManager instance for frame access and control.

    Configuration is loaded from the global config module.
    """
    logger = logging.getLogger(__name__)

    output_dir = config["OUTPUT_DIR"]
    output_resize_width = config["STREAM_WIDTH_OUTPUT_RESIZE"]
    config["CONFIDENCE_THRESHOLD_DETECTION"]
    config["CLASSIFIER_CONFIDENCE_THRESHOLD"]
    EDIT_PASSWORD = config["EDIT_PASSWORD"]
    logger.info(
        f"Loaded EDIT_PASSWORD: {'***' if EDIT_PASSWORD and EDIT_PASSWORD != 'default_pass' else '<Not Set or Default>'}"
    )

    if not EDIT_PASSWORD or EDIT_PASSWORD in ["SECRET_PASSWORD", "default_pass"]:
        logger.warning(
            "EDIT_PASSWORD not set securely in .env or settings.yaml. Access might be restricted or insecure."
        )

    IMAGE_WIDTH = 150
    PAGE_SIZE = 50

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    common_names_file = os.path.join(project_root, "assets", "common_names_DE.json")
    try:
        with open(common_names_file, encoding="utf-8") as f:
            COMMON_NAMES = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load common names from {common_names_file}: {e}")
        COMMON_NAMES = {"Cyanistes_caeruleus": "Eurasian blue tit"}

    def get_detections_for_date(date_str_iso):
        rows = fetch_detections_for_gallery(db_conn, date_str_iso)
        # Convert to list of dicts immediately for easier handling
        return [dict(row) for row in rows]

    def delete_detections(detection_ids):
        """
        [SEMANTIC DELETE]
        Rejects specific detections.
        """
        reject_detections(db_conn, detection_ids)
        return True

    def get_all_detections():
        """
        Reads all active detections from SQLite.
        Returns list of dicts.
        """
        try:
            rows = fetch_detections_for_gallery(db_conn, order_by="score")
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error reading detections from SQLite: {e}")
            return []

    def get_daily_covers():
        """Returns a dict of {YYYY-MM-DD: {path, bbox}} for gallery overview."""
        covers = {}
        gallery_threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)
        try:
            rows = fetch_daily_covers(db_conn, min_score=gallery_threshold)
            for row in rows:
                date_key = row["date_key"]  # Already YYYY-MM-DD
                optimized_name = row["optimized_name_virtual"]
                if not date_key or not optimized_name:
                    continue

                # Use virtual thumbnail path corresponding to new route structure
                thumb_path_virtual = row[
                    "thumbnail_path_virtual"
                ]  # YYYYMMDD/filename.webp

                # Construct display path pointing to new routes
                if thumb_path_virtual:
                    display_path = f"/uploads/derivatives/thumbs/{thumb_path_virtual}"
                    is_thumb = True
                else:
                    display_path = f"/uploads/derivatives/optimized/{optimized_name}"
                    is_thumb = False

                bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])

                covers[date_key] = {
                    "path": display_path,
                    "bbox": bbox,
                    "is_thumb": is_thumb,
                    "count": row["image_count"],
                }
        except Exception as e:
            logger.error(f"Error reading daily covers from SQLite: {e}")
        return covers

    def get_captured_detections():
        """
        Returns a list of captured detections (dicts).
        Uses caching to avoid repeated DB hits if frequent.
        """
        now = time.time()
        if (
            _cached_images["images"] is not None
            and (now - _cached_images["timestamp"]) < _CACHE_TIMEOUT
        ):
            return _cached_images["images"]

        detections = []
        try:
            rows = fetch_detections_for_gallery(
                db_conn, order_by="time"
            )  # Most recent first
            detections = [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error reading detections from SQLite: {e}")

        _cached_images["images"] = detections
        _cached_images["timestamp"] = now
        return detections

    def get_captured_detections_by_date():
        """
        Returns a dictionary grouping detections by date (YYYY-MM-DD).
        """
        detections = get_captured_detections()
        detections_by_date = {}
        for det in detections:
            ts = det["image_timestamp"]
            # ts format YYYYMMDD_HHMMSS
            if len(ts) >= 8:
                date_str = ts[:8]
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                if formatted_date not in detections_by_date:
                    detections_by_date[formatted_date] = []
                detections_by_date[formatted_date].append(det)
        return detections_by_date

    def get_daily_species_summary(date_iso: str):
        """Returns per-species counts for a given date (YYYY-MM-DD) - always fresh from DB."""
        try:
            rows = fetch_detection_species_summary(db_conn, date_iso)
        except Exception as e:
            logger.error(f"Error fetching daily species summary for {date_iso}: {e}")
            rows = []

        summary = []
        for row in rows:
            species = row["species"]
            count = row["count"]
            if not species:
                continue
            common_name = COMMON_NAMES.get(species, species.replace("_", " "))
            summary.append(
                {"species": species, "common_name": common_name, "count": int(count)}
            )
        return summary

    def generate_video_feed():
        # Load placeholder once
        static_placeholder_path = "assets/static_placeholder.jpg"
        if os.path.exists(static_placeholder_path):
            static_placeholder = cv2.imread(static_placeholder_path)
            if static_placeholder is not None:
                original_h, original_w = static_placeholder.shape[:2]
                ratio = original_h / float(original_w)
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * ratio)
                static_placeholder = cv2.resize(
                    static_placeholder, (placeholder_w, placeholder_h)
                )
            else:
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * 9 / 16)
                static_placeholder = np.zeros(
                    (placeholder_h, placeholder_w, 3), dtype=np.uint8
                )
        else:
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros(
                (placeholder_h, placeholder_w, 3), dtype=np.uint8
            )

        # Parameters for text overlay
        padding_x_percent = 0.005
        padding_y_percent = 0.04

        while True:
            start_time = time.time()
            # Retrieve the most recent display frame (raw or optimized)
            frame = detection_manager.get_display_frame()
            if frame is not None:
                # Derive size from the current frame to avoid blocking on VideoCapture availability.
                h, w = frame.shape[:2]
                output_resize_height = int(h * output_resize_width / w) if w else h
                resized_frame = cv2.resize(
                    frame, (output_resize_width, output_resize_height)
                )
                # Overlay current timestamp
                pil_image = Image.fromarray(
                    cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                )
            else:
                # If no frame, use the placeholder.
                pil_image = Image.fromarray(
                    cv2.cvtColor(static_placeholder, cv2.COLOR_BGR2RGB)
                )

            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            padding_x = int(img_width * padding_x_percent)
            padding_y = int(img_height * padding_y_percent)

            # Dynamic font size: ~2.5% of image width, clamped 12-24px
            target_font_size = max(12, min(24, int(img_width * 0.025)))
            try:
                custom_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    target_font_size,
                )
            except OSError:
                try:
                    custom_font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        target_font_size,
                    )
                except OSError:
                    try:
                        # macOS fallback
                        custom_font = ImageFont.truetype(
                            "/System/Library/Fonts/Helvetica.ttc", target_font_size
                        )
                    except OSError:
                        custom_font = ImageFont.load_default()

            # Classic top-right timestamp (Mac style)
            # German weekday/month names
            weekdays_de = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
            months_de = [
                "",
                "Jan",
                "Feb",
                "Mär",
                "Apr",
                "Mai",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Okt",
                "Nov",
                "Dez",
            ]

            now = datetime.now()
            weekday = weekdays_de[now.weekday()]
            month = months_de[now.month]

            # Single line: "Fr 30. Jan 22:25:45"
            timestamp_text = f"{weekday} {now.day}. {month} {now.strftime('%H:%M:%S')}"

            # Position: top-right
            bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img_width - text_width - padding_x - 5
            text_y = padding_y

            # Auto-contrast: sample background luminance
            # Get the region where text will be drawn
            region_x1 = max(0, text_x - 2)
            region_y1 = max(0, text_y - 2)
            region_x2 = min(img_width, text_x + text_width + 2)
            region_y2 = min(img_height, text_y + text_height + 2)

            # Crop region and calculate average brightness
            region = pil_image.crop((region_x1, region_y1, region_x2, region_y2))
            pixels = list(region.getdata())
            if pixels:
                # Calculate luminance (perceived brightness)
                avg_lum = sum(
                    0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in pixels
                ) / len(pixels)
            else:
                avg_lum = 128  # Default to mid-gray

            # Choose text color based on background
            if avg_lum > 128:
                # Light background → dark text
                text_color = (0, 0, 0)
                shadow_color = (255, 255, 255)
            else:
                # Dark background → light text
                text_color = (255, 255, 255)
                shadow_color = (0, 0, 0)

            # Draw shadow + text
            draw.text(
                (text_x + 1, text_y + 1),
                timestamp_text,
                font=custom_font,
                fill=shadow_color,
            )
            draw.text(
                (text_x, text_y), timestamp_text, font=custom_font, fill=text_color
            )
            # Convert back to OpenCV BGR format
            frame_with_timestamp = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode(
                ".jpg", frame_with_timestamp, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if ret:
                frame_data = buffer.tobytes()
                # Include Content-Length for Safari compatibility
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(frame_data)).encode()
                    + b"\r\n\r\n"
                    + frame_data
                    + b"\r\n"
                )
            elapsed = time.time() - start_time
            stream_fps = config.get("STREAM_FPS", 0)
            if stream_fps and stream_fps > 0:
                desired_frame_time = 1.0 / stream_fps
                if elapsed < desired_frame_time:
                    time.sleep(desired_frame_time - elapsed)

    # -----------------------------
    # Flask Server and Routes
    # -----------------------------
    # Create Flask server with Jinja2 template support for server-first rendering.
    # Template folder is at project root level, not in web/ subdirectory.
    project_root = os.path.dirname(os.path.dirname(__file__))
    template_folder = os.path.join(project_root, "templates")
    assets_folder = os.path.join(project_root, "assets")
    server = Flask(__name__, template_folder=template_folder)

    # Configure Flask Session for server-side auth
    server.secret_key = os.environ.get(
        "FLASK_SECRET_KEY", "watchmybirds-dev-key-change-in-production"
    )

    # Load version info if available (Only for Dev Sync)
    version_info = ""
    try:
        version_file = os.path.join(
            output_dir, "..", "version.txt"
        )  # Look in /opt/app/version.txt
        if os.path.exists("version.txt"):
            with open("version.txt") as f:
                version_info = f"Synced: {f.read().strip()}"
        elif os.path.exists(version_file):
            with open(version_file) as f:
                version_info = f"Synced: {f.read().strip()}"
    except Exception:
        pass

    @server.context_processor
    def inject_version():
        return dict(version_info=version_info)

    @server.context_processor
    def inject_counts():
        """Injects global counts for Navbar badges (Trash, Untagged)."""
        counts = {"trash_count": 0, "untagged_count": 0}
        try:
            # Use a fresh connection or the shared one?
            # Shared db_conn is created at module level. SQLite allows sharing read-only if thread safe.
            # But let's use a fresh connection pattern or the helper if it's quick.
            # fetch_trash_count uses 'conn'.
            # Since this runs on EVERY request, we must be careful.
            # For now, let's just query. Optimisation: Cache this for 60s?

            # Simple caching mechanism for counts
            now = time.time()
            # Use a global cache dict attached to the function or module
            if not hasattr(inject_counts, "cache"):
                inject_counts.cache = {"time": 0, "data": counts}

            if now - inject_counts.cache["time"] < 30:  # 30s cache
                return inject_counts.cache["data"]

            with get_connection() as conn:
                counts["trash_count"] = fetch_trash_count(conn)
                # Use SAVE_THRESHOLD for Review Queue count
                save_threshold = config.get("SAVE_THRESHOLD", 0.65)
                counts["untagged_count"] = fetch_review_queue_count(
                    conn, save_threshold
                )

            inject_counts.cache = {"time": now, "data": counts}
        except Exception as e:
            logger.error(f"Error fetching global counts: {e}")

        return counts

    # --- Auth Helper ---
    def login_required(f):
        """Decorator to require authentication for Flask routes."""
        from functools import wraps

        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import redirect, url_for

            if not session.get("authenticated"):
                return redirect(url_for("login", next=request.path))
            return f(*args, **kwargs)

        return decorated_function

    def setup_web_routes(server):
        path_mgr = get_path_manager(output_dir)

        # --- Helper for Regeneration ---
        def regenerate_derivative(filename_rel, type="thumb"):
            """
            Attempts to regenerate a missing derivative.
            filename_rel: YYYYMMDD/basename.webp (path from route)
            type: 'thumb' | 'optimized'
            """
            try:
                # 1. Parse Path
                # filename_rel is like "20240120/20240120_120000_123456_crop_1.webp" or "20240120/timestamp.webp"
                # PathManager expects just the filename, it resolves date from correct format.
                # Route passes <path:filename>, which might include slashes if user requests "20240120/file.webp"
                # But path_manager.get_derivative_path expects filename only if it contains date.

                # Let's extract just the filename
                filename = os.path.basename(filename_rel)

                # 2. Check source (Original)
                # Recover original filename logic.
                # Optimized: original name is same usually (just .jpg)
                # Thumb: {basename}_crop_{i}.webp -> need to find original {basename}.jpg

                # Regex for Thumb: (.*)_crop_(\d+)\.webp
                # Regex for Optimized: (.*)\.webp -> (.*).jpg

                original_filename = None
                crop_index = None

                if type == "thumb":
                    match = re.match(r"(.*)_crop_(\d+)\.webp$", filename)
                    if match:
                        base_no_ext = match.group(1)
                        crop_index = int(match.group(2))
                        original_filename = f"{base_no_ext}.jpg"
                elif type == "optimized":
                    base_no_ext = os.path.splitext(filename)[0]
                    original_filename = f"{base_no_ext}.jpg"

                if not original_filename:
                    return False

                original_path = path_mgr.get_original_path(original_filename)

                if not original_path.exists():
                    logger.error(
                        f"Cannot regenerate {filename}: Original missing at {original_path}"
                    )
                    return False

                # 3. Load Original
                img = cv2.imread(str(original_path))
                if img is None:
                    return False

                # 4. Process
                if type == "optimized":
                    # Resize logic
                    if img.shape[1] > 1920:
                        scale = 1920 / img.shape[1]
                        new_h = int(img.shape[0] * scale)
                        out_img = cv2.resize(img, (1920, new_h))
                    else:
                        out_img = img

                    target_path = path_mgr.get_derivative_path(filename, "optimized")

                elif type == "thumb":
                    # BBox Lookup from DB
                    # We need the detection corresponding to index
                    # Note: crop_index is 1-based index from detection list
                    # Order by detection_id ASC for that image

                    # Need clean filename for query? "base_no_ext.jpg"
                    # image_filename is original_filename

                    cursor = db_conn.execute(
                        """
                        SELECT bbox_x, bbox_y, bbox_w, bbox_h
                        FROM detections
                        WHERE image_filename = ?
                        ORDER BY detection_id ASC
                        LIMIT 1 OFFSET ?
                    """,
                        (original_filename, crop_index - 1),
                    )

                    row = cursor.fetchone()
                    if not row:
                        logger.error(
                            f"Cannot regenerate thumb: No detection found for {original_filename} index {crop_index}"
                        )
                        return False

                    # Crop Logic
                    h, w = img.shape[:2]
                    x1 = int(row[0] * w)
                    y1 = int(row[1] * h)
                    bw = int(row[2] * w)
                    bh = int(row[3] * h)
                    x2 = x1 + bw
                    y2 = y1 + bh

                    # Expand & Square (Copy logic from DetectionManager or simplify)
                    # Simplified slightly for robustness
                    TARGET_SIZE = 256
                    EXPANSION = 0.1
                    side = int(max(bw, bh) * (1 + EXPANSION))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    sq_x1, sq_y1 = cx - side // 2, cy - side // 2
                    sq_x2, sq_y2 = sq_x1 + side, sq_y1 + side

                    # Clamp
                    sq_x1, sq_y1 = max(0, sq_x1), max(0, sq_y1)
                    sq_x2, sq_y2 = min(w, sq_x2), min(h, sq_y2)

                    if sq_x2 > sq_x1 and sq_y2 > sq_y1:
                        crop_img = img[sq_y1:sq_y2, sq_x1:sq_x2]
                        out_img = cv2.resize(
                            crop_img,
                            (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                        target_path = path_mgr.get_derivative_path(filename, "thumb")
                    else:
                        return False

                # 5. Save
                path_mgr.ensure_date_structure(
                    path_mgr.extract_date_from_filename(filename)
                )
                cv2.imwrite(
                    str(target_path), out_img, [int(cv2.IMWRITE_WEBP_QUALITY), 80]
                )
                logger.info(f"Regenerated missing derivative: {target_path}")
                return True

            except Exception as e:
                logger.error(f"Regeneration failed for {filename}: {e}")
                return False

        # --- Routes ---

        @server.route("/uploads/originals/<path:filename>")
        def serve_original(filename):
            # filename typically includes date folder e.g. "20240120/file.jpg"
            # path_manager logic: get_original_path expects JUST filename generally,
            # OR we construct path manually relative to originals dir.
            # Let's trust flask's safe_join usually but we have structure.
            # Best: use path_mgr.resolve? No, path_mgr constructs absolute paths.

            # If request is "20240120/file.jpg", we can map it.
            full_path = path_mgr.originals_dir / filename
            if not full_path.exists():
                return "Not found", 404
            return send_from_directory(
                os.path.dirname(full_path), os.path.basename(full_path)
            )

        @server.route("/uploads/derivatives/thumbs/<path:filename>")
        def serve_thumb(filename):
            full_path = path_mgr.thumbs_dir / filename
            if not full_path.exists():
                # Trigger Regeneration
                # filename is e.g. "20240120/foo.webp"
                if regenerate_derivative(filename, "thumb"):
                    if not full_path.exists():  # Double check
                        return "Regeneration failed", 500
                else:
                    return "Not found and could not regenerate", 404
            return send_from_directory(
                os.path.dirname(full_path), os.path.basename(full_path)
            )

        @server.route("/uploads/derivatives/optimized/<path:filename>")
        def serve_optimized(filename):
            full_path = path_mgr.optimized_dir / filename
            if not full_path.exists():
                if regenerate_derivative(filename, "optimized"):
                    if not full_path.exists():
                        return "Regeneration failed", 500
                else:
                    return "Not found and could not regenerate", 404
            return send_from_directory(
                os.path.dirname(full_path), os.path.basename(full_path)
            )

        def daily_species_summary_route():
            date_iso = request.args.get("date")
            if not date_iso:
                date_iso = datetime.now().strftime("%Y-%m-%d")
            try:
                datetime.strptime(date_iso, "%Y-%m-%d")
            except ValueError:
                return (
                    jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}),
                    400,
                )
            summary = get_daily_species_summary(date_iso)
            return jsonify({"date": date_iso, "summary": summary})

        server.route("/assets/<path:filename>")(
            lambda filename: send_from_directory(assets_folder, filename)
        )

        def video_feed_route():
            return Response(
                generate_video_feed(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        # Snapshot API - returns current frame as downloadable JPEG
        @server.route("/api/snapshot")
        def snapshot_api():
            """Return current video frame as JPEG for download."""
            from datetime import datetime

            # Get frame from detection_manager (same source as video feed)
            frame = detection_manager.get_display_frame()
            if frame is None:
                return jsonify({"error": "No frame available"}), 503

            # Resize to output dimensions
            h, w = frame.shape[:2]
            output_h = int(h * output_resize_width / w) if w else h
            resized = cv2.resize(frame, (output_resize_width, output_h))

            # Convert to JPEG
            _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

            filename = f"snapshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

            return Response(
                buffer.tobytes(),
                mimetype="image/jpeg",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

        # --------------------------------------------------------------------------
        # Ingest API Route
        # --------------------------------------------------------------------------
        @server.route("/api/ingest/start", methods=["POST"])
        @login_required
        def start_ingest_endpoint():
            try:
                # Determine ingest path (Docker vs Local) with logging
                # Use configured path from config.py (Source of Truth)
                env_path = config.get("INGEST_DIR")
                cwd_path = os.path.abspath(os.path.join(os.getcwd(), "ingest"))

                logger.info(
                    f"Ingest Request: CWD={os.getcwd()}, Configured: {env_path}, Local: {cwd_path}"
                )

                if os.path.exists(env_path):
                    ingest_path = env_path
                    logger.info(f"Using configured ingest path: {ingest_path}")
                elif os.path.exists(cwd_path):
                    ingest_path = cwd_path
                    logger.info(
                        f"Configured path not found. Using local CWD fallback: {ingest_path}"
                    )
                else:
                    ingest_path = env_path  # Fallback to default
                    logger.warning(
                        f"No valid ingest dir found. Falling back to configured: {ingest_path}"
                    )

                # Trigger User Ingest in background
                import threading

                def run_ingest():
                    detection_manager.start_user_ingest(ingest_path)

                t = threading.Thread(target=run_ingest)
                t.start()

                return (
                    jsonify(
                        {
                            "status": "success",
                            "message": "User Ingest started. Stream will pause.",
                        }
                    ),
                    200,
                )
            except Exception as e:
                logger.error(f"Error starting ingest: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def settings_get_route():
            return jsonify(get_settings_payload())

        @login_required
        def settings_post_route():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid payload"}), 400
            valid, errors = validate_runtime_updates(payload)
            if errors:
                return jsonify({"errors": errors}), 400
            update_runtime_settings(valid)

            # --- Dynamic Reload Logic ---
            # Delegate component updates to DetectionManager (Video Source, Debug Mode, Models, etc.)
            detection_manager.update_configuration(valid)

            return jsonify(get_settings_payload())

        server.add_url_rule(
            "/video_feed", endpoint="video_feed", view_func=video_feed_route
        )
        server.add_url_rule(
            "/api/daily_species_summary",
            endpoint="daily_species_summary",
            view_func=daily_species_summary_route,
            methods=["GET"],
        )
        server.add_url_rule(
            "/api/settings",
            endpoint="settings_get",
            view_func=settings_get_route,
            methods=["GET"],
        )
        server.add_url_rule(
            "/api/settings",
            endpoint="settings_post",
            view_func=settings_post_route,
            methods=["POST"],
        )

        # --- Analytics API Routes (All-Time, Read-Only) ---
        def analytics_summary_route():
            with get_connection() as conn:
                summary = fetch_analytics_summary(conn)
            return jsonify(summary)

        def analytics_time_of_day_route():
            with get_connection() as conn:
                rows = fetch_all_detection_times(conn)

            if not rows:
                return jsonify({"points": [], "peak_hour": None, "histogram": []})

            # 1. Parse Times to Float Hours
            # HHMMSS -> H + M/60 + S/3600
            hours_float = []
            for row in rows:
                t_str = row["time_str"]  # "HHMMSS"
                if len(t_str) == 6:
                    h = int(t_str[0:2])
                    m = int(t_str[2:4])
                    s = int(t_str[4:6])
                    val = h + m / 60.0 + s / 3600.0
                    hours_float.append(val)
                elif len(t_str) == 8:  # HH:MM:SS fallback
                    try:
                        h = int(t_str[0:2])
                        m = int(t_str[3:5])
                        s = int(t_str[6:8])
                        val = h + m / 60.0 + s / 3600.0
                        hours_float.append(val)
                    except Exception:
                        pass

            if not hours_float:
                return jsonify({"points": [], "peak_hour": None, "histogram": []})

            # 2. KDE Approximation via Histogram + Gaussian Smoothing
            # We use 1440 bins (minutes) for high resolution
            # This avoids adding scipy dependency
            import numpy as np

            bins = 144
            hist, bin_edges = np.histogram(
                hours_float, bins=bins, range=(0, 24), density=True
            )

            # Simple Gaussian Smoothing (Window size approx 1-2 hours)
            # Standard deviation for smoothing kernel
            sigma = 1.6  # Adjust for smoothness vs detail
            x_vals = np.linspace(-3 * sigma, 3 * sigma, int(6 * sigma) + 1)
            kernel = np.exp(-(x_vals**2) / (2 * sigma**2))
            kernel = kernel / np.sum(kernel)

            smooth_density = np.convolve(hist, kernel, mode="same")

            # Normalize peak to 1.0 (relative density) or use probability density?
            # User asked for "Relative activity density" and "Fläche auf 1".
            # Density=True in histogram ensures integral is 1.
            # Convoluting with normalized kernel preserves integral approx 1.

            # Generate Output Points
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            points = []
            max_y = 0
            peak_hour = 0

            for x, y in zip(bin_centers, smooth_density, strict=False):
                points.append({"x": round(float(x), 2), "y": float(y)})
                if y > max_y:
                    max_y = y
                    peak_hour = x

            # Subsampled Histogram for "Backdrop" (Optional)
            # Use coarser bins for the visual histogram (e.g. 24 hours)
            hist_coarse, edges_coarse = np.histogram(
                hours_float, bins=48, range=(0, 24), density=True
            )
            histogram_points = []
            for i in range(len(hist_coarse)):
                histogram_points.append(
                    {
                        "x": float((edges_coarse[i] + edges_coarse[i + 1]) / 2),
                        "y": float(hist_coarse[i]),
                    }
                )

            return jsonify(
                {
                    "points": points,
                    "peak_hour": round(float(peak_hour), 2),
                    "peak_density": float(max_y),
                    "histogram": histogram_points,
                }
            )

        def analytics_species_activity_route():
            with get_connection() as conn:
                rows = fetch_species_timestamps(conn)

            import numpy as np

            # Group by species
            species_times = {}
            for r in rows:
                sp = r["species"]
                t_str = (
                    r["image_timestamp"][9:15]
                    if len(r["image_timestamp"]) >= 15
                    else ""
                )  # YYYYMMDD_HHMMSS
                if len(t_str) == 6:
                    try:
                        h = (
                            int(t_str[0:2])
                            + int(t_str[2:4]) / 60.0
                            + int(t_str[4:6]) / 3600.0
                        )
                        if sp not in species_times:
                            species_times[sp] = []
                        species_times[sp].append(h)
                    except Exception:
                        pass

            series = []
            for sp, times in species_times.items():
                # Rule: n >= 10 for KDE, else Histogram
                # Max normalization (Ridgeplot style)

                if len(times) < 10:
                    # Histogram (1h bins)
                    hist, edges = np.histogram(
                        times, bins=24, range=(0, 24), density=False
                    )
                    # Normalize to max 1.0
                    max_val = np.max(hist)
                    if max_val > 0:
                        hist = hist / max_val

                    centers = (edges[:-1] + edges[1:]) / 2
                    points = [
                        {"x": float(x), "y": float(y)} for x, y in zip(centers, hist, strict=False)
                    ]
                    peak = centers[np.argmax(hist)]
                else:
                    # Numpy Gaussian Smoothing (Manual KDE)
                    # 144 bins (10 min)
                    bins = 144
                    hist, edges = np.histogram(
                        times, bins=bins, range=(0, 24), density=True
                    )

                    # Bandwidth: Explicitly set ~1.5 hours (sigma)
                    # 1.5h in bins (10m) = 9 bins
                    sigma = 9
                    x_vals = np.linspace(-3 * sigma, 3 * sigma, int(6 * sigma) + 1)
                    kernel = np.exp(-(x_vals**2) / (2 * sigma**2))
                    kernel = kernel / np.sum(kernel)
                    smooth = np.convolve(hist, kernel, mode="same")

                    # Max Normalization
                    max_val = np.max(smooth)
                    if max_val > 0:
                        smooth = smooth / max_val

                    centers = (edges[:-1] + edges[1:]) / 2
                    points = [
                        {"x": float(x), "y": float(y)} for x, y in zip(centers, smooth, strict=False)
                    ]
                    peak = centers[np.argmax(smooth)]

                series.append(
                    {
                        "species": sp,
                        "points": points,
                        "peak_hour": float(peak),
                        "count": len(times),
                    }
                )

            # Sort species by Peak Hour (median might be better but peak is requested in prompt context A vs B)
            # Prompt said: "A. Sortiere Arten nach Median-Aktivitätszeit, nicht Peak."
            # Let's calculate median for sort.
            for s in series:
                sp = s["species"]
                times = species_times[sp]
                s["median_hour"] = float(np.median(times))

            series.sort(key=lambda x: x["median_hour"])

            return jsonify(series)

        server.add_url_rule(
            "/api/analytics/summary",
            endpoint="analytics_summary",
            view_func=analytics_summary_route,
            methods=["GET"],
        )

        server.add_url_rule(
            "/api/analytics/time-of-day",
            endpoint="analytics_time_of_day",
            view_func=analytics_time_of_day_route,
            methods=["GET"],
        )
        server.add_url_rule(
            "/api/analytics/species-activity",
            endpoint="analytics_species_activity",
            view_func=analytics_species_activity_route,
            methods=["GET"],
        )

        # --- Phase 5: Trash Routes (Flask) ---
        @login_required
        def trash_route():
            page = request.args.get("page", 1, type=int)
            limit = 50

            with get_connection() as conn:
                items, total_count = fetch_trash_items(conn, page=page, limit=limit)

            processed_items = []
            for item in items:
                ts = item.get("image_timestamp", "")
                trash_type = item.get("trash_type", "detection")

                # Handle display path based on trash type
                if trash_type == "detection":
                    # Detection: use thumbnail or optimized image
                    full_path = item.get("relative_path") or item.get(
                        "image_optimized", ""
                    )
                    thumb_virtual = item.get("thumbnail_path_virtual")

                    if thumb_virtual:
                        display_path = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_path = f"/uploads/derivatives/optimized/{full_path}"

                    common_name = (
                        item.get("cls_class_name")
                        or item.get("od_class_name")
                        or "Unknown"
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

        @login_required
        def trash_restore_route():
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

                with get_connection() as conn:
                    # Restore detections
                    if detection_ids:
                        restore_detections(conn, detection_ids)
                        restored_count += len(detection_ids)

                    # Restore no_bird images (back to untagged)
                    if image_filenames:
                        restored_count += restore_no_bird_images(conn, image_filenames)

                return jsonify(
                    {"status": "success", "result": {"restored": restored_count}}
                )
            except Exception as e:
                logger.error(f"Error restoring trash: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @login_required
        def trash_purge_route():
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

                with get_connection() as conn:
                    # Purge detections
                    if detection_ids:
                        result = hard_delete_detections(
                            conn, detection_ids=detection_ids
                        )
                        det_deleted = result.get("rows_deleted", 0)
                        files_deleted = result.get("files_deleted", 0)

                    # Purge no_bird images (with full file cleanup)
                    if image_filenames:
                        from utils.file_gc import hard_delete_images

                        img_result = hard_delete_images(conn, filenames=image_filenames)
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

        @login_required
        def trash_empty_route():
            """Empties entire trash (detections + no_bird images)."""
            try:
                det_deleted = 0
                img_deleted = 0
                files_deleted = 0

                with get_connection() as conn:
                    # Empty rejected detections
                    result = hard_delete_detections(conn, before_date="2099-12-31")
                    det_deleted = result.get("rows_deleted", 0)
                    files_deleted = result.get("files_deleted", 0)

                    # Empty no_bird images (with full file cleanup)
                    from utils.file_gc import hard_delete_images

                    img_result = hard_delete_images(conn, delete_all=True)
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

        @login_required
        def reject_detection_route():
            data = request.get_json() or {}
            ids = data.get("ids", [])
            if not ids:
                return jsonify({"error": "No IDs provided"}), 400
            with get_connection() as conn:
                reject_detections(conn, ids)
            return jsonify({"status": "success"})

        server.add_url_rule(
            "/trash", endpoint="trash", view_func=trash_route, methods=["GET"]
        )
        server.add_url_rule(
            "/api/trash/restore",
            endpoint="trash_restore",
            view_func=trash_restore_route,
            methods=["POST"],
        )
        server.add_url_rule(
            "/api/trash/purge",
            endpoint="trash_purge",
            view_func=trash_purge_route,
            methods=["POST"],
        )
        server.add_url_rule(
            "/api/trash/empty",
            endpoint="trash_empty",
            view_func=trash_empty_route,
            methods=["POST"],
        )
        server.add_url_rule(
            "/api/detections/reject",
            endpoint="reject_detection",
            view_func=reject_detection_route,
            methods=["POST"],
        )

        # --- Auth Routes (Server-Side Session) ---
        def login_route():
            error = None
            next_url = request.args.get("next", "/gallery")

            if request.method == "POST":
                password = request.form.get("password", "")
                next_url = request.form.get("next", "/gallery")

                if password == (EDIT_PASSWORD or ""):
                    session["authenticated"] = True
                    logger.info("User authenticated successfully.")
                    return redirect(next_url)
                else:
                    error = "Invalid password. Please try again."
                    logger.warning("Failed login attempt.")

            return render_template("login.html", error=error, next_url=next_url)

        def logout_route():
            session.pop("authenticated", None)
            return redirect("/gallery")

        server.add_url_rule(
            "/login", endpoint="login", view_func=login_route, methods=["GET", "POST"]
        )
        server.add_url_rule(
            "/logout", endpoint="logout", view_func=logout_route, methods=["GET"]
        )

        # --- Phase 6: Edit Page (server-rendered) ---
        @login_required
        def edit_route(date_iso):
            """Server-rendered edit page with filtering and batch actions."""
            # Validate date
            try:
                datetime.strptime(date_iso, "%Y-%m-%d")
            except ValueError:
                return "Invalid date format. Use YYYY-MM-DD", 400

            # Get filters from query params
            filters = {
                "status": request.args.get("status", "all"),
                "species": request.args.get("species", "all"),
                "sort": request.args.get("sort", "time_desc"),
                "min_conf": request.args.get("min_conf", "0.0"),
            }

            detections = get_detections_for_date(date_iso)
            if not detections:
                return render_template(
                    "edit.html",
                    date_iso=date_iso,
                    detections=[],
                    filters=filters,
                    species_list=[],
                    image_width=IMAGE_WIDTH,
                )

            # Extract unique species for the dropdown
            species_list = sorted(
                list(
                    set(
                        det["cls_class_name"] or det["od_class_name"] or "Unknown"
                        for det in detections
                    )
                )
            )

            # Apply Filters
            filtered = []
            try:
                min_conf_val = float(filters["min_conf"])
            except ValueError:
                min_conf_val = 0.0

            for det in detections:
                # Status filter
                is_downloaded = bool(det.get("downloaded_timestamp"))
                if filters["status"] == "downloaded" and not is_downloaded:
                    continue
                if filters["status"] == "not_downloaded" and is_downloaded:
                    continue

                # Species filter
                sp = det["cls_class_name"] or det["od_class_name"] or "Unknown"
                if filters["species"] != "all" and sp != filters["species"]:
                    continue

                # Confidence filter
                conf = max(
                    det.get("od_confidence") or 0, det.get("cls_confidence") or 0
                )
                if conf < min_conf_val:
                    continue

                # Add display_path for template - Use pre-computed virtual paths from DB
                thumb_virtual = det.get("thumbnail_path_virtual")
                relative_path = det.get("relative_path", "")
                original_name = det.get("original_name", "")

                ts = det.get("image_timestamp", "")
                date_folder = (
                    f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else date_iso
                )

                if thumb_virtual:
                    det["display_path"] = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    det["display_path"] = (
                        f"/uploads/derivatives/optimized/{relative_path}"
                    )

                det["full_path"] = f"/uploads/derivatives/optimized/{relative_path}"
                det["original_path"] = (
                    f"/uploads/originals/{date_folder}/{original_name}"
                )
                det["common_name"] = COMMON_NAMES.get(
                    det.get("cls_class_name") or det.get("od_class_name"), "Unknown"
                )
                det["latin_name"] = (
                    det.get("cls_class_name") or det.get("od_class_name") or ""
                )

                filtered.append(det)

            # Apply Sorting
            if filters["sort"] == "time_asc":
                filtered.sort(key=lambda x: x["image_timestamp"])
            elif filters["sort"] == "time_desc":
                filtered.sort(key=lambda x: x["image_timestamp"], reverse=True)
            elif filters["sort"] == "score":
                filtered.sort(key=lambda x: x["score"] or 0.0, reverse=True)
            elif filters["sort"] == "confidence":
                filtered.sort(
                    key=lambda x: max(
                        x.get("od_confidence") or 0, x.get("cls_confidence") or 0
                    ),
                    reverse=True,
                )

            # --- Pagination ---
            page = request.args.get("page", 1, type=int)
            per_page = 100
            total_items = len(filtered)
            total_pages = math.ceil(total_items / per_page)

            if page < 1:
                page = 1
            if page > total_pages and total_pages > 0:
                page = total_pages

            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            detections_page = filtered[start_idx:end_idx]

            # Generate pagination range (simple version)
            # Show mostly current, prev, next, first, last logic can be done in template or simple list here
            # Let's keep it simple: just pass numbers

            return render_template(
                "edit.html",
                date_iso=date_iso,
                detections=detections_page,
                filters=filters,
                species_list=species_list,
                image_width=IMAGE_WIDTH,
                pagination={
                    "page": page,
                    "total_pages": total_pages,
                    "total_items": total_items,
                    "has_prev": page > 1,
                    "has_next": page < total_pages,
                    "prev_num": page - 1,
                    "next_num": page + 1,
                },
            )

        @login_required
        def edit_actions_route():
            """Handles POST actions from the edit page (reject, download)."""
            action = request.form.get("action")
            date_iso = request.form.get("date_iso")
            det_ids = request.form.getlist("ids")

            if not det_ids:
                return redirect(f"/edit/{date_iso}")

            ids_int = [int(i) for i in det_ids]

            if action == "reject":
                with get_connection() as conn:
                    reject_detections(conn, ids_int)
                # Reset caches
                _cached_images["images"] = None
                # _daily_gallery_summary_cache was removed
                return redirect(f"/edit/{date_iso}")

            elif action == "download":
                import io
                import zipfile

                from flask import send_file

                with get_connection() as conn:
                    # Resolve IDs to paths
                    placeholders = ",".join("?" for _ in ids_int)
                    query = f"""
                        SELECT d.detection_id, i.filename as original_name, i.timestamp
                        FROM detections d
                        JOIN images i ON d.image_filename = i.filename
                        WHERE d.detection_id IN ({placeholders})
                    """
                    rows = conn.execute(query, ids_int).fetchall()

                    output_dir = config.get("OUTPUT_DIR", "detections")
                    files_to_zip = []

                    for r in rows:
                        original_name, ts = r["original_name"], r["timestamp"]
                        if not original_name or not ts:
                            continue

                        # Build YYYY-MM-DD folder format
                        date_folder = (
                            f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                        )

                        # Clean Slate: originals are in originals/YYYY-MM-DD/
                        abs_path = os.path.join(
                            output_dir, "originals", date_folder, original_name
                        )
                        files_to_zip.append((abs_path, original_name))

                    # Update downloaded timestamp using original filenames
                    if files_to_zip:
                        download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        filenames = [f[1] for f in files_to_zip]
                        update_downloaded_timestamp(conn, filenames, download_time)

                # Create Zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for abs_path, arcname in files_to_zip:
                        if os.path.exists(abs_path):
                            zf.write(abs_path, arcname=arcname)

                zip_buffer.seek(0)
                download_name = f"watchmybirds_{date_iso.replace('-','')}_download.zip"
                return send_file(
                    zip_buffer,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=download_name,
                )

            return redirect(f"/edit/{date_iso}")

        server.add_url_rule(
            "/edit/<date_iso>",
            endpoint="edit_page",
            view_func=edit_route,
            methods=["GET"],
        )
        server.add_url_rule(
            "/api/edit/actions",
            endpoint="edit_actions",
            view_func=edit_actions_route,
            methods=["POST"],
        )

        # --- Review Queue Page (was Orphan Images Admin Page) ---
        @login_required
        def orphans_route():
            """
            Review Queue: Images needing user decision.
            Shows orphans (no detections) AND low-confidence detections.
            Sorted oldest first.
            """
            from utils.db import fetch_review_queue_images
            from utils.path_manager import get_path_manager

            output_dir = config.get("OUTPUT_DIR", "output")
            save_threshold = config.get("SAVE_THRESHOLD", 0.65)
            pm = get_path_manager(output_dir)

            with get_connection() as conn:
                rows = fetch_review_queue_images(conn, save_threshold)

            orphans = []
            for row in rows:
                filename = row["filename"]
                timestamp = row["timestamp"] or ""
                review_reason = row["review_reason"]  # 'orphan' or 'low_confidence'
                max_od_conf = row["max_od_conf"]

                # Format date/time from timestamp (YYYYMMDD_HHMMSS)
                formatted_date = ""
                if len(timestamp) >= 15:
                    try:
                        dt = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        formatted_date = timestamp[:15]

                # Resolve paths using path_manager
                original_path = pm.get_original_path(filename)
                pm.get_derivative_path(filename, "thumb")

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
                # Assuming standard storage: originals/YYYY-MM-DD/filename
                # Timestamp is YYYYMMDD_HHMMSS
                full_url = ""
                if len(timestamp) >= 8:
                    date_folder_str = (
                        f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                    )
                    full_url = f"/uploads/originals/{date_folder_str}/{filename}"

                # Badge label for review reason
                if review_reason == "orphan":
                    reason_label = "No Detection"
                else:
                    conf_pct = round((max_od_conf or 0) * 100)
                    reason_label = f"Low Conf ({conf_pct}%)"

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
                        "max_od_conf": max_od_conf,
                    }
                )

            return render_template(
                "orphans.html", orphans=orphans, current_path="/admin/review"
            )

        server.add_url_rule(
            "/admin/review",
            endpoint="review_page",
            view_func=orphans_route,
            methods=["GET"],
        )
        # Note: Legacy delete route removed. Review Queue uses /api/review/decision.
        # File deletion is handled via Trash tab only (no file deletion in Review Queue).

        # --- Orphan Thumbnail On-Demand Generation ---
        @login_required
        def orphan_thumb_route(filename):
            """On-demand thumbnail generation for orphan images."""
            from flask import abort, send_file

            from utils.image_ops import generate_preview_thumbnail
            from utils.path_manager import get_path_manager

            output_dir = config.get("OUTPUT_DIR", "output")
            pm = get_path_manager(output_dir)

            # Resolve paths via PathManager
            original_path = pm.get_original_path(filename)
            preview_path = pm.get_preview_thumb_path(filename)

            # If preview already cached, serve it
            if preview_path.exists():
                return send_file(str(preview_path), mimetype="image/webp")

            # Original must exist to generate preview
            if not original_path.exists():
                abort(404)

            # Generate preview thumbnail
            success = generate_preview_thumbnail(
                str(original_path), str(preview_path), size=256
            )

            if success and preview_path.exists():
                return send_file(str(preview_path), mimetype="image/webp")
            else:
                abort(500)

        server.add_url_rule(
            "/api/review-thumb/<filename>",
            endpoint="review_thumb",
            view_func=orphan_thumb_route,
            methods=["GET"],
        )

        # --- Review Queue API ---
        @login_required
        def review_decision_route():
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
                        jsonify(
                            {"status": "error", "message": "No filenames provided"}
                        ),
                        400,
                    )

                if action not in ("confirm", "no_bird", "skip"):
                    return (
                        jsonify(
                            {"status": "error", "message": f"Invalid action: {action}"}
                        ),
                        400,
                    )

                # Skip action: no database change
                if action == "skip":
                    return jsonify(
                        {"status": "success", "updated": 0, "action": "skip"}
                    )

                # Map action to review_status
                status_map = {"confirm": "confirmed_bird", "no_bird": "no_bird"}
                new_status = status_map[action]

                with get_connection() as conn:
                    updated = update_review_status(conn, filenames, new_status)

                logger.info(f"Review decision: {action} -> {updated} images updated")
                return jsonify(
                    {"status": "success", "updated": updated, "action": action}
                )

            except Exception as e:
                logger.error(f"Error in review decision: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        server.add_url_rule(
            "/api/review/decision",
            endpoint="review_decision",
            view_func=review_decision_route,
            methods=["POST"],
        )

        # --- Analytics Dashboard (Svelte-rendered, read-only) ---
        def analytics_page_route():
            """Serves the analytics dashboard page with minimal HTML for Svelte mount."""
            return render_template("analytics.html")

        server.add_url_rule(
            "/analytics",
            endpoint="analytics_page",
            view_func=analytics_page_route,
            methods=["GET"],
        )

        # --- Analytics Dashboard (Pure Jinja2/CSS - No JavaScript) ---
        def analytics_pure_route():
            """Server-rendered analytics dashboard without Svelte - pure Jinja2/CSS."""
            import numpy as np

            # 1. Summary Stats
            summary = {
                "total_detections": 0,
                "total_species": 0,
                "date_range": {"first": None, "last": None},
            }
            try:
                with get_connection() as conn:
                    summary = fetch_analytics_summary(conn)
            except Exception as e:
                logger.error(f"Error fetching analytics summary: {e}")

            # 2. Time of Day Histogram (24 hourly bins)
            time_of_day = {
                "histogram": [],
                "peak_hour": None,
                "peak_hour_formatted": "—",
            }
            try:
                with get_connection() as conn:
                    rows = fetch_all_detection_times(conn)

                hours_float = []
                for row in rows:
                    t_str = row["time_str"]
                    if len(t_str) == 6:
                        h = int(t_str[0:2])
                        m = int(t_str[2:4])
                        hours_float.append(h + m / 60.0)

                if hours_float:
                    # Create 24 hourly bins
                    hist, edges = np.histogram(hours_float, bins=24, range=(0, 24))
                    max_count = max(hist) if max(hist) > 0 else 1

                    histogram_data = []
                    for i, count in enumerate(hist):
                        histogram_data.append(
                            {
                                "hour": i,
                                "count": int(count),
                                "height_pct": (
                                    round((count / max_count) * 100, 1)
                                    if max_count > 0
                                    else 0
                                ),
                            }
                        )
                    time_of_day["histogram"] = histogram_data

                    # Peak hour
                    peak_idx = np.argmax(hist)
                    time_of_day["peak_hour"] = peak_idx
                    time_of_day["peak_hour_formatted"] = f"{peak_idx:02d}:00"
            except Exception as e:
                logger.error(f"Error fetching time of day data: {e}")

            # 3. Species Activity with Sparklines
            species_activity = []
            try:
                with get_connection() as conn:
                    rows = fetch_species_timestamps(conn)

                # Group by species
                species_times = {}
                for r in rows:
                    sp = r["species"]
                    t_str = (
                        r["image_timestamp"][9:15]
                        if len(r["image_timestamp"]) >= 15
                        else ""
                    )
                    if len(t_str) == 6:
                        try:
                            h = int(t_str[0:2]) + int(t_str[2:4]) / 60.0
                            if sp not in species_times:
                                species_times[sp] = []
                            species_times[sp].append(h)
                        except Exception:
                            pass

                for sp, times in species_times.items():
                    if len(times) < 3:
                        continue  # Skip species with very few detections

                    # Create histogram for sparkline
                    hist, edges = np.histogram(times, bins=24, range=(0, 24))
                    max_val = max(hist) if max(hist) > 0 else 1
                    normalized = hist / max_val

                    # Generate SVG path for sparkline
                    points = []
                    for i, y in enumerate(normalized):
                        x = (i / 23) * 200  # Scale to SVG viewBox width
                        y_coord = 30 - (y * 28)  # Invert Y, leave some margin
                        prefix = "M" if i == 0 else "L"
                        points.append(f"{prefix} {x:.1f} {y_coord:.1f}")
                    sparkline_path = " ".join(points)

                    # Peak hour
                    peak_idx = np.argmax(hist)
                    peak_formatted = f"{peak_idx:02d}:00"

                    species_activity.append(
                        {
                            "species": sp,
                            "count": len(times),
                            "peak_hour_formatted": peak_formatted,
                            "sparkline_path": sparkline_path,
                            "median_hour": float(np.median(times)),
                        }
                    )

                # Sort by median activity time
                species_activity.sort(key=lambda x: x["median_hour"])
            except Exception as e:
                logger.error(f"Error fetching species activity: {e}")

            return render_template(
                "analytics_pure.html",
                summary=summary,
                time_of_day=time_of_day,
                species_activity=species_activity,
                current_path="/analytics-pure",
            )

        server.add_url_rule(
            "/analytics-pure",
            endpoint="analytics_pure",
            view_func=analytics_pure_route,
            methods=["GET"],
        )

        # --- Phase 2: Species Summary (server-rendered) ---
        def species_route():
            """Server-rendered species summary page using Jinja2 templates."""

            # Get all detections
            all_detections = get_captured_detections()

            # Get threshold from query param or config (query param overrides)
            try:
                min_score_param = request.args.get("min_score", type=float)
            except (ValueError, TypeError):
                min_score_param = None

            if min_score_param is not None:
                current_threshold = min_score_param
            else:
                current_threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)

            # Apply display threshold
            if current_threshold > 0:
                all_detections = [
                    d
                    for d in all_detections
                    if (d.get("score") or 0.0) >= current_threshold
                ]

            # Aggregate: best per species
            species_groups = {}
            for det in all_detections:
                cls_class = det.get("cls_class_name")
                od_class = det.get("od_class_name")
                species_key = (
                    cls_class
                    if cls_class
                    else (od_class if od_class else "Unclassified")
                )

                score = det.get("score") or 0.0
                if species_key not in species_groups or score > (
                    species_groups[species_key].get("score") or 0.0
                ):
                    species_groups[species_key] = det

            # Convert to list and enrich for template
            detections = []
            for species, det in sorted(
                species_groups.items(), key=lambda x: COMMON_NAMES.get(x[0], x[0])
            ):
                # Build enriched detection dict for template with Clean Slate URLs
                full_path = det.get("relative_path") or det.get(
                    "optimized_name_virtual", ""
                )
                thumb_virtual = det.get("thumbnail_path_virtual")

                if thumb_virtual:
                    display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    display_url = f"/uploads/derivatives/optimized/{full_path}"

                full_url = f"/uploads/derivatives/optimized/{full_path}"
                original_url = (
                    f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                )

                # Extract date/time from image_timestamp (YYYYMMDD_HHMMSS)
                ts = det.get("image_timestamp", "")
                if len(ts) >= 15:
                    date_str = ts[:8]
                    time_str = ts[9:15]
                    formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
                    formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    gallery_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    formatted_date = ""
                    formatted_time = ""
                    gallery_date = ""

                detections.append(
                    {
                        "detection_id": det.get("detection_id"),
                        "display_path": display_url,
                        "full_path": full_url,
                        "original_path": original_url,
                        "common_name": COMMON_NAMES.get(
                            species, species.replace("_", " ")
                        ),
                        "od_class_name": det.get("od_class_name", ""),
                        "od_confidence": det.get("od_confidence", 0.0) or 0.0,
                        "cls_class_name": det.get("cls_class_name", ""),
                        "cls_confidence": det.get("cls_confidence", 0.0) or 0.0,
                        "score": det.get("score", 0.0) or 0.0,
                        "formatted_date": formatted_date,
                        "formatted_time": formatted_time,
                        "gallery_date": gallery_date,
                        # Fields required by detection_modal.html
                        "siblings": [],
                        "sibling_count": 1,
                        "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                        "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                        "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                        "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                    }
                )

            return render_template(
                "species.html",
                current_path="/species",
                detections=detections,
                image_width=IMAGE_WIDTH,
                current_threshold=current_threshold,
                species_count=len(detections),
            )

        server.add_url_rule(
            "/species", endpoint="species", view_func=species_route, methods=["GET"]
        )

        # --- Phase 3: Gallery Routes (server-rendered) ---
        def gallery_route():
            """Server-rendered main gallery page with daily covers."""
            daily_covers = get_daily_covers()
            sorted_dates = sorted(daily_covers.keys(), reverse=True)

            days = []
            for date_str in sorted_dates:
                data = daily_covers.get(date_str)
                if not data:
                    continue
                # Format date nicely
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    display_date = dt.strftime("%A, %d. %B %Y")
                except Exception:
                    display_date = date_str

                days.append(
                    {
                        "date": date_str,
                        "display_date": display_date,
                        "cover_path": data.get("path", ""),
                        "count": data.get("count", 0),
                    }
                )

            return render_template(
                "gallery.html",
                current_path="/gallery",
                days=days,
                image_width=IMAGE_WIDTH,
            )

        server.add_url_rule(
            "/gallery", endpoint="gallery", view_func=gallery_route, methods=["GET"]
        )

        def subgallery_route(date):
            """Server-rendered subgallery page for a specific date."""
            # Validate date format
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                return "Invalid date format.", 400

            # Get query params
            page = request.args.get("page", 1, type=int)
            sort_by = request.args.get("sort", "score")

            # Get threshold from query param or config
            try:
                min_score_param = request.args.get("min_score", type=float)
            except (ValueError, TypeError):
                min_score_param = None

            if min_score_param is not None:
                current_threshold = min_score_param
            else:
                current_threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)

            # Fetch detections
            sql_order = "time" if sort_by in ("time_asc", "time_desc") else "score"
            rows = fetch_detections_for_gallery(db_conn, date, order_by=sql_order)
            detections_raw = [dict(row) for row in rows]

            # Apply score filter
            if current_threshold > 0:
                detections_raw = [
                    d
                    for d in detections_raw
                    if (d.get("score") or 0.0) >= current_threshold
                ]

            # Python-side sorting adjustments
            if sort_by == "species":
                detections_raw.sort(
                    key=lambda x: (
                        x.get("cls_class_name") or x.get("od_class_name") or ""
                    ).lower()
                )
            elif sort_by == "time_asc":
                detections_raw.sort(key=lambda x: x.get("image_timestamp", ""))
            elif sort_by == "time_desc":
                detections_raw.sort(
                    key=lambda x: x.get("image_timestamp", ""), reverse=True
                )
            # score comes sorted from DB

            total_items = len(detections_raw)
            total_pages = math.ceil(total_items / PAGE_SIZE) or 1
            page = max(1, min(page, total_pages))
            start_index = (page - 1) * PAGE_SIZE
            end_index = page * PAGE_SIZE

            page_detections_raw = detections_raw[start_index:end_index]

            # Enrich detections for template
            def enrich_detection(det):
                full_path = det.get("relative_path") or det.get(
                    "optimized_name_virtual", ""
                )  # YYYYMMDD/foo.webp
                thumb_virtual = det.get("thumbnail_path_virtual")

                if thumb_virtual:
                    display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    display_url = f"/uploads/derivatives/optimized/{full_path}"

                full_url = f"/uploads/derivatives/optimized/{full_path}"
                original_url = (
                    f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                )

                ts = det.get("image_timestamp", "")
                if len(ts) >= 15:
                    date_str = ts[:8]
                    time_str = ts[9:15]
                    formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
                    formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    formatted_date = ""
                    formatted_time = ""

                species_key = (
                    det.get("cls_class_name") or det.get("od_class_name") or ""
                )
                sibling_count = det.get("sibling_count", 1) or 1

                # Load sibling detections if multiple birds on same image
                siblings = []
                if sibling_count > 1:
                    original_name = det.get("original_name", "")
                    if original_name:
                        from utils.db import fetch_sibling_detections

                        sibling_rows = fetch_sibling_detections(db_conn, original_name)
                        for sib in sibling_rows:
                            sib_species_key = (
                                sib["cls_class_name"] or sib["od_class_name"] or ""
                            )
                            sib_thumb = sib["thumbnail_path_virtual"]
                            siblings.append(
                                {
                                    "detection_id": sib["detection_id"],
                                    "common_name": COMMON_NAMES.get(
                                        sib_species_key,
                                        sib_species_key.replace("_", " "),
                                    ),
                                    "od_class_name": sib["od_class_name"] or "",
                                    "od_confidence": sib["od_confidence"] or 0.0,
                                    "cls_class_name": sib["cls_class_name"] or "",
                                    "cls_confidence": sib["cls_confidence"] or 0.0,
                                    "score": sib["score"] or 0.0,
                                    "thumb_url": (
                                        f"/uploads/derivatives/thumbs/{sib_thumb}"
                                        if sib_thumb
                                        else ""
                                    ),
                                    "bbox_x": sib["bbox_x"] or 0.0,
                                    "bbox_y": sib["bbox_y"] or 0.0,
                                    "bbox_w": sib["bbox_w"] or 0.0,
                                    "bbox_h": sib["bbox_h"] or 0.0,
                                }
                            )

                return {
                    "detection_id": det.get("detection_id"),
                    "display_url": display_url,
                    "full_url": full_url,
                    "original_url": original_url,
                    "display_path": display_url,
                    "full_path": full_url,
                    "original_path": original_url,
                    "common_name": COMMON_NAMES.get(
                        species_key, species_key.replace("_", " ")
                    ),
                    "od_class_name": det.get("od_class_name", ""),
                    "od_confidence": det.get("od_confidence", 0.0) or 0.0,
                    "cls_class_name": det.get("cls_class_name", ""),
                    "cls_confidence": det.get("cls_confidence", 0.0) or 0.0,
                    "score": det.get("score", 0.0) or 0.0,
                    "formatted_date": formatted_date,
                    "formatted_time": formatted_time,
                    "sibling_count": sibling_count,
                    "siblings": siblings,
                    "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                    "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                    "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                    "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                }

            detections = [enrich_detection(d) for d in page_detections_raw]

            # Species of the Day (page 1 only)
            species_of_day = []
            if page == 1:
                species_groups = {}
                for det in detections_raw:
                    cls_class = det.get("cls_class_name")
                    od_class = det.get("od_class_name")
                    species_key = (
                        cls_class
                        if cls_class
                        else (od_class if od_class else "Unclassified")
                    )
                    score = det.get("score") or 0.0
                    if species_key not in species_groups or score > (
                        species_groups[species_key].get("score") or 0.0
                    ):
                        species_groups[species_key] = det
                species_of_day = [
                    enrich_detection(d)
                    for d in sorted(
                        species_groups.values(),
                        key=lambda x: x.get("score", 0),
                        reverse=True,
                    )
                ]

            # Build pagination range
            window = 2
            pagination_range = []
            range_start = max(1, page - window)
            range_end = min(total_pages, page + window)

            if range_start > 1:
                pagination_range.append(1)
                if range_start > 2:
                    pagination_range.append("...")

            for p in range(range_start, range_end + 1):
                pagination_range.append(p)

            if range_end < total_pages:
                if range_end < total_pages - 1:
                    pagination_range.append("...")
                pagination_range.append(total_pages)

            return render_template(
                "subgallery.html",
                current_path=f"/gallery/{date}",
                date=date,
                page=page,
                total_pages=total_pages,
                total_items=total_items,
                sort_by=sort_by,
                current_threshold=current_threshold,
                detections=detections,
                species_of_day=species_of_day,
                pagination_range=pagination_range,
                image_width=IMAGE_WIDTH,
            )

        server.add_url_rule(
            "/gallery/<date>",
            endpoint="subgallery",
            view_func=subgallery_route,
            methods=["GET"],
        )

        @server.route("/logs")
        @login_required
        def logs_route():
            """Displays the last 200 lines of the app log."""
            log_file = Path(config["OUTPUT_DIR"]) / "logs" / "app.log"
            logs = ""
            if log_file.exists():
                try:
                    with open(log_file, encoding="utf-8") as f:
                        # Use a more efficient way to get last N lines if file is huge,
                        # but for now, reading all and slicing is simple for app logs.
                        lines = f.readlines()
                        logs = "".join(lines[-200:])
                except Exception as e:
                    logs = f"Error reading log file: {e}"
            else:
                logs = f"Log file not found at {log_file}"

            return render_template("logs.html", logs=logs)

    setup_web_routes(server)

    # --- Phase 4: Homepage (Live Stream) Migrated to Flask (Final Phase) ---
    def index_route():
        """Server-rendered Homepage / Live Stream."""
        today_iso = datetime.now().strftime("%Y-%m-%d")

        # 1. Day Count & Title
        today_count = 0
        try:
            with get_connection() as conn:
                today_count = fetch_day_count(conn, today_iso)
            title = f"Live Stream (Today's Detections: {today_count})"
        except Exception as e:
            logger.error(f"Error fetching day count: {e}")
            title = "Live Stream"

        # 1b. Dashboard Stats (All-time stats for engagement)
        dashboard_stats = {
            "total_detections": 0,
            "total_species": 0,
            "today_count": today_count,
            "first_date": None,
            "last_date": None,
        }
        try:
            with get_connection() as conn:
                summary = fetch_analytics_summary(conn)
                dashboard_stats["total_detections"] = summary.get("total_detections", 0)
                dashboard_stats["total_species"] = summary.get("total_species", 0)
                date_range = summary.get("date_range", {})
                dashboard_stats["first_date"] = date_range.get("first")
                dashboard_stats["last_date"] = date_range.get("last")
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {e}")

        # 2. Latest Detections (Top 5)
        latest_detections = []
        try:
            with get_connection() as conn:
                rows = fetch_detections_for_gallery(
                    conn, today_iso, limit=5, order_by="time"
                )
                # Enrich for template
                for row in rows:
                    det = dict(row)
                    full_path = det.get("relative_path") or det.get(
                        "optimized_name_virtual", ""
                    )
                    thumb_virtual = det.get("thumbnail_path_virtual")

                    if thumb_virtual:
                        display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_url = f"/uploads/derivatives/optimized/{full_path}"

                    full_url = f"/uploads/derivatives/optimized/{full_path}"
                    original_url = (
                        f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                    )

                    # Compat variables
                    display_path = display_url
                    original_path = original_url
                    ts = det.get("image_timestamp", "")

                    formatted_time = ""
                    formatted_date = ""
                    if len(ts) >= 15:
                        date_str = ts[:8]
                        time_str = ts[9:15]  # HHMMSS
                        formatted_time = (
                            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        )
                        formatted_date = (
                            f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
                        )

                    latest_detections.append(
                        {
                            "detection_id": det.get("detection_id"),
                            "common_name": COMMON_NAMES.get(
                                det.get("cls_class_name") or det.get("od_class_name"),
                                "Unknown",
                            ),
                            "latin_name": det.get("cls_class_name")
                            or det.get("od_class_name")
                            or "",
                            "od_class_name": det.get("od_class_name") or "",
                            "od_confidence": det.get("od_confidence") or 0.0,
                            "cls_class_name": det.get("cls_class_name") or "",
                            "cls_confidence": det.get("cls_confidence") or 0.0,
                            "score": det.get("score", 0.0) or 0.0,
                            "display_path": display_path,
                            "full_path": full_url,
                            "original_path": original_path,
                            "formatted_time": formatted_time,
                            "formatted_date": formatted_date,
                            "sibling_count": det.get("sibling_count", 1) or 1,
                            "siblings": [],  # Stream page doesn't need to load all siblings for now, just prevent crash
                            "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                            "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                            "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                            "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                        }
                    )
        except Exception as e:
            logger.error(f"Error fetching latest detections: {e}")

        # 3. Daily Summary (Gallery Style - Best per species)
        # Used for "Today's Summary" section - visual grid
        visual_summary = []
        try:
            with get_connection() as conn:
                # fetch all for today to group by species
                rows = fetch_detections_for_gallery(conn, today_iso, order_by="score")
                all_today = [dict(r) for r in rows]

                species_groups = {}
                for det in all_today:
                    cls = det.get("cls_class_name")
                    od = det.get("od_class_name")
                    s_key = cls if cls else (od if od else "Unclassified")
                    score = det.get("score", 0.0) or 0.0
                    if s_key not in species_groups or score > species_groups[s_key].get(
                        "score", 0.0
                    ):
                        species_groups[s_key] = det

                # Sort by score desc for display
                sorted_summary = sorted(
                    species_groups.values(),
                    key=lambda x: x.get("score", 0.0),
                    reverse=True,
                )

                for det in sorted_summary:
                    # Clean Slate URL construction
                    full_path = det.get("relative_path") or det.get(
                        "optimized_name_virtual", ""
                    )
                    thumb_virtual = det.get("thumbnail_path_virtual")

                    if thumb_virtual:
                        display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_url = f"/uploads/derivatives/optimized/{full_path}"

                    full_url = f"/uploads/derivatives/optimized/{full_path}"
                    original_url = (
                        f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                    )

                    ts = det.get("image_timestamp", "")
                    formatted_time = ""
                    formatted_date = ""
                    if len(ts) >= 15:
                        date_str = ts[:8]
                        time_str = ts[9:15]
                        formatted_time = (
                            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        )
                        formatted_date = (
                            f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
                        )

                    visual_summary.append(
                        {
                            "detection_id": det.get("detection_id"),
                            "common_name": COMMON_NAMES.get(
                                det.get("cls_class_name") or det.get("od_class_name"),
                                "Unknown",
                            ),
                            "latin_name": det.get("cls_class_name")
                            or det.get("od_class_name")
                            or "",
                            "od_class_name": det.get("od_class_name") or "",
                            "od_confidence": det.get("od_confidence") or 0.0,
                            "cls_class_name": det.get("cls_class_name") or "",
                            "cls_confidence": det.get("cls_confidence") or 0.0,
                            "score": det.get("score", 0.0) or 0.0,
                            "display_path": display_url,
                            "full_path": full_url,
                            "original_path": original_url,
                            "formatted_time": formatted_time,
                            "formatted_date": formatted_date,
                            "gallery_date": today_iso,  # For "Go to Day" button in modal
                            "sibling_count": det.get("sibling_count", 1) or 1,
                            "siblings": [],
                            "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                            "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                            "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                            "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                        }
                    )

        except Exception as e:
            logger.error(f"Error fetching visual summary: {e}")

        # 4. Species Summary Table (enriched with best image from visual_summary)
        species_summary_table = []
        try:
            species_summary_table = get_daily_species_summary(today_iso)

            # Create lookup for count per species
            species_count_map = {}
            for item in species_summary_table:
                species_key = item.get("species", "")
                if species_key:
                    species_count_map[species_key] = item.get("count", 0)

            # Enrich visual_summary with count
            for det in visual_summary:
                species_key = det.get("species") or det.get("latin_name", "")
                det["count"] = species_count_map.get(species_key, 0)

        except Exception as e:
            logger.error(f"Error fetching species summary table: {e}")

        return render_template(
            "stream.html",
            current_path="/",  # Active nav state
            title=title,
            latest_detections=latest_detections,
            visual_summary=visual_summary,
            species_summary=species_summary_table,
            dashboard_stats=dashboard_stats,
            empty_latest_message="No detections yet today.",
            image_width=IMAGE_WIDTH,
            today_iso=today_iso,
        )

    server.add_url_rule("/", endpoint="index", view_func=index_route, methods=["GET"])

    RUNTIME_BOOL_KEYS = {
        "DAY_AND_NIGHT_CAPTURE",
        "TELEGRAM_ENABLED",
        "DEBUG_MODE",
        "EXIF_GPS_ENABLED",
    }
    RUNTIME_NUMBER_KEYS = {
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "DETECTION_INTERVAL_SECONDS",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "TELEGRAM_COOLDOWN",
        "GALLERY_DISPLAY_THRESHOLD",
    }

    SETTING_LABELS = {
        "CONFIDENCE_THRESHOLD_DETECTION": "OD Confidence Threshold (Minimum detector confidence to identify a bird)",
        "SAVE_THRESHOLD": "OD Save Threshold (Minimum detector confidence to save image to disk)",
        "DETECTION_INTERVAL_SECONDS": "Detection Interval (Seconds between AI analysis cycles - higher = less CPU)",
        "CLASSIFIER_CONFIDENCE_THRESHOLD": "CLS Confidence Threshold (Minimum classifier confidence for species naming)",
        "DAY_AND_NIGHT_CAPTURE": "24/7 Capture (Enable or disable night-time detection)",
        "DAY_AND_NIGHT_CAPTURE_LOCATION": "Sun Event Location (City name for sunrise/sunset checks)",
        "TELEGRAM_ENABLED": "Telegram Alerts (Enable notification messages with photos)",
        "TELEGRAM_COOLDOWN": "Telegram Cooldown (Minimum seconds between consecutive alerts)",
        "STREAM_FPS": "Stream Display FPS (Visual smoothness in the web browser)",
        "STREAM_FPS_CAPTURE": "Stream Capture FPS (Internal capture rate - affects CPU impact)",
        "EDIT_PASSWORD": "Edit Password (Required to apply changes or delete images)",
        "INGEST_DIR": "Ingest Directory (Source path for manual image imports)",
        "OUTPUT_DIR": "Output Directory (Main storage for images and database)",
        "MODEL_BASE_PATH": "Model Base Path (Storage directory for AI model weights)",
        "DEBUG_MODE": "Debug Mode (Enables verbose logging and developer features)",
        "CPU_LIMIT": "CPU Core Limit (Maximum number of cores allowed for processing)",
        "VIDEO_SOURCE": "Video Source (Input RTSP URL or secondary camera ID)",
        "DETECTOR_MODEL_CHOICE": "Primary Detector Model (The AI engine selected at boot)",
        "STREAM_WIDTH_OUTPUT_RESIZE": "Stream Output Width (Visual resolution for the live feed)",
        "LOCATION_DATA": "Geographic Coordinates (Latitude/Longitude for metadata)",
        "GALLERY_DISPLAY_THRESHOLD": "Species Summary Min. Score (Quality score threshold: 50% OD + 50% CLS confidence, or OD-only if no classification)",
        "TELEGRAM_BOT_TOKEN": "Telegram Bot Token (From BotFather)",
        "TELEGRAM_CHAT_ID": "Telegram Chat ID (Numeric ID or JSON list of IDs)",
        "EXIF_GPS_ENABLED": "Write GPS to Exif (Safe to disable for privacy)",
    }

    RUNTIME_KEYS_ORDER = [
        "VIDEO_SOURCE",
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "DETECTION_INTERVAL_SECONDS",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "DAY_AND_NIGHT_CAPTURE",
        "DAY_AND_NIGHT_CAPTURE_LOCATION",
        "LOCATION_DATA",
        "EXIF_GPS_ENABLED",
        "TELEGRAM_COOLDOWN",
        "TELEGRAM_ENABLED",
        "GALLERY_DISPLAY_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "EDIT_PASSWORD",
        "DEBUG_MODE",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]

    ADVANCED_KEYS = {
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "EDIT_PASSWORD",
        "DEBUG_MODE",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    }

    SYSTEM_KEYS_ORDER = [
        "OUTPUT_DIR",
        "INGEST_DIR",
        "MODEL_BASE_PATH",
    ]

    def _format_value(value):
        if isinstance(value, dict) and "latitude" in value and "longitude" in value:
            return f"{value['latitude']}, {value['longitude']}"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    # ---------------------------------------------------------
    # FLASK ROUTES FOR SETTINGS (Phase 7)
    # ---------------------------------------------------------

    @server.route("/settings", methods=["GET"])
    @login_required
    def settings_route():
        trash_count = 0
        try:
            with get_connection() as conn:
                trash_count = fetch_trash_count(conn)
        except Exception as e:
            logger.error(f"Failed to fetch trash count: {e}")

        payload = get_settings_payload()
        return render_template(
            "settings.html",
            payload=payload,
            runtime_keys=RUNTIME_KEYS_ORDER,
            bool_keys=RUNTIME_BOOL_KEYS,
            number_keys=RUNTIME_NUMBER_KEYS,
            advanced_keys=ADVANCED_KEYS,
            system_keys=SYSTEM_KEYS_ORDER,
            labels=SETTING_LABELS,
            trash_count=trash_count,
        )

    @server.route("/api/settings/update", methods=["POST"])
    @login_required
    def update_settings_route():
        updates = {}
        for key in RUNTIME_KEYS_ORDER:
            val = request.form.get(key)
            if val is not None:
                # Type Conversion based on Keys
                if key in RUNTIME_BOOL_KEYS:
                    updates[key] = val.lower() == "true"
                elif key in RUNTIME_NUMBER_KEYS:
                    try:
                        val_clean = val.replace(",", ".")
                        if "." in val_clean:
                            updates[key] = float(val_clean)
                        else:
                            updates[key] = int(val_clean)
                    except ValueError:
                        pass  # Ignore execution
                else:
                    updates[key] = val

        logger.info(f"Received settings update request: {updates}")

        valid, errors = validate_runtime_updates(updates)
        if errors:
            flash(f"Errors: {errors}", "error")
        else:
            update_runtime_settings(valid)
            # Trigger component updates (Debug Mode, Video Source, etc.)
            detection_manager.update_configuration(valid)
            flash("Settings updated successfully.", "success")

        return redirect(url_for("settings_route"))

    @server.route("/api/settings/ingest", methods=["POST"])
    @login_required
    def ingest_route():
        import threading

        # Determine Path (Logic copied from trigger_ingest)
        env_path = config.get("INGEST_DIR")
        cwd_path = os.path.abspath(os.path.join(os.getcwd(), "ingest"))
        if os.path.exists(env_path):
            ingest_path = env_path
        elif os.path.exists(cwd_path):
            ingest_path = cwd_path
        else:
            ingest_path = env_path

        def run_ingest():
            detection_manager.start_user_ingest(ingest_path)

        t = threading.Thread(target=run_ingest)
        t.start()

        flash("Ingest process started in background.", "info")
        return redirect(url_for("settings_route"))

    @server.route("/api/system/versions", methods=["GET"])
    @login_required
    def system_versions_route():
        data = {
            "app_version": "Unknown",
            "kernel": "Unknown",
            "os": "Unknown",
            "bootloader": "Unknown",
        }

        # 1. App Version
        try:
            if os.path.exists("version.txt"):
                with open("version.txt") as f:
                    data["app_version"] = f.read().strip()
        except Exception:
            pass

        # 2. Kernel
        try:
            data["kernel"] = platform.release()
        except Exception:
            pass

        # 3. OS
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            data["os"] = line.split("=")[1].strip().strip('"')
                            break
        except Exception:
            pass

        # 4. Bootloader
        # Note: rpi-eeprom-update may require elevated privileges. With NNP=true,
        # sudo is blocked. If unprivileged access fails, bootloader info stays "Unknown".
        try:
            import shutil

            if shutil.which("rpi-eeprom-update"):
                res = subprocess.run(
                    ["rpi-eeprom-update"], capture_output=True, text=True, timeout=5
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if "CURRENT:" in line:
                            parts = line.split("CURRENT:", 1)
                            if len(parts) > 1:
                                data["bootloader"] = parts[1].strip()
                                break
        except Exception:
            pass

        return jsonify(data)

    # --- Shutdown / Restart Routes ---
    @server.route("/api/system/shutdown", methods=["POST"])
    @login_required
    def shutdown_route():
        # Security: Only authenticated users can trigger shutdown
        try:
            logger.warning("System shutdown initiated via Web UI.")

            import shutil

            if not (os.path.isdir("/run/systemd/system") and shutil.which("systemctl")):
                logger.warning(
                    "Shutdown ignored: systemd not available (likely container)."
                )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Power management is intentionally disabled in non-systemd environments.",
                        }
                    ),
                    400,
                )

            def delayed_shutdown():
                time.sleep(2)
                try:
                    import subprocess

                    subprocess.run(
                        ["systemctl", "--no-ask-password", "--no-wall", "poweroff"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except subprocess.CalledProcessError as e:
                    details = (e.stderr or e.stdout or "").strip()
                    logger.error(
                        f"Shutdown command failed (Exit {e.returncode}): {details}"
                    )
                except Exception as e:
                    logger.error(f"Shutdown command failed: {e}")

            import threading

            t = threading.Thread(target=delayed_shutdown)
            t.start()

            return (
                jsonify({"status": "success", "message": "System is shutting down..."}),
                200,
            )
        except Exception as e:
            logger.error(f"Error initiating shutdown: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @server.route("/api/system/restart", methods=["POST"])
    @login_required
    def restart_route():
        # Security: Only authenticated users can trigger restart
        try:
            logger.warning("System restart initiated via Web UI.")

            import shutil

            if not (os.path.isdir("/run/systemd/system") and shutil.which("systemctl")):
                logger.warning(
                    "Restart ignored: systemd not available (likely container)."
                )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Power management is intentionally disabled in non-systemd environments.",
                        }
                    ),
                    400,
                )

            def delayed_restart():
                time.sleep(2)
                try:
                    import subprocess

                    subprocess.run(
                        ["systemctl", "--no-ask-password", "--no-wall", "reboot"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except subprocess.CalledProcessError as e:
                    details = (e.stderr or e.stdout or "").strip()
                    logger.error(
                        f"Restart command failed (Exit {e.returncode}): {details}"
                    )
                except Exception as e:
                    logger.error(f"Restart command failed: {e}")

            import threading

            t = threading.Thread(target=delayed_restart)
            t.start()

            return (
                jsonify({"status": "success", "message": "System is restarting..."}),
                200,
            )
        except Exception as e:
            logger.error(f"Error initiating restart: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @server.route("/api/system/stats", methods=["GET"])
    @login_required
    def system_stats_route():
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()

            # Disk usage for OUTPUT_DIR partition
            disk = None
            try:
                output_dir = config.get("OUTPUT_DIR", "./data/output")
                disk_usage = psutil.disk_usage(output_dir)
                disk = {
                    "total_gb": round(disk_usage.total / (1024**3), 1),
                    "used_gb": round(disk_usage.used / (1024**3), 1),
                    "free_gb": round(disk_usage.free / (1024**3), 1),
                    "percent": disk_usage.percent,
                }
            except Exception:
                pass

            # Get CPU temperature
            temp = None
            try:
                # Raspberry Pi: use vcgencmd
                import subprocess

                result = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    # Output: "temp=45.0'C"
                    temp_str = result.stdout.strip()
                    temp = float(temp_str.replace("temp=", "").replace("'C", ""))
            except Exception:
                # Fallback: try psutil sensors
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for _name, entries in temps.items():
                            if entries:
                                temp = entries[0].current
                                break
                except Exception:
                    pass

            response = {"status": "success", "cpu": cpu_percent, "ram": mem.percent}
            if temp is not None:
                response["temp"] = temp
            if disk is not None:
                response["disk"] = disk
            return jsonify(response)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # ==========================================================================
    # INBOX ROUTES (Web Upload with Explicit Processing)
    # ==========================================================================

    # Thread state for inbox processing
    _inbox_processing = {"active": False, "lock": __import__("threading").Lock()}

    @server.route("/inbox")
    @login_required
    def inbox_page():
        return render_template("inbox.html")

    @server.route("/api/inbox", methods=["POST"])
    @login_required
    def inbox_upload():
        """
        Handle file uploads to inbox/pending.
        Max 20 files, max 50MB each.
        """
        try:
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
            path_mgr = get_path_manager(output_dir)
            pending_dir = path_mgr.get_inbox_pending_dir()

            for f in files:
                if not f or not f.filename:
                    continue

                # Extension check (Source of Truth)
                ext = os.path.splitext(f.filename.lower())[1]
                if ext not in ALLOWED_EXTENSIONS:
                    errors.append(f"{f.filename}: Ungültiges Format (nur JPG/PNG)")
                    continue

                # Size check (read first, check size)
                f.seek(0, 2)  # Seek to end
                size = f.tell()
                f.seek(0)  # Reset

                if size > MAX_FILE_SIZE:
                    errors.append(f"{f.filename}: Datei zu groß (max 50 MB)")
                    continue

                # Safe filename
                safe_name = secure_filename(f.filename)
                if not safe_name:
                    safe_name = f"upload_{int(time.time() * 1000)}{ext}"

                # SKIP duplicates instead of renaming
                target_path = pending_dir / safe_name
                if target_path.exists():
                    skipped.append(safe_name)
                    continue  # Skip this file entirely

                try:
                    f.save(str(target_path))
                    uploaded.append(safe_name)
                except Exception as e:
                    errors.append(f"{f.filename}: Speichern fehlgeschlagen ({e})")

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

    @server.route("/api/inbox/status", methods=["GET"])
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
            path_mgr = get_path_manager(output_dir)

            pending_dir = path_mgr.inbox_pending_dir
            processed_dir = path_mgr.inbox_dir / "processed" / today
            skipped_dir = path_mgr.inbox_dir / "skipped" / today
            error_dir = path_mgr.inbox_error_dir

            pending_count = (
                len(list(pending_dir.iterdir())) if pending_dir.exists() else 0
            )
            processed_today = (
                len(list(processed_dir.iterdir())) if processed_dir.exists() else 0
            )
            skipped_today = (
                len(list(skipped_dir.iterdir())) if skipped_dir.exists() else 0
            )
            error_count = len(list(error_dir.iterdir())) if error_dir.exists() else 0

            # Detection state check
            detection_running = not detection_manager.paused

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

    @server.route("/api/inbox/process", methods=["POST"])
    @login_required
    def inbox_process():
        """
        Start processing of inbox/pending files.
        Policy: Detection MUST be stopped (paused) before processing.
        Returns 409 if detection running or processing already active.
        """
        try:
            # Check 1: Already processing?
            with _inbox_processing["lock"]:
                if _inbox_processing["active"]:
                    return jsonify({"error": "Verarbeitung läuft bereits"}), 409

            # Note: Detection state no longer blocks import
            # The ingest process queues files for the detection manager

            # Get path manager
            output_dir = config.get("OUTPUT_DIR", "./data/output")
            path_mgr = get_path_manager(output_dir)

            # Take snapshot of pending files
            pending_dir = path_mgr.get_inbox_pending_dir()
            snapshot = list(pending_dir.iterdir())
            file_count = len([f for f in snapshot if f.is_file()])

            if file_count == 0:
                return (
                    jsonify({"message": "Keine Dateien zu verarbeiten", "count": 0}),
                    200,
                )

            # Start background processing
            import threading

            def run_inbox_ingest():
                from utils.ingest import ingest_inbox_folder

                try:
                    with _inbox_processing["lock"]:
                        _inbox_processing["active"] = True

                    logger.info(f"Starting inbox ingest for {file_count} files")
                    ingest_inbox_folder(
                        str(pending_dir), [str(f) for f in snapshot if f.is_file()]
                    )
                    logger.info("Inbox ingest completed")

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
                        "message": f"Verarbeitung von {file_count} Dateien gestartet",
                        "count": file_count,
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Inbox process error: {e}")
            return jsonify({"error": str(e)}), 500

    # ==========================================================================
    # BACKUP ROUTES (Streaming Archive Generation)
    # ==========================================================================

    @server.route("/backup")
    @login_required
    def backup_page():
        return render_template("backup.html")

    @server.route("/api/backup/stats", methods=["GET"])
    @login_required
    def backup_stats():
        """Returns statistics about data available for backup."""
        try:
            from utils.backup import get_backup_stats

            stats = get_backup_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Backup stats error: {e}")
            return jsonify({"error": str(e)}), 500

    @server.route("/api/backup/create", methods=["POST"])
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

            # Stream the archive with auto-pause/resume
            from utils.backup import stream_backup_archive

            def generate_with_pause():
                """Generator that auto-pauses detection during backup streaming."""
                # Store previous state to restore correctly
                was_paused = detection_manager.paused
                
                try:
                    # Pause detection if not already paused
                    if not was_paused:
                        logger.info("Backup: Pausing detection for backup...")
                        detection_manager.paused = True
                        time.sleep(1)  # Brief wait for detection loop to pause
                    
                    yield from stream_backup_archive(
                        include_db=include_db,
                        include_originals=include_originals,
                        include_derivatives=include_derivatives,
                        include_settings=include_settings,
                    )
                finally:
                    # Restore previous state
                    if not was_paused:
                        logger.info("Backup: Resuming detection after backup.")
                        detection_manager.paused = False

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

    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    return server
