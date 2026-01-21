# ------------------------------------------------------------------------------
# web_interface.py
# ------------------------------------------------------------------------------

import os
import json
import math
from urllib.parse import parse_qs
import re
import logging
from flask import Flask, send_from_directory, Response, request, jsonify, render_template, session, redirect, url_for, flash, has_request_context
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from config import (
    get_config,
    get_settings_payload,
    validate_runtime_updates,
    update_runtime_settings,
    save_settings_yaml,
)
from utils.db import (
    get_connection,
    fetch_detections_for_gallery,
    fetch_daily_covers,
    reject_detections,
    update_downloaded_timestamp,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_hourly_counts,
    fetch_trash_items,
    fetch_trash_count,
    restore_detections,
    purge_detections,
    # Analytics functions
    fetch_all_time_daily_counts,
    fetch_analytics_summary,
    fetch_all_detection_times,
    fetch_species_timestamps,
)
from utils.path_manager import get_path_manager
from utils.file_gc import hard_delete_detections

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
    CONFIDENCE_THRESHOLD_DETECTION = config["CONFIDENCE_THRESHOLD_DETECTION"]
    CLASSIFIER_CONFIDENCE_THRESHOLD = config["CLASSIFIER_CONFIDENCE_THRESHOLD"]
    EDIT_PASSWORD = config["EDIT_PASSWORD"]
    logger.info(
        f"Loaded EDIT_PASSWORD: {'***' if EDIT_PASSWORD and EDIT_PASSWORD != 'default_pass' else '<Not Set or Default>'}"
    )



    if not EDIT_PASSWORD or EDIT_PASSWORD in ["SECRET_PASSWORD", "default_pass"]:
        logger.warning(
            "EDIT_PASSWORD not set securely in .env or settings.yaml. Access might be restricted or insecure."
        )

    RECENT_IMAGES_COUNT = 10
    IMAGE_WIDTH = 150
    PAGE_SIZE = 50

    common_names_file = os.path.join(os.getcwd(), "assets", "common_names_DE.json")
    try:
        with open(common_names_file, "r", encoding="utf-8") as f:
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
        try:
            rows = fetch_daily_covers(db_conn)
            for row in rows:
                date_key = row["date_key"] # Already YYYY-MM-DD
                optimized_name = row["optimized_name_virtual"]
                if not date_key or not optimized_name:
                    continue
                
                # Use virtual thumbnail path corresponding to new route structure
                thumb_path_virtual = row["thumbnail_path_virtual"] # YYYYMMDD/filename.webp
                
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
                    "count": row["detection_count"]
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
            rows = fetch_detections_for_gallery(db_conn, order_by="time") # Most recent first
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
        min_font_size = 12
        min_font_size_percent = 0.05

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
            custom_font = ImageFont.load_default()
            timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
            bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img_width - text_width - padding_x
            text_y = img_height - text_height - padding_y
            draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")
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
                    b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                    + frame_data + b"\r\n"
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
    server.secret_key = os.environ.get("FLASK_SECRET_KEY", "watchmybirds-dev-key-change-in-production")

    # --- Auth Helper ---
    def login_required(f):
        """Decorator to require authentication for Flask routes."""
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import session, redirect, url_for
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
                    logger.error(f"Cannot regenerate {filename}: Original missing at {original_path}")
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
                    
                    cursor = db_conn.execute("""
                        SELECT bbox_x, bbox_y, bbox_w, bbox_h 
                        FROM detections 
                        WHERE image_filename = ? 
                        ORDER BY detection_id ASC 
                        LIMIT 1 OFFSET ?
                    """, (original_filename, crop_index - 1))
                    
                    row = cursor.fetchone()
                    if not row:
                        logger.error(f"Cannot regenerate thumb: No detection found for {original_filename} index {crop_index}")
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
                    sq_x1, sq_y1 = cx - side//2, cy - side//2
                    sq_x2, sq_y2 = sq_x1 + side, sq_y1 + side
                    
                    # Clamp
                    sq_x1, sq_y1 = max(0, sq_x1), max(0, sq_y1)
                    sq_x2, sq_y2 = min(w, sq_x2), min(h, sq_y2)
                    
                    if sq_x2 > sq_x1 and sq_y2 > sq_y1:
                        crop_img = img[sq_y1:sq_y2, sq_x1:sq_x2]
                        out_img = cv2.resize(crop_img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                        target_path = path_mgr.get_derivative_path(filename, "thumb")
                    else:
                        return False

                # 5. Save
                path_mgr.ensure_date_structure(path_mgr.extract_date_from_filename(filename))
                cv2.imwrite(str(target_path), out_img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
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
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))

        @server.route("/uploads/derivatives/thumbs/<path:filename>")
        def serve_thumb(filename):
            full_path = path_mgr.thumbs_dir / filename
            if not full_path.exists():
                # Trigger Regeneration
                # filename is e.g. "20240120/foo.webp"
                if regenerate_derivative(filename, "thumb"):
                    if not full_path.exists(): # Double check
                         return "Regeneration failed", 500
                else:
                    return "Not found and could not regenerate", 404
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))

        @server.route("/uploads/derivatives/optimized/<path:filename>")
        def serve_optimized(filename):
            full_path = path_mgr.optimized_dir / filename
            if not full_path.exists():
                if regenerate_derivative(filename, "optimized"):
                     if not full_path.exists():
                         return "Regeneration failed", 500
                else:
                    return "Not found and could not regenerate", 404
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
            
        def daily_species_summary_route():
            date_iso = request.args.get("date")
            if not date_iso:
                date_iso = datetime.now().strftime("%Y-%m-%d")
            try:
                datetime.strptime(date_iso, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400
            summary = get_daily_species_summary(date_iso)
            return jsonify({"date": date_iso, "summary": summary})

        server.route("/assets/<path:filename>")(lambda filename: send_from_directory(assets_folder, filename))
        def video_feed_route():
            return Response(
                generate_video_feed(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
        
        # --------------------------------------------------------------------------
        # Ingest API Route
        # --------------------------------------------------------------------------
        @server.route("/api/ingest/start", methods=["POST"])
        def start_ingest_endpoint():
            try:
                # Determine ingest path (Docker vs Local) with logging
                # Use configured path from config.py (Source of Truth)
                env_path = config.get("INGEST_DIR")
                cwd_path = os.path.abspath(os.path.join(os.getcwd(), "ingest"))
                
                logger.info(f"Ingest Request: CWD={os.getcwd()}, Configured: {env_path}, Local: {cwd_path}")

                if os.path.exists(env_path):
                    ingest_path = env_path
                    logger.info(f"Using configured ingest path: {ingest_path}")
                elif os.path.exists(cwd_path):
                    ingest_path = cwd_path
                    logger.info(f"Configured path not found. Using local CWD fallback: {ingest_path}")
                else:
                    ingest_path = env_path # Fallback to default
                    logger.warning(f"No valid ingest dir found. Falling back to configured: {ingest_path}")
                
                # Trigger User Ingest in background
                import threading
                def run_ingest():
                    detection_manager.start_user_ingest(ingest_path)
                
                t = threading.Thread(target=run_ingest)
                t.start()
                
                return jsonify({"status": "success", "message": "User Ingest started. Stream will pause."}), 200
            except Exception as e:
                logger.error(f"Error starting ingest: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def settings_get_route():
            return jsonify(get_settings_payload())

        def settings_post_route():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid payload"}), 400
            valid, errors = validate_runtime_updates(payload)
            if errors:
                return jsonify({"errors": errors}), 400
            update_runtime_settings(valid)
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
            "/api/settings", endpoint="settings_get", view_func=settings_get_route, methods=["GET"]
        )
        server.add_url_rule(
            "/api/settings", endpoint="settings_post", view_func=settings_post_route, methods=["POST"]
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
                t_str = row["time_str"] # "HHMMSS"
                if len(t_str) == 6:
                    h = int(t_str[0:2])
                    m = int(t_str[2:4])
                    s = int(t_str[4:6])
                    val = h + m/60.0 + s/3600.0
                    hours_float.append(val)
                elif len(t_str) == 8: # HH:MM:SS fallback
                     try:
                        h = int(t_str[0:2])
                        m = int(t_str[3:5])
                        s = int(t_str[6:8])
                        val = h + m/60.0 + s/3600.0
                        hours_float.append(val)
                     except: pass

            if not hours_float:
                return jsonify({"points": [], "peak_hour": None, "histogram": []})

            # 2. KDE Approximation via Histogram + Gaussian Smoothing
            # We use 1440 bins (minutes) for high resolution
            # This avoids adding scipy dependency
            import numpy as np
            
            bins = 144
            hist, bin_edges = np.histogram(hours_float, bins=bins, range=(0, 24), density=True)
            
            # Simple Gaussian Smoothing (Window size approx 1-2 hours)
            # Standard deviation for smoothing kernel
            sigma = 1.6  # Adjust for smoothness vs detail
            x_vals = np.linspace(-3*sigma, 3*sigma, int(6*sigma)+1)
            kernel = np.exp(-x_vals**2 / (2*sigma**2))
            kernel = kernel / np.sum(kernel)
            
            smooth_density = np.convolve(hist, kernel, mode='same')
            
            # Normalize peak to 1.0 (relative density) or use probability density?
            # User asked for "Relative activity density" and "Fläche auf 1".
            # Density=True in histogram ensures integral is 1.
            # Convoluting with normalized kernel preserves integral approx 1.
            
            # Generate Output Points
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            points = []
            max_y = 0
            peak_hour = 0
            
            for x, y in zip(bin_centers, smooth_density):
                points.append({"x": round(float(x), 2), "y": float(y)})
                if y > max_y:
                    max_y = y
                    peak_hour = x

            # Subsampled Histogram for "Backdrop" (Optional)
            # Use coarser bins for the visual histogram (e.g. 24 hours)
            hist_coarse, edges_coarse = np.histogram(hours_float, bins=48, range=(0, 24), density=True)
            histogram_points = []
            for i in range(len(hist_coarse)):
                histogram_points.append({
                    "x": float((edges_coarse[i] + edges_coarse[i+1])/2),
                    "y": float(hist_coarse[i])
                })

            return jsonify({
                "points": points, 
                "peak_hour": round(float(peak_hour), 2),
                "peak_density": float(max_y),
                "histogram": histogram_points
            })


        
        def analytics_species_activity_route():
             with get_connection() as conn:
                rows = fetch_species_timestamps(conn)
             
             import numpy as np
             
             # Group by species
             species_times = {}
             for r in rows:
                 sp = r["species"]
                 t_str = r["image_timestamp"][9:15] if len(r["image_timestamp"]) >= 15 else "" # YYYYMMDD_HHMMSS
                 if len(t_str) == 6:
                     try:
                        h = int(t_str[0:2]) + int(t_str[2:4])/60.0 + int(t_str[4:6])/3600.0
                        if sp not in species_times: species_times[sp] = []
                        species_times[sp].append(h)
                     except: pass
             
             series = []
             for sp, times in species_times.items():
                 # Rule: n >= 10 for KDE, else Histogram
                 # Max normalization (Ridgeplot style)
                 
                 if len(times) < 10:
                     # Histogram (1h bins)
                     hist, edges = np.histogram(times, bins=24, range=(0, 24), density=False)
                     # Normalize to max 1.0
                     max_val = np.max(hist)
                     if max_val > 0: hist = hist / max_val
                     
                     centers = (edges[:-1] + edges[1:]) / 2
                     points = [{"x": float(x), "y": float(y)} for x, y in zip(centers, hist)]
                     peak = centers[np.argmax(hist)]
                 else:
                     # Numpy Gaussian Smoothing (Manual KDE)
                     # 144 bins (10 min)
                     bins = 144
                     hist, edges = np.histogram(times, bins=bins, range=(0, 24), density=True)
                     
                     # Bandwidth: Explicitly set ~1.5 hours (sigma)
                     # 1.5h in bins (10m) = 9 bins
                     sigma = 9
                     x_vals = np.linspace(-3*sigma, 3*sigma, int(6*sigma)+1)
                     kernel = np.exp(-x_vals**2 / (2*sigma**2))
                     kernel = kernel / np.sum(kernel)
                     smooth = np.convolve(hist, kernel, mode='same')
                     
                     # Max Normalization
                     max_val = np.max(smooth)
                     if max_val > 0: smooth = smooth / max_val
                     
                     centers = (edges[:-1] + edges[1:]) / 2
                     points = [{"x": float(x), "y": float(y)} for x, y in zip(centers, smooth)]
                     peak = centers[np.argmax(smooth)]
                 
                 series.append({
                     "species": sp,
                     "points": points,
                     "peak_hour": float(peak),
                     "count": len(times)
                 })
             
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
        def trash_route():
            page = request.args.get("page", 1, type=int)
            limit = 50
            
            with get_connection() as conn:
                items, total_count = fetch_trash_items(conn, page=page, limit=limit)
             
            processed_items = []
            for item in items:
                ts = item["image_timestamp"]
                # Clean Slate: Use relative_path and thumbnail_path_virtual from DB
                full_path = item.get("relative_path") or item.get("optimized_name_virtual", "")
                thumb_virtual = item.get("thumbnail_path_virtual")
                
                if thumb_virtual:
                    display_path = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    display_path = f"/uploads/derivatives/optimized/{full_path}"
                
                try:
                    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    formatted_time = ts
                    
                processed_items.append({
                    "detection_id": item["detection_id"],
                    "display_path": display_path,
                    "common_name": item.get("cls_class_name") or item.get("od_class_name") or "Unknown",
                    "formatted_time": formatted_time
                })
            
            total_pages = math.ceil(total_count / limit) if limit > 0 else 1
            
            return render_template(
                "trash.html",
                items=processed_items,
                page=page,
                total_pages=total_pages,
                total_items=total_count,
                image_width=IMAGE_WIDTH
            )

        def trash_restore_route():
            try:
                data = request.get_json() or {}
                ids = data.get("ids", [])
                with get_connection() as conn:
                    restore_detections(conn, ids)
                return jsonify({"status": "success", "count": len(ids)})
            except Exception as e:
                logger.error(f"Error restoring trash: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def trash_purge_route():
            try:
                data = request.get_json() or {}
                ids = data.get("ids", [])
                with get_connection() as conn:
                    result = hard_delete_detections(conn, detection_ids=ids)
                if result.get("purged"):
                    logger.info(f"Trash purge: {result.get('rows_deleted', 0)} detections, {result.get('files_deleted', 0)} files deleted")
                return jsonify({"status": "success", "result": result})
            except Exception as e:
                logger.error(f"Error purging trash: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def trash_empty_route():
            try:
                with get_connection() as conn:
                    result = hard_delete_detections(conn, before_date="2099-12-31")
                if result.get("purged"):
                    logger.info(f"Trash emptied: {result.get('rows_deleted', 0)} detections, {result.get('files_deleted', 0)} files deleted")
                return jsonify({"status": "success", "result": result})
            except Exception as e:
                logger.error(f"Error emptying trash: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def reject_detection_route():
            data = request.get_json() or {}
            ids = data.get("ids", [])
            if not ids:
                return jsonify({"error": "No IDs provided"}), 400
            with get_connection() as conn:
                reject_detections(conn, ids)
            return jsonify({"status": "success"})

        server.add_url_rule("/trash", endpoint="trash", view_func=trash_route, methods=["GET"])
        server.add_url_rule("/api/trash/restore", endpoint="trash_restore", view_func=trash_restore_route, methods=["POST"])
        server.add_url_rule("/api/trash/purge", endpoint="trash_purge", view_func=trash_purge_route, methods=["POST"])
        server.add_url_rule("/api/trash/empty", endpoint="trash_empty", view_func=trash_empty_route, methods=["POST"])
        server.add_url_rule("/api/detections/reject", endpoint="reject_detection", view_func=reject_detection_route, methods=["POST"])

        # --- Auth Routes (Server-Side Session) ---
        def login_route():
            from flask import session, flash, url_for
            error = None
            next_url = request.args.get("next", "/gallery")
            
            if request.method == "POST":
                password = request.form.get("password", "")
                next_url = request.form.get("next", "/gallery")
                
                if password == EDIT_PASSWORD:
                    session["authenticated"] = True
                    logger.info("User authenticated successfully.")
                    return redirect(next_url)
                else:
                    error = "Invalid password. Please try again."
                    logger.warning("Failed login attempt.")
            
            return render_template("login.html", error=error, next_url=next_url)
        
        def logout_route():
            from flask import session
            session.pop("authenticated", None)
            return redirect("/gallery")

        server.add_url_rule("/login", endpoint="login", view_func=login_route, methods=["GET", "POST"])
        server.add_url_rule("/logout", endpoint="logout", view_func=logout_route, methods=["GET"])

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
                return render_template("edit.html", date_iso=date_iso, detections=[], filters=filters, species_list=[], image_width=IMAGE_WIDTH)

            # Extract unique species for the dropdown
            species_list = sorted(list(set(
                det["cls_class_name"] or det["od_class_name"] or "Unknown" 
                for det in detections
            )))

            # Apply Filters
            filtered = []
            try:
                min_conf_val = float(filters["min_conf"])
            except ValueError:
                min_conf_val = 0.0

            for det in detections:
                # Status filter
                is_downloaded = bool(det.get("downloaded_timestamp"))
                if filters["status"] == "downloaded" and not is_downloaded: continue
                if filters["status"] == "not_downloaded" and is_downloaded: continue

                # Species filter
                sp = det["cls_class_name"] or det["od_class_name"] or "Unknown"
                if filters["species"] != "all" and sp != filters["species"]: continue

                # Confidence filter
                conf = max(det.get("od_confidence") or 0, det.get("cls_confidence") or 0)
                if conf < min_conf_val: continue

                # Add display_path for template - Use pre-computed virtual paths from DB
                thumb_virtual = det.get("thumbnail_path_virtual")
                relative_path = det.get("relative_path", "")
                original_name = det.get("original_name", "")
                
                ts = det.get("image_timestamp", "")
                date_folder = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else date_iso
                
                if thumb_virtual:
                    det["display_path"] = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    det["display_path"] = f"/uploads/derivatives/optimized/{relative_path}"
                
                det["full_path"] = f"/uploads/derivatives/optimized/{relative_path}"
                det["original_path"] = f"/uploads/originals/{date_folder}/{original_name}"
                det["common_name"] = COMMON_NAMES.get(det.get("cls_class_name") or det.get("od_class_name"), "Unknown")
                det["latin_name"] = det.get("cls_class_name") or det.get("od_class_name") or ""
                
                filtered.append(det)

            # Apply Sorting
            if filters["sort"] == "time_asc":
                filtered.sort(key=lambda x: x["image_timestamp"])
            elif filters["sort"] == "time_desc":
                filtered.sort(key=lambda x: x["image_timestamp"], reverse=True)
            elif filters["sort"] == "score":
                filtered.sort(key=lambda x: x["score"] or 0.0, reverse=True)
            elif filters["sort"] == "confidence":
                filtered.sort(key=lambda x: max(x.get("od_confidence") or 0, x.get("cls_confidence") or 0), reverse=True)

            # --- Pagination ---
            page = request.args.get("page", 1, type=int)
            per_page = 100
            total_items = len(filtered)
            total_pages = math.ceil(total_items / per_page)
            
            if page < 1: page = 1
            if page > total_pages and total_pages > 0: page = total_pages
            
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
                }
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
                import zipfile, io
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
                    optimized_names = []
                    
                    for r in rows:
                        original_name, ts = r["original_name"], r["timestamp"]
                        if not original_name or not ts: continue
                        
                        # Build YYYY-MM-DD folder format
                        date_folder = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                        
                        # Clean Slate: originals are in originals/YYYY-MM-DD/
                        abs_path = os.path.join(output_dir, "originals", date_folder, original_name)
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
                return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=download_name)

            return redirect(f"/edit/{date_iso}")

        server.add_url_rule("/edit/<date_iso>", endpoint="edit_page", view_func=edit_route, methods=["GET"])
        server.add_url_rule("/api/edit/actions", endpoint="edit_actions", view_func=edit_actions_route, methods=["POST"])

        # --- Orphan Images Admin Page ---
        @login_required
        def orphans_route():
            """Admin page for viewing and deleting orphan images (no active detections)."""
            from utils.db import fetch_orphan_images
            from utils.path_manager import get_path_manager
            
            output_dir = config.get("OUTPUT_DIR", "output")
            pm = get_path_manager(output_dir)
            
            with get_connection() as conn:
                rows = fetch_orphan_images(conn)
            
            orphans = []
            for row in rows:
                filename = row["filename"]
                timestamp = row["timestamp"] or ""
                
                # Format date/time from timestamp (YYYYMMDD_HHMMSS)
                formatted_date = ""
                if len(timestamp) >= 15:
                    try:
                        dt = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_date = timestamp[:15]
                
                # Resolve paths using path_manager
                original_path = pm.get_original_path(filename)
                thumb_path = pm.get_derivative_path(filename, "thumb")
                
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
                thumb_url = f"/api/orphan-thumb/{filename}"
                
                orphans.append({
                    "filename": filename,
                    "timestamp": timestamp,
                    "formatted_date": formatted_date,
                    "file_size": file_size,
                    "file_size_str": file_size_str,
                    "thumb_url": thumb_url,
                })
            
            return render_template("orphans.html", orphans=orphans, current_path="/admin/orphans")
        
        @login_required
        def orphans_delete_route():
            """Handle deletion of orphan images."""
            from utils.db import delete_orphan_images
            from utils.path_manager import get_path_manager
            from flask import flash
            
            action = request.form.get("action", "")
            filenames = request.form.getlist("filenames")
            
            output_dir = config.get("OUTPUT_DIR", "output")
            pm = get_path_manager(output_dir)
            
            with get_connection() as conn:
                # If delete_all, fetch all orphan filenames
                if action == "delete_all":
                    from utils.db import fetch_orphan_images
                    rows = fetch_orphan_images(conn)
                    filenames = [row["filename"] for row in rows]
                
                if not filenames:
                    return redirect("/admin/orphans")
                
                # Delete files first (idempotent - missing files are OK)
                files_deleted = 0
                bytes_deleted = 0
                for filename in filenames:
                    # Delete original
                    original_path = pm.get_original_path(filename)
                    if original_path.exists():
                        try:
                            bytes_deleted += original_path.stat().st_size
                            original_path.unlink()
                            files_deleted += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete original {original_path}: {e}")
                    
                    # Delete optimized derivative
                    try:
                        opt_path = pm.get_derivative_path(filename, "optimized")
                        if opt_path.exists():
                            bytes_deleted += opt_path.stat().st_size
                            opt_path.unlink()
                    except Exception as e:
                        logger.debug(f"Could not delete optimized for {filename}: {e}")
                    
                    # Delete thumb derivative
                    try:
                        thumb_path = pm.get_derivative_path(filename, "thumb")
                        if thumb_path.exists():
                            bytes_deleted += thumb_path.stat().st_size
                            thumb_path.unlink()
                    except Exception as e:
                        logger.debug(f"Could not delete thumb for {filename}: {e}")
                    
                    # Delete preview thumbnail (orphan-specific)
                    try:
                        preview_path = pm.get_preview_thumb_path(filename)
                        if preview_path.exists():
                            bytes_deleted += preview_path.stat().st_size
                            preview_path.unlink()
                    except Exception as e:
                        logger.debug(f"Could not delete preview thumb for {filename}: {e}")
                
                # Delete from database
                rows_deleted = delete_orphan_images(conn, filenames)
                
                logger.info(f"Orphan cleanup: {rows_deleted} DB rows, {files_deleted} files deleted, {bytes_deleted} bytes freed")
                
                # Format bytes for display
                if bytes_deleted >= 1024 * 1024:
                    size_str = f"{bytes_deleted / (1024 * 1024):.1f} MB"
                elif bytes_deleted >= 1024:
                    size_str = f"{bytes_deleted / 1024:.1f} KB"
                else:
                    size_str = f"{bytes_deleted} B"
                
                flash(f"Deleted {rows_deleted} orphan image(s), freed {size_str}.", "success")
            
            return redirect("/admin/orphans")
        
        server.add_url_rule("/admin/orphans", endpoint="orphans_page", view_func=orphans_route, methods=["GET"])
        server.add_url_rule("/admin/orphans/delete", endpoint="orphans_delete", view_func=orphans_delete_route, methods=["POST"])

        # --- Orphan Thumbnail On-Demand Generation ---
        @login_required
        def orphan_thumb_route(filename):
            """On-demand thumbnail generation for orphan images."""
            from flask import send_file, abort
            from utils.path_manager import get_path_manager
            from utils.image_ops import generate_preview_thumbnail
            
            output_dir = config.get("OUTPUT_DIR", "output")
            pm = get_path_manager(output_dir)
            
            # Resolve paths via PathManager
            original_path = pm.get_original_path(filename)
            preview_path = pm.get_preview_thumb_path(filename)
            
            # If preview already cached, serve it
            if preview_path.exists():
                return send_file(str(preview_path), mimetype='image/webp')
            
            # Original must exist to generate preview
            if not original_path.exists():
                abort(404)
            
            # Generate preview thumbnail
            success = generate_preview_thumbnail(
                str(original_path),
                str(preview_path),
                size=256
            )
            
            if success and preview_path.exists():
                return send_file(str(preview_path), mimetype='image/webp')
            else:
                abort(500)
        
        server.add_url_rule("/api/orphan-thumb/<filename>", endpoint="orphan_thumb", view_func=orphan_thumb_route, methods=["GET"])



        # --- Analytics Dashboard (Svelte-rendered, read-only) ---
        def analytics_page_route():
            """Serves the analytics dashboard page with minimal HTML for Svelte mount."""
            return render_template("analytics.html")
        
        server.add_url_rule(
            "/analytics", endpoint="analytics_page", view_func=analytics_page_route, methods=["GET"]
        )

        # --- Phase 2: Species Summary (server-rendered) ---
        def species_route():
            """Server-rendered species summary page using Jinja2 templates."""
            import re
            
            # Get all detections
            all_detections = get_captured_detections()
            
            # Apply display threshold
            current_threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.0)
            if current_threshold > 0:
                all_detections = [d for d in all_detections if (d.get("score") or 0.0) >= current_threshold]
            
            # Aggregate: best per species
            species_groups = {}
            for det in all_detections:
                cls_class = det.get("cls_class_name")
                od_class = det.get("od_class_name")
                species_key = cls_class if cls_class else (od_class if od_class else "Unclassified")
                
                score = det.get("score") or 0.0
                if species_key not in species_groups or score > (species_groups[species_key].get("score") or 0.0):
                    species_groups[species_key] = det
            
            # Convert to list and enrich for template
            detections = []
            for species, det in sorted(species_groups.items(), key=lambda x: COMMON_NAMES.get(x[0], x[0])):
                # Build enriched detection dict for template with Clean Slate URLs
                full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
                thumb_virtual = det.get("thumbnail_path_virtual")
                
                if thumb_virtual:
                    display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    display_url = f"/uploads/derivatives/optimized/{full_path}"
                
                full_url = f"/uploads/derivatives/optimized/{full_path}"
                original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                
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
                
                detections.append({
                    "detection_id": det.get("detection_id"),
                    "display_path": display_url,
                    "full_path": full_url,
                    "original_path": original_url,
                    "common_name": COMMON_NAMES.get(species, species.replace("_", " ")),
                    "od_class_name": det.get("od_class_name", ""),
                    "od_confidence": det.get("od_confidence", 0.0) or 0.0,
                    "cls_class_name": det.get("cls_class_name", ""),
                    "cls_confidence": det.get("cls_confidence", 0.0) or 0.0,
                    "score": det.get("score", 0.0) or 0.0,
                    "formatted_date": formatted_date,
                    "formatted_time": formatted_time,
                    "gallery_date": gallery_date,
                })
            
            return render_template(
                "species.html",
                current_path="/species",
                detections=detections,
                image_width=IMAGE_WIDTH,
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
                except:
                    display_date = date_str
                
                days.append({
                    "date": date_str,
                    "display_date": display_date,
                    "cover_path": data.get("path", ""),
                    "count": data.get("count", 0),
                })
            
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
            sort_by = request.args.get("sort", "time_desc")
            
            # Fetch detections
            sql_order = "time" if sort_by in ("time_asc", "time_desc") else "score"
            rows = fetch_detections_for_gallery(db_conn, date, order_by=sql_order)
            detections_raw = [dict(row) for row in rows]
            
            # Python-side sorting adjustments
            if sort_by == "species":
                detections_raw.sort(key=lambda x: (x.get("cls_class_name") or x.get("od_class_name") or "").lower())
            elif sort_by == "time_asc":
                detections_raw.sort(key=lambda x: x.get("image_timestamp", ""))
            # time_desc and score come sorted from DB
            
            total_items = len(detections_raw)
            total_pages = math.ceil(total_items / PAGE_SIZE) or 1
            page = max(1, min(page, total_pages))
            start_index = (page - 1) * PAGE_SIZE
            end_index = page * PAGE_SIZE
            
            page_detections_raw = detections_raw[start_index:end_index]
            
            # Enrich detections for template
            def enrich_detection(det):
                full_path = det.get("relative_path") or det.get("optimized_name_virtual", "") # YYYYMMDD/foo.webp
                thumb_virtual = det.get("thumbnail_path_virtual")
                
                if thumb_virtual:
                    display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                else:
                    display_url = f"/uploads/derivatives/optimized/{full_path}"
                
                full_url = f"/uploads/derivatives/optimized/{full_path}"
                original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                
                ts = det.get("image_timestamp", "")
                if len(ts) >= 15:
                    date_str = ts[:8]
                    time_str = ts[9:15]
                    formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
                    formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    formatted_date = ""
                    formatted_time = ""
                
                species_key = det.get("cls_class_name") or det.get("od_class_name") or ""
                
                return {
                    "detection_id": det.get("detection_id"),
                    "display_url": display_url,
                    "full_url": full_url,
                    "original_url": original_url,
                    "display_path": display_url,
                    "full_path": full_url,
                    "original_path": original_url,
                    "common_name": COMMON_NAMES.get(species_key, species_key.replace("_", " ")),
                    "od_class_name": det.get("od_class_name", ""),
                    "od_confidence": det.get("od_confidence", 0.0) or 0.0,
                    "cls_class_name": det.get("cls_class_name", ""),
                    "cls_confidence": det.get("cls_confidence", 0.0) or 0.0,
                    "score": det.get("score", 0.0) or 0.0,
                    "formatted_date": formatted_date,
                    "formatted_time": formatted_time,
                }
            
            detections = [enrich_detection(d) for d in page_detections_raw]
            
            # Species of the Day (page 1 only)
            species_of_day = []
            if page == 1:
                species_groups = {}
                for det in detections_raw:
                    cls_class = det.get("cls_class_name")
                    od_class = det.get("od_class_name")
                    species_key = cls_class if cls_class else (od_class if od_class else "Unclassified")
                    score = det.get("score") or 0.0
                    if species_key not in species_groups or score > (species_groups[species_key].get("score") or 0.0):
                        species_groups[species_key] = det
                species_of_day = [enrich_detection(d) for d in sorted(species_groups.values(), key=lambda x: x.get("score", 0), reverse=True)]
            
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
                detections=detections,
                species_of_day=species_of_day,
                pagination_range=pagination_range,
                image_width=IMAGE_WIDTH,
            )
        
        server.add_url_rule(
            "/gallery/<date>", endpoint="subgallery", view_func=subgallery_route, methods=["GET"]
        )

    setup_web_routes(server)


    # --- Phase 4: Homepage (Live Stream) Migrated to Flask (Final Phase) ---
    def index_route():
        """Server-rendered Homepage / Live Stream."""
        today_iso = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Day Count & Title
        try:
            with get_connection() as conn:
                count = fetch_day_count(conn, today_iso)
            title = f"Live Stream (Today's Detections: {count})"
        except Exception as e:
            logger.error(f"Error fetching day count: {e}")
            title = "Live Stream"

        # 2. Latest Detections (Top 5)
        latest_detections = []
        try:
            with get_connection() as conn:
                rows = fetch_detections_for_gallery(conn, today_iso, limit=5, order_by="time")
                # Enrich for template
                for row in rows:
                    det = dict(row)
                    full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
                    thumb_virtual = det.get("thumbnail_path_virtual")
                    
                    if thumb_virtual:
                        display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_url = f"/uploads/derivatives/optimized/{full_path}"
                    
                    full_url = f"/uploads/derivatives/optimized/{full_path}"
                    original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                    
                    # Compat variables
                    display_path = display_url
                    original_path = original_url
                    ts = det.get("image_timestamp", "")
                    
                    formatted_time = ""
                    formatted_date = ""
                    if len(ts) >= 15:
                            date_str = ts[:8]
                            time_str = ts[9:15] # HHMMSS
                            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"

                    latest_detections.append({
                        "detection_id": det.get("detection_id"),
                        "common_name": COMMON_NAMES.get(det.get("cls_class_name") or det.get("od_class_name"), "Unknown"),
                        "latin_name": det.get("cls_class_name") or det.get("od_class_name") or "",
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
                    })
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
                    if s_key not in species_groups or score > species_groups[s_key].get("score", 0.0):
                            species_groups[s_key] = det
                
                # Sort by score desc for display
                sorted_summary = sorted(species_groups.values(), key=lambda x: x.get("score", 0.0), reverse=True)
                
                for det in sorted_summary:
                    # Clean Slate URL construction
                    full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
                    thumb_virtual = det.get("thumbnail_path_virtual")
                    
                    if thumb_virtual:
                        display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_url = f"/uploads/derivatives/optimized/{full_path}"
                    
                    full_url = f"/uploads/derivatives/optimized/{full_path}"
                    original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                    
                    ts = det.get("image_timestamp", "")
                    formatted_time = ""
                    formatted_date = ""
                    if len(ts) >= 15:
                            date_str = ts[:8]
                            time_str = ts[9:15]
                            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"

                    visual_summary.append({
                        "detection_id": det.get("detection_id"),
                        "common_name": COMMON_NAMES.get(det.get("cls_class_name") or det.get("od_class_name"), "Unknown"),
                        "latin_name": det.get("cls_class_name") or det.get("od_class_name") or "",
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
                    })

        except Exception as e:
            logger.error(f"Error fetching visual summary: {e}")

        # 4. Species Summary Table
        species_summary_table = []
        try:
            species_summary_table = get_daily_species_summary(today_iso)
        except Exception as e:
            logger.error(f"Error fetching species summary table: {e}")


        return render_template(
            "stream.html",
            current_path="/", # Active nav state
            title=title,
            latest_detections=latest_detections,
            visual_summary=visual_summary,
            species_summary=species_summary_table,
            empty_latest_message="No detections yet today.",
            image_width=IMAGE_WIDTH,
            today_iso=today_iso,
        )

    server.add_url_rule("/", endpoint="index", view_func=index_route, methods=["GET"])

    RUNTIME_BOOL_KEYS = {"DAY_AND_NIGHT_CAPTURE", "TELEGRAM_ENABLED"}
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
    }

    RUNTIME_KEYS_ORDER = [
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "DETECTION_INTERVAL_SECONDS",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "DAY_AND_NIGHT_CAPTURE",
        "DAY_AND_NIGHT_CAPTURE_LOCATION",
        "TELEGRAM_COOLDOWN",
        "TELEGRAM_ENABLED",
        "GALLERY_DISPLAY_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "EDIT_PASSWORD",
    ]

    ADVANCED_KEYS = {"STREAM_FPS", "STREAM_FPS_CAPTURE", "EDIT_PASSWORD"}

    SYSTEM_KEYS_ORDER = [
        "OUTPUT_DIR",
        "INGEST_DIR",
        "MODEL_BASE_PATH",
        "DEBUG_MODE",
        "CPU_LIMIT",
        "VIDEO_SOURCE",
        "DETECTOR_MODEL_CHOICE",
        "STREAM_WIDTH_OUTPUT_RESIZE",
        "LOCATION_DATA",
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
            trash_count=trash_count
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
                    updates[key] = (val.lower() == "true")
                elif key in RUNTIME_NUMBER_KEYS:
                    try:
                        val_clean = val.replace(",", ".")
                        if "." in val_clean:
                            updates[key] = float(val_clean)
                        else:
                            updates[key] = int(val_clean)
                    except ValueError:
                        pass # Ignore execution
                else:
                    updates[key] = val
        
        logger.info(f"Received settings update request: {updates}")
        
        valid, errors = validate_runtime_updates(updates)
        if errors:
            flash(f"Errors: {errors}", "error")
        else:
            update_runtime_settings(valid)
            flash("Settings updated successfully.", "success")
        
        return redirect(url_for('settings_route'))

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
        return redirect(url_for('settings_route'))






    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    return server
