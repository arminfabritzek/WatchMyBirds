# ------------------------------------------------------------------------------
# web_interface.py
# ------------------------------------------------------------------------------

import logging
import math
import os
import platform
import random
import re
import secrets
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from config import (
    ensure_go2rtc_stream_synced,
    get_config,
    get_settings_payload,
    update_runtime_settings,
)
from utils.review_metadata import (
    BBOX_REVIEW_CORRECT,
    BBOX_REVIEW_WRONG,
    REVIEW_STATUS_CONFIRMED_BIRD,
)
from utils.settings import mask_rtsp_url
from web.blueprints.auth import auth_bp, login_required
from web.power_actions import (
    POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
    get_power_action_success_message,
    is_power_management_available,
    schedule_power_action,
)
from web.services import (
    db_service,
    gallery_service,
    onvif_service,
    path_service,
)

# In-Memory Caches
_species_summary_cache = {"timestamp": 0, "payload": None}


def create_web_interface(detection_manager, system_monitor=None):
    """
    Creates and returns a Flask web server for the project.

    Args:
        detection_manager: The DetectionManager instance for frame access and control.
        system_monitor: Optional SystemMonitor instance for vitals API.

    Configuration is loaded from the global config module.
    """
    logger = logging.getLogger(__name__)

    # V-03: Local scope access instead of module-level globals
    config = get_config()

    output_dir = config["OUTPUT_DIR"]
    output_resize_width = config["STREAM_WIDTH_OUTPUT_RESIZE"]
    config["CONFIDENCE_THRESHOLD_DETECTION"]
    config["CLASSIFIER_CONFIDENCE_THRESHOLD"]
    EDIT_PASSWORD = config["EDIT_PASSWORD"]
    logger.info(
        "Loaded EDIT_PASSWORD: %s",
        "***"
        if EDIT_PASSWORD
        and EDIT_PASSWORD not in ["watchmybirds", "SECRET_PASSWORD", "default_pass"]
        else "<Not Set or Default>",
    )

    if not EDIT_PASSWORD or EDIT_PASSWORD in [
        "watchmybirds",
        "SECRET_PASSWORD",
        "default_pass",
    ]:
        logger.warning(
            "EDIT_PASSWORD not set securely in .env or settings.yaml. Access might be restricted or insecure."
        )

    # Clear restart-required marker on fresh app start
    try:
        from web.services import backup_restore_service

        backup_restore_service.clear_restart_marker(output_dir)
    except Exception as e:
        logger.debug(f"Could not clear restart marker: {e}")

    IMAGE_WIDTH = 150
    PAGE_SIZE = 50

    from utils.species_names import load_common_names

    _species_locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    COMMON_NAMES = load_common_names(_species_locale)
    UNKNOWN_SPECIES_KEY = "Unknown_species"

    def _get_species_key_local(det: dict | None) -> str:
        if not det:
            return UNKNOWN_SPECIES_KEY

        species_key = det.get("species_key")
        if species_key:
            return species_key

        manual_species = det.get("manual_species_override")
        if manual_species:
            return manual_species

        cls_species = det.get("cls_class_name")
        if cls_species:
            return cls_species

        od_species = det.get("od_class_name")
        if od_species and str(od_species).strip().lower() not in {
            "bird",
            "unknown",
            "unclassified",
        }:
            return od_species

        return UNKNOWN_SPECIES_KEY

    def _get_common_name_local(species_key: str | None) -> str:
        if not species_key:
            return COMMON_NAMES.get(UNKNOWN_SPECIES_KEY, "Unknown species")
        return COMMON_NAMES.get(species_key, species_key.replace("_", " "))

    def _compute_auto_rating_local(od_confidence, cls_confidence, bbox_w, bbox_h):
        """
        Local auto-rating fallback.
        Kept in web_interface to avoid hard runtime coupling to trash blueprint helpers.
        """
        od_conf = od_confidence or 0
        cls_conf = cls_confidence or 0
        bbox_area = (bbox_w or 0) * (bbox_h or 0)

        visual_score = od_conf * 0.4 + cls_conf * 0.6
        if bbox_area > 0.05:
            visual_score += 0.1
        elif bbox_area < 0.005:
            visual_score -= 0.15

        if visual_score >= 0.65:
            return 4
        if visual_score >= 0.45:
            return 3
        if visual_score >= 0.25:
            return 2
        return 1

    def _compute_rating_lazy(det):
        """Compute auto-rating for a detection that has no rating yet. Write back to DB."""
        rating = _compute_auto_rating_local(
            det.get("od_confidence", 0),
            det.get("cls_confidence", 0),
            det.get("bbox_w", 0),
            det.get("bbox_h", 0),
        )
        # Lazy write-back (best-effort, non-blocking)
        try:
            det_id = det.get("detection_id")
            if det_id:
                conn = db_service.get_connection()
                try:
                    conn.execute(
                        "UPDATE detections SET rating = ?, rating_source = 'auto' WHERE detection_id = ? AND rating IS NULL",
                        (rating, det_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            pass  # Non-critical
        return rating

    def get_detections_for_date(date_str_iso):
        with db_service.closing_connection() as conn:
            rows = db_service.fetch_detections_for_gallery(
                conn, date_str_iso, order_by="time"
            )
            # Convert to list of dicts immediately for easier handling
            return [dict(row) for row in rows]

    def delete_detections(detection_ids):
        """
        [SEMANTIC DELETE]
        Rejects specific detections.
        """
        with db_service.closing_connection() as conn:
            db_service.reject_detections(conn, detection_ids)
        return True

    def get_all_detections():
        """
        Reads all active detections from SQLite.
        Returns list of dicts.
        """
        try:
            with db_service.closing_connection() as conn:
                rows = db_service.fetch_detections_for_gallery(conn, order_by="time")
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error reading detections from SQLite: {e}")
            return []

    def get_daily_covers():
        """Returns a dict of {YYYY-MM-DD: {path, bbox}} for gallery overview."""
        covers = {}
        try:
            detections = get_captured_detections()
            by_date = {}
            for det in detections:
                date_key = _date_iso_from_timestamp(det.get("image_timestamp", ""))
                if not date_key:
                    continue
                by_date.setdefault(date_key, []).append(det)

            for date_key, day_detections in by_date.items():
                chosen = _pick_cover_for_group(
                    day_detections, seed_key=f"day:{date_key}", date_iso=date_key
                )
                if not chosen:
                    continue

                full_path = chosen.get("relative_path") or chosen.get(
                    "optimized_name_virtual", ""
                )
                thumb_path_virtual = chosen.get("thumbnail_path_virtual")
                if not full_path and not thumb_path_virtual:
                    continue

                if thumb_path_virtual:
                    display_path = f"/uploads/derivatives/thumbs/{thumb_path_virtual}"
                    is_thumb = True
                else:
                    display_path = f"/uploads/derivatives/optimized/{full_path}"
                    is_thumb = False

                # Count = number of observations (grouped visits)
                obs = gallery_service.group_detections_into_observations(day_detections)
                count = len(obs)

                bbox = (
                    chosen.get("bbox_x", 0.0),
                    chosen.get("bbox_y", 0.0),
                    chosen.get("bbox_w", 0.0),
                    chosen.get("bbox_h", 0.0),
                )

                covers[date_key] = {
                    "path": display_path,
                    "bbox": bbox,
                    "is_thumb": is_thumb,
                    "count": count,
                    "detection_id": chosen.get("detection_id"),
                }
        except Exception as e:
            logger.error(f"Error reading daily covers from SQLite: {e}")
        return covers

    def _bbox_touches_edge(det: dict, margin: float = 0.01) -> bool:
        """Return True if the detection's bounding box touches or exceeds the image edge.

        Normalized bbox coords are in 0..1 range.
        A bbox is considered edge-touching if any side is within *margin* of
        the image boundary (default 1%).
        """
        bx = det.get("bbox_x") or 0.0
        by = det.get("bbox_y") or 0.0
        bw = det.get("bbox_w") or 0.0
        bh = det.get("bbox_h") or 0.0
        if bw <= 0 or bh <= 0:
            return True  # no valid bbox → treat as edge-touching
        return (
            bx <= margin
            or by <= margin
            or (bx + bw) >= (1.0 - margin)
            or (by + bh) >= (1.0 - margin)
        )

    def _cover_quality_tuple(det: dict) -> tuple[int, float]:
        """
        Quality key for species cover and summary selection.
        Priority: favorite first, then score.
        """
        is_fav = 1 if int(det.get("is_favorite") or 0) else 0
        score = float(det.get("score") or 0.0)
        return (is_fav, score)

    def _is_favorite(det: dict) -> bool:
        """True when a detection is marked as ❤️ favorite."""
        return bool(int(det.get("is_favorite") or 0))

    def _build_detection_view_dict(
        det: dict,
        *,
        species_key: str,
        common_name: str,
        formatted_date: str = "",
        formatted_time: str = "",
        gallery_date: str = "",
        siblings: list | None = None,
        sibling_count: int = 1,
        include_decision_state: bool = False,
        extra: dict | None = None,
    ) -> dict:
        """Build the shared detection view payload used by frontend templates."""
        payload = {
            "detection_id": det.get("detection_id"),
            "species_key": species_key,
            "common_name": common_name,
            "od_class_name": det.get("od_class_name", ""),
            "od_confidence": det.get("od_confidence", 0.0) or 0.0,
            "cls_class_name": det.get("cls_class_name", ""),
            "cls_confidence": det.get("cls_confidence", 0.0) or 0.0,
            "score": det.get("score", 0.0) or 0.0,
            "review_status": det.get("review_status"),
            "manual_species_override": det.get("manual_species_override"),
            "species_source": det.get("species_source"),
            "formatted_date": formatted_date,
            "formatted_time": formatted_time,
            "gallery_date": gallery_date,
            "siblings": siblings or [],
            "sibling_count": sibling_count,
            "bbox_x": det.get("bbox_x", 0.0) or 0.0,
            "bbox_y": det.get("bbox_y", 0.0) or 0.0,
            "bbox_w": det.get("bbox_w", 0.0) or 0.0,
            "bbox_h": det.get("bbox_h", 0.0) or 0.0,
            "rating": det.get("rating"),
            "rating_source": det.get("rating_source", "auto"),
            "is_favorite": _is_favorite(det),
        }
        if include_decision_state:
            payload["decision_state"] = det.get("decision_state")
        if extra:
            payload.update(extra)
        return payload

    def _date_iso_from_timestamp(ts: str) -> str:
        """Convert YYYYMMDD_HHMMSS -> YYYY-MM-DD."""
        if not ts or len(ts) < 8:
            return ""
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

    def _format_stream_timestamp(ts: str) -> str:
        """Format stream timestamps for compact card metadata."""
        if not ts:
            return "Unknown"
        try:
            return datetime.strptime(ts[:15], "%Y%m%d_%H%M%S").strftime(
                "%d.%m.%Y %H:%M"
            )
        except ValueError:
            return "Unknown"

    def _pick_cover_for_group(candidates: list[dict], **_kwargs) -> dict | None:
        """
        Pick one cover candidate.
        Priority:
        1) If ❤️ favorites exist → random.choice() among them
        2) Prefer non-edge images
        3) Otherwise pick single best by score
        """
        if not candidates:
            return None

        # 1. Favorites win over everything else
        fav_pool = [d for d in candidates if _is_favorite(d)]
        if fav_pool:
            return random.choice(fav_pool)

        # 2. Prefer non-edge images for the fallback
        interior = [d for d in candidates if not _bbox_touches_edge(d)]
        pool = interior if interior else candidates

        # 3. Fallback: best by score
        ranked = sorted(
            pool,
            key=lambda d: float(d.get("score") or 0.0),
            reverse=True,
        )
        return ranked[0] if ranked else None

    def get_captured_detections():
        """
        Returns a list of captured detections (dicts).
        Uses fresh DB reads for UI correctness after manual relabel/rating updates.
        """
        try:
            return gallery_service.get_all_detections()
        except Exception as e:
            logger.error(f"Error reading detections from SQLite: {e}")
            return []

    def _build_stream_media_payload(det: dict | None) -> dict:
        """Return stream-friendly media URLs and basic detection metadata."""
        if not det:
            return {
                "detection_id": None,
                "display_path": "",
                "gallery_date": "",
                "is_favorite": False,
                "score": 0.0,
            }

        full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
        thumb_virtual = det.get("thumbnail_path_virtual")
        if thumb_virtual:
            display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
        else:
            display_url = f"/uploads/derivatives/optimized/{full_path}"

        ts = det.get("image_timestamp", "") or ""
        return {
            "detection_id": det.get("detection_id"),
            "display_path": display_url,
            "gallery_date": _date_iso_from_timestamp(ts),
            "is_favorite": bool(int(det.get("is_favorite") or 0)),
            "score": float(det.get("score") or 0.0),
        }

    def _enrich_species_board(board: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """Attach stream/UI fields to the species story board payload."""
        enriched_board = {"featured": [], "grid": []}
        for section in ("featured", "grid"):
            for item in board.get(section, []):
                species_key = item.get("species_key") or UNKNOWN_SPECIES_KEY
                primary = item.get("primary_detection")
                primary_payload = _build_stream_media_payload(primary)
                story_frames = []
                for story_det in item.get("story_detections", []):
                    frame_payload = _build_stream_media_payload(story_det)
                    frame_payload["detection_id"] = story_det.get("detection_id")
                    story_frames.append(frame_payload)

                enriched_board[section].append(
                    {
                        "species_key": species_key,
                        "common_name": _get_common_name_local(species_key),
                        "latin_name": species_key,
                        "visit_count": int(item.get("visit_count") or 0),
                        "last_seen_timestamp": item.get("last_seen_timestamp") or "",
                        "last_seen_display": _format_stream_timestamp(
                            item.get("last_seen_timestamp") or ""
                        ),
                        "is_favorite_available": bool(
                            item.get("is_favorite_available")
                        ),
                        "best_cover_score": float(item.get("best_cover_score") or 0.0),
                        "detection_id": primary_payload["detection_id"],
                        "display_path": primary_payload["display_path"],
                        "gallery_date": primary_payload["gallery_date"],
                        "is_favorite": primary_payload["is_favorite"],
                        "score": primary_payload["score"],
                        "story_frames": story_frames,
                    }
                )
        return enriched_board

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
            with db_service.closing_connection() as conn:
                rows = db_service.fetch_detection_species_summary(conn, date_iso)
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

    # -----------------------------
    # Flask Server and Routes
    # -----------------------------
    # Create Flask server with Jinja2 template support for server-first rendering.
    # Template folder is at project root level, not in web/ subdirectory.
    project_root = os.path.dirname(os.path.dirname(__file__))
    template_folder = os.path.join(project_root, "templates")
    assets_folder = os.path.join(project_root, "assets")
    server = Flask(__name__, template_folder=template_folder)
    # Expose helper globally for imported Jinja macros (works without "with context").
    server.jinja_env.globals["wikipedia_species_url"] = (
        gallery_service.get_species_wikipedia_url
    )
    server.jinja_env.globals["REVIEW_STATUS_CONFIRMED_BIRD"] = (
        REVIEW_STATUS_CONFIRMED_BIRD
    )
    server.jinja_env.globals["BBOX_REVIEW_CORRECT"] = BBOX_REVIEW_CORRECT
    server.jinja_env.globals["BBOX_REVIEW_WRONG"] = BBOX_REVIEW_WRONG

    # Configure Flask for large file uploads (backups can be several GB)
    server.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB

    # Configure Flask Session for server-side auth
    def _load_or_create_secret_key(key_file: Path) -> str:
        """Load persisted secret key or generate a new one."""
        if key_file.exists():
            return key_file.read_text().strip()
        key = secrets.token_hex(32)
        key_file.write_text(key)
        key_file.chmod(0o600)
        return key

    secret_key_path = Path(output_dir) / ".flask_secret_key"
    server.secret_key = os.environ.get("FLASK_SECRET_KEY") or _load_or_create_secret_key(secret_key_path)
    server.config["PERMANENT_SESSION_LIFETIME"] = 28800  # 8 hours in seconds
    server.config["SESSION_COOKIE_HTTPONLY"] = True
    server.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Inject a default-password warning and CSRF token into every template context
    @server.context_processor
    def inject_security_context():
        from web.services import auth_service as _auth

        # CSRF token: generate once per session, reuse until session ends
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_hex(32)

        warn = (
            session.get("authenticated")
            and _auth.is_default_password()
        )
        return {
            "warn_default_password": warn,
            "setup_password_required": _auth.should_require_password_setup(),
            "csrf_token": session["_csrf_token"],
        }

    # CSRF validation for state-changing requests
    _CSRF_EXEMPT_PATHS = frozenset()  # add paths here if needed

    @server.before_request
    def check_csrf_token():
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return
        if request.path in _CSRF_EXEMPT_PATHS:
            return

        # Token from form field or X-CSRF-Token header
        token = (
            request.form.get("_csrf_token")
            or request.headers.get("X-CSRF-Token")
        )
        if not token or token != session.get("_csrf_token"):
            from flask import abort
            abort(403)

    # Register Blueprints
    server.register_blueprint(auth_bp)

    # Register API v1 Blueprint
    from web.blueprints.api_v1 import init_api_v1

    def _on_runtime_settings_applied(valid_updates: dict) -> None:
        """React to runtime setting changes dispatched from API v1."""
        nonlocal COMMON_NAMES
        if "SPECIES_COMMON_NAME_LOCALE" in valid_updates:
            new_names = load_common_names(valid_updates["SPECIES_COMMON_NAME_LOCALE"])
            COMMON_NAMES.clear()
            COMMON_NAMES.update(new_names)

    init_api_v1(
        server,
        detection_manager,
        system_monitor=system_monitor,
        on_runtime_settings_applied=_on_runtime_settings_applied,
    )

    # Register Trash Blueprint
    from web.blueprints.trash import trash_bp

    server.register_blueprint(trash_bp)

    # Register Review Blueprint
    from web.blueprints.review import review_bp

    server.register_blueprint(review_bp)

    # Register Inbox Blueprint
    from web.blueprints.inbox import inbox_bp, init_inbox_bp

    init_inbox_bp(detection_manager)
    server.register_blueprint(inbox_bp)

    # Register Analytics Blueprint
    from web.blueprints.analytics import analytics_bp

    server.register_blueprint(analytics_bp)

    # Register Backup Blueprint
    from web.blueprints.backup import backup_bp, init_backup_bp

    init_backup_bp(detection_manager)
    server.register_blueprint(backup_bp)

    # Register Moderation Blueprint
    from web.blueprints.moderation import moderation_bp

    server.register_blueprint(moderation_bp)

    # Auth helper is now imported from web.blueprints.auth

    def setup_web_routes(server):
        path_mgr = path_service.get_path_manager(output_dir)

        # --- Routes ---

        @server.route("/uploads/originals/<path:filename>")
        def serve_original(filename):
            # filename typically includes date folder e.g. "20240120/file.jpg"
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
                # Trigger Regeneration via Service
                if gallery_service.regenerate_derivative(output_dir, filename, "thumb"):
                    if not full_path.exists():  # Double check
                        return "Regeneration failed", 500
                else:
                    # Fallback: try _preview.webp if _crop_N.webp missing
                    import re

                    preview_name = re.sub(
                        r"_crop_\d+\.webp$", "_preview.webp", filename
                    )
                    preview_path = path_mgr.thumbs_dir / preview_name
                    if preview_name != filename and preview_path.exists():
                        return send_from_directory(
                            os.path.dirname(preview_path),
                            os.path.basename(preview_path),
                        )
                    return "Not found and could not regenerate", 404
            return send_from_directory(
                os.path.dirname(full_path), os.path.basename(full_path)
            )

        @server.route("/uploads/derivatives/optimized/<path:filename>")
        def serve_optimized(filename):
            full_path = path_mgr.optimized_dir / filename
            if not full_path.exists():
                # Trigger Regeneration via Service
                if gallery_service.regenerate_derivative(
                    output_dir, filename, "optimized"
                ):
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

        @server.route("/video_feed")
        def video_feed():
            """Compatibility MJPEG stream for browsers that cannot reach Go2RTC."""

            stream_fps = float(config.get("STREAM_FPS", 5.0) or 5.0)
            stream_fps = max(1.0, stream_fps)
            frame_interval = 1.0 / stream_fps

            def generate():
                while True:
                    loop_start = time.time()
                    frame = detection_manager.get_display_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue

                    try:
                        h, w = frame.shape[:2]
                        output_h = int(h * output_resize_width / w) if w else h
                        resized = cv2.resize(frame, (output_resize_width, output_h))
                        ok, buffer = cv2.imencode(
                            ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        if not ok:
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to encode MJPEG frame: {e}")
                        time.sleep(0.05)
                        continue

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )

                    elapsed = time.time() - loop_start
                    sleep_for = frame_interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
                },
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

        # --- ONVIF Discovery API ---
        @server.route("/api/onvif/discover", methods=["GET"])
        @login_required
        def onvif_discover_route():
            """Scans network for ONVIF cameras and returns results."""
            try:
                cameras = onvif_service.discover_cameras(fast=False)

                if not cameras:
                    # Return empty success list rather than error if just nothing found
                    return jsonify({"status": "success", "cameras": []})

                return jsonify({"status": "success", "cameras": cameras})
            except Exception as e:
                logger.error(f"ONVIF Discovery route failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/onvif/get_stream_uri", methods=["POST"])
        @login_required
        def onvif_get_stream_uri_route():
            """Retrieves RTSP stream URI for a specific camera with credentials."""
            try:
                data = request.get_json() or {}
                ip = data.get("ip")
                port = int(data.get("port", 80))
                user = data.get("username", "")
                password = data.get("password", "")

                if not ip:
                    return jsonify(
                        {"status": "error", "message": "IP is required"}
                    ), 400

                uri = onvif_service.get_stream_uri(ip, port, user, password)

                if uri:
                    return jsonify({"status": "success", "uri": uri})
                else:
                    return jsonify(
                        {"status": "error", "message": "Could not retrieve URI"}
                    ), 404
            except Exception as e:
                logger.error(f"ONVIF Stream URI route failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        # --- Camera Management API ---
        @server.route("/api/cameras", methods=["GET"])
        @login_required
        def cameras_list_route():
            """List all saved cameras."""
            try:
                cameras = onvif_service.get_saved_cameras()
                return jsonify({"status": "success", "cameras": cameras})
            except Exception as e:
                logger.error(f"Camera list failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/cameras", methods=["POST"])
        @login_required
        def cameras_add_route():
            """Add a new camera."""
            try:
                data = request.get_json() or {}
                ip = data.get("ip")
                port = int(data.get("port", 80))
                username = data.get("username", "")
                password = data.get("password", "")
                name = data.get("name", "")

                if not ip:
                    return jsonify(
                        {"status": "error", "message": "IP is required"}
                    ), 400

                result = onvif_service.save_camera(
                    ip=ip,
                    port=port,
                    username=username,
                    password=password,
                    name=name,
                )
                return jsonify({"status": "success", "camera": result})
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 400
            except Exception as e:
                logger.error(f"Camera add failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
        @login_required
        def cameras_delete_route(camera_id: int):
            """Delete a camera."""
            try:
                if onvif_service.delete_camera(camera_id):
                    return jsonify({"status": "success"})
                else:
                    return jsonify(
                        {"status": "error", "message": "Camera not found"}
                    ), 404
            except Exception as e:
                logger.error(f"Camera delete failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/cameras/<int:camera_id>", methods=["PUT"])
        @login_required
        def cameras_update_route(camera_id: int):
            """Update a camera."""
            try:
                data = request.get_json() or {}
                if onvif_service.update_camera(
                    camera_id,
                    ip=data.get("ip"),
                    port=data.get("port"),
                    username=data.get("username"),
                    password=data.get("password"),
                    name=data.get("name"),
                ):
                    return jsonify({"status": "success"})
                else:
                    return jsonify(
                        {"status": "error", "message": "Camera not found"}
                    ), 404
            except Exception as e:
                logger.error(f"Camera update failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/cameras/<int:camera_id>/test", methods=["POST"])
        @login_required
        def cameras_test_route(camera_id: int):
            """Test connection to a camera using ONVIF GetDeviceInformation."""
            try:
                cam = onvif_service.get_camera(camera_id, include_password=True)

                if not cam:
                    return jsonify(
                        {"status": "error", "message": "Camera not found"}
                    ), 404

                details = onvif_service.get_device_info(
                    ip=cam["ip"],
                    port=cam.get("port", 80),
                    username=cam.get("username", ""),
                    password=cam.get("password", ""),
                )

                if details:
                    # Update test result via service
                    onvif_service.update_test_result(
                        camera_id,
                        success=True,
                        manufacturer=details.get("manufacturer", ""),
                        model=details.get("model", ""),
                    )
                    return jsonify(
                        {
                            "status": "success",
                            "details": {
                                "manufacturer": details.get("manufacturer"),
                                "model": details.get("model"),
                                "firmware": details.get("firmware"),
                                "has_ptz": False,
                            },
                        }
                    )
                else:
                    onvif_service.update_test_result(camera_id, success=False)
                    return jsonify(
                        {"status": "error", "message": "Connection failed."}
                    ), 400
            except Exception as e:
                logger.error(f"Camera test failed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        @server.route("/api/cameras/<int:camera_id>/use", methods=["POST"])
        @login_required
        def cameras_use_route(camera_id: int):
            """
            Use a saved camera: Fetches RTSP URI using stored credentials
            and sets it as CAMERA_URL. The resolver derives effective VIDEO_SOURCE.
            """
            try:
                from config import resolve_effective_sources

                cam = onvif_service.get_camera(camera_id, include_password=True)

                if not cam:
                    return jsonify(
                        {"status": "error", "message": "Camera not found"}
                    ), 404

                logger.info(
                    f"Activating camera {camera_id}: {cam.get('name')} @ {cam['ip']}:{cam.get('port', 80)}"
                )

                # Check if credentials are present
                if not cam.get("username") or not cam.get("password"):
                    logger.warning(f"Camera {camera_id} has no credentials stored")
                    return jsonify(
                        {
                            "status": "error",
                            "message": "No credentials stored for this camera. Please edit the camera and add username/password.",
                        }
                    ), 400

                # Get RTSP URI using stored credentials via service
                try:
                    uri = onvif_service.get_stream_uri(
                        camera_ip=cam["ip"],
                        port=cam.get("port", 80),
                        username=cam.get("username", ""),
                        password=cam.get("password", ""),
                    )
                    logger.info(f"Retrieved RTSP URI for camera {camera_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to get stream URI for camera {camera_id}: {e}"
                    )
                    return jsonify(
                        {"status": "error", "message": f"ONVIF connection failed: {e}"}
                    ), 400

                if not uri:
                    return jsonify(
                        {
                            "status": "error",
                            "message": "Could not retrieve stream URI from camera",
                        }
                    ), 400

                # Set CAMERA_URL (not VIDEO_SOURCE directly) and resolve
                logger.info(f"Setting CAMERA_URL to: {uri[:50]}...")
                update_runtime_settings({"CAMERA_URL": uri})

                cfg = get_config()
                ensure_go2rtc_stream_synced(cfg)
                resolved = resolve_effective_sources(cfg)
                cfg["VIDEO_SOURCE"] = resolved["video_source"]

                detection_manager.update_configuration(
                    {"VIDEO_SOURCE": resolved["video_source"]}
                )

                logger.info(
                    f"Camera {camera_id} activated: mode={resolved['effective_mode']} "
                    f"video_source={resolved['video_source'][:40]}"
                )

                return jsonify(
                    {
                        "status": "success",
                        "message": f"Camera '{cam.get('name', 'Camera')}' is now active",
                        "uri_set": True,
                    }
                )

            except Exception as e:
                logger.error(f"Camera use failed: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500

        server.add_url_rule(
            "/api/daily_species_summary",
            endpoint="daily_species_summary",
            view_func=daily_species_summary_route,
            methods=["GET"],
        )

        # --- Analytics API Routes --- MOVED TO web/blueprints/analytics.py ---

        # --- Phase 5: Trash Routes --- MOVED TO web/blueprints/trash.py ---

        # --- Auth Routes now handled by auth_bp (registered in create_web_interface) ---

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
            if filters["species"] in {"Unknown", "Unclassified"}:
                filters["species"] = UNKNOWN_SPECIES_KEY

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
                        _get_species_key_local(det)
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
                sp = _get_species_key_local(det)
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
                species_key = _get_species_key_local(det)
                det["species_key"] = species_key
                det["common_name"] = _get_common_name_local(species_key)
                det["latin_name"] = species_key or ""

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

            # Handle reject_all BEFORE checking det_ids (it doesn't need selections)
            if action == "reject_all":
                if not date_iso:
                    return redirect("/gallery")

                with db_service.closing_connection() as conn:
                    # Get all detection IDs for this date
                    date_prefix = date_iso.replace("-", "")  # 2026-02-06 -> 20260206
                    query = """
                        SELECT d.detection_id
                        FROM detections d
                        JOIN images i ON d.image_filename = i.filename
                        WHERE i.timestamp LIKE ?
                        AND (d.status IS NULL OR d.status != 'rejected')
                    """
                    rows = conn.execute(query, (f"{date_prefix}%",)).fetchall()
                    all_ids = [r["detection_id"] for r in rows]

                    if all_ids:
                        db_service.reject_detections(conn, all_ids)
                        logger.info(
                            f"Rejected ALL {len(all_ids)} detections for {date_iso}"
                        )

                # Reset caches
                gallery_service.invalidate_cache()
                return redirect(f"/gallery/{date_iso}")

            if not det_ids:
                return redirect(f"/edit/{date_iso}")

            ids_int = [int(i) for i in det_ids]

            if action == "reject":
                with db_service.closing_connection() as conn:
                    db_service.reject_detections(conn, ids_int)
                # Reset caches
                gallery_service.invalidate_cache()
                # _daily_gallery_summary_cache was removed
                return redirect(f"/edit/{date_iso}")

            elif action == "download":
                import io
                import zipfile

                from flask import send_file

                with db_service.closing_connection() as conn:
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
                        db_service.update_downloaded_timestamp(
                            conn, filenames, download_time
                        )

                # Create Zip
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for abs_path, arcname in files_to_zip:
                        if os.path.exists(abs_path):
                            zf.write(abs_path, arcname=arcname)

                zip_buffer.seek(0)
                download_name = f"watchmybirds_{date_iso.replace('-', '')}_download.zip"
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

        # --- Review Queue Routes --- MOVED TO web/blueprints/review.py ---

        # --- Analytics Dashboard Routes --- MOVED TO web/blueprints/analytics.py ---

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
                current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

            # Apply display threshold
            if current_threshold > 0:
                all_detections = [
                    d
                    for d in all_detections
                    if (d.get("score") or 0.0) >= current_threshold
                ]

            # Group detections per species and pick one curated cover per species.
            species_candidates = {}
            for det in all_detections:
                species_key = _get_species_key_local(det)
                species_candidates.setdefault(species_key, []).append(det)

            # One cover per species (favorites preferred, random rotation).
            today_iso = datetime.now().strftime("%Y-%m-%d")
            species_groups = {}
            for s_key, candidates in species_candidates.items():
                chosen = _pick_cover_for_group(
                    candidates, seed_key=f"species:{s_key}", date_iso=today_iso
                )
                if chosen:
                    species_groups[s_key] = chosen

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
                    _build_detection_view_dict(
                        det,
                        species_key=species,
                        common_name=_get_common_name_local(species),
                        formatted_date=formatted_date,
                        formatted_time=formatted_time,
                        gallery_date=gallery_date,
                        extra={
                            "display_path": display_url,
                            "full_path": full_url,
                            "original_path": original_url,
                        },
                    )
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

        def species_overview_route():
            """Species-specific overview page with all detections for one species."""
            raw_species_key = request.args.get("species_key", type=str) or ""
            species_key = raw_species_key.strip().replace(" ", "_")
            if species_key in {"Unknown", "Unclassified"}:
                species_key = UNKNOWN_SPECIES_KEY
            if not species_key:
                return redirect(url_for("species"))

            page = request.args.get("page", 1, type=int)

            # Get threshold from query param or config (query param overrides)
            try:
                min_score_param = request.args.get("min_score", type=float)
            except (ValueError, TypeError):
                min_score_param = None

            if min_score_param is not None:
                current_threshold = min_score_param
            else:
                current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

            all_detections = get_captured_detections()

            filtered = []
            for det in all_detections:
                det_species_key = _get_species_key_local(det)
                if det_species_key != species_key:
                    continue
                if (
                    current_threshold > 0
                    and (det.get("score") or 0.0) < current_threshold
                ):
                    continue
                filtered.append(det)

            total_items = len(filtered)
            total_pages = math.ceil(total_items / PAGE_SIZE) or 1
            page = max(1, min(page, total_pages))
            start_index = (page - 1) * PAGE_SIZE
            end_index = page * PAGE_SIZE
            page_detections_raw = filtered[start_index:end_index]

            detections = []
            for det in page_detections_raw:
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
                    _build_detection_view_dict(
                        det,
                        species_key=species_key,
                        common_name=_get_common_name_local(species_key),
                        formatted_date=formatted_date,
                        formatted_time=formatted_time,
                        gallery_date=gallery_date,
                        include_decision_state=True,
                        extra={
                            "display_path": display_url,
                            "full_path": full_url,
                            "original_path": original_url,
                        },
                    )
                )

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
                "species_overview.html",
                current_path="/species",
                species_key=species_key,
                species_common_name=_get_common_name_local(species_key),
                current_threshold=current_threshold,
                detections=detections,
                page=page,
                total_pages=total_pages,
                total_items=total_items,
                pagination_range=pagination_range,
                image_width=IMAGE_WIDTH,
            )

        server.add_url_rule(
            "/species/overview",
            endpoint="species_overview",
            view_func=species_overview_route,
            methods=["GET"],
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
                        "cover_detection_id": data.get("detection_id"),
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
            """Server-rendered subgallery page for a specific date (observation-based)."""
            # Validate date format
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                return "Invalid date format.", 400

            # Get query params
            page = request.args.get("page", 1, type=int)
            sort_by = request.args.get("sort", "time_desc")

            # Get threshold from query param or config
            try:
                min_score_param = request.args.get("min_score", type=float)
            except (ValueError, TypeError):
                min_score_param = None

            if min_score_param is not None:
                current_threshold = min_score_param
            else:
                current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

            # Fetch all detections for this date
            with db_service.closing_connection() as conn:
                rows = db_service.fetch_detections_for_gallery(
                    conn, date, order_by="time"
                )
            detections_raw = [dict(row) for row in rows]

            # ── Group into observations ────────────────────────────────
            observations_all = gallery_service.group_detections_into_observations(
                detections_raw
            )
            total_obs_unfiltered = len(observations_all)

            # Apply min_score filter on observation.best_score
            focus_id_param = request.args.get("focus", type=int)
            if current_threshold > 0:
                observations_filtered = []
                for obs in observations_all:
                    if obs["best_score"] >= current_threshold:
                        observations_filtered.append(obs)
                    elif focus_id_param and focus_id_param in obs["detection_ids"]:
                        # Always include the observation containing the focused detection
                        observations_filtered.append(obs)
                observations_all = observations_filtered

            # ── Sort observations ──────────────────────────────────────
            if sort_by == "time_asc":
                observations_all.sort(key=lambda o: o["end_time"])
            elif sort_by == "score":
                observations_all.sort(key=lambda o: o["best_score"], reverse=True)
            elif sort_by == "species":
                observations_all.sort(
                    key=lambda o: COMMON_NAMES.get(o["species"], o["species"]).lower()
                )
            else:
                # Default: time_desc — newest detection in the observation first.
                observations_all.sort(key=lambda o: o["end_time"], reverse=True)

            total_items = len(observations_all)
            total_pages = math.ceil(total_items / PAGE_SIZE) or 1

            # Auto-focus: if ?focus=DETECTION_ID, find the observation & page
            if focus_id_param and page == 1:
                for idx, obs in enumerate(observations_all):
                    if focus_id_param in obs["detection_ids"]:
                        page = (idx // PAGE_SIZE) + 1
                        break

            page = max(1, min(page, total_pages))
            start_index = (page - 1) * PAGE_SIZE
            end_index = page * PAGE_SIZE
            page_observations = observations_all[start_index:end_index]
            focus_observation_id = None
            if focus_id_param:
                for obs in page_observations:
                    if focus_id_param in obs["detection_ids"]:
                        focus_observation_id = obs["observation_id"]
                        break

            # ── Build detection lookup for enrichment ──────────────────
            det_by_id = {d.get("detection_id"): d for d in detections_raw}

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

                species_key = _get_species_key_local(det)
                sibling_count = det.get("sibling_count", 1) or 1

                # Load sibling detections if multiple birds on same image
                siblings = []
                if sibling_count > 1:
                    original_name = det.get("original_name", "")
                    if original_name:
                        sibling_rows = gallery_service.get_sibling_detections(
                            original_name
                        )
                        for sib in sibling_rows:
                            sib_species_key = _get_species_key_local(sib)
                            sib_thumb = sib["thumbnail_path_virtual"]
                            siblings.append(
                                _build_detection_view_dict(
                                    sib,
                                    species_key=sib_species_key,
                                    common_name=_get_common_name_local(
                                        sib_species_key
                                    ),
                                    include_decision_state=True,
                                    extra={
                                        "thumb_url": (
                                            f"/uploads/derivatives/thumbs/{sib_thumb}"
                                            if sib_thumb
                                            else ""
                                        ),
                                    },
                                )
                            )

                return _build_detection_view_dict(
                    det,
                    species_key=species_key,
                    common_name=_get_common_name_local(species_key),
                    formatted_date=formatted_date,
                    formatted_time=formatted_time,
                    gallery_date=date,
                    siblings=siblings,
                    sibling_count=sibling_count,
                    include_decision_state=True,
                    extra={
                        "image_timestamp": ts,
                        "display_url": display_url,
                        "full_url": full_url,
                        "original_url": original_url,
                        "display_path": display_url,
                        "full_path": full_url,
                        "original_path": original_url,
                    },
                )

            # ── Enrich observations for template ───────────────────────
            def _format_duration(sec: float) -> str:
                """Format seconds to human-readable duration."""
                sec = max(0, int(sec))
                if sec < 60:
                    return f"{sec}s"
                minutes = sec // 60
                remaining = sec % 60
                return f"{minutes}m {remaining:02d}s"

            enriched_observations = []
            for obs in page_observations:
                cover_det = det_by_id.get(obs["cover_detection_id"])
                if not cover_det:
                    continue
                enriched_cover = enrich_detection(cover_det)

                # Enrich all detections in this observation for modals
                all_dets_enriched = []
                for did in obs["detection_ids"]:
                    raw = det_by_id.get(did)
                    if raw:
                        all_dets_enriched.append(enrich_detection(raw))
                all_dets_enriched.sort(
                    key=lambda det: (
                        det.get("image_timestamp", ""),
                        int(det.get("detection_id") or 0),
                    ),
                    reverse=True,
                )

                enriched_observations.append(
                    {
                        "observation_id": obs["observation_id"],
                        "species": obs["species"],
                        "common_name": _get_common_name_local(obs["species"]),
                        "photo_count": obs["photo_count"],
                        "duration_sec": obs["duration_sec"],
                        "duration_display": _format_duration(obs["duration_sec"]),
                        "best_score": obs["best_score"],
                        "cover_detection": enriched_cover,
                        "all_detections": all_dets_enriched,
                        "detection_ids": obs["detection_ids"],
                        "start_time": obs["start_time"],
                        "end_time": obs["end_time"],
                    }
                )

            # Modal navigation should follow the visible gallery sequence:
            # observation cards stay in grid order, and filmstrip detections
            # remain adjacent to their observation instead of being interleaved
            # globally across other observations.
            nav_index_by_detection_id: dict[int, int] = {}
            nav_order = [
                det
                for obs in enriched_observations
                for det in obs["all_detections"]
            ]
            for idx, det in enumerate(nav_order):
                det_id = int(det.get("detection_id") or 0)
                if det_id > 0:
                    nav_index_by_detection_id[det_id] = idx

            for obs in enriched_observations:
                obs["cover_detection"]["nav_index"] = nav_index_by_detection_id.get(
                    int(obs["cover_detection"].get("detection_id") or 0)
                )
                for det in obs["all_detections"]:
                    det["nav_index"] = nav_index_by_detection_id.get(
                        int(det.get("detection_id") or 0)
                    )

            # Species of the Day (page 1 only) — unchanged logic
            species_of_day = []
            if page == 1:
                species_candidates = {}
                for det in detections_raw:
                    species_key = _get_species_key_local(det)
                    species_candidates.setdefault(species_key, []).append(det)

                species_groups = {}
                for s_key, candidates in species_candidates.items():
                    chosen = _pick_cover_for_group(
                        candidates, seed_key=f"daydetail:{s_key}", date_iso=date
                    )
                    if chosen:
                        species_groups[s_key] = chosen

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
                total_items_unfiltered=total_obs_unfiltered,
                sort_by=sort_by,
                current_threshold=current_threshold,
                observations=enriched_observations,
                species_of_day=species_of_day,
                pagination_range=pagination_range,
                image_width=IMAGE_WIDTH,
                focus_observation_id=focus_observation_id,
                focus_detection_id=focus_id_param,
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
            """Admin diagnostics view with app log and runtime snapshots."""

            from collections import deque

            def _tail_text(path: Path, max_lines: int) -> str:
                if not path.exists():
                    return f"File not found: {path}"
                try:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        return "".join(deque(f, maxlen=max_lines))
                except Exception as e:
                    return f"Error reading {path.name}: {e}"

            logs_dir = Path(config["OUTPUT_DIR"]) / "logs"
            app_log_path = logs_dir / "app.log"
            vitals_path = logs_dir / "vital_signs.csv"

            return render_template(
                "logs.html",
                current_path="/logs",
                app_logs=_tail_text(app_log_path, 300),
                vitals_logs=_tail_text(vitals_path, 240),
                app_log_path=str(app_log_path),
                vitals_log_path=str(vitals_path),
            )

    setup_web_routes(server)

    # --- Phase 4: Homepage (Live Stream) Migrated to Flask (Final Phase) ---
    def index_route():
        """Server-rendered Homepage / Live Stream."""
        # Calculate 24h rolling window threshold
        from datetime import timedelta

        now = datetime.now()
        threshold_24h = now - timedelta(hours=24)
        threshold_str = threshold_24h.strftime("%Y%m%d_%H%M%S")
        today_iso = now.strftime("%Y-%m-%d")
        gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

        def _format_ts_parts(ts: str) -> tuple[str, str, str]:
            if not ts or len(ts) < 15:
                return "", "", ""
            date_str = ts[:8]
            time_str = ts[9:15]
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            gallery_date_iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return formatted_time, formatted_date, gallery_date_iso

        def _format_ts_human(ts: str) -> str:
            if not ts:
                return "Unknown"
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                return dt.strftime("%d.%m.%Y %H:%M")
            except ValueError:
                return "Unknown"

        def _format_epoch_human(ts: float) -> str:
            if ts <= 0:
                return "Never"
            try:
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError, OSError):
                return "Unknown"

        def _format_age_short(seconds: float) -> str:
            if seconds < 0:
                return "0s ago"
            if seconds < 60:
                return f"{int(seconds)}s ago"
            if seconds < 3600:
                return f"{int(seconds // 60)}m ago"
            if seconds < 86400:
                return f"{int(seconds // 3600)}h ago"
            return f"{int(seconds // 86400)}d ago"

        def _format_iso_human(ts: str | None) -> str:
            if not ts:
                return "n/a"
            try:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return "n/a"

        # 1. 24h Count (kept for downstream preview/feed usage)
        last_24h_count = 0
        last_24h_rows: list[dict] = []
        try:
            with db_service.closing_connection() as conn:
                last_24h_rows = [
                    dict(row)
                    for row in db_service.fetch_detections_last_24h(
                        conn, threshold_str, order_by="time"
                    )
                ]
            last_24h_summary = gallery_service.summarize_observations(
                last_24h_rows, min_score=gallery_threshold
            )
            last_24h_count = last_24h_summary["summary"]["total_observations"]
        except Exception as e:
            logger.error(f"Error fetching 24h count: {e}")

        # 1b. Dashboard Stats (All-time stats for engagement)
        dashboard_stats = {
            "total_detections": 0,
            "total_species": 0,
            "last_24h_count": last_24h_count,  # Renamed from today_count
            "today_count": 0,
            "first_date": None,
            "last_date": None,
        }
        try:
            with db_service.closing_connection() as conn:
                summary = db_service.fetch_analytics_summary(conn)
                dashboard_stats["total_detections"] = summary.get("total_detections", 0)
                dashboard_stats["total_species"] = summary.get("total_species", 0)
                date_range = summary.get("date_range", {})
                dashboard_stats["first_date"] = date_range.get("first")
                dashboard_stats["last_date"] = date_range.get("last")
                dashboard_stats["today_count"] = db_service.fetch_day_count(
                    conn, today_iso
                )
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {e}")

        # 1c. Today Observation Stats (aligned with gallery grouping)
        species_visit_counts: dict[str, int] = {}
        today_rows: list[dict] = []
        try:
            with db_service.closing_connection() as conn:
                today_rows = [
                    dict(row)
                    for row in db_service.fetch_detections_for_gallery(
                        conn, today_iso, order_by="time"
                    )
                ]
            today_summary = gallery_service.summarize_observations(
                today_rows, min_score=gallery_threshold
            )
            today_summary_stats = today_summary["summary"]
            dashboard_stats["today_visits"] = today_summary_stats["total_observations"]
            dashboard_stats["today_avg_confidence"] = today_summary_stats["avg_score"]
            species_visit_counts = today_summary_stats["species_counts"]
            today_rows = today_summary["detections"]
        except Exception as e:
            logger.error(f"Error fetching today observation stats: {e}")

        title = f"Live • {dashboard_stats.get('today_visits', 0)} Observations Today"

        # 2. Latest Detections (Top 5 from last 24h)
        latest_detections = []
        try:
            for det in last_24h_rows[:5]:
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
                formatted_time, formatted_date, _ = _format_ts_parts(ts)

                latest_detections.append(
                    {
                        "detection_id": det.get("detection_id"),
                        "species_key": _get_species_key_local(det),
                        "common_name": _get_common_name_local(
                            _get_species_key_local(det)
                        ),
                        "latin_name": _get_species_key_local(det),
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
                        "gallery_date": today_iso,  # For "Go to Day" button in modal
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

        # 3. Today's Visitors Summary (gallery-aligned observation scope)
        visual_summary = []
        try:
            species_candidates = {}
            for det in today_rows:
                s_key = _get_species_key_local(det)
                species_candidates.setdefault(s_key, []).append(det)

            species_groups = {}
            for s_key, candidates in species_candidates.items():
                chosen = _pick_cover_for_group(
                    candidates, seed_key=f"day:{s_key}", date_iso=today_iso
                )
                if chosen:
                    species_groups[s_key] = chosen

            # Sort by quality (favorite first, then score)
            sorted_summary = sorted(
                species_groups.values(),
                key=_cover_quality_tuple,
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
                formatted_time, formatted_date, _ = _format_ts_parts(ts)

                visual_summary.append(
                    {
                        "detection_id": det.get("detection_id"),
                        "species_key": _get_species_key_local(det),
                        "common_name": _get_common_name_local(
                            _get_species_key_local(det)
                        ),
                        "latin_name": _get_species_key_local(det),
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
                        "is_favorite": bool(int(det.get("is_favorite") or 0)),
                    }
                )

        except Exception as e:
            logger.error(f"Error fetching visual summary: {e}")

        # 4. Species Summary Table (enriched with best image from visual_summary)
        species_summary_table = []
        try:
            species_summary_table = get_daily_species_summary(today_iso)

            # Enrich visual_summary with visit count (not detection count)
            for det in visual_summary:
                species_key = det.get("species_key") or det.get("latin_name", "")
                det["count"] = species_visit_counts.get(species_key, 0)

        except Exception as e:
            logger.error(f"Error fetching species summary table: {e}")

        # 5. Archive Preview – latest 5 UNIQUE species (deduplicated)
        recent_archive_preview = []
        try:
            with db_service.closing_connection() as conn:
                # Fetch more rows so we can deduplicate by species
                rows = db_service.fetch_detections_for_gallery(
                    conn, limit=30, order_by="time"
                )
                seen_species = set()
                for row in rows:
                    det = dict(row)
                    species_key = _get_species_key_local(det)

                    # Skip if we already have this species
                    if species_key in seen_species:
                        continue
                    seen_species.add(species_key)

                    full_path = det.get("relative_path") or det.get(
                        "optimized_name_virtual", ""
                    )
                    thumb_virtual = det.get("thumbnail_path_virtual")
                    if thumb_virtual:
                        display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    else:
                        display_url = f"/uploads/derivatives/optimized/{full_path}"

                    ts = det.get("image_timestamp", "")
                    formatted_time, formatted_date, gallery_date_iso = _format_ts_parts(
                        ts
                    )
                    recent_archive_preview.append(
                        {
                            "detection_id": det.get("detection_id"),
                            "common_name": _get_common_name_local(species_key),
                            "display_path": display_url,
                            "formatted_time": formatted_time,
                            "formatted_date": formatted_date,
                            "gallery_date_iso": gallery_date_iso,
                            "image_timestamp": ts,
                        }
                    )

                    if len(recent_archive_preview) >= 5:
                        break
        except Exception as e:
            logger.error(f"Error fetching archive preview: {e}")

        # 5b. Best Species Story Board
        best_species_board = {"featured": [], "grid": []}
        best_species_preview = []
        try:
            best_species_board = _enrich_species_board(
                gallery_service.build_species_story_board(
                    get_captured_detections(),

                    total_limit=12,
                    featured_count=3,
                    excluded_species={UNKNOWN_SPECIES_KEY},
                )
            )
            best_species_preview = (
                best_species_board.get("featured", [])
                + best_species_board.get("grid", [])
            )
        except Exception as e:
            logger.error(f"Error fetching best species board: {e}")

        # 6. Landing Status Row
        landing_status = {
            "last_detection": "No detections yet",
            "stream_state": "Unknown",
            "stream_tone": "neutral",
            "stream_detail": "No successful frame yet",
        }

        if recent_archive_preview:
            latest = recent_archive_preview[0]
            landing_status["last_detection"] = (
                f"{latest.get('common_name', 'Unknown')} · "
                f"{_format_ts_human(latest.get('image_timestamp', ''))}"
            )

        try:
            frame_ts = 0.0
            last_good_frame_ts = 0.0
            first_frame_received = False
            frame_lock = getattr(detection_manager, "frame_lock", None)
            if frame_lock is not None:
                with frame_lock:
                    frame_ts = float(
                        getattr(detection_manager, "latest_raw_timestamp", 0.0) or 0.0
                    )
                    last_good_frame_ts = float(
                        getattr(detection_manager, "last_good_frame_timestamp", 0.0)
                        or 0.0
                    )
                    first_frame_received = bool(
                        getattr(detection_manager, "_first_frame_received", False)
                    )
            else:
                frame_ts = float(
                    getattr(detection_manager, "latest_raw_timestamp", 0.0) or 0.0
                )
                last_good_frame_ts = float(
                    getattr(detection_manager, "last_good_frame_timestamp", 0.0) or 0.0
                )
                first_frame_received = bool(
                    getattr(detection_manager, "_first_frame_received", False)
                )

            if first_frame_received and last_good_frame_ts <= 0 and frame_ts > 0:
                last_good_frame_ts = frame_ts

            now_ts = time.time()
            if frame_ts <= 0:
                landing_status["stream_state"] = "Offline"
                landing_status["stream_tone"] = "bad"
            else:
                frame_age = now_ts - frame_ts
                if first_frame_received and frame_age <= 5:
                    landing_status["stream_state"] = "Online"
                    landing_status["stream_tone"] = "ok"
                elif frame_age <= 20:
                    landing_status["stream_state"] = "Starting"
                    landing_status["stream_tone"] = "warn"
                else:
                    landing_status["stream_state"] = "Offline"
                    landing_status["stream_tone"] = "bad"

            if last_good_frame_ts > 0:
                last_good_age = max(0.0, now_ts - last_good_frame_ts)
                landing_status["stream_detail"] = (
                    f"Last good frame: {_format_age_short(last_good_age)}"
                )
            elif landing_status["stream_state"] == "Starting":
                landing_status["stream_detail"] = ""
        except Exception as e:
            logger.debug(f"Could not compute stream status for landing: {e}")

        today_detection_count = dashboard_stats.get("today_count", 0)
        is_quiet_today = today_detection_count == 0

        noise_hourly = {"noise": [], "total_noise": 0, "total_birds": 0}

        return render_template(
            "stream.html",
            title=title,
            current_path="/",  # Active nav state
            latest_detections=latest_detections,
            visual_summary=visual_summary,
            species_summary=species_summary_table,
            dashboard_stats=dashboard_stats,
            empty_latest_message="No detections in the last 24 hours.",
            image_width=IMAGE_WIDTH,
            today_iso=today_iso,
            today_detection_count=today_detection_count,
            is_quiet_today=is_quiet_today,
            recent_archive_preview=recent_archive_preview,
            best_species_board=best_species_board,
            best_species_preview=best_species_preview,
            landing_status=landing_status,
            noise_hourly=noise_hourly,
        )

    def tbwd_presentation_route():
        return render_template(
            "tbwd_presentation.html",
            current_path="/tbwd",
        )

    server.add_url_rule(
        "/tbwd",
        endpoint="tbwd_presentation",
        view_func=tbwd_presentation_route,
        methods=["GET"],
    )

    def tbwd_vision_route():
        return render_template(
            "tbwd_vision.html",
            current_path="/tbwd-vision",
        )

    server.add_url_rule(
        "/tbwd-vision",
        endpoint="tbwd_vision",
        view_func=tbwd_vision_route,
        methods=["GET"],
    )

    def tbwd_habitat_route():
        return render_template(
            "tbwd_habitat.html",
            current_path="/tbwd-habitat",
        )

    server.add_url_rule(
        "/tbwd-habitat",
        endpoint="tbwd_habitat",
        view_func=tbwd_habitat_route,
        methods=["GET"],
    )
    server.add_url_rule("/", endpoint="index", view_func=index_route, methods=["GET"])

    RUNTIME_BOOL_KEYS = {
        "DAY_AND_NIGHT_CAPTURE",
        "TELEGRAM_ENABLED",
        "DEBUG_MODE",
        "EXIF_GPS_ENABLED",
        "INBOX_REQUIRE_EXIF_DATETIME",
        "INBOX_REQUIRE_EXIF_GPS",
        "MOTION_DETECTION_ENABLED",
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
        "CPU_LIMIT": "CPU Core Limit (0 disables affinity, >0 pins to first N available cores)",
        "VIDEO_SOURCE": "Video Source (Input RTSP URL or secondary camera ID)",
        "DETECTOR_MODEL_CHOICE": "Primary Detector Model (The AI engine selected at boot)",
        "STREAM_WIDTH_OUTPUT_RESIZE": "Stream Output Width (Visual resolution for the live feed)",
        "LOCATION_DATA": "Geographic Coordinates (Latitude/Longitude for metadata)",
        "GALLERY_DISPLAY_THRESHOLD": "Species Summary Min. Score (Quality score threshold: 50% OD + 50% CLS confidence, or OD-only if no classification)",
        "TELEGRAM_BOT_TOKEN": "Telegram Bot Token (From BotFather)",
        "TELEGRAM_CHAT_ID": "Telegram Chat ID (Numeric ID or JSON list of IDs)",
        "EXIF_GPS_ENABLED": "Write GPS to Exif (Safe to disable for privacy)",
        "INBOX_REQUIRE_EXIF_DATETIME": "Inbox Require EXIF Date/Time (Skip imports without DateTimeOriginal/DateTimeDigitized)",
        "INBOX_REQUIRE_EXIF_GPS": "Inbox Require EXIF GPS (Skip imports without GPSLatitude/GPSLongitude)",
        "SPECIES_COMMON_NAME_LOCALE": "Species Common Names (Language for display names: DE=Deutsch, NO=Norsk)",
    }

    # Keys ordered for UI display purposes
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
        "INBOX_REQUIRE_EXIF_DATETIME",
        "INBOX_REQUIRE_EXIF_GPS",
        "SPECIES_COMMON_NAME_LOCALE",
        "TELEGRAM_COOLDOWN",
        "TELEGRAM_ENABLED",
        "GALLERY_DISPLAY_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "EDIT_PASSWORD",
        "DEBUG_MODE",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "MOTION_DETECTION_ENABLED",
        "MOTION_SENSITIVITY",
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
            with db_service.closing_connection() as conn:
                trash_count = db_service.fetch_trash_count(conn)
        except Exception as e:
            logger.error(f"Failed to fetch trash count: {e}")

        payload = get_settings_payload()

        # Security: Mask RTSP password in UI
        if "VIDEO_SOURCE" in payload:
            payload["VIDEO_SOURCE"]["value"] = mask_rtsp_url(
                payload["VIDEO_SOURCE"]["value"]
            )
        if "CAMERA_URL" in payload:
            payload["CAMERA_URL"]["value"] = mask_rtsp_url(
                payload["CAMERA_URL"]["value"]
            )

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

    @server.route("/api/system/versions", methods=["GET"])
    @login_required
    def system_versions_route():
        from utils.deploy_info import read_build_metadata

        # Build metadata from shared helper
        meta = read_build_metadata()

        data = {
            "app_version": meta["app_version"],
            "git_commit": meta["git_commit"],
            "build_date": meta["build_date"],
            "deploy_type": meta["deploy_type"],
            "kernel": "Unknown",
            "os": "Unknown",
            "bootloader": "Unknown",
        }

        # Kernel
        try:
            data["kernel"] = platform.release()
        except Exception:
            pass

        # OS
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            data["os"] = line.split("=")[1].strip().strip('"')
                            break
        except Exception:
            pass

        # Bootloader
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

            if not is_power_management_available():
                logger.warning(
                    "Shutdown ignored: systemd not available (likely container)."
                )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                        }
                    ),
                    400,
                )
            schedule_power_action("shutdown", logger)

            return (
                jsonify(
                    {
                        "status": "success",
                        "message": get_power_action_success_message("shutdown"),
                    }
                ),
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

            if not is_power_management_available():
                logger.warning(
                    "Restart ignored: systemd not available (likely container)."
                )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                        }
                    ),
                    400,
                )
            schedule_power_action("restart", logger)

            return (
                jsonify(
                    {
                        "status": "success",
                        "message": get_power_action_success_message("restart"),
                    }
                ),
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
    # INBOX ROUTES --- MOVED TO web/blueprints/inbox.py ---
    # ==========================================================================

    # ==========================================================================
    # DETECTION CONTROL ROUTES
    # ==========================================================================

    @server.route("/api/status", methods=["GET"])
    @login_required
    def api_status():
        """
        Returns general system status including detection and deep scan state.
        Used by various pages to check detection status.
        """
        try:
            output_dir = config.get("OUTPUT_DIR", "./data/output")

            response = {
                "detection_paused": detection_manager.paused,
                "detection_running": not detection_manager.paused,
                "restart_required": backup_restore_service.is_restart_required(
                    output_dir
                ),
            }

            # Scope 3: Deep Scan visibility
            try:
                from core.analysis_queue import analysis_queue
                from web.services.analysis_service import count_deep_scan_candidates

                response["deep_scan_active"] = detection_manager.is_deep_scan_active()
                response["deep_scan_queue_pending"] = analysis_queue.pending_count()
                response["deep_scan_candidates_remaining"] = (
                    count_deep_scan_candidates()
                )
            except Exception as ds_err:
                logger.debug(f"Could not compute deep scan status: {ds_err}")

            # P1-03: Session-level decision state counters (zero DB cost)
            try:
                response["decision_state_counts"] = dict(
                    detection_manager.decision_state_counts
                )
            except Exception:
                pass

            return jsonify(response)
        except Exception as e:
            logger.error(f"Status API error: {e}")
            return jsonify({"error": str(e)}), 500

    @server.route("/api/detection/pause", methods=["POST"])
    @login_required
    def detection_pause():
        """Pauses the detection loop."""
        try:
            if detection_manager.paused:
                return jsonify(
                    {
                        "status": "paused",
                        "message": "Detection was already paused",
                    }
                )

            detection_manager.paused = True
            logger.info("Detection paused via API")

            return jsonify(
                {
                    "status": "success",
                    "message": "Detection paused",
                }
            )

        except Exception as e:
            logger.error(f"Detection pause error: {e}")
            return jsonify({"error": str(e)}), 500

    @server.route("/api/detection/resume", methods=["POST"])
    @login_required
    def detection_resume():
        """Resumes the detection loop."""
        try:
            # Block during restore - MIGRATED to service
            if backup_restore_service.is_restore_active():
                return (
                    jsonify(
                        {"error": "Cannot resume detection during restore operation"}
                    ),
                    409,
                )

            if not detection_manager.paused:
                return jsonify(
                    {
                        "status": "running",
                        "message": "Detection was already running",
                    }
                )

            detection_manager.paused = False
            logger.info("Detection resumed via API")

            return jsonify(
                {
                    "status": "success",
                    "message": "Detection resumed",
                }
            )

        except Exception as e:
            logger.error(f"Detection resume error: {e}")
            return jsonify({"error": str(e)}), 500

    # ==========================================================================
    # BACKUP/RESTORE ROUTES --- MOVED TO web/blueprints/backup.py ---
    # ==========================================================================

    # -----------------------------
    # Security: Request Audit Logging
    # -----------------------------
    security_logger = logging.getLogger("security.access")

    # Paths that are polled by the frontend or serve static content
    _LOG_SKIP_PREFIXES = (
        "/assets/", "/favicon", "/uploads/",
        "/api/review-thumb/", "/api/thumb/",
        "/api/v1/health", "/api/v1/weather/",
        "/api/v1/system/versions", "/api/v1/system/diagnostics",
        "/api/v1/public/go2rtc/health", "/api/v1/cameras",
        "/logs",
    )

    @server.after_request
    def log_request(response):
        if not request.path.startswith(_LOG_SKIP_PREFIXES):
            security_logger.info(
                "%s %s %s ip=%s user=%s",
                request.method,
                request.path,
                response.status_code,
                request.remote_addr,
                "authenticated" if session.get("authenticated") else "anonymous",
            )
        return response

    # -----------------------------
    # Security Headers
    # -----------------------------
    @server.after_request
    def set_security_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # go2rtc runs on the same host but port 1984 — allow it in CSP
        host = request.host.split(":")[0]
        go2rtc_origin = f"http://{host}:1984"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            f"connect-src 'self' {go2rtc_origin}; "
            f"frame-src 'self' {go2rtc_origin}; "
            "frame-ancestors 'none';"
        )
        return response

    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    return server
