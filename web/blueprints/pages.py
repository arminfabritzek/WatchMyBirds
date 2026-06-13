import os

from flask import Blueprint, current_app, jsonify, render_template, request

from config import get_config, get_settings_payload
from logging_config import get_logger
from utils.settings import mask_rtsp_url
from web import view_helpers
from web.blueprints.auth import login_required
from web.services import db_service

logger = get_logger(__name__)
config = get_config()

pages_bp = Blueprint("pages", __name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_ASSETS_FOLDER = os.path.join(_PROJECT_ROOT, "assets")


RUNTIME_BOOL_KEYS = {
    "DAY_AND_NIGHT_CAPTURE",
    "PTZ_TRACKING_OVERLAY_ENABLED",
    "TELEGRAM_ENABLED",
    "DEBUG_MODE",
    "EXIF_GPS_ENABLED",
    "INBOX_REQUIRE_EXIF_DATETIME",
    "INBOX_REQUIRE_EXIF_GPS",
    "MOTION_DETECTION_ENABLED",
    "TRAINING_EXPORT_AUTO_OPT_IN",
    "EXPORT_BURN_IN_METADATA",
}
RUNTIME_NUMBER_KEYS = {
    "SAVE_THRESHOLD",
    "NON_BIRD_CONFIRM_THRESHOLD",
    "DETECTION_INTERVAL_SECONDS",
    "STREAM_FPS",
    "STREAM_FPS_CAPTURE",
    "TELEGRAM_COOLDOWN",
    "GALLERY_DISPLAY_THRESHOLD",
    "MAX_DETECTIONS_PER_BURST",
    "BURST_WINDOW_SECONDS",
    "OD_NIGHT_START_OFFSET_MIN",
    "OD_NIGHT_END_OFFSET_MIN",
    "GALLERY_QUALITY_BOTTOM_PCT",
    "GALLERY_QUALITY_MIN_SCORED",
}

SETTING_LABELS = {
    "SAVE_THRESHOLD": "OD Save Threshold (Minimum detector confidence to save image to disk; ignored in Auto mode)",
    "SAVE_THRESHOLD_MODE": "Save Threshold Mode (Auto: derive from active model · Manual: use Save Threshold value)",
    "NON_BIRD_CONFIRM_THRESHOLD": "Non-bird Confidence Floor (Minimum OD confidence for marten/cat/squirrel/hedgehog to be saved and confirmed; birds use a separate CLS-based gate and are unaffected)",
    "NON_BIRD_DROP_BELOW_CONFIRM": "Drop Non-bird Below Floor (When on, weak non-bird detections are dropped pre-persist — no DB row, no image saved. Off keeps them as UNCERTAIN for bbox-cluster analysis)",
    "DETECTION_INTERVAL_SECONDS": "Detection Interval (Seconds between AI analysis cycles - higher = less CPU)",
    "DAY_AND_NIGHT_CAPTURE": "24/7 Capture (When ON, OD runs day and night. When OFF, OD pauses outside the daytime window — see offsets below)",
    "PTZ_TRACKING_OVERLAY_ENABLED": "Live Tracking Overlay (Draws the auto-PTZ target box and tracking state over the live stream; a diagnostic aid, off by default)",
    "DAY_AND_NIGHT_CAPTURE_LOCATION": "Sun Event Location (City name fallback; primary source is LOCATION_DATA lat/lon)",
    "OD_NIGHT_START_OFFSET_MIN": "OD Night Start Offset (Minutes added to civil dusk; positive = OD keeps running into the evening, default 30 for late-active species)",
    "OD_NIGHT_END_OFFSET_MIN": "OD Night End Offset (Minutes added to civil dawn; negative = OD starts earlier in the morning, default -45 for the dawn chorus)",
    "OD_NIGHT_TWILIGHT_MODE": "OD Night Twilight Mode (civil / nautical / geometric — defines what 'dawn' and 'dusk' mean)",
    "GALLERY_QUALITY_BOTTOM_PCT": "Gallery Quality Floor (Hide the blurriest N% of crops from gallery thumbnails, relative to this station's own sharpness range; 0 disables. Crops stay in detail views and export)",
    "GALLERY_QUALITY_MIN_SCORED": "Gallery Quality Min. Scored (Only apply the quality floor once at least this many crops have a sharpness score — protects small stations from hiding birds on a thin sample)",
    "TELEGRAM_ENABLED": "Live Alerts (Send Telegram messages when birds are detected)",
    "TELEGRAM_COOLDOWN": "Alert Cooldown (Minimum seconds between live alerts)",
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
    "EXPORT_BURN_IN_METADATA": "Burn Metadata Into Downloads (Embed species name + location into downloaded copies for iNaturalist re-import; the stored original is never changed. Off downloads the raw original)",
    "INBOX_REQUIRE_EXIF_DATETIME": "Inbox Require EXIF Date/Time (Skip imports without DateTimeOriginal/DateTimeDigitized)",
    "INBOX_REQUIRE_EXIF_GPS": "Inbox Require EXIF GPS (Skip imports without GPSLatitude/GPSLongitude)",
    "SPECIES_COMMON_NAME_LOCALE": "Species Common Names (Language for display names: DE=Deutsch, NO=Norsk)",
    "TRAINING_EXPORT_AUTO_OPT_IN": "Auto-queue approvals for training export (Each approved event immediately joins the training export pool — visible in /admin/export)",
    "MAX_DETECTIONS_PER_BURST": "Burst Cap (Max detections persisted per rolling window — protects review queue from flocks; 0 disables)",
    "BURST_WINDOW_SECONDS": "Burst Window (Seconds for the rolling burst-cap window)",
}

RUNTIME_KEYS_ORDER = [
    "VIDEO_SOURCE",
    "SAVE_THRESHOLD_MODE",
    "SAVE_THRESHOLD",
    "NON_BIRD_CONFIRM_THRESHOLD",
    "NON_BIRD_DROP_BELOW_CONFIRM",
    "MAX_DETECTIONS_PER_BURST",
    "BURST_WINDOW_SECONDS",
    "DETECTION_INTERVAL_SECONDS",
    "DAY_AND_NIGHT_CAPTURE",
    "DAY_AND_NIGHT_CAPTURE_LOCATION",
    "PTZ_TRACKING_OVERLAY_ENABLED",
    "OD_NIGHT_START_OFFSET_MIN",
    "OD_NIGHT_END_OFFSET_MIN",
    "OD_NIGHT_TWILIGHT_MODE",
    "GALLERY_QUALITY_BOTTOM_PCT",
    "GALLERY_QUALITY_MIN_SCORED",
    "LOCATION_DATA",
    "EXIF_GPS_ENABLED",
    "EXPORT_BURN_IN_METADATA",
    "INBOX_REQUIRE_EXIF_DATETIME",
    "INBOX_REQUIRE_EXIF_GPS",
    "SPECIES_COMMON_NAME_LOCALE",
    "TRAINING_EXPORT_AUTO_OPT_IN",
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
    "MAX_DETECTIONS_PER_BURST",
    "BURST_WINDOW_SECONDS",
}

SYSTEM_KEYS_ORDER = [
    "OUTPUT_DIR",
    "INGEST_DIR",
    "MODEL_BASE_PATH",
]


@pages_bp.route("/assets/<path:filename>")
def serve_asset(filename):
    return view_helpers.send_cached_static_file(
        _ASSETS_FOLDER,
        filename,
        max_age=view_helpers.ASSET_CACHE_SECONDS,
        private=False,
        immutable=False,
    )


@pages_bp.route("/tbwd", methods=["GET"])
def tbwd_presentation_route():
    return render_template("tbwd_presentation.html", current_path="/tbwd")


@pages_bp.route("/tbwd-vision", methods=["GET"])
def tbwd_vision_route():
    return render_template("tbwd_vision.html", current_path="/tbwd-vision")


@pages_bp.route("/privacy", methods=["GET"])
def privacy_route():
    return render_template("privacy.html")


@pages_bp.route("/settings", methods=["GET"])
@login_required
def settings_route():
    trash_count = 0
    try:
        with db_service.closing_connection() as conn:
            trash_count = db_service.fetch_trash_count(conn)
    except Exception as e:
        logger.error(f"Failed to fetch trash count: {e}")

    payload = get_settings_payload()

    if "VIDEO_SOURCE" in payload:
        payload["VIDEO_SOURCE"]["value"] = mask_rtsp_url(payload["VIDEO_SOURCE"]["value"])
    if "CAMERA_URL" in payload:
        payload["CAMERA_URL"]["value"] = mask_rtsp_url(payload["CAMERA_URL"]["value"])

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


@pages_bp.route("/admin/_profile", methods=["GET"])
@login_required
def admin_profile():
    import cProfile
    import io
    import pstats
    import time

    ALLOWED_ROUTES = {"/", "/analytics"}
    target = request.args.get("route", "/").strip()
    if target not in ALLOWED_ROUTES:
        return jsonify(
            {"error": "route not whitelisted", "allowed": sorted(ALLOWED_ROUTES)}
        ), 400

    try:
        top_n = max(1, min(100, int(request.args.get("top", "30"))))
    except ValueError:
        top_n = 30
    sort_key = request.args.get("sort", "cumulative")
    if sort_key not in ("cumulative", "tottime", "calls"):
        sort_key = "cumulative"

    client = current_app.test_client()
    with client.session_transaction() as sess:
        sess["authenticated"] = True

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    try:
        resp = client.get(target)
    finally:
        profiler.disable()
    wall_ms = int((time.perf_counter() - wall_start) * 1000)

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats(sort_key)
    stats.print_stats(top_n)
    text_dump = buf.getvalue()

    rows = []
    for func, (_cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append(
            {
                "file": func[0],
                "line": func[1],
                "func": func[2],
                "ncalls": nc,
                "tottime": round(tt, 4),
                "cumtime": round(ct, 4),
            }
        )
    rows.sort(
        key=lambda r: r["cumtime" if sort_key == "cumulative" else "tottime"],
        reverse=True,
    )
    rows = rows[:top_n]

    return jsonify(
        {
            "route": target,
            "status": resp.status_code,
            "response_bytes": len(resp.data),
            "wall_ms": wall_ms,
            "sort": sort_key,
            "top": rows,
            "text": text_dump,
        }
    )
