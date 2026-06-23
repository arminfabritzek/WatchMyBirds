


import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    request,
    session,
)
from flask_compress import Compress

from config import (
    get_config,
)
from utils.review_metadata import (
    BBOX_REVIEW_CORRECT,
    BBOX_REVIEW_WRONG,
    REVIEW_STATUS_CONFIRMED_BIRD,
)
from web.blueprints.auth import auth_bp
from web.security import safe_log_value as _slv
from web.services import (
    gallery_service,
)

_BEST_SPECIES_CACHE_TTL_SECONDS = 30
_best_species_cache: dict = {"timestamp": 0.0, "payload": None}


def invalidate_best_species_cache() -> None:


    _best_species_cache["timestamp"] = 0.0
    _best_species_cache["payload"] = None


_TICKER_STATS_CACHE_TTL_SECONDS = 60
_ticker_stats_cache: dict = {"key": None, "timestamp": 0.0, "payload": None}


_module_logger = logging.getLogger(__name__)


def _compute_ticker_dashboard_stats() -> dict:

    today_iso = datetime.now().strftime("%Y-%m-%d")
    try:
        threshold = float(get_config()["GALLERY_DISPLAY_THRESHOLD"])
    except Exception:
        threshold = 0.0
    cache_key = (today_iso, threshold)

    now_ts = time.time()
    if (
        _ticker_stats_cache["key"] == cache_key
        and (now_ts - float(_ticker_stats_cache["timestamp"] or 0.0))
        < _TICKER_STATS_CACHE_TTL_SECONDS
        and isinstance(_ticker_stats_cache["payload"], dict)
    ):
        return _ticker_stats_cache["payload"]

    stats = {
        "today_visits": 0,
        "total_species": 0,
        "today_busiest_hour": "",
        "today_species_count": 0,
    }
    try:
        from core import gallery_core as _gc
        from web.services import db_service as _dbs

        with _dbs.closing_connection() as conn:
            try:
                stats["total_species"] = _dbs.fetch_gallery_total_species_count(conn)
            except Exception:


                _module_logger.debug(
                    "ticker stats: total_species fetch failed", exc_info=True
                )
            try:
                today_rows = [
                    dict(row)
                    for row in _dbs.fetch_detections_for_gallery(
                        conn, today_iso, order_by="time"
                    )
                ]
                summary = _gc.summarize_observations(today_rows, min_score=threshold)
                s = summary["summary"]
                stats["today_visits"] = int(s.get("total_observations", 0) or 0)
                stats["today_species_count"] = len(s.get("species_counts", {}) or {})
                hour_buckets: dict[str, int] = {}
                for det in summary.get("detections", today_rows):
                    ts = det.get("image_timestamp", "") or ""
                    if len(ts) >= 11:
                        hh = ts[9:11]
                        hour_buckets[hh] = hour_buckets.get(hh, 0) + 1
                if hour_buckets:
                    peak = max(hour_buckets.items(), key=lambda kv: kv[1])[0]
                    stats["today_busiest_hour"] = f"{peak}:00"
            except Exception:
                _module_logger.debug(
                    "ticker stats: today aggregation failed", exc_info=True
                )
    except Exception:
        _module_logger.debug("ticker stats: outer wrapper failed", exc_info=True)

    _ticker_stats_cache["key"] = cache_key
    _ticker_stats_cache["timestamp"] = now_ts
    _ticker_stats_cache["payload"] = stats
    return stats


def create_web_interface(detection_manager, system_monitor=None):


    logger = logging.getLogger(__name__)


    config = get_config()

    output_dir = config["OUTPUT_DIR"]
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


    try:
        from web.services import backup_restore_service

        backup_restore_service.clear_restart_marker(output_dir)
    except Exception as e:
        logger.debug(f"Could not clear restart marker: {e}")

    from web import view_helpers

    _species_locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    view_helpers.init_common_names(_species_locale)


    project_root = os.path.dirname(os.path.dirname(__file__))
    template_folder = os.path.join(project_root, "templates")
    server = Flask(__name__, template_folder=template_folder)


    Compress(server)
    server.config["COMPRESS_MIMETYPES"] = ["text/html", "application/json"]


    server.jinja_env.globals["wikipedia_species_url"] = (
        gallery_service.get_species_wikipedia_url
    )
    server.jinja_env.globals["REVIEW_STATUS_CONFIRMED_BIRD"] = (
        REVIEW_STATUS_CONFIRMED_BIRD
    )
    server.jinja_env.globals["BBOX_REVIEW_CORRECT"] = BBOX_REVIEW_CORRECT
    server.jinja_env.globals["BBOX_REVIEW_WRONG"] = BBOX_REVIEW_WRONG


    server.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024


    def _load_or_create_secret_key(key_file: Path) -> str:


        if key_file.exists():
            return key_file.read_text().strip()
        key = secrets.token_hex(32)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return key_file.read_text().strip()
        with os.fdopen(fd, "w") as fh:
            fh.write(key)
        return key

    secret_key_path = Path(output_dir) / ".flask_secret_key"
    server.secret_key = os.environ.get(
        "FLASK_SECRET_KEY"
    ) or _load_or_create_secret_key(secret_key_path)
    server.config["PERMANENT_SESSION_LIFETIME"] = 28800

    server.config["SESSION_COOKIE_HTTPONLY"] = True
    server.config["SESSION_COOKIE_SAMESITE"] = "Lax"


    from utils.deploy_info import read_build_metadata as _read_build_metadata

    _footer_meta = _read_build_metadata()
    _footer_commit = (
        "" if _footer_meta["git_commit"] == "Unknown" else _footer_meta["git_commit"]
    )
    _footer_build_date = (
        ""
        if _footer_meta["build_date"] == "Unknown"
        else _footer_meta["build_date"][:10]
    )


    @server.context_processor
    def inject_security_context():
        from web.services import auth_service as _auth


        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_hex(32)

        is_authenticated = bool(session.get("authenticated"))
        warn = is_authenticated and _auth.is_default_password()
        return {
            "warn_default_password": warn,
            "setup_password_required": _auth.should_require_password_setup(),
            "csrf_token": session["_csrf_token"],
            "is_authenticated": is_authenticated,
            "can_moderate": is_authenticated,
            "station_name": str(config.get("STATION_NAME", "")),


            "ticker_dashboard_stats": _compute_ticker_dashboard_stats(),


            "footer_commit": _footer_commit,
            "footer_build_date": _footer_build_date,
        }


    _CSRF_EXEMPT_PATHS = frozenset()


    @server.before_request
    def check_csrf_token():
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return
        if request.path in _CSRF_EXEMPT_PATHS:
            return


        token = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or token != session.get("_csrf_token"):
            from flask import abort

            abort(403)


    server.register_blueprint(auth_bp)


    from web.blueprints.api_v1 import init_api_v1

    def _on_runtime_settings_applied(valid_updates: dict) -> None:

        if "SPECIES_COMMON_NAME_LOCALE" in valid_updates:
            view_helpers.refresh_common_names(
                valid_updates["SPECIES_COMMON_NAME_LOCALE"]
            )

    init_api_v1(
        server,
        detection_manager,
        system_monitor=system_monitor,
        on_runtime_settings_applied=_on_runtime_settings_applied,
    )


    from web.blueprints.media import media_bp

    server.register_blueprint(media_bp)


    from web.blueprints.onvif_ingest import init_onvif_ingest_bp, onvif_ingest_bp

    init_onvif_ingest_bp(detection_manager=detection_manager)
    server.register_blueprint(onvif_ingest_bp)


    from web.blueprints.cameras import cameras_bp, init_cameras_bp

    init_cameras_bp(detection_manager=detection_manager)
    server.register_blueprint(cameras_bp)


    from web.blueprints.system import init_system_bp, system_bp

    init_system_bp(detection_manager=detection_manager)
    server.register_blueprint(system_bp)


    from web.blueprints.stream import init_stream_bp, stream_bp

    init_stream_bp(detection_manager=detection_manager)
    server.register_blueprint(stream_bp)


    from web.blueprints.pages import pages_bp

    server.register_blueprint(pages_bp)


    from web.blueprints.gallery import gallery_bp, init_gallery_bp

    init_gallery_bp(detection_manager=detection_manager)
    server.register_blueprint(gallery_bp)


    from web.blueprints.trash import trash_bp

    server.register_blueprint(trash_bp)


    from web.blueprints.live_stream import live_stream_bp

    server.register_blueprint(live_stream_bp)


    from web.blueprints.review import review_bp

    server.register_blueprint(review_bp)


    from web.blueprints.inbox import inbox_bp, init_inbox_bp

    init_inbox_bp(detection_manager)
    server.register_blueprint(inbox_bp)


    from web.blueprints.analytics import analytics_bp

    server.register_blueprint(analytics_bp)


    from web.blueprints.backup import backup_bp, init_backup_bp

    init_backup_bp(detection_manager)
    server.register_blueprint(backup_bp)


    from web.blueprints.moderation import moderation_bp

    server.register_blueprint(moderation_bp)


    from utils.deploy_info import read_build_metadata
    from web.blueprints.training_export import (
        init_training_export_bp,
        training_export_bp,
    )


    _build_meta = read_build_metadata()
    init_training_export_bp(
        output_dir=output_dir,
        app_config=config,
        app_version=str(_build_meta.get("app_version") or ""),
    )
    server.register_blueprint(training_export_bp)


    from web.blueprints.user_groundtruth_export import (
        init_user_groundtruth_export_bp,
        user_groundtruth_export_bp,
    )

    init_user_groundtruth_export_bp(
        output_dir=output_dir,
        app_version=str(_build_meta.get("app_version") or ""),
    )
    server.register_blueprint(user_groundtruth_export_bp)

    from web.blueprints.retention import retention_bp

    server.register_blueprint(retention_bp)


    try:
        from web.blueprints.companion import (
            companion_bp,
            init_companion_bp,
        )
        from web.services.companion.factory import build_inference_client
        from web.services.companion.recorder import CompanionRecorder
        from web.services.companion.service import CompanionService
        from web.services.compute_lease_service import init_compute_lease_service

        lease_service = init_compute_lease_service(detection_manager)
        companion_client = build_inference_client(
            config,
            model_base_path=str(config.get("MODEL_BASE_PATH", "./data/models")),
        )
        companion_recorder = CompanionRecorder(base_dir=output_dir)
        companion_service = CompanionService(
            client=companion_client,
            recorder=companion_recorder,
            lease=lease_service,
            enabled=bool(config.get("COMPANION_ENABLED", False)),
            pause_detection=bool(
                config.get("COMPANION_PAUSE_DETECTION_DURING_INFERENCE", True)
            ),
            timeout_s=float(config.get("COMPANION_INFERENCE_TIMEOUT_S", 60)),
        )
        init_companion_bp(companion_service)
        server.register_blueprint(companion_bp)
        backend = str(config.get("COMPANION_INFERENCE_BACKEND", "llama_cpp"))
        model_id = getattr(companion_client, "model_id", "<unconfigured>")
        logger.info(
            "Companion v1 backend registered (enabled=%s, backend=%s, model=%s)",
            companion_service.enabled,
            backend,
            model_id,
        )
    except Exception as exc:
        logger.error("Companion v1 backend registration failed: %s", exc, exc_info=True)




    from web.blueprints.index import index_bp, init_index_bp

    init_index_bp(detection_manager=detection_manager)
    server.register_blueprint(index_bp)

    security_logger = logging.getLogger("security.access")


    _LOG_SKIP_PREFIXES = (
        "/assets/",
        "/favicon",
        "/uploads/",
        "/api/review-thumb/",
        "/api/thumb/",
        "/api/v1/health",
        "/api/v1/weather/",
        "/api/v1/system/versions",
        "/api/v1/system/diagnostics",
        "/api/v1/public/go2rtc/health",
        "/api/v1/cameras",
        "/healthz",
        "/logs",

        "/api/inbox/status",
        "/api/v1/system/backup/status",
        "/api/v1/system/backup/format/status",
        "/api/v1/system/backup/format/devices",
        "/api/v1/system/updates/status",
        "/api/v1/system/updates/check",
        "/api/v1/models/detector",
        "/api/v1/models/classifier",
    )

    @server.after_request
    def log_request(response):
        if not request.path.startswith(_LOG_SKIP_PREFIXES):
            authenticated = session.get("authenticated")


            is_routine_get = (
                authenticated
                and request.method == "GET"
                and 200 <= response.status_code < 300
            )
            log_fn = security_logger.debug if is_routine_get else security_logger.info
            log_fn(
                "%s %s %s ip=%s user=%s",
                _slv(request.method),
                _slv(request.path),
                response.status_code,
                _slv(request.remote_addr),
                "authenticated" if authenticated else "anonymous",
            )
        return response


    @server.after_request
    def set_security_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

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


    return server
