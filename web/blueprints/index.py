"""Dashboard landing route (``/``).

Extracted from the ``create_web_interface`` factory closure in
``web/web_interface.py``. Follows the ``init_*_bp()`` pattern used across
``web/blueprints/`` (reference: ``web/blueprints/gallery.py``): a module-level
``Blueprint``, an ``init_index_bp(detection_manager=...)`` injector, and the
route plus its formatting helpers as top-level functions.

The data helpers this route leans on (``enrich_species_board``,
``fetch_best_species_pools``, ``render_best_species_board``,
``get_daily_species_summary``, ``pick_cover_for_group``,
``build_detection_view_dict``, ``date_iso_from_timestamp``, ``get_species_key``,
``get_common_name``, plus ``COMMON_NAMES`` / ``UNKNOWN_SPECIES_KEY``) live in
``web/view_helpers.py`` and are read live (``COMMON_NAMES`` is the same dict
object mutated in place on a locale change).

The best-species board cache is deliberately *not* owned here: its canonical
home stays ``web.web_interface`` because ``web/blueprints/trash.py`` imports
``invalidate_best_species_cache`` from there and a test inspects
``web_interface._best_species_cache`` directly. This module reads and writes the
*same* dict object via ``web_interface`` so invalidation stays coherent.

The dashboard route is registered via ``add_url_rule`` with ``endpoint="index"``
on the blueprint, so under Flask's blueprint namespacing it resolves as
``index.index``. The URL path (``/``) is unchanged; no caller references the
endpoint by name (there are no ``url_for("index")`` callers in the codebase).
"""

import time
from datetime import datetime, timedelta

from flask import Blueprint, render_template

from config import get_config
from core.gallery_core import cover_quality_tuple as _cover_quality_tuple
from core.species_colours import assign_species_colours as _assign_species_colours
from logging_config import get_logger
from web import view_helpers
from web import web_interface as _web_interface
from web.services import db_service, gallery_service

logger = get_logger(__name__)
config = get_config()

index_bp = Blueprint("index", __name__)

IMAGE_WIDTH = 150

_ANALYTICS_SUMMARY_CACHE_TTL_SECONDS = 5 * 60
_analytics_summary_cache: dict = {"timestamp": 0.0, "payload": None}


_detection_manager = None


def init_index_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


def index_route():

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

    sibling_rows_cache: dict[str, list] = {}

    def _load_modal_siblings(raw: dict) -> list[dict]:

        sibling_count = raw.get("sibling_count", 1) or 1
        if sibling_count <= 1:
            return []
        original_name = raw.get("original_name", "")
        if not original_name:
            return []
        cached = sibling_rows_cache.get(original_name)
        if cached is not None:
            sib_rows = cached
        else:
            sib_rows = gallery_service.get_sibling_detections(original_name)
        out: list[dict] = []
        for sib in sib_rows:
            sib_sk = view_helpers.get_species_key(sib)
            sib_thumb = sib.get("thumbnail_path_virtual")
            out.append(
                view_helpers.build_detection_view_dict(
                    sib,
                    species_key=sib_sk,
                    common_name=view_helpers.get_common_name(sib_sk),
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
        return out

    def _build_modal_detection(raw: dict, gallery_date: str = "") -> dict:

        fp = raw.get("relative_path") or raw.get("optimized_name_virtual", "")
        tv = raw.get("thumbnail_path_virtual")
        d_url = (
            f"/uploads/derivatives/thumbs/{tv}"
            if tv
            else f"/uploads/derivatives/optimized/{fp}"
        )
        f_url = f"/uploads/derivatives/optimized/{fp}"
        o_url = f"/uploads/originals/{fp.replace('.webp', '.jpg')}"
        ts = raw.get("image_timestamp", "")
        ft, fd, _ = _format_ts_parts(ts)
        sk = view_helpers.get_species_key(raw)
        return {
            "detection_id": raw.get("detection_id"),
            "species_key": sk,
            "common_name": view_helpers.get_common_name(sk),
            "latin_name": sk,
            "od_class_name": raw.get("od_class_name") or "",
            "od_confidence": raw.get("od_confidence") or 0.0,
            "cls_class_name": raw.get("cls_class_name") or "",
            "cls_confidence": raw.get("cls_confidence") or 0.0,
            "score": float(raw.get("score") or 0.0),
            "display_path": d_url,
            "full_path": f_url,
            "original_path": o_url,
            "formatted_time": ft,
            "formatted_date": fd,
            "gallery_date": gallery_date or view_helpers.date_iso_from_timestamp(ts),
            "sibling_count": raw.get("sibling_count", 1) or 1,
            "siblings": _load_modal_siblings(raw),
            "bbox_x": raw.get("bbox_x", 0.0) or 0.0,
            "bbox_y": raw.get("bbox_y", 0.0) or 0.0,
            "bbox_w": raw.get("bbox_w", 0.0) or 0.0,
            "bbox_h": raw.get("bbox_h", 0.0) or 0.0,
            "is_favorite": bool(int(raw.get("is_favorite") or 0)),
            "is_gallery_eligible": bool(int(raw.get("is_gallery_eligible") or 0)),
        }

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

    last_24h_count = 0
    last_24h_rows: list[dict] = []
    last_24h_summary: dict = {}
    dashboard_stats = {
        "total_detections": 0,
        "total_species": 0,
        "last_24h_count": 0,
        "today_count": 0,
        "first_date": None,
        "last_date": None,
    }
    species_visit_counts: dict[str, int] = {}
    today_rows: list[dict] = []
    today_summary: dict = {}

    with db_service.closing_connection() as conn:
        try:
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
            dashboard_stats["last_24h_count"] = last_24h_count
        except Exception as e:
            logger.error(f"Error fetching 24h count: {e}")

        try:
            _species_total_cached = _analytics_summary_cache.get("payload")
            _summary_age = time.time() - float(
                _analytics_summary_cache.get("timestamp") or 0.0
            )
            if (
                isinstance(_species_total_cached, int)
                and _summary_age < _ANALYTICS_SUMMARY_CACHE_TTL_SECONDS
            ):
                total_species = _species_total_cached
            else:
                total_species = db_service.fetch_gallery_total_species_count(conn)
                _analytics_summary_cache["timestamp"] = time.time()
                _analytics_summary_cache["payload"] = total_species
            dashboard_stats["total_species"] = total_species
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {e}")

        try:
            today_rows = [
                dict(row)
                for row in db_service.fetch_detections_for_gallery(
                    conn, today_iso, order_by="time"
                )
            ]
            today_summary = gallery_service.summarize_observations(
                today_rows, min_score=gallery_threshold
            )
        except Exception as e:
            logger.error(f"Error fetching today observation stats: {e}")

    try:
        today_summary_stats = today_summary["summary"]
        dashboard_stats["today_visits"] = today_summary_stats["total_observations"]
        species_visit_counts = today_summary_stats["species_counts"]
        today_rows = today_summary["detections"]

        hour_buckets: dict[str, int] = {}
        for _det in today_rows:
            _ts = _det.get("image_timestamp", "") or ""
            if len(_ts) >= 11:
                hour_buckets[_ts[9:11]] = hour_buckets.get(_ts[9:11], 0) + 1
        if hour_buckets:
            _peak = max(hour_buckets.items(), key=lambda kv: kv[1])[0]
            dashboard_stats["today_busiest_hour"] = f"{_peak}:00"
    except Exception as e:
        logger.error(f"Error fetching today observation stats: {e}")

    title = f"Live • {dashboard_stats.get('today_visits', 0)} Observations Today"

    try:
        _sibling_keys: set[str] = set()
        for _det in last_24h_rows[:5]:
            if (_det.get("sibling_count") or 1) > 1:
                _name = _det.get("original_name") or ""
                if _name:
                    _sibling_keys.add(_name)
        for _det in today_rows:
            if (_det.get("sibling_count") or 1) > 1:
                _name = _det.get("original_name") or ""
                if _name:
                    _sibling_keys.add(_name)
        if _sibling_keys:
            sibling_rows_cache.update(
                gallery_service.get_sibling_detections_batch(sorted(_sibling_keys))
            )
    except Exception as e:
        logger.error(f"Error pre-fetching sibling rows: {e}")

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
            original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"

            display_path = display_url
            original_path = original_url
            ts = det.get("image_timestamp", "")
            formatted_time, formatted_date, _ = _format_ts_parts(ts)

            latest_detections.append(
                {
                    "detection_id": det.get("detection_id"),
                    "species_key": view_helpers.get_species_key(det),
                    "common_name": view_helpers.get_common_name(
                        view_helpers.get_species_key(det)
                    ),
                    "latin_name": view_helpers.get_species_key(det),
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
                    "gallery_date": today_iso,
                    "sibling_count": det.get("sibling_count", 1) or 1,
                    "siblings": _load_modal_siblings(det),
                    "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                    "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                    "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                    "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                }
            )
    except Exception as e:
        logger.error(f"Error fetching latest detections: {e}")

    visual_summary = []
    try:
        species_candidates = {}
        for det in today_rows:
            s_key = view_helpers.get_species_key(det)
            species_candidates.setdefault(s_key, []).append(det)

        species_groups = {}
        for s_key, candidates in species_candidates.items():
            chosen = view_helpers.pick_cover_for_group(
                candidates, seed_key=f"day:{s_key}", date_iso=today_iso
            )
            if chosen:
                species_groups[s_key] = chosen

        sorted_summary = sorted(
            species_groups.values(),
            key=_cover_quality_tuple,
            reverse=True,
        )

        for det in sorted_summary:
            full_path = det.get("relative_path") or det.get(
                "optimized_name_virtual", ""
            )
            thumb_virtual = det.get("thumbnail_path_virtual")

            if thumb_virtual:
                display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
            else:
                display_url = f"/uploads/derivatives/optimized/{full_path}"

            full_url = f"/uploads/derivatives/optimized/{full_path}"
            original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"

            ts = det.get("image_timestamp", "")
            formatted_time, formatted_date, _ = _format_ts_parts(ts)

            visual_summary.append(
                {
                    "detection_id": det.get("detection_id"),
                    "species_key": view_helpers.get_species_key(det),
                    "common_name": view_helpers.get_common_name(
                        view_helpers.get_species_key(det)
                    ),
                    "latin_name": view_helpers.get_species_key(det),
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
                    "gallery_date": today_iso,
                    "sibling_count": det.get("sibling_count", 1) or 1,
                    "siblings": _load_modal_siblings(det),
                    "bbox_x": det.get("bbox_x", 0.0) or 0.0,
                    "bbox_y": det.get("bbox_y", 0.0) or 0.0,
                    "bbox_w": det.get("bbox_w", 0.0) or 0.0,
                    "bbox_h": det.get("bbox_h", 0.0) or 0.0,
                    "is_favorite": bool(int(det.get("is_favorite") or 0)),
                    "is_gallery_eligible": bool(
                        int(det.get("is_gallery_eligible") or 0)
                    ),
                }
            )

    except Exception as e:
        logger.error(f"Error fetching visual summary: {e}")

    species_summary_table = []
    try:
        species_summary_table = view_helpers.get_daily_species_summary(today_iso)

        for det in visual_summary:
            species_key = det.get("species_key") or det.get("latin_name", "")
            det["count"] = species_visit_counts.get(species_key, 0)

        _stream_vis_colour_map = _assign_species_colours(
            [d.get("species_key") or "" for d in visual_summary]
        )
        for det in visual_summary:
            det["species_colour"] = _stream_vis_colour_map.get(
                det.get("species_key") or "", None
            )

    except Exception as e:
        logger.error(f"Error fetching species summary table: {e}")

    today_visitors_board = {"featured": [], "grid": []}
    today_visitors_modal_dets = []
    try:
        if today_rows:
            today_visitors_board = view_helpers.enrich_species_board(
                gallery_service.build_species_story_board(
                    today_rows,
                    total_limit=12,
                    featured_count=3,
                    excluded_species={view_helpers.UNKNOWN_SPECIES_KEY},
                )
            )

            board_det_ids = set()
            for section in ("featured", "grid"):
                for item in today_visitors_board.get(section, []):
                    if item.get("detection_id"):
                        board_det_ids.add(item["detection_id"])
                    for frame in item.get("story_frames", []):
                        if frame.get("detection_id"):
                            board_det_ids.add(frame["detection_id"])

            vs_ids = {d.get("detection_id") for d in visual_summary}
            extra_ids = board_det_ids - vs_ids
            if extra_ids:
                today_rows_by_id = {d.get("detection_id"): d for d in today_rows}
                for det_id in extra_ids:
                    raw = today_rows_by_id.get(det_id)
                    if raw:
                        today_visitors_modal_dets.append(
                            _build_modal_detection(raw, today_iso)
                        )
    except Exception as e:
        logger.error(f"Error fetching today visitors board: {e}")

    recent_archive_preview = []
    try:
        with db_service.closing_connection() as conn:
            rows = db_service.fetch_detections_for_gallery(
                conn, limit=30, order_by="time"
            )
            seen_species = set()
            for row in rows:
                det = dict(row)
                species_key = view_helpers.get_species_key(det)

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
                formatted_time, formatted_date, gallery_date_iso = _format_ts_parts(ts)
                recent_archive_preview.append(
                    {
                        "detection_id": det.get("detection_id"),
                        "common_name": view_helpers.get_common_name(species_key),
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

    best_species_board = {"featured": [], "grid": []}
    best_species_preview = []
    best_species_modal_dets = []
    _best_species_cache = _web_interface._best_species_cache
    cache_payload = _best_species_cache.get("payload")
    cache_age = time.time() - float(_best_species_cache.get("timestamp") or 0.0)
    species_pools: list[dict] | None = None
    if (
        cache_payload is not None
        and cache_age < _web_interface._BEST_SPECIES_CACHE_TTL_SECONDS
    ):
        species_pools = cache_payload["pools"]
        best_species_modal_dets = cache_payload["modal_dets"]
    else:
        try:
            raw_pools, raw_modal_rows = view_helpers.fetch_best_species_pools(
                total_limit=12,
                frames_per_species=20,
            )
            species_pools = raw_pools
            best_species_modal_dets = [
                _build_modal_detection(raw) for raw in raw_modal_rows
            ]
            _best_species_cache["timestamp"] = time.time()
            _best_species_cache["payload"] = {
                "pools": species_pools,
                "modal_dets": best_species_modal_dets,
            }
        except Exception as e:
            logger.error(f"Error fetching best species board: {e}")

    if species_pools:
        try:
            raw_best_board = view_helpers.render_best_species_board(
                species_pools,
                total_limit=12,
                featured_count=3,
                frame_count=3,
            )
            best_species_board = view_helpers.enrich_species_board(raw_best_board)
            best_species_preview = best_species_board.get(
                "featured", []
            ) + best_species_board.get("grid", [])
        except Exception as e:
            logger.error(f"Error rendering best species board: {e}")

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
        frame_lock = getattr(_detection_manager, "frame_lock", None)
        if frame_lock is not None:
            with frame_lock:
                frame_ts = float(
                    getattr(_detection_manager, "latest_raw_timestamp", 0.0) or 0.0
                )
                last_good_frame_ts = float(
                    getattr(_detection_manager, "last_good_frame_timestamp", 0.0) or 0.0
                )
                first_frame_received = bool(
                    getattr(_detection_manager, "_first_frame_received", False)
                )
        else:
            frame_ts = float(
                getattr(_detection_manager, "latest_raw_timestamp", 0.0) or 0.0
            )
            last_good_frame_ts = float(
                getattr(_detection_manager, "last_good_frame_timestamp", 0.0) or 0.0
            )
            first_frame_received = bool(
                getattr(_detection_manager, "_first_frame_received", False)
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
        current_path="/",
        latest_detections=latest_detections,
        visual_summary=visual_summary,
        today_visitors_board=today_visitors_board,
        today_visitors_modal_dets=today_visitors_modal_dets,
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
        best_species_modal_dets=best_species_modal_dets,
        landing_status=landing_status,
        noise_hourly=noise_hourly,
    )


index_bp.add_url_rule("/", endpoint="index", view_func=index_route, methods=["GET"])
