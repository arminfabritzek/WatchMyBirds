"""Dashboard landing route (``/``).

Extracted from the ``create_web_interface`` factory closure in
``web/web_interface.py``. Follows the ``init_*_bp()`` pattern used across
``web/blueprints/`` (reference: ``web/blueprints/gallery.py``): a module-level
``Blueprint``, an ``init_index_bp(detection_manager=...)`` injector, and the
route plus its formatting helpers as top-level functions.

The data helpers this route leans on (``enrich_species_board``,
``fetch_best_species_pools``, ``render_best_species_board``,
``pick_cover_for_group``,
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
from datetime import datetime

from flask import Blueprint, Response, current_app, g, render_template

from config import get_config
from core.species_colours import assign_species_colours as _assign_species_colours
from logging_config import get_logger
from utils.cover_selection import cover_sort_key
from web import view_helpers
from web import web_interface as _web_interface
from web.services import db_service, gallery_service

logger = get_logger(__name__)
config = get_config()

index_bp = Blueprint("index", __name__)

IMAGE_WIDTH = 150

_detection_manager = None


def init_index_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


def index_route(*, archive_only: bool = False) -> str | tuple[str, int]:

    now = datetime.now()
    today_iso = now.strftime("%Y-%m-%d")

    def _format_ts_parts(ts: str) -> tuple[str, str, str]:
        if not ts or len(ts) < 15:
            return "", "", ""
        date_str = ts[:8]
        time_str = ts[9:15]
        formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
        gallery_date_iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return formatted_time, formatted_date, gallery_date_iso

    sibling_rows_cache: dict[str, list] = {}

    def _load_modal_siblings(raw: dict) -> list[dict]:

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
            "manual_object_id": raw.get("manual_object_id"),
            "object_key": raw.get("object_key"),
            "object_kind": "manual" if raw.get("manual_object_id") else "detection",
            "original_name": raw.get("original_name") or "",
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

    dashboard_stats = {
        "total_species": None,
        "today_visits": 0,
        "today_busiest_hour": "",
    }
    species_visit_counts: dict[str, int] = {}
    today_rows: list[dict] = []
    today_summary: dict = {}

    title = "Live"
    visual_summary = []
    today_visitors_board = {"featured": [], "grid": []}
    today_visitors_modal_dets = []
    if not archive_only:
        with db_service.closing_connection() as conn:
            try:
                today_rows = [
                    dict(row)
                    for row in db_service.fetch_detections_for_gallery(
                        conn, today_iso, order_by="time"
                    )
                ]
                today_summary = gallery_service.summarize_observations(today_rows)
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
            for _det in today_rows:
                _name = _det.get("original_name") or ""
                if _name:
                    _sibling_keys.add(_name)
            if _sibling_keys:
                sibling_rows_cache.update(
                    gallery_service.get_sibling_detections_batch(sorted(_sibling_keys))
                )
        except Exception as e:
            logger.error(f"Error pre-fetching sibling rows: {e}")

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
                key=cover_sort_key,
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
                original_url = (
                    f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"
                )

                ts = det.get("image_timestamp", "")
                formatted_time, formatted_date, _ = _format_ts_parts(ts)

                visual_summary.append(
                    {
                        "detection_id": det.get("detection_id"),
                        "original_name": det.get("original_name") or "",
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

        try:
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

    best_species_board = {"featured": [], "grid": []}
    best_species_preview = []
    best_species_modal_dets = []
    cache = _web_interface._best_species_cache
    database_key = str(get_config()["OUTPUT_DIR"])
    if archive_only:
        try:
            # Concurrent browsers share one rebuild, instead of each sorting
            # the entire archive. Cache photo pools, never current human answers.
            with current_app.extensions["homepage_archive_lock"]:
                payload = cache.get("payload")
                if (
                    payload is None
                    or payload.get("database") != database_key
                    or time.time() - float(cache.get("timestamp") or 0)
                    >= _web_interface._BEST_SPECIES_CACHE_TTL_SECONDS
                ):
                    generation = cache.get("generation")
                    pools, modal_rows = view_helpers.fetch_best_species_pools(
                        total_limit=12, frames_per_species=20
                    )
                    payload = {
                        "database": database_key,
                        "pools": pools,
                        "modal_rows": modal_rows,
                        "total_species": pools[0].get("total_species_count", 0)
                        if pools
                        else 0,
                    }
                    if cache.get("generation") is generation:
                        cache.update(timestamp=time.time(), payload=payload)
            raw_best_board = view_helpers.render_best_species_board(
                payload["pools"], total_limit=12, featured_count=3, frame_count=4
            )
            best_species_board = view_helpers.enrich_species_board(raw_best_board)
            best_species_preview = (
                best_species_board["featured"] + best_species_board["grid"]
            )
            visible_ids = {
                det.get("detection_id")
                for card in best_species_preview
                for det in (card, *card.get("story_frames", []))
            }
            selected_rows = [
                row
                for row in payload["modal_rows"]
                if row.get("detection_id") in visible_ids
            ]
            sibling_keys = sorted(
                {
                    row["original_name"]
                    for row in selected_rows
                    if row.get("original_name")
                    and row["original_name"] not in sibling_rows_cache
                }
            )
            if sibling_keys:
                companions = gallery_service.get_sibling_detections_batch(sibling_keys)
                sibling_rows_cache.update(
                    (name, companions.get(name, [])) for name in sibling_keys
                )
            best_species_modal_dets = [
                _build_modal_detection(row) for row in selected_rows
            ]
            dashboard_stats["total_species"] = payload["total_species"]
        except Exception:
            logger.exception("Error fetching homepage archive")
            return "Best of Species could not load. Please retry.", 503
    else:
        payload = cache.get("payload")
        if (
            payload is not None
            and payload.get("database") == database_key
            and time.time() - float(cache.get("timestamp") or 0)
            < _web_interface._BEST_SPECIES_CACHE_TTL_SECONDS
        ):
            dashboard_stats["total_species"] = payload["total_species"]

    # Cached photo pools must not cache the current human answers.
    view_helpers.enrich_modal_review_states(
        visual_summary + today_visitors_modal_dets + best_species_modal_dets
    )

    # The shared ticker renders on this page too. Reuse this request's computed
    # totals instead of scanning the same observation history a second time.
    g.ticker_dashboard_stats = (
        {}
        if archive_only
        else {
            "today_visits": dashboard_stats.get("today_visits", 0),
            "total_species": dashboard_stats.get("total_species", 0),
            "today_busiest_hour": dashboard_stats.get("today_busiest_hour", ""),
            "today_species_count": len(species_visit_counts),
        }
    )

    return render_template(
        "partials/homepage_archive.html" if archive_only else "stream.html",
        title=title,
        current_path="/",
        visual_summary=visual_summary,
        today_visitors_board=today_visitors_board,
        today_visitors_modal_dets=today_visitors_modal_dets,
        dashboard_stats=dashboard_stats,
        image_width=IMAGE_WIDTH,
        today_iso=today_iso,
        stream_source_mode=config.get("STREAM_SOURCE_MODE", "auto"),
        best_species_board=best_species_board,
        best_species_preview=best_species_preview,
        best_species_modal_dets=best_species_modal_dets,
    )


index_bp.add_url_rule("/", endpoint="index", view_func=index_route, methods=["GET"])


@index_bp.get("/homepage/archive")
def homepage_archive_route() -> Response:
    """Load archive-wide cards after the live page is already usable."""
    result = index_route(archive_only=True)
    response = current_app.make_response(result)
    response.headers["Cache-Control"] = "no-store"
    return response
