from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

from flask import send_from_directory

from config import get_config
from core.species_colours import assign_species_colours
from utils.species_names import (
    load_common_names,
    resolve_common_name,
    species_key_from_candidates,
)
from web.security import safe_log_value as _slv
from web.services import db_service, gallery_service

logger = logging.getLogger(__name__)

UNKNOWN_SPECIES_KEY = "Unknown_species"
IMAGE_CACHE_SECONDS = 30 * 24 * 60 * 60
ASSET_CACHE_SECONDS = 7 * 24 * 60 * 60

# Mutable runtime state: a single dict object, mutated in place so that every
# importer observing ``view_helpers.COMMON_NAMES`` sees locale changes live.
COMMON_NAMES: dict[str, str] = {}


def init_common_names(locale: str | None = None) -> None:
    if locale is None:
        locale = get_config().get("SPECIES_COMMON_NAME_LOCALE", "DE")
    refresh_common_names(locale)


def refresh_common_names(locale: str) -> None:
    new_names = load_common_names(locale)
    COMMON_NAMES.clear()
    COMMON_NAMES.update(new_names)


# ---- static file serving (shared by media_bp + pages_bp /assets) -----------


def send_cached_static_file(
    directory: str | os.PathLike,
    filename: str,
    *,
    max_age: int,
    private: bool = True,
    immutable: bool = False,
):
    response = send_from_directory(
        directory,
        filename,
        conditional=True,
        max_age=max_age,
    )
    visibility = "private" if private else "public"
    cache_control = f"{visibility}, max-age={max_age}"
    if immutable:
        cache_control += ", immutable"
    response.headers["Cache-Control"] = cache_control
    return response


def send_contained_upload(
    root: os.PathLike,
    relative: str,
    *,
    max_age: int,
    private: bool = True,
    immutable: bool = False,
):
    root_path = Path(root).resolve()
    if (
        not isinstance(relative, str)
        or not relative
        or "\x00" in relative
        or ".." in relative.split("/")
        or ".." in relative.split("\\")
        or relative.startswith("/")
        or relative.startswith("\\")
    ):
        return "Not found", 404
    safe_rel = os.path.normpath(relative).replace("\\", "/").lstrip("/")
    if safe_rel.startswith("..") or "/../" in safe_rel:
        return "Not found", 404
    try:
        candidate = (root_path / safe_rel).resolve()
        candidate.relative_to(root_path)
    except (ValueError, OSError):
        return "Not found", 404
    if not candidate.is_file():
        return "Not found", 404
    rel = candidate.relative_to(root_path).as_posix()
    return send_cached_static_file(
        root_path,
        rel,
        max_age=max_age,
        private=private,
        immutable=immutable,
    )


# ---- detection view-model helpers ------------------------------------------


def get_species_key(det: dict | None) -> str:
    if not det:
        return UNKNOWN_SPECIES_KEY

    species_key = det.get("species_key")
    if species_key:
        return species_key

    manual_species = det.get("manual_species_override")
    if manual_species:
        return manual_species

    return species_key_from_candidates(
        cls_class_name=det.get("cls_class_name"),
        od_class_name=det.get("od_class_name"),
    )


def get_common_name(species_key: str | None) -> str:
    return resolve_common_name(species_key or UNKNOWN_SPECIES_KEY, COMMON_NAMES)


def compute_auto_rating(od_confidence, cls_confidence, bbox_w, bbox_h):
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


def compute_rating_lazy(det):
    rating = compute_auto_rating(
        det.get("od_confidence", 0),
        det.get("cls_confidence", 0),
        det.get("bbox_w", 0),
        det.get("bbox_h", 0),
    )

    try:
        det_id = det.get("detection_id")
        if det_id:
            conn = db_service.get_connection()
            try:
                conn.execute(
                    "UPDATE detections SET rating = ?, rating_source = 'auto' "
                    "WHERE detection_id = ? AND rating IS NULL",
                    (rating, det_id),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass

    return rating


def bbox_touches_edge(det: dict, margin: float = 0.01) -> bool:
    bx = det.get("bbox_x") or 0.0
    by = det.get("bbox_y") or 0.0
    bw = det.get("bbox_w") or 0.0
    bh = det.get("bbox_h") or 0.0
    if bw <= 0 or bh <= 0:
        return True

    return (
        bx <= margin
        or by <= margin
        or (bx + bw) >= (1.0 - margin)
        or (by + bh) >= (1.0 - margin)
    )


def is_favorite(det: dict) -> bool:
    return bool(int(det.get("is_favorite") or 0))


def is_gallery_eligible(det: dict) -> bool:
    return bool(int(det.get("is_gallery_eligible") or 0))


def date_iso_from_timestamp(ts: str) -> str:
    if not ts or len(ts) < 8:
        return ""
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"


def build_detection_view_dict(
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
        "is_favorite": is_favorite(det),
        "is_gallery_eligible": is_gallery_eligible(det),
        "image_filename": (
            det.get("image_filename") or det.get("original_name") or ""
        ),
    }
    if include_decision_state:
        payload["decision_state"] = det.get("decision_state")
    if extra:
        payload.update(extra)
    return payload


# ---- DB-read page-data helpers ---------------------------------------------


def get_detections_for_date(date_str_iso):
    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(
            conn, date_str_iso, order_by="time"
        )

        return [dict(row) for row in rows]


def delete_detections(detection_ids):
    with db_service.closing_connection() as conn:
        db_service.reject_detections(conn, detection_ids)
    return True


def get_all_detections():
    try:
        with db_service.closing_connection() as conn:
            rows = db_service.fetch_detections_for_gallery(conn, order_by="time")
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error reading detections from SQLite: {e}")
        return []


def get_daily_covers():
    try:
        return gallery_service.get_daily_covers(COMMON_NAMES)
    except Exception as e:
        logger.error(f"Error reading daily covers from SQLite: {e}")
    return {}


def get_daily_species_summary(date_iso: str):
    try:
        with db_service.closing_connection() as conn:
            rows = db_service.fetch_detection_species_summary(conn, date_iso)
    except Exception as exc:
        logger.error(
            "Error fetching daily species summary for %s [%s]",
            _slv(date_iso),
            type(exc).__name__,
            exc_info=True,
        )
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


def get_captured_detections():
    try:
        return gallery_service.get_all_detections()
    except Exception as e:
        logger.error(f"Error reading detections from SQLite: {e}")
        return []


def get_captured_detections_by_date():
    detections = get_captured_detections()
    detections_by_date = {}
    for det in detections:
        ts = det["image_timestamp"]

        if len(ts) >= 8:
            date_str = ts[:8]
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if formatted_date not in detections_by_date:
                detections_by_date[formatted_date] = []
            detections_by_date[formatted_date].append(det)
    return detections_by_date


# ---- stream-media page-data helpers ----------------------------------------


def format_stream_timestamp(ts: str) -> str:
    if not ts:
        return "Unknown"
    try:
        return datetime.strptime(ts[:15], "%Y%m%d_%H%M%S").strftime(
            "%d.%m.%Y %H:%M"
        )
    except ValueError:
        return "Unknown"


def pick_cover_for_group(candidates: list[dict], **_kwargs) -> dict | None:
    from core.gallery_core import pick_cover_for_group as _pick_cover_for_group

    return _pick_cover_for_group(candidates)


def build_stream_media_payload(det: dict | None) -> dict:
    if not det:
        return {
            "detection_id": None,
            "display_path": "",
            "gallery_date": "",
            "is_favorite": False,
            "is_gallery_eligible": False,
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
        "gallery_date": date_iso_from_timestamp(ts),
        "is_favorite": bool(int(det.get("is_favorite") or 0)),
        "is_gallery_eligible": bool(int(det.get("is_gallery_eligible") or 0)),
        "score": float(det.get("score") or 0.0),
    }


# ---- best-species story-board cluster --------------------------------------


def enrich_species_board(board: dict[str, list[dict]]) -> dict[str, list[dict]]:
    enriched_board = {"featured": [], "grid": []}
    for section in ("featured", "grid"):
        for item in board.get(section, []):
            species_key = item.get("species_key") or UNKNOWN_SPECIES_KEY
            primary = item.get("primary_detection")
            primary_payload = build_stream_media_payload(primary)
            story_frames = []
            for story_det in item.get("story_detections", []):
                frame_payload = build_stream_media_payload(story_det)
                frame_payload["detection_id"] = story_det.get("detection_id")
                story_frames.append(frame_payload)

            enriched_board[section].append(
                {
                    "species_key": species_key,
                    "common_name": get_common_name(species_key),
                    "latin_name": species_key,
                    "visit_count": int(item.get("visit_count") or 0),
                    "last_seen_timestamp": item.get("last_seen_timestamp") or "",
                    "last_seen_display": format_stream_timestamp(
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
                    "is_gallery_eligible": primary_payload["is_gallery_eligible"],
                    "score": primary_payload["score"],
                    "story_frames": story_frames,
                }
            )
    colour_map = assign_species_colours(
        [
            item.get("species_key") or UNKNOWN_SPECIES_KEY
            for section in ("featured", "grid")
            for item in enriched_board.get(section, [])
        ]
    )
    for section in ("featured", "grid"):
        for item in enriched_board.get(section, []):
            slot = colour_map.get(item.get("species_key") or UNKNOWN_SPECIES_KEY)
            item["species_colour"] = slot
            for frame in item.get("story_frames", []):
                frame["species_colour"] = slot
    return enriched_board


def fetch_best_species_pools(
    *,
    total_limit: int = 12,
    frames_per_species: int = 20,
) -> tuple[list[dict], list[dict]]:
    with db_service.closing_connection() as conn:
        rows = db_service.fetch_species_story_board_candidates(
            conn,
            total_limit=total_limit,
            frames_per_species=frames_per_species,
            excluded_species={UNKNOWN_SPECIES_KEY},
        )

    pools_by_species: dict[str, dict] = {}
    ordered_species: list[str] = []
    modal_rows_by_id: dict[int, dict] = {}

    for row in rows:
        det = dict(row)
        species_key = det.get("species_key") or UNKNOWN_SPECIES_KEY
        if species_key not in pools_by_species:
            pools_by_species[species_key] = {
                "species_key": species_key,
                "visit_count": int(det.get("visit_count") or 0),
                "last_seen_timestamp": det.get("last_seen_timestamp") or "",
                "best_cover_score": float(det.get("best_cover_score") or 0.0),
                "is_favorite_available": bool(
                    int(det.get("is_favorite_available") or 0)
                ),
                "candidates": [],
            }
            ordered_species.append(species_key)

        pools_by_species[species_key]["candidates"].append(det)

        det_id = int(det.get("detection_id") or 0)
        if det_id > 0:
            modal_rows_by_id.setdefault(det_id, det)

    species_pools = [pools_by_species[species] for species in ordered_species]
    return species_pools, list(modal_rows_by_id.values())


def render_best_species_board(
    species_pools: list[dict],
    *,
    total_limit: int = 12,
    featured_count: int = 3,
    frame_count: int = 3,
) -> dict[str, list[dict]]:
    from core.gallery_core import _choose_story_board_frames

    rng = random.Random()
    board_items: list[dict] = []
    for pool in species_pools:
        candidates = pool.get("candidates") or []
        primary, frames = _choose_story_board_frames(
            candidates, rng=rng, frame_count=frame_count
        )
        board_items.append(
            {
                "species_key": pool["species_key"],
                "visit_count": pool["visit_count"],
                "last_seen_timestamp": pool["last_seen_timestamp"],
                "best_cover_score": pool["best_cover_score"],
                "is_favorite_available": pool["is_favorite_available"],
                "primary_detection": primary
                or (candidates[0] if candidates else None),
                "story_detections": frames,
            }
        )

    return {
        "featured": board_items[:featured_count],
        "grid": board_items[featured_count:total_limit],
    }
