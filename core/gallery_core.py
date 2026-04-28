"""
Gallery Core - Gallery Business Logic.

Provides all gallery-related operations separated from the web layer.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2

from config import get_config
from utils.db import (
    closing_connection,
    fetch_daily_covers,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
)
from utils.db import (
    fetch_sibling_detections as db_fetch_sibling_detections,
)
from utils.image_ops import generate_preview_thumbnail as _generate_preview_thumbnail
from utils.log_safety import safe_log_value as _slv
from utils.path_manager import get_path_manager
from utils.wikipedia import (
    build_species_wikipedia_url as _build_species_wikipedia_url,
)

logger = logging.getLogger(__name__)
config = get_config()

# Cache timeout in seconds
_CACHE_TIMEOUT = 60
_cached_images: dict[str, Any] = {"images": None, "timestamp": 0}


def get_detections_for_date(date_str_iso: str) -> list[dict]:
    """
    Fetch all detections for a specific date.

    Args:
        date_str_iso: Date in YYYY-MM-DD format

    Returns:
        List of detection dictionaries
    """
    with closing_connection() as conn:
        rows = fetch_detections_for_gallery(conn, date_str_iso, order_by="time")
        return [dict(row) for row in rows]


def get_all_detections() -> list[dict]:
    """
    Reads all active detections from SQLite.

    Returns:
        List of detection dictionaries
    """
    try:
        with closing_connection() as conn:
            rows = fetch_detections_for_gallery(conn, order_by="time")
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error reading detections from SQLite: {e}")
        return []


def get_captured_detections() -> list[dict]:
    """
    Returns a list of captured detections with caching.

    Uses in-memory caching to avoid repeated DB hits.

    Returns:
        List of detection dictionaries
    """
    now = time.time()
    if (
        _cached_images["images"] is not None
        and (now - _cached_images["timestamp"]) < _CACHE_TIMEOUT
    ):
        return _cached_images["images"]

    detections = []
    try:
        with closing_connection() as conn:
            rows = fetch_detections_for_gallery(conn, order_by="time")
            detections = [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error reading detections from SQLite: {e}")

    _cached_images["images"] = detections
    _cached_images["timestamp"] = now
    return detections


def get_captured_detections_by_date() -> dict[str, list]:
    """
    Returns a dictionary grouping detections by date (YYYY-MM-DD).

    Returns:
        Dictionary mapping date strings to lists of detections
    """
    detections = get_captured_detections()
    detections_by_date: dict[str, list] = {}
    for det in detections:
        ts = det.get("image_timestamp", "")
        # ts format YYYYMMDD_HHMMSS
        if len(ts) >= 8:
            date_str = ts[:8]
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if formatted_date not in detections_by_date:
                detections_by_date[formatted_date] = []
            detections_by_date[formatted_date].append(det)
    return detections_by_date


def get_daily_covers(common_names: dict[str, str] | None = None) -> dict[str, dict]:
    """
    Returns a dict of {YYYY-MM-DD: {path, bbox, count}} for gallery overview.

    Args:
        common_names: Optional dict for species name translation

    Returns:
        Dictionary mapping dates to cover image metadata
    """
    if common_names is None:
        common_names = {}

    covers: dict[str, dict] = {}
    gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

    try:
        with closing_connection() as conn:
            rows = fetch_daily_covers(conn, min_score=gallery_threshold)
            for row in rows:
                date_key = row["date_key"]
                optimized_name = row["optimized_name_virtual"]
                if not date_key or not optimized_name:
                    continue

                thumb_path_virtual = row["thumbnail_path_virtual"]

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


def get_daily_species_summary(
    date_iso: str, common_names: dict[str, str] | None = None
) -> list[dict]:
    """
    Returns per-species counts for a given date (YYYY-MM-DD).

    Always returns fresh data from DB (no caching).

    Args:
        date_iso: Date in YYYY-MM-DD format
        common_names: Optional dict for species name translation

    Returns:
        List of species summary dictionaries with species, common_name, count
    """
    if common_names is None:
        common_names = {}

    try:
        with closing_connection() as conn:
            rows = fetch_detection_species_summary(conn, date_iso)
    except Exception as e:
        logger.error(f"Error fetching daily species summary for {date_iso}: {e}")
        rows = []

    summary = []
    for row in rows:
        species = row["species"]
        count = row["count"]
        if not species:
            continue
        common_name = common_names.get(species, species.replace("_", " "))
        summary.append(
            {"species": species, "common_name": common_name, "count": int(count)}
        )
    return summary


# ── Observation Grouping (Issue #12) ────────────────────────────────

# Clustering constants – must match utils/db/analytics.py
_OBS_MAX_GAP_SEC = 60
_OBS_MAX_BBOX_DIST = 0.25
_OBS_MIN_BBOX_IOU = 0.02
_OBS_MIN_AREA_SIMILARITY = 0.2


def _ts_to_epoch(ts: str) -> float:
    """Convert YYYYMMDD_HHMMSS (or YYYYMMDD_HHMMSS_ffffff) to epoch seconds.

    The images table stores timestamps with optional microsecond suffixes
    (e.g. ``20260225_084116_427121``).  We only need second-level precision,
    so we truncate to the first 15 characters before parsing.

    Returns 0.0 on failure.
    """
    from datetime import datetime as _dt

    try:
        # Truncate to YYYYMMDD_HHMMSS (15 chars), discarding _ffffff
        return _dt.strptime(ts[:15], "%Y%m%d_%H%M%S").timestamp()
    except (ValueError, TypeError):
        return 0.0


def _bbox_dist(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Euclidean distance between bbox centres (normalised coords)."""
    import math

    cx_a = (ax or 0) + (aw or 0) / 2.0
    cy_a = (ay or 0) + (ah or 0) / 2.0
    cx_b = (bx or 0) + (bw or 0) / 2.0
    cy_b = (by or 0) + (bh or 0) / 2.0
    return math.hypot(cx_a - cx_b, cy_a - cy_b)


def _bbox_iou_local(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """IoU for normalised xywh boxes."""
    ax1, ay1 = (ax or 0), (ay or 0)
    ax2, ay2 = ax1 + (aw or 0), ay1 + (ah or 0)
    bx1, by1 = (bx or 0), (by or 0)
    bx2, by2 = bx1 + (bw or 0), by1 + (bh or 0)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(0.0, (aw or 0) * (ah or 0)) + max(0.0, (bw or 0) * (bh or 0)) - inter
    return inter / union if union > 0 else 0.0


def _bbox_area_sim(aw: float, ah: float, bw: float, bh: float) -> float:
    """Area similarity ratio [0..1]."""
    a = max(0.0, (aw or 0) * (ah or 0))
    b = max(0.0, (bw or 0) * (bh or 0))
    if a <= 0 or b <= 0:
        return 1.0
    return min(a, b) / max(a, b)


def group_detections_into_observations(
    detections: list[dict],
    max_gap_sec: float = _OBS_MAX_GAP_SEC,
    max_bbox_dist: float = _OBS_MAX_BBOX_DIST,
) -> list[dict]:
    """Group detection dicts into biological observations.

    This function now delegates to :func:`core.events.build_bird_events`.
    The legacy 60 s + bbox proximity merge has been retired in favour of
    the biological event window (30 min, same species, no bbox split).
    The return shape is preserved so existing gallery, stream, and
    analytics callers do not need to change.

    The ``max_gap_sec`` and ``max_bbox_dist`` parameters are kept for
    backwards compatibility. ``max_gap_sec`` is converted to minutes for
    the event builder; ``max_bbox_dist`` is unused under the new policy
    and silently ignored. New callers should call ``build_bird_events``
    directly with ``gap_minutes``.

    Returns a list of observation dicts sorted by ``start_time`` desc:

    .. code-block:: python

        {
            "observation_id": int,       # 1-based index
            "species": str,
            "detection_ids": list[int],
            "photo_count": int,
            "duration_sec": float,
            "best_score": float,
            "cover_detection_id": int,
            "start_time": str,
            "end_time": str,
        }
    """
    if not detections:
        return []

    # Local import to avoid an import cycle: ``core.events`` imports
    # helpers from this module.
    from core.events import EVENT_GAP_MINUTES_DEFAULT, build_bird_events

    # Re-key any detections that store their timestamp under
    # ``image_timestamp`` (gallery / analytics surfaces) so the event
    # builder can read them through its standard ``timestamp`` field
    # without forcing every caller to remap upstream.
    normalised_detections: list[dict] = []
    for det in detections:
        if not det.get("timestamp") and det.get("image_timestamp"):
            det = {**det, "timestamp": det["image_timestamp"]}
        normalised_detections.append(det)

    # Honour the legacy ``max_gap_sec`` parameter when callers tighten
    # the window manually; otherwise fall back to the biological default.
    if max_gap_sec is None or float(max_gap_sec) <= 0:
        gap_minutes = EVENT_GAP_MINUTES_DEFAULT
    elif max_gap_sec == _OBS_MAX_GAP_SEC:
        gap_minutes = EVENT_GAP_MINUTES_DEFAULT
    else:
        gap_minutes = float(max_gap_sec) / 60.0

    events = build_bird_events(normalised_detections, gap_minutes=gap_minutes)

    # Compute best_score per event from the source rows so the
    # gallery's score-threshold filter still has the metric it needs.
    score_by_id: dict[int, float] = {}
    for det in detections:
        det_id = det.get("detection_id")
        if det_id is None:
            continue
        try:
            score_by_id[int(det_id)] = float(det.get("score") or 0.0)
        except (TypeError, ValueError):
            score_by_id[int(det_id)] = 0.0

    # Match the legacy ordering: start_time descending so the newest
    # observation comes first.
    sorted_events = sorted(
        events,
        key=lambda event: event.start_time,
        reverse=True,
    )

    observations: list[dict] = []
    for idx, event in enumerate(sorted_events, start=1):
        member_scores = [
            score_by_id.get(int(det_id), 0.0) for det_id in event.detection_ids
        ]
        best_score = max(member_scores) if member_scores else 0.0
        observations.append(
            {
                "observation_id": idx,
                "species": event.species or "unknown",
                "detection_ids": list(event.detection_ids),
                "photo_count": event.photo_count,
                "duration_sec": event.duration_sec,
                "best_score": best_score,
                "cover_detection_id": event.cover_detection_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
            }
        )

    # Touch the unused legacy gates so ruff treats them as referenced.
    _ = (_OBS_MAX_BBOX_DIST, _OBS_MIN_BBOX_IOU, _OBS_MIN_AREA_SIMILARITY, max_bbox_dist)
    return observations


# Concurrent-visit grouping for the Subgallery UI.
#
# Two species in the same time window stay separate observations in the
# data layer. The Subgallery groups them only as a visual visit window.

_CONCURRENT_VISIT_WINDOW_MINUTES_DEFAULT = 5.0


def group_concurrent_observations(
    observations: list[dict],
    *,
    window_minutes: float = _CONCURRENT_VISIT_WINDOW_MINUTES_DEFAULT,
) -> list[list[dict]]:
    """Bucket observations into visit windows for the Subgallery UI.

    A visit window is a list of observations whose ``[start_time, end_time]``
    intervals overlap within ``window_minutes`` tolerance. Same-species
    observations stay separate inside a window — this helper only groups
    them visually and never merges them in the data layer.

    The function is pure: no DB calls, no mutation of the input list.

    Determinism: observations are sorted internally by
    ``(start_time, end_time, observation_id)`` before bucketing, so the
    returned list is stable for the same input. Sort order of the returned
    visit windows is preserved from that internal sort (i.e. earliest
    ``start_time`` first). Callers that want newest-first should sort by
    the first observation's ``end_time`` after the fact — the helper's
    docstring picks "first observation" for deterministic re-sort keys,
    which keeps the helper deterministic for callers that want to
    re-sort afterwards.

    Args:
        observations: list of observation dicts as returned by
            :func:`group_detections_into_observations`. Each dict must
            carry ``start_time`` and ``end_time`` in the
            ``YYYYMMDD_HHMMSS`` string form used by the rest of the
            pipeline.
        window_minutes: tolerance in minutes for two intervals to be
            considered concurrent. Defaults to 5 min — tight enough that
            the common "one species at a time" case never gets grouped.

    Returns:
        List of visit windows. Each visit window is a list of
        observation dicts (references to the originals, not copies).
        Single-observation windows are returned as 1-element lists.
    """
    if not observations:
        return []

    tolerance_sec = max(0.0, float(window_minutes) * 60.0)

    # Sort deterministically. Ties on start_time fall back to end_time
    # then observation_id so the output stays stable across runs.
    def _sort_key(obs: dict) -> tuple[float, float, int]:
        return (
            _ts_to_epoch(obs.get("start_time", "") or ""),
            _ts_to_epoch(obs.get("end_time", "") or ""),
            int(obs.get("observation_id") or 0),
        )

    ordered = sorted(observations, key=_sort_key)

    visit_windows: list[list[dict]] = []
    current: list[dict] = []
    current_end_epoch = 0.0

    for obs in ordered:
        start_epoch = _ts_to_epoch(obs.get("start_time", "") or "")
        end_epoch = _ts_to_epoch(obs.get("end_time", "") or "")
        if not current:
            current = [obs]
            current_end_epoch = end_epoch
            continue

        # Two intervals are "concurrent" when the next start falls
        # within tolerance of the running max end. This also naturally
        # handles true overlaps (next_start <= current_end).
        if start_epoch - current_end_epoch <= tolerance_sec:
            current.append(obs)
            if end_epoch > current_end_epoch:
                current_end_epoch = end_epoch
        else:
            visit_windows.append(current)
            current = [obs]
            current_end_epoch = end_epoch

    if current:
        visit_windows.append(current)

    return visit_windows


def summarize_observations(
    detections: list[dict],
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Summarize detections through the gallery observation model.

    This keeps stream-side counters aligned with the observation grouping used
    by the day gallery. The optional ``min_score`` matches subgallery behavior:
    filter on ``observation.best_score`` after grouping, not per detection row.
    """
    if not detections:
        return {
            "observations": [],
            "detections": [],
            "summary": {
                "total_observations": 0,
                "total_detections": 0,
                "species_counts": {},
                "avg_score": 0.0,
            },
        }

    observations = group_detections_into_observations(detections)
    if min_score > 0:
        observations = [
            obs
            for obs in observations
            if float(obs.get("best_score") or 0.0) >= min_score
        ]

    included_ids: set[int] = set()
    species_counts: dict[str, int] = {}
    for obs in observations:
        species = obs.get("species") or ""
        if species:
            species_counts[species] = species_counts.get(species, 0) + 1
        for det_id in obs.get("detection_ids") or []:
            if det_id is not None:
                included_ids.add(int(det_id))

    included_detections: list[dict] = []
    total_score = 0.0
    scored_count = 0
    for det in detections:
        det_id = det.get("detection_id")
        if det_id is None or int(det_id) not in included_ids:
            continue
        included_detections.append(det)
        try:
            total_score += float(det.get("score") or 0.0)
            scored_count += 1
        except (TypeError, ValueError):
            continue

    avg_score = round(total_score / scored_count, 2) if scored_count else 0.0

    return {
        "observations": observations,
        "detections": included_detections,
        "summary": {
            "total_observations": len(observations),
            "total_detections": len(included_detections),
            "species_counts": species_counts,
            "avg_score": avg_score,
        },
    }


def _story_board_bbox_touches_edge(det: dict, margin: float = 0.01) -> bool:
    """Return True if the bbox touches the image edge."""
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


def _story_board_candidate_quality(det: dict) -> tuple[int, int, float, float, str, int]:
    """Quality key for story-board cover candidates."""
    is_favorite = 1 if int(det.get("is_favorite") or 0) else 0
    is_interior = 0 if _story_board_bbox_touches_edge(det) else 1
    score = float(det.get("score") or 0.0)
    bbox_quality = float(det.get("bbox_quality") or 0.0)
    ts = det.get("image_timestamp", "") or ""
    det_id = int(det.get("detection_id") or 0)
    return (is_favorite, is_interior, score, bbox_quality, ts, det_id)


def _rank_story_board_candidates(candidates: list[dict]) -> list[dict]:
    """Sort cover candidates by favorite/interior/score/recency quality."""
    return sorted(
        candidates,
        key=_story_board_candidate_quality,
        reverse=True,
    )


def _build_story_board_candidate_pool(
    detections: list[dict],
    cover_detection_ids: set[int],
    limit: int = 12,
) -> list[dict]:
    """Build a species-level candidate pool with favorites ahead of covers.

    The story board still uses observation-derived ranking for visit counts, but
    image rotation should not be restricted to one cover per observation. We
    therefore promote:
    - favorited detections first
    - observation cover detections second
    - then other high-quality detections as fallback
    """
    if not detections:
        return []

    ranked = _rank_story_board_candidates(detections)
    pool: list[dict] = []
    seen_ids: set[int] = set()

    def _append_unique(items: list[dict]) -> None:
        for det in items:
            det_id = int(det.get("detection_id") or 0)
            if det_id <= 0 or det_id in seen_ids:
                continue
            pool.append(det)
            seen_ids.add(det_id)
            if len(pool) >= limit:
                break

    favorites = [d for d in ranked if int(d.get("is_favorite") or 0)]
    covers = [d for d in ranked if int(d.get("detection_id") or 0) in cover_detection_ids]

    _append_unique(favorites)
    if len(pool) < limit:
        _append_unique(covers)
    if len(pool) < limit:
        _append_unique(ranked)

    return pool


def _choose_story_board_frames(
    candidates: list[dict],
    rng: random.Random | None = None,
    frame_count: int = 3,
) -> tuple[dict | None, list[dict]]:
    """Pick one primary cover and up to ``frame_count`` rotating frames."""
    if not candidates:
        return None, []

    if rng is None:
        rng = random.Random()

    ranked = _rank_story_board_candidates(candidates)
    favorites = [d for d in ranked if int(d.get("is_favorite") or 0)]
    fallback_pool = ranked[: min(3, len(ranked))]
    if len(favorites) >= 2:
        primary_pool = favorites
    elif len(favorites) == 1 and len(ranked) > 1:
        primary_pool = [favorites[0], favorites[0]]
        primary_pool.extend(
            det
            for det in fallback_pool
            if det.get("detection_id") != favorites[0].get("detection_id")
        )
    else:
        primary_pool = fallback_pool
    primary = rng.choice(primary_pool)

    frames = [primary]
    remaining = [
        det for det in ranked if det.get("detection_id") != primary.get("detection_id")
    ]
    if remaining and frame_count > 1:
        extra_pool = remaining[: min(len(remaining), 6)]
        shuffled = extra_pool[:]
        rng.shuffle(shuffled)
        frames.extend(shuffled[: frame_count - 1])

    return primary, frames


def build_species_story_board(
    detections: list[dict],
    since_timestamp: str = "",
    total_limit: int = 12,
    featured_count: int = 3,
    excluded_species: set[str] | None = None,
    rng: random.Random | None = None,
) -> dict[str, list[dict]]:
    """Build a stable species board with rotating imagery.

    The board ranks species deterministically by visit count, last seen, and
    best cover score, while allowing per-render image rotation within the
    chosen species set.
    """
    if not detections or total_limit <= 0:
        return {"featured": [], "grid": []}

    if excluded_species is None:
        excluded_species = set()

    if rng is None:
        rng = random.Random()

    # Use the central species fallback helper so that "bird" (the OD
    # category, not a species) cannot leak into the storyboard grouping
    # when CLS is missing. Non-bird OD class names like "squirrel" still
    # pass through as valid species keys.
    from utils.species_names import UNKNOWN_SPECIES_KEY, species_key_from_candidates

    filtered: list[dict] = []
    species_detections: dict[str, list[dict]] = {}
    for det in detections:
        ts = det.get("image_timestamp", "") or ""
        if not ts:
            continue
        if since_timestamp and ts < since_timestamp:
            continue

        species = species_key_from_candidates(
            manual_override=det.get("manual_species_override"),
            species_key=det.get("species_key"),
            cls_class_name=det.get("cls_class_name"),
            od_class_name=det.get("od_class_name"),
        )
        if species == UNKNOWN_SPECIES_KEY or species in excluded_species:
            continue
        filtered.append(det)
        species_detections.setdefault(species, []).append(det)

    if not filtered:
        return {"featured": [], "grid": []}

    observations = group_detections_into_observations(filtered)
    if not observations:
        return {"featured": [], "grid": []}

    det_by_id = {det.get("detection_id"): det for det in filtered}
    species_rows: dict[str, dict[str, Any]] = {}

    for obs in observations:
        species_key = obs.get("species") or ""
        if not species_key or species_key in excluded_species:
            continue

        cover_det = det_by_id.get(obs.get("cover_detection_id"))
        if not cover_det:
            continue

        row = species_rows.setdefault(
            species_key,
            {
                "species_key": species_key,
                "visit_count": 0,
                "last_seen_timestamp": "",
                "best_cover_score": 0.0,
                "is_favorite_available": False,
                "_cover_detection_ids": set(),
            },
        )
        row["visit_count"] += 1
        row["last_seen_timestamp"] = max(
            row["last_seen_timestamp"],
            obs.get("end_time", "") or "",
        )
        row["best_cover_score"] = max(
            float(row["best_cover_score"] or 0.0),
            float(obs.get("best_score") or 0.0),
        )
        row["_cover_detection_ids"].add(int(cover_det.get("detection_id") or 0))

    ranked_species = sorted(
        species_rows.values(),
        key=lambda row: (
            -int(row["visit_count"]),
            -(int(row["last_seen_timestamp"][:8]) if row["last_seen_timestamp"] else 0),
            row["last_seen_timestamp"],
            -float(row["best_cover_score"] or 0.0),
            row["species_key"],
        ),
    )

    board_items: list[dict[str, Any]] = []
    for row in ranked_species:
        species_key = row["species_key"]
        candidates = _build_story_board_candidate_pool(
            species_detections.get(species_key, []),
            row.get("_cover_detection_ids") or set(),
        )
        primary, frames = _choose_story_board_frames(candidates, rng=rng, frame_count=3)
        if not primary:
            continue

        board_items.append(
            {
                "species_key": species_key,
                "visit_count": int(row["visit_count"]),
                "last_seen_timestamp": row["last_seen_timestamp"],
                "best_cover_score": float(row["best_cover_score"] or 0.0),
                "is_favorite_available": any(
                    int(det.get("is_favorite") or 0) for det in candidates
                ),
                "primary_detection": primary,
                "story_detections": frames,
            }
        )
        if len(board_items) >= total_limit:
            break

    return {
        "featured": board_items[:featured_count],
        "grid": board_items[featured_count:total_limit],
    }


def invalidate_cache() -> None:
    """Invalidates the detection cache, forcing a refresh on next access."""
    global _cached_images
    _cached_images = {"images": None, "timestamp": 0}


# --- Thumbnail Generation ---


def generate_preview_thumbnail(
    original_path: str | Path, preview_path: str | Path, size: int = 256
) -> bool:
    """
    Generate a preview thumbnail for an image.

    Args:
        original_path: Path to the original image
        preview_path: Path where the preview should be saved
        size: Thumbnail size in pixels

    Returns:
        True on success, False on failure
    """
    return _generate_preview_thumbnail(str(original_path), str(preview_path), size)


def get_image_paths(output_dir: str, filename: str) -> dict[str, Path]:
    """
    Get resolved paths for an image file.

    Args:
        output_dir: Base output directory
        filename: Image filename

    Returns:
        Dictionary with 'original' and 'preview' paths
    """
    pm = get_path_manager(output_dir)
    return {
        "original": pm.get_original_path(filename),
        "preview": pm.get_preview_thumb_path(filename),
    }


# --- Sibling Detections ---


def get_sibling_detections(original_name: str) -> list[dict]:
    """
    Get sibling detections for an image (multiple birds on same image).

    Args:
        original_name: The original image filename

    Returns:
        List of sibling detection dictionaries
    """
    with closing_connection() as conn:
        rows = db_fetch_sibling_detections(conn, original_name)
        return [dict(row) for row in rows]


# --- External Links ---


def get_species_wikipedia_url(
    common_name: str | None,
    scientific_name: str | None = None,
    locale: str = "de",
) -> str | None:
    """
    Build a robust Wikipedia species search URL.

    Args:
        common_name: Species common name
        scientific_name: Species scientific name
        locale: Wikipedia locale subdomain (default: "de")

    Returns:
        URL string or None
    """
    return _build_species_wikipedia_url(common_name, scientific_name, locale)


# --- Derivative Regeneration ---


def regenerate_derivative(
    output_dir: str, filename_rel: str, type: str = "thumb"
) -> bool:
    """
    Attempts to regenerate a missing derivative.

    Args:
        output_dir: Base output directory
        filename_rel: YYYYMMDD/basename.webp (path from route)
        type: 'thumb' | 'optimized'

    Returns:
        True if successful, False otherwise
    """
    try:
        path_mgr = get_path_manager(output_dir)

        # 1. Parse Path
        filename = os.path.basename(filename_rel)

        # 2. Check source (Original)
        original_filename = None
        crop_index = None

        if type == "thumb":
            # Equivalent to r"(.*)_crop_(\d+)\.webp$" but without the
            # polynomial backtracking on adversarial input.
            if filename.endswith(".webp") and "_crop_" in filename:
                base_with_idx = filename[: -len(".webp")]
                base_no_ext, _, idx_str = base_with_idx.rpartition("_crop_")
                if base_no_ext and idx_str.isdigit():
                    crop_index = int(idx_str)
                    original_filename = f"{base_no_ext}.jpg"
        elif type == "optimized":
            base_no_ext = os.path.splitext(filename)[0]
            original_filename = f"{base_no_ext}.jpg"

        if not original_filename:
            return False

        original_path = path_mgr.get_original_path(original_filename)

        if not original_path.exists():
            logger.error(
                f"Cannot regenerate {_slv(filename)}: Original missing at {_slv(original_path)}"
            )
            return False

        # 3. Load Original
        img = cv2.imread(str(original_path))
        if img is None:
            return False

        # 4. Process
        target_path = None
        out_img = None

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
            with closing_connection() as conn:
                cursor = conn.execute(
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
                        f"Cannot regenerate thumb: No detection found for {_slv(original_filename)} index {crop_index}"
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

                # Expand & Square
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
        if target_path and out_img is not None:
            path_mgr.ensure_date_structure(
                path_mgr.extract_date_from_filename(filename)
            )
            cv2.imwrite(str(target_path), out_img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
            logger.info(f"Regenerated missing derivative: {target_path}")
            return True

    except Exception as e:
        logger.error(f"Regeneration failed for {_slv(filename_rel)}: {e}")
        return False

    return False
