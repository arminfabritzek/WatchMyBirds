"""
Gallery Core - Gallery Business Logic.

Provides all gallery-related operations separated from the web layer.
"""

import logging
import time
from pathlib import Path
from typing import Any

from config import get_config
from utils.db import (
    fetch_daily_covers,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    get_connection,
)
from utils.db import (
    fetch_sibling_detections as db_fetch_sibling_detections,
)
from utils.image_ops import generate_preview_thumbnail as _generate_preview_thumbnail
from utils.path_manager import get_path_manager

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
    with get_connection() as conn:
        rows = fetch_detections_for_gallery(conn, date_str_iso, order_by="time")
        return [dict(row) for row in rows]


def get_all_detections() -> list[dict]:
    """
    Reads all active detections from SQLite.

    Returns:
        List of detection dictionaries
    """
    try:
        with get_connection() as conn:
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
        with get_connection() as conn:
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
    gallery_threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)

    try:
        with get_connection() as conn:
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
        with get_connection() as conn:
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
    with get_connection() as conn:
        rows = db_fetch_sibling_detections(conn, original_name)
        return [dict(row) for row in rows]
