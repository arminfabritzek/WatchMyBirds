"""
Gallery Core - Gallery Business Logic.

Provides all gallery-related operations separated from the web layer.
"""

import logging
import os
import re
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
        logger.error(f"Regeneration failed for {filename_rel}: {e}")
        return False

    return False
