"""
Detections Core - Detection Data Operations.

Provides all detection-related database operations and business logic.
"""

import logging
from pathlib import Path
from typing import Any

from utils.db import (
    fetch_review_queue_count,
    fetch_trash_count,
    fetch_trash_items,
    get_connection,
)
from utils.db import (
    fetch_review_queue_images as db_fetch_review_queue_images,
)
from utils.db import (
    reject_detections as db_reject_detections,
)
from utils.db import (
    restore_detections as db_restore_detections,
)
from utils.db import (
    update_review_status as db_update_review_status,
)
from utils.file_gc import (
    hard_delete_detections as gc_hard_delete_detections,
)
from utils.file_gc import (
    hard_delete_images as gc_hard_delete_images,
)
from utils.path_manager import get_path_manager

logger = logging.getLogger(__name__)


def reject_detections(detection_ids: list[int]) -> bool:
    """
    Semantic delete - marks detections as rejected (moves to trash).

    Args:
        detection_ids: List of detection IDs to reject

    Returns:
        True on success
    """
    with get_connection() as conn:
        db_reject_detections(conn, detection_ids)
    return True


def restore_detections(detection_ids: list[int]) -> bool:
    """
    Restores rejected detections from trash.

    Args:
        detection_ids: List of detection IDs to restore

    Returns:
        True on success
    """
    with get_connection() as conn:
        db_restore_detections(conn, detection_ids)
    return True


def hard_delete_detections(detection_ids: list[int]) -> dict[str, Any]:
    """
    Permanently deletes detections from database and filesystem.

    Args:
        detection_ids: List of detection IDs to delete

    Returns:
        Dictionary with deletion results
    """
    return gc_hard_delete_detections(detection_ids)


def update_review_status(
    detection_id: int, save_threshold: float | None = None, reviewed: bool = True
) -> bool:
    """
    Updates the review status of a detection.

    Args:
        detection_id: ID of the detection
        save_threshold: Threshold value if applicable
        reviewed: Whether the detection has been reviewed

    Returns:
        True on success
    """
    with get_connection() as conn:
        db_update_review_status(conn, detection_id, save_threshold, reviewed)
    return True


def get_trash_items() -> list[dict]:
    """
    Retrieves all items in trash.

    Returns:
        List of trash item dictionaries
    """
    with get_connection() as conn:
        rows = fetch_trash_items(conn)
        return [dict(row) for row in rows]


def get_trash_count() -> int:
    """
    Returns the number of items in trash.

    Returns:
        Count of trash items
    """
    with get_connection() as conn:
        return fetch_trash_count(conn)


def get_review_queue_count(save_threshold: float) -> int:
    """
    Returns the number of items in the review queue.

    Args:
        save_threshold: Minimum score threshold for review queue

    Returns:
        Count of items needing review
    """
    with get_connection() as conn:
        return fetch_review_queue_count(conn, save_threshold)


# --- Hard Delete Operations (with connection) ---


def hard_delete_detections_with_conn(
    conn,
    detection_ids: list[int] | None = None,
    before_date: str | None = None,
) -> dict[str, Any]:
    """
    Permanently deletes detections using an existing connection.

    Args:
        conn: Database connection
        detection_ids: Optional list of detection IDs to delete
        before_date: Optional date string for deleting all before this date

    Returns:
        Dictionary with deletion results
    """
    return gc_hard_delete_detections(
        conn, detection_ids=detection_ids, before_date=before_date
    )


def hard_delete_images(
    filenames: list[str] | None = None,
    delete_all: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes images from database and filesystem.

    Args:
        filenames: Optional list of image filenames to delete
        delete_all: If True, delete all images

    Returns:
        Dictionary with deletion results
    """
    with get_connection() as conn:
        return gc_hard_delete_images(conn, filenames=filenames, delete_all=delete_all)


def hard_delete_images_with_conn(
    conn,
    filenames: list[str] | None = None,
    delete_all: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes images using an existing connection.

    Args:
        conn: Database connection
        filenames: Optional list of image filenames to delete
        delete_all: If True, delete all images

    Returns:
        Dictionary with deletion results
    """
    return gc_hard_delete_images(conn, filenames=filenames, delete_all=delete_all)


# --- Review Queue ---


def get_review_queue_images(output_dir: str, save_threshold: float) -> list[dict]:
    """
    Get images for the review queue with path resolution.

    Args:
        output_dir: Base output directory
        save_threshold: Minimum confidence threshold

    Returns:
        List of image dictionaries with resolved paths
    """
    get_path_manager(output_dir)

    with get_connection() as conn:
        rows = db_fetch_review_queue_images(conn, save_threshold)

    images = []
    for row in rows:
        images.append(dict(row))

    return images


def get_preview_path(output_dir: str, filename: str) -> Path:
    """
    Get the preview thumbnail path for a filename.

    Args:
        output_dir: Base output directory
        filename: Image filename

    Returns:
        Path to the preview thumbnail
    """
    pm = get_path_manager(output_dir)
    return pm.get_preview_thumb_path(filename)
