"""
Detections Core - Detection Data Operations.

Provides all detection-related database operations and business logic.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from utils.db import (
    closing_connection,
    fetch_review_queue_count,
    fetch_trash_count,
    fetch_trash_items,
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
from utils.file_gc import (
    hard_delete_detections as gc_hard_delete_detections,
)
from utils.path_manager import get_path_manager

logger = logging.getLogger(__name__)


def move_to_trash(conn: sqlite3.Connection, data: dict) -> dict[str, int]:
    ids = data.get("detection_ids", data.get("ids", []))
    filenames = data.get("image_filenames", [])
    if (
        not isinstance(ids, list)
        or not isinstance(filenames, list)
        or len(ids) + len(filenames) > 1000
        or any(type(did) is not int or did <= 0 for did in ids)
        or any(not isinstance(name, str) or not name for name in filenames)
        or not (ids or filenames)
    ):
        raise ValueError("Choose between 1 and 1000 objects or photos")
    conn.execute("BEGIN IMMEDIATE")
    selected = set(ids)
    for filename in filenames:
        selected.update(
            row[0]
            for row in conn.execute(
                "SELECT detection_id FROM detections WHERE image_filename=?",
                (filename,),
            )
        )
    active = []
    images = set()
    for did in sorted(selected):
        row = conn.execute(
            "SELECT status, image_filename FROM detections WHERE detection_id=?", (did,)
        ).fetchone()
        if row is None:
            raise ValueError("The selection has changed; reload before moving objects")
        if row[0] == "active":
            active.append(did)
            images.add(row[1])
    db_reject_detections(conn, active, commit=False)
    return {"rejected_detections": len(active), "rejected_images": len(images)}


def reject_detections(detection_ids: list[int]) -> bool:
    """
    Semantic delete - marks detections as rejected (moves to trash).

    Args:
        detection_ids: List of detection IDs to reject

    Returns:
        True on success
    """
    with closing_connection() as conn:
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
    with closing_connection() as conn:
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


def get_trash_items() -> list[dict]:
    """
    Retrieves all items in trash.

    Returns:
        List of trash item dictionaries
    """
    with closing_connection() as conn:
        rows = fetch_trash_items(conn)
        return [dict(row) for row in rows]


def get_trash_count() -> int:
    """
    Returns the number of items in trash.

    Returns:
        Count of trash items
    """
    with closing_connection() as conn:
        return fetch_trash_count(conn)


def get_review_queue_count(gallery_threshold: float) -> int:
    """
    Returns the number of items in the review queue.

    Args:
        gallery_threshold: Minimum score threshold used for gallery display.
            The review queue includes images below this threshold (plus orphans).

    Returns:
        Count of items needing review
    """
    with closing_connection() as conn:
        return fetch_review_queue_count(conn, gallery_threshold)


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
    from utils.db.connection import human_answers_migrated

    if not human_answers_migrated(conn):
        raise ValueError("Human answer migration must finish before deleting trash")
    return gc_hard_delete_detections(
        conn, detection_ids=detection_ids, before_date=before_date
    )


# --- Review Queue ---


def get_review_queue_images(output_dir: str, gallery_threshold: float) -> list[dict]:
    """
    Get images for the review queue with path resolution.

    Args:
        output_dir: Base output directory
        gallery_threshold: Minimum score threshold used for gallery display.
            The review queue includes images below this threshold (plus orphans).

    Returns:
        List of image dictionaries with resolved paths
    """
    get_path_manager(output_dir)

    with closing_connection() as conn:
        rows = db_fetch_review_queue_images(conn, gallery_threshold)

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
