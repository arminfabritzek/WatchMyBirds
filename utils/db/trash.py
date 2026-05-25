"""
Trash Management Operations.

This module handles trash-related database operations including
fetching, counting, and managing rejected items.
"""

import sqlite3
from typing import Any

from utils.db.detections import (
    _top1_confidence_sql,
    _top1_species_sql,
    effective_species_sql,
)


def fetch_trash_items(
    conn: sqlite3.Connection,
    page: int = 1,
    limit: int = 50,
    species: str = None,
    before_date: str = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetches trashed items with pagination and filters.

    Trash contains only rejected detections (``trash_type='detection'``).

    ``review_status='no_bird'`` images are **not** trash — they are
    user-verified false-positive crops kept for the training-export
    pipeline (``utils.db.user_groundtruth.fetch_hard_negatives``).
    Surfacing them here would invite an "Empty Trash" sweep that
    silently wipes the hard-negative corpus. The legacy UNION ALL
    over ``images.review_status='no_bird'`` was removed for that
    reason. Restore/purge endpoints accepting ``image_filenames``
    still exist (``web/blueprints/trash.py``) for an Export-Dashboard
    surface, but no_bird images do not appear in the trash grid.

    Returns (items, total_count).
    """
    offset = (page - 1) * limit
    items = []

    # === Rejected Detections (only trash content) ===
    det_where = ["d.status = 'rejected'"]
    det_params = []

    if species:
        det_where.append("""
            (d.od_class_name = ? OR EXISTS (
                SELECT 1 FROM classifications c
                WHERE c.detection_id = d.detection_id AND c.cls_class_name = ?
            ))
        """)
        det_params.extend([species, species])

    if before_date:
        date_prefix = before_date.replace("-", "")
        det_where.append("d.image_filename < ?")
        det_params.append(date_prefix)

    det_where_sql = " AND ".join(det_where)

    # Count detections
    det_count_row = conn.execute(
        f"SELECT COUNT(*) FROM detections d WHERE {det_where_sql}", det_params
    ).fetchone()
    det_count = det_count_row[0] if det_count_row else 0

    total_count = det_count

    # === Fetch Items (sorted by timestamp DESC, paginated) ===
    query = f"""
        SELECT
            'detection' as trash_type,
            CAST(d.detection_id AS TEXT) as item_id,
            i.timestamp as image_timestamp,
            i.filename as filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.manual_species_override,
            d.species_source,
            d.created_at,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) as relative_path,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) as thumbnail_path_virtual,
            {_top1_species_sql("d")} as cls_class_name,
            {_top1_confidence_sql("d")} as cls_confidence,
            {effective_species_sql("d")} as species_key
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {det_where_sql}

        ORDER BY image_timestamp DESC
        LIMIT ? OFFSET ?
    """

    all_params = det_params + [limit, offset]
    rows = conn.execute(query, all_params).fetchall()

    for row in rows:
        items.append(
            {
                "trash_type": row["trash_type"],
                "item_id": row["item_id"],  # detection_id (as str) or filename
                "detection_id": (
                    int(row["item_id"]) if row["trash_type"] == "detection" else None
                ),
                "filename": row["filename"],
                "image_timestamp": row["image_timestamp"],
                "image_optimized": row["optimized_name_virtual"],
                "relative_path": row["relative_path"],
                "thumbnail_path_virtual": row["thumbnail_path_virtual"],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_w": row["bbox_w"],
                "bbox_h": row["bbox_h"],
                "od_class_name": row["od_class_name"],
                "od_confidence": row["od_confidence"],
                "manual_species_override": row["manual_species_override"],
                "species_source": row["species_source"],
                "cls_class_name": row["cls_class_name"],
                "cls_confidence": row["cls_confidence"],
                "species_key": row["species_key"],
                "created_at": row["created_at"],
            }
        )

    return items, total_count


def fetch_trash_count(conn: sqlite3.Connection) -> int:
    """
    Returns number of trashed items (for badge).

    Counts only rejected detections. ``review_status='no_bird'`` images
    are deliberately excluded — see ``fetch_trash_items`` docstring.
    """
    det_row = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'rejected'"
    ).fetchone()
    return det_row[0] if det_row else 0
