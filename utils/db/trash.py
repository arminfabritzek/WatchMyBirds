"""
Trash Management Operations.

This module handles trash-related database operations including
fetching, counting, and managing rejected items.
"""

import sqlite3
from typing import Any


def fetch_trash_items(
    conn: sqlite3.Connection,
    page: int = 1,
    limit: int = 50,
    species: str = None,
    before_date: str = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetches trashed items with pagination and filters.
    Returns heterogeneous items:
    - Rejected detections (trash_type='detection')
    - No-bird images (trash_type='image')

    Returns (items, total_count).
    """
    offset = (page - 1) * limit
    items = []

    # === Part 1: Rejected Detections ===
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

    # === Part 2: No-Bird Images ===
    img_where = ["i.review_status = 'no_bird'"]
    img_params = []

    if before_date:
        date_prefix = before_date.replace("-", "")
        img_where.append("i.timestamp < ?")
        img_params.append(date_prefix)

    # Species filter doesn't apply to no-bird images (they have no species)
    img_where_sql = " AND ".join(img_where)

    img_count_row = conn.execute(
        f"SELECT COUNT(*) FROM images i WHERE {img_where_sql}", img_params
    ).fetchone()
    img_count = img_count_row[0] if img_count_row else 0

    total_count = det_count + img_count

    # === Fetch Items with UNION (sorted by timestamp DESC, paginated) ===
    # We use a UNION ALL to combine both types

    union_query = f"""
        SELECT
            'detection' as trash_type,
            CAST(d.detection_id AS TEXT) as item_id,
            i.timestamp as image_timestamp,
            i.filename as filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.created_at,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) as relative_path,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) as thumbnail_path_virtual,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {det_where_sql}

        UNION ALL

        SELECT
            'image' as trash_type,
            i.filename as item_id,
            i.timestamp as image_timestamp,
            i.filename as filename,
            NULL as bbox_x, NULL as bbox_y, NULL as bbox_w, NULL as bbox_h,
            NULL as od_class_name,
            NULL as od_confidence,
            i.review_updated_at as created_at,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) as relative_path,
            NULL as thumbnail_path_virtual,
            NULL as cls_class_name,
            NULL as cls_confidence
        FROM images i
        WHERE {img_where_sql}

        ORDER BY image_timestamp DESC
        LIMIT ? OFFSET ?
    """

    all_params = det_params + img_params + [limit, offset]
    rows = conn.execute(union_query, all_params).fetchall()

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
                "cls_class_name": row["cls_class_name"],
                "cls_confidence": row["cls_confidence"],
                "created_at": row["created_at"],
            }
        )

    return items, total_count


def fetch_trash_count(conn: sqlite3.Connection) -> int:
    """
    Returns total number of trashed items (for badge).
    Includes: rejected detections + images with review_status='no_bird'.
    """
    # Count rejected detections
    det_row = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'rejected'"
    ).fetchone()
    det_count = det_row[0] if det_row else 0

    # Count no_bird images
    img_row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE review_status = 'no_bird'"
    ).fetchone()
    img_count = img_row[0] if img_row else 0

    return det_count + img_count
