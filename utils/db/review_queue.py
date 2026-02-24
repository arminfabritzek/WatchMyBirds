"""
Review Queue Database Operations.

This module handles review queue-related database operations including
orphan images, review status updates, and queue management.
"""

import sqlite3
from collections.abc import Iterable
from datetime import datetime


def fetch_orphan_images(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns images that have no detections at all (true orphan images).
    Images with rejected detections (in trash) are NOT orphans.
    These are candidates for cleanup.
    """
    query = """
    SELECT
        i.filename,
        i.timestamp
    FROM images i
    WHERE NOT EXISTS (
        SELECT 1 FROM detections d
        WHERE d.image_filename = i.filename
    )
    AND i.filename IS NOT NULL
    ORDER BY i.timestamp DESC;
    """
    cur = conn.execute(query)
    return cur.fetchall()


def delete_orphan_images(conn: sqlite3.Connection, filenames: Iterable[str]) -> int:
    """
    Deletes image rows from the database by filename.
    Returns the number of rows deleted.
    File deletion must be handled separately by file_gc.
    """
    names = list(filenames)
    if not names:
        return 0
    placeholders = ",".join("?" for _ in names)
    cur = conn.execute(f"DELETE FROM images WHERE filename IN ({placeholders})", names)
    conn.commit()
    return cur.rowcount


def fetch_orphan_count(conn: sqlite3.Connection) -> int:
    """Returns count of images with zero detections."""
    query = """
    SELECT COUNT(*)
    FROM images i
    WHERE NOT EXISTS (
        SELECT 1 FROM detections d
        WHERE d.image_filename = i.filename
    )
    """
    row = conn.execute(query).fetchone()
    return row[0] if row else 0


def fetch_review_queue_images(
    conn: sqlite3.Connection,
    gallery_threshold: float = 0.7,
    exclude_deep_scanned: bool = False,
) -> list[sqlite3.Row]:
    """
    Returns images that need review:
    - review_status = 'untagged' AND
    - (no detections OR max(score) < gallery_threshold)

    Sorted by timestamp ASC (oldest first).
    """
    where_clauses = [
        "(i.review_status IS NULL OR i.review_status = 'untagged')",
        "(NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename) OR (SELECT MAX(COALESCE(d.score, 0.0)) FROM detections d WHERE d.image_filename = i.filename) < ?)",
        "i.filename IS NOT NULL",
    ]
    params = [gallery_threshold]

    if exclude_deep_scanned:
        where_clauses.append(
            "NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename AND d.od_model_id LIKE 'deep_scan_%')"
        )

    where_sql = " AND ".join(where_clauses)

    query = f"""
    SELECT
        i.filename,
        i.timestamp,
        i.review_status,
        (SELECT MAX(COALESCE(d.score, 0.0)) FROM detections d WHERE d.image_filename = i.filename) as max_score,
        (
            SELECT d.bbox_x
            FROM detections d
            WHERE d.image_filename = i.filename
            ORDER BY COALESCE(d.score, 0.0) DESC, d.detection_id DESC
            LIMIT 1
        ) as bbox_x,
        (
            SELECT d.bbox_y
            FROM detections d
            WHERE d.image_filename = i.filename
            ORDER BY COALESCE(d.score, 0.0) DESC, d.detection_id DESC
            LIMIT 1
        ) as bbox_y,
        (
            SELECT d.bbox_w
            FROM detections d
            WHERE d.image_filename = i.filename
            ORDER BY COALESCE(d.score, 0.0) DESC, d.detection_id DESC
            LIMIT 1
        ) as bbox_w,
        (
            SELECT d.bbox_h
            FROM detections d
            WHERE d.image_filename = i.filename
            ORDER BY COALESCE(d.score, 0.0) DESC, d.detection_id DESC
            LIMIT 1
        ) as bbox_h,
        CASE
            WHEN NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
            THEN 'orphan'
            ELSE 'low_score'
        END as review_reason
    FROM images i
    WHERE {where_sql}
    ORDER BY i.timestamp ASC;
    """
    cur = conn.execute(query, params)
    return cur.fetchall()


def fetch_review_queue_count(
    conn: sqlite3.Connection, gallery_threshold: float = 0.7
) -> int:
    """
    Returns count of images needing review (for badge).
    Same criteria as fetch_review_queue_images.
    """
    query = """
    SELECT COUNT(*)
    FROM images i
    WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
    AND (
        NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
        OR (SELECT MAX(COALESCE(d.score, 0.0)) FROM detections d WHERE d.image_filename = i.filename) < ?
    )
    AND i.filename IS NOT NULL;
    """
    row = conn.execute(query, (gallery_threshold,)).fetchone()
    return row[0] if row else 0


def restore_no_bird_images(conn: sqlite3.Connection, filenames: Iterable[str]) -> int:
    """
    Restores 'no_bird' images back to 'untagged' (returns them to Review Queue).
    Returns: number of rows updated.
    """
    names = list(filenames)
    if not names:
        return 0

    updated_at = datetime.now().isoformat()

    placeholders = ",".join("?" for _ in names)
    params = [updated_at] + names

    cur = conn.execute(
        f"""
        UPDATE images
        SET review_status = 'untagged', review_updated_at = ?
        WHERE filename IN ({placeholders})
        AND review_status = 'no_bird';
        """,
        params,
    )
    conn.commit()
    return cur.rowcount


def delete_no_bird_images(
    conn: sqlite3.Connection, filenames: Iterable[str] = None, delete_all: bool = False
) -> int:
    """
    Permanently deletes 'no_bird' images from the database.
    File deletion must be handled separately.

    Args:
        filenames: Specific files to delete (if None and delete_all=True, deletes all)
        delete_all: If True and filenames is None, deletes ALL no_bird images

    Returns: number of rows deleted.
    """
    if not filenames and not delete_all:
        return 0

    if filenames:
        names = list(filenames)
        placeholders = ",".join("?" for _ in names)
        cur = conn.execute(
            f"DELETE FROM images WHERE filename IN ({placeholders}) AND review_status = 'no_bird'",
            names,
        )
    else:
        # Delete all no_bird images
        cur = conn.execute("DELETE FROM images WHERE review_status = 'no_bird'")

    conn.commit()
    return cur.rowcount


def update_review_status(
    conn: sqlite3.Connection,
    filenames: Iterable[str],
    new_status: str,
    updated_at: str = None,
) -> int:
    """
    Updates review_status for specified images.
    Only updates images that are currently 'untagged' (no way back).

    new_status: 'confirmed_bird' | 'no_bird'
    Returns: number of rows updated.
    """
    names = list(filenames)
    if not names:
        return 0

    if new_status not in ("confirmed_bird", "no_bird"):
        raise ValueError(f"Invalid review status: {new_status}")

    if updated_at is None:
        updated_at = datetime.now().isoformat()

    placeholders = ",".join("?" for _ in names)
    params = [new_status, updated_at] + names

    cur = conn.execute(
        f"""
        UPDATE images
        SET review_status = ?, review_updated_at = ?
        WHERE filename IN ({placeholders})
        AND (review_status IS NULL OR review_status = 'untagged');
        """,
        params,
    )
    conn.commit()
    return cur.rowcount
