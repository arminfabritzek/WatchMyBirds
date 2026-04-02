"""
Review Queue Database Operations.

This module handles review queue-related database operations including
orphan images, review status updates, and queue management.
"""

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta

from utils.db.detections import UNKNOWN_SPECIES_KEY, effective_species_sql


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
    filename: str | None = None,
) -> list[sqlite3.Row]:
    """
    Returns review items that need review:
    - true orphan images as image-backed items
    - active unresolved detections as detection-backed items

    Sorted by source image timestamp DESC (newest first).
    """
    species_sql = effective_species_sql("d")

    orphan_where = [
        "(i.review_status IS NULL OR i.review_status = 'untagged')",
        "NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)",
        "i.filename IS NOT NULL",
    ]
    orphan_params: list[object] = []

    detection_where = [
        "COALESCE(d.status, 'active') = 'active'",
        "COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')",
        "(i.review_status IS NULL OR i.review_status = 'untagged')",
        """(
            COALESCE(d.score, 0.0) < ?
            OR d.decision_state IN ('uncertain', 'unknown')
        )""",
        "i.filename IS NOT NULL",
    ]
    detection_params: list[object] = [gallery_threshold]

    if exclude_deep_scanned:
        orphan_where.append(
            "NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename AND d.od_model_id LIKE 'deep_scan_%')"
        )
        detection_where.append("COALESCE(d.od_model_id, '') NOT LIKE 'deep_scan_%'")

    if filename:
        orphan_where.append("i.filename = ?")
        orphan_params.append(filename)
        detection_where.append("i.filename = ?")
        detection_params.append(filename)

    orphan_where_sql = " AND ".join(orphan_where)
    detection_where_sql = " AND ".join(detection_where)

    query = f"""
    SELECT *
    FROM (
        SELECT
            'image' as item_kind,
            i.filename as item_id,
            i.filename,
            i.filename as source_image_filename,
            i.timestamp,
            i.review_status,
            NULL as max_score,
            NULL as best_detection_id,
            NULL as active_detection_id,
            NULL as bbox_x,
            NULL as bbox_y,
            NULL as bbox_w,
            NULL as bbox_h,
            'orphan' as review_reason,
            NULL as decision_state,
            NULL as bbox_quality,
            NULL as unknown_score,
            NULL as decision_reasons,
            NULL as od_confidence,
            NULL as cls_confidence,
            NULL as species_key,
            NULL as manual_species_override,
            NULL as species_source,
            NULL as manual_bbox_review,
            0 as sibling_detection_count
        FROM images i
        WHERE {orphan_where_sql}

        UNION ALL

        SELECT
            'detection' as item_kind,
            CAST(d.detection_id AS TEXT) as item_id,
            i.filename,
            i.filename as source_image_filename,
            i.timestamp,
            i.review_status,
            d.score as max_score,
            d.detection_id as best_detection_id,
            d.detection_id as active_detection_id,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            CASE
                WHEN d.decision_state = 'unknown' THEN 'unknown_species'
                WHEN d.decision_state = 'uncertain' THEN 'uncertain'
                ELSE 'low_score'
            END as review_reason,
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons,
            d.od_confidence,
            (
                SELECT c.cls_confidence
                FROM classifications c
                WHERE c.detection_id = d.detection_id
                  AND c.rank = 1
                  AND COALESCE(c.status, 'active') = 'active'
                LIMIT 1
            ) as cls_confidence,
            {species_sql} as species_key,
            d.manual_species_override,
            d.species_source,
            d.manual_bbox_review,
            (
                SELECT COUNT(*)
                FROM detections ds
                WHERE ds.image_filename = d.image_filename
                  AND COALESCE(ds.status, 'active') = 'active'
                  AND COALESCE(ds.decision_state, '') NOT IN ('confirmed', 'rejected')
            ) as sibling_detection_count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {detection_where_sql}
    ) review_items
    ORDER BY timestamp DESC, item_kind DESC, CAST(COALESCE(active_detection_id, 0) AS INTEGER) ASC;
    """
    cur = conn.execute(query, [*orphan_params, *detection_params])
    return cur.fetchall()


def fetch_review_queue_image(
    conn: sqlite3.Connection,
    filename: str,
    gallery_threshold: float = 0.7,
    exclude_deep_scanned: bool = False,
) -> sqlite3.Row | None:
    """Return the first review-queue row for a filename, or ``None`` if absent."""
    rows = fetch_review_queue_images(
        conn,
        gallery_threshold=gallery_threshold,
        exclude_deep_scanned=exclude_deep_scanned,
        filename=filename,
    )
    return rows[0] if rows else None


def fetch_recent_review_species(
    conn: sqlite3.Connection,
    limit: int = 8,
    lookback_days: int = 7,
) -> list[sqlite3.Row]:
    """Return recently common active species for review quick-picks."""
    species_sql = effective_species_sql("d")
    cutoff = (datetime.now() - timedelta(days=max(1, lookback_days))).strftime(
        "%Y%m%d_%H%M%S"
    )

    query = f"""
    SELECT
        {species_sql} as species_key,
        COUNT(*) as hit_count,
        MAX(i.timestamp) as last_seen
    FROM detections d
    JOIN images i ON i.filename = d.image_filename
    WHERE COALESCE(d.status, 'active') = 'active'
      AND i.timestamp IS NOT NULL
      AND i.timestamp >= ?
      AND {species_sql} IS NOT NULL
      AND {species_sql} != ?
    GROUP BY {species_sql}
    ORDER BY hit_count DESC, last_seen DESC
    LIMIT ?
    """
    cur = conn.execute(query, (cutoff, UNKNOWN_SPECIES_KEY, limit))
    return cur.fetchall()


def fetch_review_queue_count(
    conn: sqlite3.Connection, gallery_threshold: float = 0.7
) -> int:
    """
    Returns count of review items needing review (for badge).
    Same criteria as fetch_review_queue_images.
    """
    query = """
    SELECT
        (
            SELECT COUNT(*)
            FROM images i
            WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
              AND NOT EXISTS (
                  SELECT 1 FROM detections d WHERE d.image_filename = i.filename
              )
              AND i.filename IS NOT NULL
        )
        +
        (
            SELECT COUNT(*)
            FROM detections d
            JOIN images i ON i.filename = d.image_filename
            WHERE COALESCE(d.status, 'active') = 'active'
              AND COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
              AND (i.review_status IS NULL OR i.review_status = 'untagged')
              AND (
                  COALESCE(d.score, 0.0) < ?
                  OR d.decision_state IN ('uncertain', 'unknown')
              )
              AND i.filename IS NOT NULL
        ) AS review_count
    """
    row = conn.execute(query, (gallery_threshold,)).fetchone()
    return row["review_count"] if row else 0


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
