"""
Detection CRUD and Query Operations.

This module handles detection-related database operations including
insert, fetch, reject, restore, and purge functionality.
"""

import sqlite3
from collections.abc import Iterable
from typing import Any


def insert_detection(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Inserts a detection record and returns its ID."""
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            od_class_name,
            od_confidence,
            od_model_id,
            created_at,
            score,
            agreement_score,
            detector_model_name,
            detector_model_version,
            classifier_model_name,
            classifier_model_version,
            thumbnail_path,
            frame_width,
            frame_height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("image_filename"),
            row.get("bbox_x"),
            row.get("bbox_y"),
            row.get("bbox_w"),
            row.get("bbox_h"),
            row.get("od_class_name"),
            row.get("od_confidence"),
            row.get("od_model_id"),
            row.get("created_at"),
            row.get("score"),
            row.get("agreement_score"),
            row.get("detector_model_name"),
            row.get("detector_model_version"),
            row.get("classifier_model_name"),
            row.get("classifier_model_version"),
            row.get("thumbnail_path"),
            row.get("frame_width"),
            row.get("frame_height"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_classification(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Inserts a classification record and returns its ID."""
    cur = conn.execute(
        """
        INSERT INTO classifications (
            detection_id,
            cls_class_name,
            cls_confidence,
            cls_model_id,
            rank,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("detection_id"),
            row.get("cls_class_name"),
            row.get("cls_confidence"),
            row.get("cls_model_id"),
            row.get("rank", 1),
            row.get("created_at"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def fetch_detections_for_gallery(
    conn: sqlite3.Connection,
    date_str_iso: str = None,
    limit: int = None,
    order_by: str = "score",
) -> list[sqlite3.Row]:
    """
    Returns detection-centric records for gallery display.
    """
    params = []
    where_clauses = ["d.status = 'active'"]

    if date_str_iso:
        date_prefix = date_str_iso.replace("-", "")
        where_clauses.append(
            "i.timestamp LIKE ? || '%'"
        )  # Using i.timestamp for filtering
        params.append(date_prefix)

    where_sql = " AND ".join(where_clauses)

    # Sort order
    if order_by == "time":
        order_clause = "ORDER BY d.created_at DESC"
    else:  # default "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"

    query = f"""
        SELECT
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.score,
            -- Virtual paths matching actual filesystem structure (YYYY-MM-DD folders)
            -- Prefer explicit thumbnail_path if available (for multi-detection support), else fallback to virtual crop
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            i.filename as original_name,
            i.downloaded_timestamp,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence,
            d.rating,
            d.rating_source,
            d.is_favorite,
            -- Count of sibling detections on the same image (for multi-bird display)
            (SELECT COUNT(*) FROM detections d2 WHERE d2.image_filename = d.image_filename AND d2.status = 'active') as sibling_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {where_sql}
        {order_clause}
    """

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(query, params)
    return cur.fetchall()


def fetch_random_favorites(
    conn: sqlite3.Connection,
    limit: int = 6,
) -> list[sqlite3.Row]:
    """
    Returns a random selection of favorite detections, diversified by species.
    Used for a quick 'best of' preview gallery on the homepage.
    """
    query = """
    WITH RankedFavorites AS (
        SELECT
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name, d.od_confidence, d.score, d.is_favorite,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            COALESCE((SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1), d.od_class_name, 'Unknown') as species_key,
            ROW_NUMBER() OVER(
                PARTITION BY COALESCE((SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1), d.od_class_name, 'Unknown')
                ORDER BY RANDOM()
            ) as rn
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active' AND d.is_favorite = 1
    )
    SELECT * FROM RankedFavorites
    WHERE rn = 1
    ORDER BY RANDOM()
    LIMIT ?
    """
    cur = conn.execute(query, (limit,))
    return cur.fetchall()


def fetch_sibling_detections(
    conn: sqlite3.Connection, image_filename: str
) -> list[sqlite3.Row]:
    """
    Returns all active detections for a given image filename.
    Used to display all birds when viewing a multi-detection image in the modal.
    Includes bbox coordinates for bounding box visualization.
    """
    query = """
        SELECT
            d.detection_id,
            d.od_class_name,
            d.od_confidence,
            d.score,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.image_filename = ? AND d.status = 'active'
        ORDER BY d.score DESC
    """
    cur = conn.execute(query, (image_filename,))
    return cur.fetchall()


def fetch_day_count(conn: sqlite3.Connection, date_str_iso: str) -> int:
    """Returns COUNT(*) for a given date (YYYY-MM-DD)."""
    date_prefix = date_str_iso.replace("-", "")
    cur = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM detections d
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active';
        """,
        (date_prefix,),
    )
    row = cur.fetchone()
    return int(row["cnt"]) if row else 0


def fetch_hourly_counts(
    conn: sqlite3.Connection, date_str_iso: str
) -> list[sqlite3.Row]:
    """Returns hourly counts for a given date (YYYY-MM-DD)."""
    date_prefix = date_str_iso.replace("-", "")
    cur = conn.execute(
        """
        SELECT
            substr(d.image_filename, 10, 2) AS hour,
            COUNT(*) AS count
        FROM detections d
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active'
        GROUP BY hour
        ORDER BY hour;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def fetch_daily_covers(
    conn: sqlite3.Connection, min_score: float = 0.0
) -> list[sqlite3.Row]:
    """
    Returns the best detection (highest rating, then score) for each day to use as a cover.
    Includes bbox for dynamic cropping and image count per day.

    Args:
        min_score: Minimum score threshold for counting images (to match gallery display filter)
    """
    query = """
    WITH DailyBest AS (
        SELECT
            (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2)) as date_iso,
            d.image_filename,
            i.filename as original_name,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.score,
            d.thumbnail_path,
            ROW_NUMBER() OVER (
                PARTITION BY (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2))
                ORDER BY COALESCE(d.rating, 0) DESC, d.score DESC
            ) as rn
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
    ),
    DayCounts AS (
        SELECT
            (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2)) as date_iso,
            COUNT(*) as image_count
        FROM detections d
        WHERE d.status = 'active'
        AND (d.score IS NULL OR d.score >= ?)
        GROUP BY (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2))
    )
    SELECT
        db.date_iso as date_key,
        REPLACE(db.original_name, '.jpg', '.webp') as optimized_name_virtual,
        (substr(db.image_filename, 1, 4) || '-' || substr(db.image_filename, 5, 2) || '-' || substr(db.image_filename, 7, 2) || '/' ||
         REPLACE(db.original_name, '.jpg', '.webp')) AS relative_path,
        db.bbox_x, db.bbox_y, db.bbox_w, db.bbox_h,
        (substr(db.image_filename, 1, 4) || '-' || substr(db.image_filename, 5, 2) || '-' || substr(db.image_filename, 7, 2) || '/' ||
         COALESCE(db.thumbnail_path, REPLACE(db.original_name, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
        dc.image_count
    FROM DailyBest db
    JOIN DayCounts dc ON db.date_iso = dc.date_iso
    WHERE db.rn = 1
    ORDER BY date_key DESC;
    """
    cur = conn.execute(query, (min_score,))
    return cur.fetchall()


def fetch_detection_species_summary(
    conn: sqlite3.Connection, date_str_iso: str
) -> list[sqlite3.Row]:
    """
    Returns counts per species for a given date (YYYY-MM-DD), based on DETECTIONS.
    Species = classification class name if present.
    If no classification class name, we consider it 'Unclassified' (or handle OD class if desired,
    but per plan we rely on CLS).
    """
    date_prefix = date_str_iso.replace("-", "")

    cur = conn.execute(
        """
        SELECT
            COALESCE(cls.cls_class_name, 'Unclassified') as species,
            COUNT(d.detection_id) as count
        FROM detections d
        LEFT JOIN classifications cls ON d.detection_id = cls.detection_id AND cls.status = 'active'
            AND cls.rank = 1 -- Assuming rank 1 is top choice
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active'
        GROUP BY species
        ORDER BY count DESC;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def reject_detections(conn: sqlite3.Connection, detection_ids: Iterable[int]) -> None:
    """
    Semantic Reject: Sets status of specific detections to 'rejected'.
    Does not delete files.
    Propagates to classifications.
    """
    ids = list(detection_ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)

    # Update status of detections
    conn.execute(
        f"UPDATE detections SET status = 'rejected' WHERE detection_id IN ({placeholders})",
        ids,
    )
    # Also reject classifications for these rejected detections
    conn.execute(
        f"UPDATE classifications SET status = 'rejected' WHERE detection_id IN ({placeholders})",
        ids,
    )
    conn.commit()


def restore_detections(conn: sqlite3.Connection, detection_ids: Iterable[int]) -> None:
    """
    Restores rejected detections to active status. also restores associated classifications.
    Triggers legacy recalculation for affected images.
    """
    ids = list(detection_ids)
    if not ids:
        return

    placeholders = ",".join("?" for _ in ids)

    # 1. Restore Detections
    conn.execute(
        f"UPDATE detections SET status = 'active' WHERE detection_id IN ({placeholders})",
        ids,
    )

    # 2. Restore Classifications (cascading restore)
    conn.execute(
        f"UPDATE classifications SET status = 'active' WHERE detection_id IN ({placeholders})",
        ids,
    )

    conn.commit()


def purge_detections(
    conn: sqlite3.Connection,
    detection_ids: Iterable[int] = None,
    before_date: str = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes rejected detections (and cascades to classifications).
    Strictly DB-only. No file deletion.
    Requires strict scoping (ids or filter).
    """
    if not detection_ids and not before_date:
        raise ValueError("Purge requires explicit detection_ids or filter")

    where_clauses = ["status = 'rejected'"]
    params = []

    if detection_ids:
        ids = list(detection_ids)
        placeholders = ",".join("?" for _ in ids)
        where_clauses.append(f"detection_id IN ({placeholders})")
        params.extend(ids)

    if before_date:
        # Expects ISO date string 'YYYY-MM-DD'
        # image_timestamp usually starts with YYYYMMDD_
        date_prefix = before_date.replace("-", "")
        where_clauses.append("image_filename < ?")
        params.append(date_prefix)

    where_sql = " AND ".join(where_clauses)

    # Dry Run: Count what would be deleted
    count_cursor = conn.execute(
        f"SELECT COUNT(*), GROUP_CONCAT(detection_id) FROM detections WHERE {where_sql}",
        params,
    )
    row = count_cursor.fetchone()
    count = row[0]
    affected_ids_str = row[1] if row[1] else ""
    affected_ids = (
        [int(x) for x in affected_ids_str.split(",")] if affected_ids_str else []
    )

    if dry_run:
        return {"purged": False, "would_purge": count, "detection_ids": affected_ids}

    # Execute Purge
    conn.execute(f"DELETE FROM detections WHERE {where_sql}", params)
    conn.commit()

    return {"purged": True, "count": count, "detection_ids": affected_ids}


def fetch_count_last_24h(conn: sqlite3.Connection, threshold_timestamp: str) -> int:
    """
    Count active detections in the last 24 hours (rolling window).

    Args:
        conn: Database connection
        threshold_timestamp: Timestamp string in formats:
            - '%Y%m%d_%H%M%S' for direct comparison with image_filename
            - Can be the result of threshold_datetime.strftime("%Y%m%d_%H%M%S")

    Returns:
        Total count of detections since threshold
    """
    cur = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
        AND i.timestamp >= ?
        """,
        (threshold_timestamp,),
    )
    row = cur.fetchone()
    return int(row["cnt"]) if row else 0


def fetch_detections_last_24h(
    conn: sqlite3.Connection,
    threshold_timestamp: str,
    limit: int | None = None,
    order_by: str = "time",
) -> list[sqlite3.Row]:
    """
    Fetch detections from the last 24 hours (rolling window).

    Args:
        conn: Database connection
        threshold_timestamp: Timestamp string (YYYYMMDD_HHMMSS format)
        limit: Optional limit
        order_by: "time" (newest first) or "score" (highest first)

    Returns:
        List of detection rows (same format as fetch_detections_for_gallery)
    """
    # Sort order
    if order_by == "time":
        order_clause = "ORDER BY i.timestamp DESC, d.score DESC"
    else:  # "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.score,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            i.filename as original_name,
            i.downloaded_timestamp,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence,
            d.is_favorite,
            (SELECT COUNT(*) FROM detections d2 WHERE d2.image_filename = d.image_filename AND d2.status = 'active') as sibling_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
        AND i.timestamp >= ?
        {order_clause}
        {limit_clause}
    """

    cur = conn.execute(query, (threshold_timestamp,))
    return cur.fetchall()


def fetch_bbox_centers(
    conn: sqlite3.Connection,
    limit: int = 1000,
) -> list[sqlite3.Row]:
    """
    Fetch bounding-box center coordinates for the heatmap strip chart.

    Returns lightweight rows with only the center-x, center-y (normalized 0-1),
    species classification, confidence score, and timestamp.
    Ordered newest-first so the frontend can render a time-series strip.

    Args:
        conn: Database connection
        limit: Maximum number of data points (default 1000, RPi-friendly)

    Returns:
        List of rows: (center_x, center_y, species, score, image_timestamp)
    """
    query = f"""
        SELECT
            (d.bbox_x + d.bbox_w / 2.0) AS center_x,
            (d.bbox_y + d.bbox_h / 2.0) AS center_y,
            d.bbox_w,
            d.bbox_h,
            COALESCE(
                (SELECT c.cls_class_name
                 FROM classifications c
                 WHERE c.detection_id = d.detection_id
                 ORDER BY c.cls_confidence DESC LIMIT 1),
                d.od_class_name
            ) AS species,
            d.score,
            i.timestamp AS image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
          AND d.bbox_x IS NOT NULL
          AND d.bbox_y IS NOT NULL
          AND d.bbox_w > 0
          AND d.bbox_h > 0
        ORDER BY i.timestamp DESC
        LIMIT {limit}
    """
    cur = conn.execute(query)
    return cur.fetchall()
