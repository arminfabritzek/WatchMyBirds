"""
Detection CRUD and Query Operations.

This module handles detection-related database operations including
insert, fetch, reject, restore, and purge functionality.
"""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from utils.review_metadata import (
    REVIEW_STATUS_NO_BIRD,
    REVIEW_STATUS_UNTAGGED,
    VALID_BBOX_REVIEW_STATES,
)
from utils.species_names import UNKNOWN_SPECIES_KEY


def _gallery_visibility_sql(det_alias: str = "d", image_alias: str = "i") -> str:
    """Shared visibility policy for gallery-like detection surfaces."""
    return f"""
        {det_alias}.status = 'active'
        AND ({image_alias}.review_status IS NULL OR {image_alias}.review_status != '{REVIEW_STATUS_NO_BIRD}')
        AND lower(COALESCE({det_alias}.decision_state, '')) NOT IN ('uncertain', 'unknown')
    """


def _normalized_detector_species_sql(det_alias: str = "d") -> str:
    return f"""
        CASE
            WHEN {det_alias}.od_class_name IS NULL OR TRIM({det_alias}.od_class_name) = ''
                THEN '{UNKNOWN_SPECIES_KEY}'
            WHEN lower({det_alias}.od_class_name) IN ('bird', 'unknown', 'unclassified')
                THEN '{UNKNOWN_SPECIES_KEY}'
            ELSE {det_alias}.od_class_name
        END
    """


def _top1_species_sql(det_alias: str = "d") -> str:
    return f"""
        (
            SELECT c.cls_class_name
            FROM classifications c
            WHERE c.detection_id = {det_alias}.detection_id
              AND c.rank = 1
              AND COALESCE(c.status, 'active') = 'active'
            LIMIT 1
        )
    """


def _top1_species_sql_for_columns(
    det_alias: str = "d", classification_columns: set[str] | None = None
) -> str:
    classification_columns = classification_columns or set()
    where_clauses = [f"c.detection_id = {det_alias}.detection_id"]
    if "rank" in classification_columns:
        where_clauses.append("c.rank = 1")
    if "status" in classification_columns:
        where_clauses.append("COALESCE(c.status, 'active') = 'active'")
    where_sql = "\n              AND ".join(where_clauses)
    order_sql = "ORDER BY c.rank ASC" if "rank" in classification_columns else ""
    return f"""
        (
            SELECT c.cls_class_name
            FROM classifications c
            WHERE {where_sql}
            {order_sql}
            LIMIT 1
        )
    """


def _top1_confidence_sql(det_alias: str = "d") -> str:
    return f"""
        (
            SELECT c.cls_confidence
            FROM classifications c
            WHERE c.detection_id = {det_alias}.detection_id
              AND c.rank = 1
              AND COALESCE(c.status, 'active') = 'active'
            LIMIT 1
        )
    """


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for a SQLite table."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe table name: {table_name!r}")

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    columns: set[str] = set()
    for row in rows:
        try:
            columns.add(row["name"])
        except (IndexError, TypeError):
            columns.add(row[1])
    return columns


def effective_species_sql(det_alias: str = "d") -> str:
    return f"""
        COALESCE(
            NULLIF({det_alias}.manual_species_override, ''),
            {_top1_species_sql(det_alias)},
            {_normalized_detector_species_sql(det_alias)}
        )
    """


def effective_species_sql_for_columns(
    det_alias: str = "d",
    detection_columns: set[str] | None = None,
    classification_columns: set[str] | None = None,
) -> str:
    """Build species SQL for tests or older DBs that lack newer columns."""
    detection_columns = detection_columns or set()
    top1_sql = (
        _top1_species_sql(det_alias)
        if classification_columns is None
        else _top1_species_sql_for_columns(det_alias, classification_columns)
    )
    manual_sql = (
        f"NULLIF({det_alias}.manual_species_override, '')"
        if "manual_species_override" in detection_columns
        else "NULL"
    )
    detector_sql = (
        _normalized_detector_species_sql(det_alias)
        if "od_class_name" in detection_columns
        else f"'{UNKNOWN_SPECIES_KEY}'"
    )
    return f"""
        COALESCE(
            {manual_sql},
            {top1_sql},
            {detector_sql}
        )
    """


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
            frame_height,
            decision_state,
            bbox_quality,
            unknown_score,
            decision_reasons,
            policy_version,
            manual_species_override,
            species_source,
            species_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            row.get("decision_state"),
            row.get("bbox_quality"),
            row.get("unknown_score"),
            row.get("decision_reasons"),
            row.get("policy_version"),
            row.get("manual_species_override"),
            row.get("species_source"),
            row.get("species_updated_at", row.get("created_at")),
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
    where_clauses = [_gallery_visibility_sql("d", "i")]

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
            i.review_status,
            d.manual_species_override,
            d.species_source,
            {_top1_species_sql("d")} as cls_class_name,
            {_top1_confidence_sql("d")} as cls_confidence,
            {effective_species_sql("d")} as species_key,
            d.rating,
            d.rating_source,
            d.is_favorite,
            -- Count of sibling detections on the same image (for multi-bird display)
            (
                SELECT COUNT(*)
                FROM detections d2
                JOIN images i2 ON i2.filename = d2.image_filename
                WHERE d2.image_filename = d.image_filename
                  AND {_gallery_visibility_sql("d2", "i2")}
            ) as sibling_count,
            -- Decision policy fields (P1-04)
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons
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
    query = f"""
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
            {effective_species_sql("d")} as species_key,
            ROW_NUMBER() OVER(
                PARTITION BY {effective_species_sql("d")}
                ORDER BY RANDOM()
            ) as rn
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")} AND d.is_favorite = 1
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
    query = f"""
        SELECT
            d.detection_id,
            d.od_class_name,
            d.od_confidence,
            d.score,
            i.review_status,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            d.decision_state,
            d.manual_species_override,
            d.species_source,
            {_top1_species_sql("d")} as cls_class_name,
            {_top1_confidence_sql("d")} as cls_confidence,
            {effective_species_sql("d")} as species_key,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.image_filename = ? AND {_gallery_visibility_sql("d", "i")}
        ORDER BY d.score DESC
    """
    cur = conn.execute(query, (image_filename,))
    return cur.fetchall()


def fetch_day_count(conn: sqlite3.Connection, date_str_iso: str) -> int:
    """Returns COUNT(*) for a given date (YYYY-MM-DD)."""
    date_prefix = date_str_iso.replace("-", "")
    cur = conn.execute(
        f"""
        SELECT COUNT(*) AS cnt
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.image_filename LIKE ? || '%'
        AND {_gallery_visibility_sql("d", "i")};
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
        f"""
        SELECT
            substr(d.image_filename, 10, 2) AS hour,
            COUNT(*) AS count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.image_filename LIKE ? || '%'
        AND {_gallery_visibility_sql("d", "i")}
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
    query = f"""
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
        WHERE {_gallery_visibility_sql("d", "i")}
    ),
    DayCounts AS (
        SELECT
            (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2)) as date_iso,
            COUNT(*) as image_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")}
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
        f"""
        SELECT
            {effective_species_sql("d")} as species,
            COUNT(d.detection_id) as count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.image_filename LIKE ? || '%'
        AND {_gallery_visibility_sql("d", "i")}
        GROUP BY species
        ORDER BY count DESC;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def fetch_active_detection_ids_in_date_range(
    conn: sqlite3.Connection, from_date: str, to_date: str
) -> list[int]:
    """
    Return active detection IDs whose capture day falls within the inclusive
    ISO date range [from_date, to_date], ordered deterministically.
    """
    from_prefix = from_date.replace("-", "")
    to_prefix = to_date.replace("-", "")

    cur = conn.execute(
        f"""
        SELECT d.detection_id
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {_gallery_visibility_sql("d", "i")}
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (from_prefix, to_prefix),
    )
    return [int(row["detection_id"]) for row in cur.fetchall()]


def fetch_active_detection_selection_in_date_range(
    conn: sqlite3.Connection, from_date: str, to_date: str
) -> dict[str, Any]:
    """
    Return active detections for an inclusive date range together with the
    distinct source images they belong to.
    """
    from_prefix = from_date.replace("-", "")
    to_prefix = to_date.replace("-", "")

    cur = conn.execute(
        f"""
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {_gallery_visibility_sql("d", "i")}
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (from_prefix, to_prefix),
    )

    rows = cur.fetchall()
    detection_ids = [int(row["detection_id"]) for row in rows]
    image_filenames = list(dict.fromkeys(str(row["image_filename"]) for row in rows))

    return {
        "detection_ids": detection_ids,
        "image_filenames": image_filenames,
        "image_count": len(image_filenames),
    }


def fetch_trash_candidate_selection_in_date_range(
    conn: sqlite3.Connection, from_date: str, to_date: str
) -> dict[str, Any]:
    """
    Return untrashed review/gallery candidates for an inclusive date range.

    This is used by Trash bulk actions, so it must operate on image-level
    eligibility (`untagged` / NULL) rather than gallery visibility rules.
    """
    from_prefix = from_date.replace("-", "")
    to_prefix = to_date.replace("-", "")

    orphan_rows = conn.execute(
        """
        SELECT i.filename AS image_filename
        FROM images i
        WHERE (i.review_status IS NULL OR i.review_status = ?)
          AND NOT EXISTS (
              SELECT 1
              FROM detections d
              WHERE d.image_filename = i.filename
          )
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, i.filename ASC
        """,
        (REVIEW_STATUS_UNTAGGED, from_prefix, to_prefix),
    ).fetchall()

    detection_rows = conn.execute(
        """
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND (i.review_status IS NULL OR i.review_status = ?)
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (REVIEW_STATUS_UNTAGGED, from_prefix, to_prefix),
    ).fetchall()

    detection_ids = [int(row["detection_id"]) for row in detection_rows]
    detection_image_filenames = [
        str(row["image_filename"]) for row in detection_rows
    ]
    orphan_image_filenames = [str(row["image_filename"]) for row in orphan_rows]
    image_filenames = list(
        dict.fromkeys([*detection_image_filenames, *orphan_image_filenames])
    )

    return {
        "detection_ids": detection_ids,
        "image_filenames": image_filenames,
        "orphan_image_filenames": orphan_image_filenames,
        "orphan_count": len(orphan_image_filenames),
        "image_count": len(image_filenames),
    }


def fetch_active_detection_selection_by_source_type(
    conn: sqlite3.Connection, source_type: str
) -> dict[str, Any]:
    """
    Return active detections for a given source type together with the number
    of distinct images those detections belong to.
    """
    cur = conn.execute(
        f"""
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        JOIN sources s ON s.source_id = i.source_id
        WHERE {_gallery_visibility_sql("d", "i")}
          AND s.type = ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (source_type,),
    )

    rows = cur.fetchall()
    detection_ids = [int(row["detection_id"]) for row in rows]
    image_filenames = list(dict.fromkeys(str(row["image_filename"]) for row in rows))

    return {
        "detection_ids": detection_ids,
        "image_filenames": image_filenames,
        "image_count": len(image_filenames),
    }


def fetch_trash_candidate_selection_by_source_type(
    conn: sqlite3.Connection, source_type: str
) -> dict[str, Any]:
    """
    Return untrashed review/gallery candidates for a given source type.

    Unlike the gallery visibility resolver, this includes review-only detections
    and orphan images as long as the image itself is still `untagged`.
    """
    orphan_rows = conn.execute(
        """
        SELECT i.filename AS image_filename
        FROM images i
        JOIN sources s ON s.source_id = i.source_id
        WHERE (i.review_status IS NULL OR i.review_status = ?)
          AND NOT EXISTS (
              SELECT 1
              FROM detections d
              WHERE d.image_filename = i.filename
          )
          AND s.type = ?
        ORDER BY i.timestamp ASC, i.filename ASC
        """,
        (REVIEW_STATUS_UNTAGGED, source_type),
    ).fetchall()

    detection_rows = conn.execute(
        """
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        JOIN sources s ON s.source_id = i.source_id
        WHERE d.status = 'active'
          AND (i.review_status IS NULL OR i.review_status = ?)
          AND s.type = ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (REVIEW_STATUS_UNTAGGED, source_type),
    ).fetchall()

    detection_ids = [int(row["detection_id"]) for row in detection_rows]
    detection_image_filenames = [
        str(row["image_filename"]) for row in detection_rows
    ]
    orphan_image_filenames = [str(row["image_filename"]) for row in orphan_rows]
    image_filenames = list(
        dict.fromkeys([*detection_image_filenames, *orphan_image_filenames])
    )

    return {
        "detection_ids": detection_ids,
        "image_filenames": image_filenames,
        "orphan_image_filenames": orphan_image_filenames,
        "orphan_count": len(orphan_image_filenames),
        "image_count": len(image_filenames),
    }


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


def apply_species_override(
    conn: sqlite3.Connection,
    detection_id: int,
    species: str,
    source: str = "manual",
) -> None:
    """Persist a final/manual species override on the detection row only."""
    conn.execute(
        """
        UPDATE detections
        SET manual_species_override = ?,
            species_source = ?,
            species_updated_at = ?
        WHERE detection_id = ?
        """,
        (species, source, datetime.now(UTC).isoformat(), detection_id),
    )
    conn.commit()


def apply_species_override_many(
    conn: sqlite3.Connection,
    detection_ids: Iterable[int],
    species: str,
    source: str = "manual",
) -> int:
    """Persist one override species for multiple detections."""
    ids = [int(det_id) for det_id in detection_ids]
    if not ids:
        return 0

    placeholders = ",".join("?" for _ in ids)
    cur = conn.execute(
        f"""
        UPDATE detections
        SET manual_species_override = ?,
            species_source = ?,
            species_updated_at = ?
        WHERE detection_id IN ({placeholders})
          AND status = 'active'
        """,
        [species, source, datetime.now(UTC).isoformat(), *ids],
    )
    conn.commit()
    return cur.rowcount


def set_manual_bbox_review(
    conn: sqlite3.Connection,
    detection_id: int,
    review_state: str | None,
) -> None:
    """Persist the manual review state for a detection bounding box."""
    normalized = (review_state or "").strip().lower() or None
    if normalized not in {None, *VALID_BBOX_REVIEW_STATES}:
        raise ValueError(f"invalid bbox review state: {review_state}")

    reviewed_at = datetime.now(UTC).isoformat() if normalized else None
    conn.execute(
        """
        UPDATE detections
        SET manual_bbox_review = ?,
            bbox_reviewed_at = ?
        WHERE detection_id = ?
        """,
        (normalized, reviewed_at, detection_id),
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
        f"""
        SELECT COUNT(*) AS cnt
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")}
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
            i.review_status,
            d.manual_species_override,
            d.species_source,
            {_top1_species_sql("d")} as cls_class_name,
            {_top1_confidence_sql("d")} as cls_confidence,
            {effective_species_sql("d")} as species_key,
            d.is_favorite,
            (
                SELECT COUNT(*)
                FROM detections d2
                JOIN images i2 ON i2.filename = d2.image_filename
                WHERE d2.image_filename = d.image_filename
                  AND {_gallery_visibility_sql("d2", "i2")}
            ) as sibling_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")}
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
            {effective_species_sql("d")} AS species,
            d.score,
            i.timestamp AS image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")}
          AND d.bbox_x IS NOT NULL
          AND d.bbox_y IS NOT NULL
          AND d.bbox_w > 0
          AND d.bbox_h > 0
        ORDER BY i.timestamp DESC
        LIMIT {limit}
    """
    cur = conn.execute(query)
    return cur.fetchall()
