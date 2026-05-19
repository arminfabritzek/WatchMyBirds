"""
Detection CRUD and Query Operations.

This module handles detection-related database operations including
insert, fetch, reject, restore, and purge functionality.
"""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from utils.review_metadata import (
    REVIEW_STATUS_NO_BIRD,
    REVIEW_STATUS_UNTAGGED,
    VALID_BBOX_REVIEW_STATES,
)
from utils.species_names import UNKNOWN_SPECIES_KEY


def _canonical_species_key_sql(expr: str) -> str:
    """SQL mirror of utils.species_names.canonical_species_key()."""
    return f"NULLIF(REPLACE(TRIM(COALESCE({expr}, '')), ' ', '_'), '')"


def _gallery_visibility_sql(det_alias: str = "d", image_alias: str = "i") -> str:
    """Shared visibility policy for gallery-like detection surfaces.

    Two-axis filter:

    1. **Temporal gate**: ``decision_state = 'confirmed'`` — only
       detections the temporal smoother has stamped as confirmed
       reach the public gallery. Uncertain / unknown / NULL live in
       the review queue instead.

    2. **Classifier decision-level gate**: exclude
       ``decision_level IN ('reject', 'species_review')``.

       - ``reject`` — top-1 below the (legacy or per-class) species
         threshold AND genus fallback declined. Empty cls_class_name,
         "Unknown species" cards — hidden.
       - ``species_review`` — between Review and Gallery thresholds in
         the two-stage gate. These have a top-1 suggestion but the
         model is not confident enough for an auto-Gallery slot; they
         live in the Unclear surface where the operator confirms or
         discards. Manual confirmation flips ``decision_level`` to
         ``species`` so they re-enter the gallery.

       ``decision_level IS NULL`` is explicitly allowed so:
       - Historical rows saved before the decision-level column
         existed stay visible when their decision_state=confirmed.
       - Classifiers that ship no YAML (legacy / dev variants)
         still produce gallery rows — their top-1 always wins,
         which is the pre-2026-04-23 behaviour.

    Rejection rationale: the classifier v2 decision layer (species /
    genus / reject) is only trustworthy when temporal smoothing has
    seen the same species across enough frames to stamp ``confirmed``.
    A single high-confidence frame can still be a false positive
    (hallucinated species, motion blur); a single low-confidence
    frame is even worse.
    """
    return f"""
        {det_alias}.status = 'active'
        AND ({image_alias}.review_status IS NULL OR {image_alias}.review_status != '{REVIEW_STATUS_NO_BIRD}')
        AND lower(COALESCE({det_alias}.decision_state, '')) = 'confirmed'
        AND (
            {det_alias}.decision_level IS NULL
            OR lower({det_alias}.decision_level) NOT IN ('reject', 'species_review')
        )
    """


def _normalized_detector_species_sql(det_alias: str = "d") -> str:
    canonical_od = _canonical_species_key_sql(f"{det_alias}.od_class_name")
    return f"""
        CASE
            WHEN {canonical_od} IS NULL
                THEN '{UNKNOWN_SPECIES_KEY}'
            WHEN lower({canonical_od}) IN ('bird', 'unknown', 'unknown_species', 'unclassified')
                THEN '{UNKNOWN_SPECIES_KEY}'
            ELSE {canonical_od}
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
            {_canonical_species_key_sql(f"NULLIF({det_alias}.manual_species_override, '')")},
            {_canonical_species_key_sql(_top1_species_sql(det_alias))},
            {_normalized_detector_species_sql(det_alias)}
        )
    """


def _effective_species_joined_sql(det_alias: str = "d", cls_alias: str = "c") -> str:
    """Species expression for queries that already join the rank-1 CLS row."""
    return f"""
        COALESCE(
            {_canonical_species_key_sql(f"NULLIF({det_alias}.manual_species_override, '')")},
            {_canonical_species_key_sql(f"{cls_alias}.cls_class_name")},
            {_normalized_detector_species_sql(det_alias)}
        )
    """


def _day_bounds(date_str_iso: str) -> tuple[str, str]:
    """Return inclusive/exclusive timestamp bounds for an ISO date."""
    day = datetime.strptime(date_str_iso, "%Y-%m-%d")
    return (
        day.strftime("%Y%m%d"),
        (day + timedelta(days=1)).strftime("%Y%m%d"),
    )


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
            {_canonical_species_key_sql(manual_sql)},
            {_canonical_species_key_sql(top1_sql)},
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
            species_updated_at,
            decision_level,
            raw_species_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            row.get("decision_level"),
            row.get("raw_species_name"),
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
    params: list[Any] = []
    where_clauses = [_gallery_visibility_sql("d", "i")]

    if date_str_iso:
        start_ts, end_ts = _day_bounds(date_str_iso)
        where_clauses.append("i.timestamp >= ? AND i.timestamp < ?")
        params.extend([start_ts, end_ts])

    where_sql = " AND ".join(where_clauses)

    # Sort order
    if order_by == "time":
        order_clause = "ORDER BY d.created_at DESC"
        outer_order_clause = "ORDER BY v.created_at DESC"
    else:  # default "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"
        outer_order_clause = "ORDER BY v.score DESC, v.image_timestamp DESC"

    species_sql = _effective_species_joined_sql("d", "c")
    select_body = f"""
            d.detection_id,
            i.timestamp as image_timestamp,
            d.created_at,
            d.image_filename,
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
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} as species_key,
            d.rating,
            d.rating_source,
            d.is_favorite,
            d.is_gallery_eligible,
            d.aesthetic_score,
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons,
            i.ptz_origin
    """

    if limit is not None:
        query = f"""
            WITH selected AS (
                SELECT d.detection_id
                FROM detections d
                JOIN images i ON d.image_filename = i.filename
                WHERE {where_sql}
                {order_clause}
                LIMIT ?
            )
            SELECT
                {select_body},
                (
                    SELECT COUNT(*)
                    FROM detections d2
                    JOIN images i2 ON i2.filename = d2.image_filename
                    WHERE d2.image_filename = d.image_filename
                      AND {_gallery_visibility_sql("d2", "i2")}
                ) as sibling_count
            FROM selected s
            JOIN detections d ON d.detection_id = s.detection_id
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            {order_clause}
        """
        params.append(limit)
        cur = conn.execute(query, params)
        return cur.fetchall()

    query = f"""
        WITH visible AS (
            SELECT
                {select_body}
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            WHERE {where_sql}
        ),
        sibling_counts AS (
            SELECT image_filename, COUNT(*) AS sibling_count
            FROM visible
            GROUP BY image_filename
        )
        SELECT
            v.*,
            COALESCE(sc.sibling_count, 1) AS sibling_count
        FROM visible v
        LEFT JOIN sibling_counts sc ON sc.image_filename = v.image_filename
        {outer_order_clause}
    """

    cur = conn.execute(query, params)
    return cur.fetchall()


def fetch_random_favorites(
    conn: sqlite3.Connection,
    limit: int = 6,
) -> list[sqlite3.Row]:
    """
    Returns a random selection of cover-worthy detections, diversified by species.
    Used for a quick 'best of' preview gallery on the homepage.

    Two sources, queried equally:
      - is_favorite=1            → HUMAN gold-label
      - is_gallery_eligible=1    → KI auto-pick (aesthetic tagger)

    Both are good cover candidates. The UI distinguishes them via a KI badge
    rendered when is_favorite=0 AND is_gallery_eligible=1. Within the random
    selection, both pool together so a station with few HUMAN favorites still
    has a populated homepage.
    """
    query = f"""
    WITH CoverCandidates AS (
        SELECT
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name, d.od_confidence, d.score,
            d.is_favorite, d.is_gallery_eligible,
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
        WHERE {_gallery_visibility_sql("d", "i")}
          AND (d.is_favorite = 1 OR d.is_gallery_eligible = 1)
    )
    SELECT * FROM CoverCandidates
    WHERE rn = 1
    ORDER BY RANDOM()
    LIMIT ?
    """
    cur = conn.execute(query, (limit,))
    return cur.fetchall()


def fetch_gallery_total_species_count(conn: sqlite3.Connection) -> int:
    """Return the number of visible gallery species without full row hydration."""
    species_sql = _effective_species_joined_sql("d", "c")
    cur = conn.execute(
        f"""
        SELECT COUNT(DISTINCT species_key) AS total
        FROM (
            SELECT {species_sql} AS species_key
            FROM detections d
            JOIN images i ON i.filename = d.image_filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            WHERE {_gallery_visibility_sql("d", "i")}
        )
        WHERE species_key IS NOT NULL
          AND species_key != ?
        """,
        (UNKNOWN_SPECIES_KEY,),
    )
    row = cur.fetchone()
    return int(row["total"] or 0) if row else 0


def fetch_species_story_board_candidates(
    conn: sqlite3.Connection,
    *,
    total_limit: int = 12,
    frames_per_species: int = 3,
    excluded_species: Iterable[str] | None = None,
) -> list[sqlite3.Row]:
    """Return a bounded, SQL-ranked candidate set for the homepage board.

    The legacy homepage path hydrated every visible detection, grouped all
    events in Python, then kept only 12 species and up to 3 images each. On
    long-running stations this made the first byte wait scale with the entire
    database. This query keeps the expensive scan inside SQLite and returns at
    most ``total_limit * frames_per_species`` rows to Python.
    """
    total_limit = max(1, int(total_limit or 1))
    frames_per_species = max(1, int(frames_per_species or 1))
    excluded = [str(species) for species in (excluded_species or []) if species]

    species_filter_sql = ""
    params: list[Any] = []
    if excluded:
        placeholders = ",".join("?" for _ in excluded)
        species_filter_sql = f"WHERE species_key NOT IN ({placeholders})"
        params.extend(excluded)

    params.extend([total_limit, frames_per_species])
    species_sql = _effective_species_joined_sql("d", "c")

    query = f"""
    WITH visible AS (
        SELECT
            d.detection_id,
            i.timestamp AS image_timestamp,
            CAST(strftime(
                '%s',
                substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2) || ' ' ||
                substr(i.timestamp, 10, 2) || ':' ||
                substr(i.timestamp, 12, 2) || ':' ||
                substr(i.timestamp, 14, 2)
            ) AS INTEGER) AS image_epoch,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.score,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            REPLACE(i.filename, '.jpg', '.webp') AS optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            i.filename AS original_name,
            i.downloaded_timestamp,
            i.review_status,
            d.manual_species_override,
            d.species_source,
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} AS species_key,
            d.rating,
            d.rating_source,
            d.is_favorite,
            d.is_gallery_eligible,
            d.aesthetic_score,
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons,
            i.ptz_origin,
            CASE
                WHEN COALESCE(d.bbox_w, 0) <= 0 OR COALESCE(d.bbox_h, 0) <= 0 THEN 0
                WHEN COALESCE(d.bbox_x, 0) <= 0.01 THEN 0
                WHEN COALESCE(d.bbox_y, 0) <= 0.01 THEN 0
                WHEN COALESCE(d.bbox_x, 0) + COALESCE(d.bbox_w, 0) >= 0.99 THEN 0
                WHEN COALESCE(d.bbox_y, 0) + COALESCE(d.bbox_h, 0) >= 0.99 THEN 0
                ELSE 1
            END AS is_interior
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
        WHERE {_gallery_visibility_sql("d", "i")}
    ),
    eligible AS (
        SELECT *
        FROM visible
        {species_filter_sql}
    ),
    ordered_events AS (
        SELECT
            species_key,
            image_epoch,
            LAG(image_epoch) OVER (
                PARTITION BY species_key
                ORDER BY image_epoch ASC, detection_id ASC
            ) AS prev_epoch
        FROM eligible
        WHERE image_epoch IS NOT NULL
    ),
    species_visits AS (
        SELECT
            species_key,
            SUM(
                CASE
                    WHEN prev_epoch IS NULL THEN 1
                    WHEN image_epoch - prev_epoch > 1800 THEN 1
                    ELSE 0
                END
            ) AS visit_count
        FROM ordered_events
        GROUP BY species_key
    ),
    species_stats AS (
        SELECT
            e.species_key,
            COALESCE(MAX(sv.visit_count), COUNT(*)) AS visit_count,
            MAX(e.image_timestamp) AS last_seen_timestamp,
            MAX(COALESCE(e.score, 0)) AS best_cover_score,
            MAX(COALESCE(e.is_favorite, 0)) AS is_favorite_available
        FROM eligible e
        LEFT JOIN species_visits sv ON sv.species_key = e.species_key
        GROUP BY e.species_key
    ),
    ranked_species AS (
        SELECT
            species_stats.*,
            ROW_NUMBER() OVER (
                ORDER BY
                    visit_count DESC,
                    substr(last_seen_timestamp, 1, 8) DESC,
                    last_seen_timestamp DESC,
                    best_cover_score DESC,
                    species_key ASC
            ) AS species_rank
        FROM species_stats
    ),
    ranked_frames AS (
        SELECT
            e.*,
            ROW_NUMBER() OVER (
                PARTITION BY e.species_key
                ORDER BY
                    COALESCE(e.is_favorite, 0) DESC,
                    CASE e.ptz_origin
                        WHEN 'preset' THEN 1
                        WHEN 'manual_drive' THEN 1
                        ELSE 0
                    END DESC,
                    COALESCE(e.is_gallery_eligible, 0) DESC,
                    e.is_interior DESC,
                    COALESCE(e.aesthetic_score, -1) DESC,
                    COALESCE(e.score, 0) DESC,
                    e.image_timestamp DESC,
                    e.detection_id DESC
            ) AS frame_rank
        FROM eligible e
        JOIN ranked_species rs ON rs.species_key = e.species_key
        WHERE rs.species_rank <= ?
    )
    SELECT
        rf.detection_id,
        rf.image_timestamp,
        rf.bbox_x,
        rf.bbox_y,
        rf.bbox_w,
        rf.bbox_h,
        rf.od_class_name,
        rf.od_confidence,
        rf.score,
        rf.thumbnail_path_virtual,
        rf.optimized_name_virtual,
        rf.relative_path,
        rf.original_name,
        rf.downloaded_timestamp,
        rf.review_status,
        rf.manual_species_override,
        rf.species_source,
        rf.cls_class_name,
        rf.cls_confidence,
        rf.species_key,
        rf.rating,
        rf.rating_source,
        rf.is_favorite,
        rf.is_gallery_eligible,
        rf.aesthetic_score,
        rf.decision_state,
        rf.bbox_quality,
        rf.unknown_score,
        rf.decision_reasons,
        rf.ptz_origin,
        (
            SELECT COUNT(*)
            FROM detections d2
            JOIN images i2 ON i2.filename = d2.image_filename
            WHERE d2.image_filename = rf.original_name
              AND {_gallery_visibility_sql("d2", "i2")}
        ) AS sibling_count,
        rs.species_rank,
        rf.frame_rank,
        rs.visit_count,
        rs.last_seen_timestamp,
        rs.best_cover_score,
        rs.is_favorite_available
    FROM ranked_frames rf
    JOIN ranked_species rs ON rs.species_key = rf.species_key
    WHERE rf.frame_rank <= ?
    ORDER BY rs.species_rank ASC, rf.frame_rank ASC
    """

    cur = conn.execute(query, params)
    return cur.fetchall()


def fetch_sibling_detections(
    conn: sqlite3.Connection, image_filename: str
) -> list[sqlite3.Row]:
    """
    Returns all active detections for a given image filename.
    Used to display all birds when viewing a multi-detection image in the modal.
    Includes bbox coordinates for bounding box visualization.
    """
    species_sql = _effective_species_joined_sql("d", "c")
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
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} as species_key,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
        WHERE d.image_filename = ? AND {_gallery_visibility_sql("d", "i")}
        ORDER BY d.score DESC
    """
    cur = conn.execute(query, (image_filename,))
    return cur.fetchall()


def fetch_sibling_detections_batch(
    conn: sqlite3.Connection, image_filenames: list[str]
) -> dict[str, list[sqlite3.Row]]:
    """
    Batch variant of fetch_sibling_detections. Returns siblings for several
    image filenames in a single query, grouped by ``image_filename``.

    Index pages render dozens of detection cards per request, each of which
    needs its multi-bird companions for the detail modal. Calling the
    single-row helper in a loop costs one round-trip per row plus
    connection setup; this batch helper collapses the lot to one query.

    Returns ``{}`` when ``image_filenames`` is empty. Image filenames not
    present in the result are absent from the returned dict (caller treats
    missing keys as "no siblings").
    """
    if not image_filenames:
        return {}
    placeholders = ",".join("?" * len(image_filenames))
    species_sql = _effective_species_joined_sql("d", "c")
    query = f"""
        SELECT
            d.image_filename,
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
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} as species_key,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
        WHERE d.image_filename IN ({placeholders}) AND {_gallery_visibility_sql("d", "i")}
        ORDER BY d.image_filename, d.score DESC
    """
    cur = conn.execute(query, tuple(image_filenames))
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in cur.fetchall():
        grouped.setdefault(row["image_filename"], []).append(row)
    return grouped


def fetch_day_count(conn: sqlite3.Connection, date_str_iso: str) -> int:
    """Returns COUNT(*) for a given date (YYYY-MM-DD)."""
    start_ts, end_ts = _day_bounds(date_str_iso)
    cur = conn.execute(
        f"""
        SELECT COUNT(*) AS cnt
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE i.timestamp >= ? AND i.timestamp < ?
        AND {_gallery_visibility_sql("d", "i")};
        """,
        (start_ts, end_ts),
    )
    row = cur.fetchone()
    return int(row["cnt"]) if row else 0


def fetch_hourly_counts(
    conn: sqlite3.Connection, date_str_iso: str
) -> list[sqlite3.Row]:
    """Returns hourly counts for a given date (YYYY-MM-DD)."""
    start_ts, end_ts = _day_bounds(date_str_iso)
    cur = conn.execute(
        f"""
        SELECT
            substr(i.timestamp, 10, 2) AS hour,
            COUNT(*) AS count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE i.timestamp >= ? AND i.timestamp < ?
        AND {_gallery_visibility_sql("d", "i")}
        GROUP BY hour
        ORDER BY hour;
        """,
        (start_ts, end_ts),
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
            d.detection_id,
            d.image_filename,
            i.filename as original_name,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.score,
            d.thumbnail_path,
            ROW_NUMBER() OVER (
                PARTITION BY (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2))
                -- Cover ranking, highest priority first:
                --   1. manual rating
                --   2. PTZ preset bias — frames captured while the camera was
                --      driven to a non-overview preset are physically closer
                --      to the bird. Auto and manual drives rank equally; legacy
                --      NULL rows sort behind anything tagged. See plan
                --      2026-05-15_PTZ_image-context-for-gallery-bias.
                --   3. nightly aesthetic score from scripts/aesthetic_tag_nightly.py
                --   4. detector confidence (legacy fallback)
                ORDER BY COALESCE(d.rating, 0) DESC,
                         CASE i.ptz_origin
                             WHEN 'preset' THEN 1
                             WHEN 'manual_drive' THEN 1
                             ELSE 0
                         END DESC,
                         COALESCE(d.aesthetic_score, -1) DESC,
                         d.score DESC
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
        db.detection_id,
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
    start_ts, end_ts = _day_bounds(date_str_iso)
    species_sql = _effective_species_joined_sql("d", "c")

    cur = conn.execute(
        f"""
        SELECT
            {species_sql} as species,
            COUNT(d.detection_id) as count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
        WHERE i.timestamp >= ? AND i.timestamp < ?
        AND {_gallery_visibility_sql("d", "i")}
        GROUP BY species
        ORDER BY count DESC;
        """,
        (start_ts, end_ts),
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
        outer_order_clause = "ORDER BY v.image_timestamp DESC, v.score DESC"
    else:  # "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"
        outer_order_clause = "ORDER BY v.score DESC, v.image_timestamp DESC"

    species_sql = _effective_species_joined_sql("d", "c")
    select_body = f"""
            d.detection_id,
            i.timestamp as image_timestamp,
            d.image_filename,
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
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} as species_key,
            d.is_favorite,
            d.is_gallery_eligible
    """

    params: list[Any] = [threshold_timestamp]
    if limit is not None:
        query = f"""
            WITH selected AS (
                SELECT d.detection_id
                FROM detections d
                JOIN images i ON d.image_filename = i.filename
                WHERE {_gallery_visibility_sql("d", "i")}
                AND i.timestamp >= ?
                {order_clause}
                LIMIT ?
            )
            SELECT
                {select_body},
                (
                    SELECT COUNT(*)
                    FROM detections d2
                    JOIN images i2 ON i2.filename = d2.image_filename
                    WHERE d2.image_filename = d.image_filename
                      AND {_gallery_visibility_sql("d2", "i2")}
                ) as sibling_count
            FROM selected s
            JOIN detections d ON d.detection_id = s.detection_id
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            {order_clause}
        """
        params.append(limit)
        cur = conn.execute(query, params)
        return cur.fetchall()

    query = f"""
        WITH visible AS (
            SELECT
                {select_body}
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            WHERE {_gallery_visibility_sql("d", "i")}
            AND i.timestamp >= ?
        ),
        sibling_counts AS (
            SELECT image_filename, COUNT(*) AS sibling_count
            FROM visible
            GROUP BY image_filename
        )
        SELECT
            v.*,
            COALESCE(sc.sibling_count, 1) AS sibling_count
        FROM visible v
        LEFT JOIN sibling_counts sc ON sc.image_filename = v.image_filename
        {outer_order_clause}
    """

    cur = conn.execute(query, params)
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
    species_sql = _effective_species_joined_sql("d", "c")
    query = f"""
        SELECT
            (d.bbox_x + d.bbox_w / 2.0) AS center_x,
            (d.bbox_y + d.bbox_h / 2.0) AS center_y,
            d.bbox_w,
            d.bbox_h,
            {species_sql} AS species,
            d.score,
            i.timestamp AS image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
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
