"""
Detection CRUD and Query Operations.

This module handles detection-related database operations including
insert, fetch, reject, restore, and purge functionality.
"""

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from utils.db.observations import (
    fact_sql,
    identity_sql,
    selection_sql,
    species_summaries,
)
from utils.species_names import (
    UNKNOWN_SPECIES_KEY,
)


def object_review_pending_sql(
    columns: set[str], alias: str = "d", conn: sqlite3.Connection | None = None
) -> str:
    level = f"{alias}.decision_level" if "decision_level" in columns else "NULL"
    unanswered = (
        f"({fact_sql(conn, 'species_identity', detection=alias)} IS NULL AND COALESCE({fact_sql(conn, 'bird_presence', detection=alias)}, '') != 'absent')"
        if conn is not None
        else "1"
    )
    return f"""({unanswered} AND (
        (COALESCE({alias}.decision_state, '') NOT IN ('confirmed', 'rejected')
         AND (COALESCE({alias}.score, 0.0) < ? OR {alias}.decision_state IN ('uncertain', 'unknown')))
        OR ({alias}.decision_state = 'confirmed'
            AND lower(COALESCE({level}, '')) IN ('reject', 'species_review'))
    ))"""


def _manual_active_count_sql(conn: sqlite3.Connection, image_expr: str) -> str:
    """Return a correlated manual-companion count for full or legacy schemas."""
    exists = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'manual_objects'
        """
    ).fetchone()
    if exists is None:
        return "0"
    return (
        "(SELECT COUNT(*) FROM manual_objects mo "
        f"WHERE mo.image_filename = {image_expr} AND mo.status = 'active')"
    )


def effective_bbox_projection_sql(
    conn: sqlite3.Connection,
    det_alias: str = "d",
    fact_alias: str = "bbox_fact",
) -> tuple[tuple[str, str, str, str], str]:
    """Return current display geometry plus its optional canonical-fact join.

    Detection geometry remains the immutable machine proposal. Once a person
    supplies a correction, image views project that current fact without
    rewriting the proposal row. Minimal legacy schemas fall back cleanly.
    """
    view_exists = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'view' AND name = 'current_human_label_facts'
        """
    ).fetchone()
    raw = tuple(f"{det_alias}.bbox_{axis}" for axis in ("x", "y", "w", "h"))
    if view_exists is None:
        return raw, ""
    projected = tuple(
        f"COALESCE({fact_alias}.bbox_{axis}, {det_alias}.bbox_{axis})"
        for axis in ("x", "y", "w", "h")
    )
    join_sql = f"""
        LEFT JOIN current_human_label_facts {fact_alias}
          ON {fact_alias}.detection_id = {det_alias}.detection_id
         AND {fact_alias}.fact_type = 'bbox_correction'
    """
    return projected, join_sql


def human_bbox_verdict_sql(conn: sqlite3.Connection, det_alias: str = "d") -> str:
    """Read human evidence only; legacy columns and model scores are not votes."""
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'view' "
            "AND name = 'current_human_label_facts'"
        ).fetchone()
        is None
    ):
        return "NULL"
    return (
        "(SELECT answer_value FROM current_human_label_facts "
        f"WHERE detection_id = {det_alias}.detection_id "
        "AND scope = 'object' AND fact_type = 'bbox_quality' "
        "AND assertion_state = 'asserted')"
    )


def _canonical_species_key_sql(expr: str) -> str:
    """SQL mirror of utils.species_names.canonical_species_key()."""
    return f"NULLIF(REPLACE(TRIM(COALESCE({expr}, '')), ' ', '_'), '')"


def is_detection_visible_in_gallery(
    conn: sqlite3.Connection, detection_id: int, min_score: float = 0.0
) -> bool:
    """Read admission without applying a second score threshold."""
    return (
        conn.execute(
            "SELECT 1 FROM current_observations WHERE detection_id = ? AND admitted",
            (detection_id,),
        ).fetchone()
        is not None
    )


def _companion_visibility_sql(
    det_alias: str = "d", image_alias: str = "i", conn: sqlite3.Connection | None = None
) -> str:
    """Show active companions without widening gallery admission."""
    presence = (
        fact_sql(conn, "bird_presence", detection=det_alias)
        if conn is not None
        else "NULL"
    )
    return f"""
        {det_alias}.status = 'active' AND COALESCE({presence}, '') != 'absent'
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


def _original_present_sql(conn: sqlite3.Connection, image_alias: str = "i") -> str:
    """SQL fragment for the original_present flag, resilient to legacy schemas.

    The column is added to images by a runtime migration; minimal test/legacy
    schemas may lack it, so degrade to a literal 1 ("present") when absent.
    """
    if "original_present" in table_columns(conn, "images"):
        return f"COALESCE({image_alias}.original_present, 1)"
    return "1"


def effective_species_sql(
    det_alias: str = "d", conn: sqlite3.Connection | None = None
) -> str:
    fallback = f"COALESCE({_canonical_species_key_sql(_top1_species_sql(det_alias))}, {_normalized_detector_species_sql(det_alias)})"
    return (
        identity_sql(conn, fallback, detection=det_alias)
        if conn is not None
        else fallback
    )


def _effective_species_joined_sql(
    det_alias: str = "d", cls_alias: str = "c", conn: sqlite3.Connection | None = None
) -> str:
    fallback = f"COALESCE({_canonical_species_key_sql(f'{cls_alias}.cls_class_name')}, {_normalized_detector_species_sql(det_alias)})"
    return (
        identity_sql(conn, fallback, detection=det_alias)
        if conn is not None
        else fallback
    )


def human_species_source_sql(conn: sqlite3.Connection, det_alias: str = "d") -> str:
    answer = fact_sql(conn, "species_identity", detection=det_alias)
    return f"CASE WHEN {answer} IN ('confirmed', 'corrected') THEN 'manual' WHEN {answer} IN ('unknown', 'wrong') THEN 'manual_' || {answer} ELSE NULL END"


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
    detector_sql = (
        _normalized_detector_species_sql(det_alias)
        if "od_class_name" in detection_columns
        else f"'{UNKNOWN_SPECIES_KEY}'"
    )
    return f"COALESCE({_canonical_species_key_sql(top1_sql)}, {detector_sql})"


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
            species_updated_at,
            decision_level,
            raw_species_name,
            geography_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            row.get("species_updated_at", row.get("created_at")),
            row.get("decision_level"),
            row.get("raw_species_name"),
            row.get("geography_json"),
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
    date_str_iso: str | None = None,
    limit: int | None = None,
    order_by: str = "score",
    detection_ids: list[int] | None = None,
    *,
    visibility: str = "gallery",
) -> list[sqlite3.Row | dict[str, Any]]:
    """
    Returns detection-centric records for gallery display.

    ``detection_ids``, when given, restricts the result to those rows (still
    subject to the usual gallery-visibility filter) instead of a date range.
    """
    if visibility == "gallery":
        where = "species_key != 'Unknown_species'"
        params: list[Any] = []
        if detection_ids is not None:
            if not detection_ids:
                return []
            where += (
                " AND COALESCE(detection_id, -manual_object_id) IN ("
                + ",".join("?" for _ in detection_ids)
                + ")"
            )
            params.extend(detection_ids)
        elif date_str_iso:
            where += " AND timestamp >= ? AND timestamp < ?"
            params.extend(_day_bounds(date_str_iso))
        order = (
            "timestamp DESC, object_key DESC"
            if order_by == "time"
            else "score DESC, timestamp DESC"
        )
        query = f"SELECT * FROM current_observations WHERE {where} ORDER BY {order}"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return hydrate_observations(
            conn, [dict(row) for row in conn.execute(query, params)]
        )
    params: list[Any] = []
    if visibility not in {"gallery", "objects", "selected"}:
        raise ValueError("invalid visibility")
    if visibility == "selected" and detection_ids is None:
        raise ValueError("selected visibility requires explicit identities")
    where_clauses = ["1"] if visibility == "selected" else ["d.status = 'active'"]

    if detection_ids is not None:
        if not detection_ids:
            return []
        placeholders = ",".join("?" for _ in detection_ids)
        where_clauses.append(f"d.detection_id IN ({placeholders})")
        params.extend(detection_ids)
    elif date_str_iso:
        start_ts, end_ts = _day_bounds(date_str_iso)
        where_clauses.append("i.timestamp >= ? AND i.timestamp < ?")
        params.extend([start_ts, end_ts])

    where_sql = " AND ".join(where_clauses)

    # Sort order
    if order_by == "time":
        order_clause = "ORDER BY d.created_at DESC, d.detection_id DESC"
        outer_order_clause = "ORDER BY v.created_at DESC, v.detection_id DESC"
    else:  # default "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"
        outer_order_clause = "ORDER BY v.score DESC, v.image_timestamp DESC"

    species_sql = _effective_species_joined_sql("d", "c", conn)
    bbox_sql, bbox_join_sql = effective_bbox_projection_sql(conn)
    verdict_sql = human_bbox_verdict_sql(conn)
    manual_count_d = _manual_active_count_sql(conn, "d.image_filename")
    manual_count_v = _manual_active_count_sql(conn, "v.image_filename")
    # original_present is added to images by a runtime migration; legacy/test
    # schemas may lack it. Introspect so the query degrades to "present" (1).
    original_present_sql = _original_present_sql(conn)
    select_body = f"""
            d.detection_id,
            i.timestamp as image_timestamp,
            d.created_at,
            d.image_filename,
            {bbox_sql[0]} AS bbox_x,
            {bbox_sql[1]} AS bbox_y,
            {bbox_sql[2]} AS bbox_w,
            {bbox_sql[3]} AS bbox_h,
            {verdict_sql} AS human_bbox_verdict,
            d.od_class_name,
            d.od_confidence,
            d.score,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            i.filename as original_name,
            {original_present_sql} as original_present,
            i.downloaded_timestamp,
            NULL AS review_status,
            {fact_sql(conn, "species_identity", "species_key")} AS human_species_key,
            {human_species_source_sql(conn)} AS human_species_source,
            c.cls_class_name,
            c.cls_confidence,
            {species_sql} as species_key,
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
                      AND {_companion_visibility_sql("d2", "i2", conn)}
                ) + {manual_count_d} as sibling_count
            FROM selected s
            JOIN detections d ON d.detection_id = s.detection_id
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
              ON c.detection_id = d.detection_id
             AND c.rank = 1
             AND COALESCE(c.status, 'active') = 'active'
            {bbox_join_sql}
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
            {bbox_join_sql}
            WHERE {where_sql}
        ),
        sibling_counts AS (
            SELECT d.image_filename, COUNT(*) AS sibling_count
            FROM detections d
            WHERE {_companion_visibility_sql(conn=conn)}
              AND d.image_filename IN (SELECT image_filename FROM visible)
            GROUP BY d.image_filename
        )
        SELECT
            v.*,
            COALESCE(sc.sibling_count, 1) + {manual_count_v} AS sibling_count
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
      - is_favorite=1            → manual gold label
      - is_gallery_eligible=1    → AI auto-pick (aesthetic tagger)

    Both are good cover candidates. The UI distinguishes them via an AI badge
    rendered when is_favorite=0 AND is_gallery_eligible=1. Within the random
    selection, both pool together so a station with few manual favorites still
    has a populated homepage.
    """
    query = """
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
            o.species_key as species_key,
            ROW_NUMBER() OVER(
                PARTITION BY o.species_key
                ORDER BY RANDOM()
            ) as rn
        FROM current_observations o
        JOIN detections d ON d.detection_id = o.detection_id
        JOIN images i ON d.image_filename = i.filename
        WHERE o.admitted
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
    """Count named species in the current observation view."""
    return conn.execute(
        "SELECT COUNT(DISTINCT species_key) FROM current_observations WHERE admitted"
    ).fetchone()[0]


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

    query = f"""
    WITH visible AS (
        SELECT o.*, o.timestamp AS image_timestamp,
            CAST(strftime('%s', substr(o.timestamp, 1, 4) || '-' ||
                substr(o.timestamp, 5, 2) || '-' || substr(o.timestamp, 7, 2) || ' ' ||
                substr(o.timestamp, 10, 2) || ':' || substr(o.timestamp, 12, 2) || ':' ||
                substr(o.timestamp, 14, 2)) AS INTEGER) AS image_epoch,
            d.aesthetic_score,
            CASE WHEN COALESCE(d.bbox_x, m.bbox_x, 0) > 0.01
                AND COALESCE(d.bbox_y, m.bbox_y, 0) > 0.01
                AND COALESCE(d.bbox_w, m.bbox_w, 0) > 0
                AND COALESCE(d.bbox_h, m.bbox_h, 0) > 0
                AND COALESCE(d.bbox_x + d.bbox_w, m.bbox_x + m.bbox_w, 1) < 0.99
                AND COALESCE(d.bbox_y + d.bbox_h, m.bbox_y + m.bbox_h, 1) < 0.99
                THEN 1 ELSE 0 END AS is_interior
        FROM current_observations o
        LEFT JOIN detections d ON d.detection_id = o.detection_id
        LEFT JOIN manual_objects m ON m.manual_object_id = o.manual_object_id
        WHERE o.admitted
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
    SELECT rf.*,
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
    return hydrate_observations(conn, [dict(row) for row in cur.fetchall()])


def fetch_sibling_detections(
    conn: sqlite3.Connection, image_filename: str, *, include_rejected: bool = False
) -> list[sqlite3.Row | dict[str, object]]:
    """
    Return active companions, optionally including dismissed boxes for inspection.
    Used to display all birds when viewing a multi-detection image in the modal.
    Includes bbox coordinates for bounding box visualization.
    """
    visibility = (
        "d.status IN ('active', 'rejected')"
        if include_rejected
        else _companion_visibility_sql(conn=conn)
    )
    species_sql = _effective_species_joined_sql("d", "c", conn)
    bbox_sql, bbox_join_sql = effective_bbox_projection_sql(conn)
    verdict_sql = human_bbox_verdict_sql(conn)
    query = f"""
        SELECT
            d.detection_id,
            d.status AS object_status,
            d.od_class_name,
            d.od_confidence,
            d.score,
            NULL AS review_status,
            {bbox_sql[0]} AS bbox_x,
            {bbox_sql[1]} AS bbox_y,
            {bbox_sql[2]} AS bbox_w,
            {bbox_sql[3]} AS bbox_h,
            {verdict_sql} AS human_bbox_verdict,
            d.decision_state,
            d.decision_level,
            (SELECT active AND NOT answered AND NOT admitted FROM current_observations o
                WHERE o.detection_id = d.detection_id) AS needs_review,
            {fact_sql(conn, "species_identity", "species_key")} AS human_species_key,
            {human_species_source_sql(conn)} AS human_species_source,
            d.is_favorite,
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
         AND (COALESCE(c.status, 'active') = 'active' OR d.status = 'rejected')
        {bbox_join_sql}
        WHERE d.image_filename = ? AND {visibility}
        ORDER BY d.score DESC
    """
    cur = conn.execute(query, (image_filename,))
    rows: list[sqlite3.Row | dict[str, object]] = list(cur.fetchall())
    rows.extend(
        _fetch_manual_companions(conn, [image_filename]).get(image_filename, [])
    )
    return rows


def fetch_sibling_detections_batch(
    conn: sqlite3.Connection,
    image_filenames: list[str],
    *,
    include_rejected: bool = False,
) -> dict[str, list[sqlite3.Row | dict[str, object]]]:
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
    visibility = (
        "d.status IN ('active', 'rejected')"
        if include_rejected
        else _companion_visibility_sql(conn=conn)
    )
    species_sql = _effective_species_joined_sql("d", "c", conn)
    bbox_sql, bbox_join_sql = effective_bbox_projection_sql(conn)
    verdict_sql = human_bbox_verdict_sql(conn)
    query = f"""
        SELECT
            d.image_filename,
            d.detection_id,
            d.status AS object_status,
            d.od_class_name,
            d.od_confidence,
            d.score,
            NULL AS review_status,
            {bbox_sql[0]} AS bbox_x,
            {bbox_sql[1]} AS bbox_y,
            {bbox_sql[2]} AS bbox_w,
            {bbox_sql[3]} AS bbox_h,
            {verdict_sql} AS human_bbox_verdict,
            d.decision_state,
            d.decision_level,
            (SELECT active AND NOT answered AND NOT admitted FROM current_observations o
                WHERE o.detection_id = d.detection_id) AS needs_review,
            {fact_sql(conn, "species_identity", "species_key")} AS human_species_key,
            {human_species_source_sql(conn)} AS human_species_source,
            d.is_favorite,
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
         AND (COALESCE(c.status, 'active') = 'active' OR d.status = 'rejected')
        {bbox_join_sql}
        WHERE d.image_filename IN ({placeholders}) AND {visibility}
        ORDER BY d.image_filename, d.score DESC
    """
    cur = conn.execute(query, image_filenames)
    grouped: dict[str, list[sqlite3.Row | dict[str, object]]] = {}
    for row in cur.fetchall():
        grouped.setdefault(row["image_filename"], []).append(row)
    for filename, manual_rows in _fetch_manual_companions(
        conn, image_filenames
    ).items():
        grouped.setdefault(filename, []).extend(manual_rows)
    return grouped


def _fetch_manual_companions(
    conn: sqlite3.Connection,
    image_filenames: list[str],
    *,
    include_retracted: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Project manual objects into the shared detail-view sibling shape."""
    names = sorted({name for name in image_filenames if name})
    if not names:
        return {}
    if _manual_active_count_sql(conn, "''") == "0":
        return {}
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT manual_object_id, image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
               species_key, species_state, revision, status
        FROM manual_objects
        WHERE image_filename IN ({placeholders}) {"" if include_retracted else "AND status = 'active'"}
        ORDER BY image_filename, manual_object_id
        """,
        names,
    ).fetchall()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        manual_object_id = int(row["manual_object_id"])
        species_key = row["species_key"]
        grouped.setdefault(str(row["image_filename"]), []).append(
            {
                "image_filename": row["image_filename"],
                "detection_id": None,
                "manual_object_id": manual_object_id,
                "object_key": f"manual:{manual_object_id}",
                "object_kind": "manual",
                "object_status": row["status"],
                "revision": int(row["revision"]),
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_w": row["bbox_w"],
                "bbox_h": row["bbox_h"],
                "species_key": species_key,
                "human_species_key": species_key,
                "human_species_source": "manual",
                "species_state": row["species_state"],
                "provenance": "manually_added",
                "od_class_name": None,
                "od_confidence": None,
                "cls_class_name": None,
                "cls_confidence": None,
                "score": None,
                "review_status": None,
                "decision_state": None,
                "thumbnail_path_virtual": None,
            }
        )
    return grouped


def fetch_day_count(conn: sqlite3.Connection, date_str_iso: str) -> int:
    """Returns COUNT(*) for a given date (YYYY-MM-DD)."""
    start_ts, end_ts = _day_bounds(date_str_iso)
    cur = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM current_observations o

        WHERE o.timestamp >= ? AND o.timestamp < ?
        AND o.admitted;
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
        """
        SELECT
            substr(o.timestamp, 10, 2) AS hour,
            COUNT(*) AS count
        FROM current_observations o

        WHERE o.timestamp >= ? AND o.timestamp < ?
        AND o.admitted
        GROUP BY hour
        ORDER BY hour;
        """,
        (start_ts, end_ts),
    )
    return cur.fetchall()


def fetch_daily_covers(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[sqlite3.Row | dict[str, Any]]:
    """
    Returns each day's cover by PTZ origin, aesthetic score, then detection score.
    Includes bbox for dynamic cropping and image count per day.

    Args:
        min_score: Minimum score threshold for counting images (to match gallery display filter)
    """
    selection = {"start": start_date or "", "end": end_date or ""}
    where, params = selection_sql(selection)
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
        WITH ranked AS (
            SELECT *, substr(REPLACE(timestamp, '-', ''), 1, 8) AS day,
                COUNT(*) OVER (PARTITION BY substr(REPLACE(timestamp, '-', ''), 1, 8)) AS image_count,
                SUM(answered) OVER (PARTITION BY substr(REPLACE(timestamp, '-', ''), 1, 8)) AS reviewed_count,
                SUM(EXISTS (SELECT 1 FROM current_human_label_facts f
                    WHERE f.detection_id = current_observations.detection_id AND f.scope = 'object'
                        AND f.fact_type = 'bbox_quality' AND f.answer_value IN ('suitable', 'unsuitable')))
                    OVER (PARTITION BY substr(REPLACE(timestamp, '-', ''), 1, 8)) AS boxes_reviewed_count,
                ROW_NUMBER() OVER (PARTITION BY substr(REPLACE(timestamp, '-', ''), 1, 8)
                    ORDER BY CASE WHEN ptz_origin IN ('preset', 'manual_drive') THEN 1 ELSE 0 END DESC,
                        COALESCE((SELECT d.aesthetic_score FROM detections d
                            WHERE d.detection_id = current_observations.detection_id), -1) DESC,
                        score DESC, detection_id DESC, object_key) AS rn
            FROM current_observations WHERE species_key != 'Unknown_species' AND {where}
        ) SELECT * FROM ranked WHERE rn = 1 ORDER BY day DESC
        """,
            params,
        )
    ]
    covers = hydrate_observations(conn, rows)
    for row in covers:
        day = row["day"]
        row["date_key"] = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        row.setdefault(
            "optimized_name_virtual", row["image_filename"].replace(".jpg", ".webp")
        )
        row.setdefault("thumbnail_path_virtual", None)
    return covers


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

    cur = conn.execute(
        """
        SELECT
            o.species_key as species,
            COUNT(*) as count
        FROM current_observations o

        WHERE o.timestamp >= ? AND o.timestamp < ?
        AND o.admitted
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
        """
        SELECT d.detection_id
        FROM current_observations o
        JOIN detections d ON d.detection_id = o.detection_id
        JOIN images i ON i.filename = d.image_filename
        WHERE o.admitted
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
        """
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM current_observations o
        JOIN detections d ON d.detection_id = o.detection_id
        JOIN images i ON i.filename = d.image_filename
        WHERE o.admitted
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
        WHERE NOT EXISTS (
              SELECT 1
              FROM detections d
              WHERE d.image_filename = i.filename
          )
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, i.filename ASC
        """,
        (from_prefix, to_prefix),
    ).fetchall()

    detection_rows = conn.execute(
        """
        SELECT
            d.detection_id,
            i.filename AS image_filename
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND substr(i.timestamp, 1, 8) >= ?
          AND substr(i.timestamp, 1, 8) <= ?
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        (from_prefix, to_prefix),
    ).fetchall()

    detection_ids = [int(row["detection_id"]) for row in detection_rows]
    detection_image_filenames = [str(row["image_filename"]) for row in detection_rows]
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


def reject_detections(
    conn: sqlite3.Connection,
    detection_ids: Iterable[int],
    *,
    commit: bool = True,
) -> None:
    """
    Semantic Reject: Sets status of specific detections to 'rejected'.
    Does not delete files.
    Propagates to classifications.
    """
    ids = list(detection_ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)

    conn.execute(
        f"UPDATE detections SET status = 'rejected', trashed_at=COALESCE(trashed_at, strftime('%Y-%m-%dT%H:%M:%fZ','now')) WHERE detection_id IN ({placeholders})",
        ids,
    )
    conn.execute(
        f"UPDATE classifications SET status = 'rejected' WHERE detection_id IN ({placeholders})",
        ids,
    )
    if commit:
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
        f"UPDATE detections SET status = 'active', trashed_at=NULL WHERE detection_id IN ({placeholders})",
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
    detection_ids: Iterable[int] | None = None,
    before_date: str | None = None,
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
        FROM current_observations o

        WHERE o.admitted
        AND o.timestamp >= ?
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
) -> list[sqlite3.Row | dict[str, Any]]:
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
    order = (
        "timestamp DESC, object_key DESC"
        if order_by == "time"
        else "score DESC, timestamp DESC"
    )
    query = f"SELECT * FROM current_observations WHERE species_key != 'Unknown_species' AND timestamp >= ? ORDER BY {order}"
    params = [threshold_timestamp]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return hydrate_observations(
        conn, [dict(row) for row in conn.execute(query, params)]
    )


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
            o.species_key AS species,
            d.score,
            i.timestamp AS image_timestamp
        FROM current_observations o
        JOIN detections d ON d.detection_id = o.detection_id
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c
          ON c.detection_id = d.detection_id
         AND c.rank = 1
         AND COALESCE(c.status, 'active') = 'active'
        WHERE o.admitted
          AND d.bbox_x IS NOT NULL
          AND d.bbox_y IS NOT NULL
          AND d.bbox_w > 0
          AND d.bbox_h > 0
        ORDER BY i.timestamp DESC
        LIMIT {limit}
    """
    cur = conn.execute(query)
    return cur.fetchall()


def hydrate_observations(
    conn: sqlite3.Connection, observations: list[dict]
) -> list[dict]:
    ids = [int(row["detection_id"]) for row in observations if row.get("detection_id")]
    hydrated = {}
    for start in range(0, len(ids), 400):
        hydrated.update(
            {
                f"detection:{row['detection_id']}": dict(row)
                for row in fetch_detections_for_gallery(
                    conn, detection_ids=ids[start : start + 400], visibility="selected"
                )
            }
        )
    names = [
        row["image_filename"] for row in observations if row.get("manual_object_id")
    ]
    for siblings in _fetch_manual_companions(
        conn, names, include_retracted=True
    ).values():
        for row in siblings:
            image = dict(
                conn.execute(
                    "SELECT timestamp, original_present FROM images WHERE filename = ?",
                    (row["image_filename"],),
                ).fetchone()
            )
            stamp = str(image["timestamp"]).replace("-", "")
            day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
            name = str(row["image_filename"])
            row.update(
                image_timestamp=image["timestamp"],
                original_name=name,
                original_present=image.get("original_present", 1),
                relative_path=f"{day}/{name.replace('.jpg', '.webp')}",
                detection_id=-int(row["manual_object_id"]),
            )
            hydrated[str(row["object_key"])] = row
    result = []
    for observation in observations:
        row = hydrated.get(observation["object_key"])
        if row is not None:
            row.update(observation)
            if observation.get("manual_object_id"):
                row["detection_id"] = -int(observation["manual_object_id"])
            result.append(row)
    return result


def fetch_observation_page(
    conn: sqlite3.Connection,
    *,
    species: str | None = None,
    answered: bool | None = None,
    selection: dict[str, str] | None = None,
    page: int = 1,
    per_page: int = 50,
) -> tuple[list[dict], int, int]:
    where, params = selection_sql(selection)
    if species == UNKNOWN_SPECIES_KEY:
        where += " AND unknown_bird"
    elif species:
        where += " AND species_key = ?"
        params.append(species)
    elif answered is not None:
        where += (
            " AND answered"
            if answered
            else " AND active AND NOT answered AND NOT admitted"
        )
    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM current_observations WHERE " + where, params
        ).fetchone()[0]
    )
    per_page = max(1, min(int(per_page), 100))
    page = max(1, min(page, max(1, (total + per_page - 1) // per_page)))
    rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM current_observations WHERE "
            + where
            + " ORDER BY (warning IS NOT NULL) DESC, timestamp DESC, COALESCE(detection_id, -manual_object_id) DESC, object_key LIMIT ? OFFSET ?",
            [*params, per_page, (page - 1) * per_page],
        )
    ]
    return hydrate_observations(conn, rows), total, page


def fetch_species_page(
    conn: sqlite3.Connection,
    species: str,
    *,
    min_score: float = 0,
    favorites_only: bool = False,
    page: int = 1,
    per_page: int = 50,
    selection: dict[str, str] | None = None,
) -> tuple[list[dict], int, int]:
    return fetch_observation_page(
        conn, species=species, selection=selection, page=page, per_page=per_page
    )


def fetch_species_covers(
    conn: sqlite3.Connection,
    *,
    min_score: float = 0,
    favorites_only: bool = False,
    selection: dict[str, str] | None = None,
) -> list[dict]:
    summaries = {row["species_key"]: row for row in species_summaries(conn, selection)}
    where, params = selection_sql(selection)
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
        WITH ranked AS (SELECT *, ROW_NUMBER() OVER (
            PARTITION BY species_key ORDER BY ({where}) DESC,
            CASE WHEN origin = 'human' THEN 1 ELSE 0 END DESC,
            original_present DESC, COALESCE(quality_gallery_ok, 1) DESC,
            (COALESCE(score, 0) >= ?) DESC,
            CASE WHEN COALESCE(is_favorite, 0) != 0 THEN 5
                WHEN ptz_origin IN ('preset', 'manual_drive') AND COALESCE(is_gallery_eligible, 0) != 0 THEN 4
                WHEN ptz_origin IN ('preset', 'manual_drive') THEN 3
                WHEN COALESCE(is_gallery_eligible, 0) != 0 THEN 2 ELSE 0 END DESC,
            COALESCE(score, 0) DESC, timestamp DESC, object_key
        ) AS rn FROM current_observations
        WHERE species_key != 'Unknown_species' OR unknown_bird)
        SELECT * FROM ranked WHERE rn = 1
    """,
            [*params, min_score],
        )
    ]
    result = []
    for row in hydrate_observations(conn, rows):
        if row["species_key"] in summaries:
            row.update(summaries[row["species_key"]])
            result.append(row)
        elif row["species_key"] == UNKNOWN_SPECIES_KEY:
            _, total, _ = fetch_observation_page(
                conn, species=UNKNOWN_SPECIES_KEY, selection=selection, per_page=1
            )
            if total:
                row["observation_count"] = total
                result.append(row)
    return result


def fetch_gallery_days(
    conn: sqlite3.Connection,
    *,
    min_score: float = 0,
    page: int = 1,
    per_page: int = 30,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return paged covers and SQL review totals, never historical detail lists."""
    days_sql = "SELECT DISTINCT substr(REPLACE(timestamp, '-', ''), 1, 8) AS day FROM current_observations WHERE species_key != 'Unknown_species'"
    total = conn.execute("SELECT COUNT(*) FROM (" + days_sql + ")").fetchone()[0]
    per_page = max(1, min(int(per_page), 100))
    page = max(1, min(page, max(1, (total + per_page - 1) // per_page)))
    days = [
        row[0]
        for row in conn.execute(
            days_sql + " ORDER BY day DESC LIMIT ? OFFSET ?",
            (per_page, (page - 1) * per_page),
        )
    ]
    if not days:
        return [], total, page
    selected = fetch_daily_covers(
        conn,
        start_date=f"{days[-1][:4]}-{days[-1][4:6]}-{days[-1][6:8]}",
        end_date=f"{days[0][:4]}-{days[0][4:6]}-{days[0][6:8]}",
    )
    for row in selected:
        row["review_progress"] = {
            "total": row["image_count"],
            "reviewed": row["reviewed_count"],
            "boxes_reviewed": row["boxes_reviewed_count"],
            "complete": row["image_count"] == row["reviewed_count"],
        }
    return selected, total, page
