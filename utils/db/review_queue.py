"""
Review Queue Database Operations.

This module handles review queue-related database operations including
orphan images, review status updates, and queue management.
"""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from utils.db.detections import (
    UNKNOWN_SPECIES_KEY,
    effective_species_sql_for_columns,
    table_columns,
)


def _detection_column_sql(
    detection_columns: set[str], column_name: str, fallback_sql: str = "NULL"
) -> str:
    return f"d.{column_name}" if column_name in detection_columns else fallback_sql


def _ex_unclear_predicate_sql(detection_columns: set[str], det_alias: str = "d") -> str:
    """Return the WHERE-fragment that admits ex-Unclear detections.

    Returns ``"0"`` (SQL false) when ``decision_level`` is absent — keeps
    Minimal-schema test fixtures working without forcing them to add the
    column. The fragment is OR-combined with the main "unresolved"
    predicate, so returning false simply means the live schema needs
    the column for the routing to apply.
    """
    if "decision_level" not in detection_columns:
        return "0"
    return (
        f"({det_alias}.decision_state = 'confirmed' "
        f"AND lower(COALESCE({det_alias}.decision_level, '')) "
        f"IN ('reject', 'species_review'))"
    )


def _ex_unclear_reason_case_sql(
    detection_columns: set[str], det_alias: str = "d"
) -> str:
    """Return the CASE-WHEN branches that label ex-Unclear review reasons.

    Returns ``""`` (no branches) when ``decision_level`` is absent. Caller
    inserts this into a larger CASE expression.
    """
    if "decision_level" not in detection_columns:
        return ""
    return (
        f"WHEN {det_alias}.decision_state = 'confirmed' "
        f"AND lower(COALESCE({det_alias}.decision_level, '')) = 'reject' "
        f"THEN 'classifier_reject' "
        f"WHEN {det_alias}.decision_state = 'confirmed' "
        f"AND lower(COALESCE({det_alias}.decision_level, '')) = 'species_review' "
        f"THEN 'classifier_species_review' "
    )


def _queue_where_clauses(
    detection_columns: set[str],
    *,
    exclude_deep_scanned: bool = False,
    filename: str | None = None,
) -> tuple[list[str], list[object], list[str], list[object]]:
    """Build the orphan/detection WHERE fragments for the review queue.

    Single source of truth for "what is in the review queue". Both the
    full render query (``fetch_review_queue_images``) and the lean summary
    path (``fetch_review_queue_summary_rows`` / ``fetch_review_queue_summary``)
    consume this, so queue membership can never drift between the Review
    page and the "Move Review Queue to Trash" action.

    Returns ``(orphan_where, orphan_params, detection_where, detection_params)``
    where the lists are AND-joinable WHERE fragments and their bound params
    in order. The detection fragment binds ``gallery_threshold`` first.
    """
    ex_unclear_predicate = _ex_unclear_predicate_sql(detection_columns)

    orphan_where = [
        "(i.review_status IS NULL OR i.review_status = 'untagged')",
        "NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)",
        "i.filename IS NOT NULL",
    ]
    orphan_params: list[object] = []

    detection_where = [
        "COALESCE(d.status, 'active') = 'active'",
        "(i.review_status IS NULL OR i.review_status = 'untagged')",
        f"""(
            (
                COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
                AND (
                    COALESCE(d.score, 0.0) < ?
                    OR d.decision_state IN ('uncertain', 'unknown')
                )
            )
            OR {ex_unclear_predicate}
        )""",
        "i.filename IS NOT NULL",
    ]
    detection_params: list[object] = []

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

    return orphan_where, orphan_params, detection_where, detection_params


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
    detection_columns = table_columns(conn, "detections")
    image_columns = table_columns(conn, "images")
    original_present_sql = (
        "COALESCE(i.original_present, 1)"
        if "original_present" in image_columns
        else "1"
    )
    species_sql = effective_species_sql_for_columns("d", detection_columns)
    is_favorite_sql = _detection_column_sql(detection_columns, "is_favorite", "0")
    is_gallery_eligible_sql = _detection_column_sql(
        detection_columns, "is_gallery_eligible", "0"
    )
    manual_species_sql = _detection_column_sql(
        detection_columns, "manual_species_override"
    )
    species_source_sql = _detection_column_sql(detection_columns, "species_source")
    manual_bbox_sql = _detection_column_sql(detection_columns, "manual_bbox_review")
    frame_width_sql = _detection_column_sql(detection_columns, "frame_width")
    frame_height_sql = _detection_column_sql(detection_columns, "frame_height")
    ex_unclear_predicate_ds = _ex_unclear_predicate_sql(detection_columns, "ds")
    ex_unclear_reason_case = _ex_unclear_reason_case_sql(detection_columns)

    orphan_where, orphan_params, detection_where, detection_params = (
        _queue_where_clauses(
            detection_columns,
            exclude_deep_scanned=exclude_deep_scanned,
            filename=filename,
        )
    )
    detection_params = [gallery_threshold, *detection_params]

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
            {original_present_sql} as original_present,
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
            NULL as cls_class_name,
            NULL as cls_confidence,
            NULL as species_key,
            NULL as manual_species_override,
            NULL as species_source,
            NULL as manual_bbox_review,
            0 as sibling_detection_count,
            0 as is_favorite,
            0 as is_gallery_eligible,
            NULL as frame_width,
            NULL as frame_height
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
            {original_present_sql} as original_present,
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
                {ex_unclear_reason_case}
                ELSE 'low_score'
            END as review_reason,
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons,
            d.od_confidence,
            (
                SELECT c.cls_class_name
                FROM classifications c
                WHERE c.detection_id = d.detection_id
                  AND c.rank = 1
                  AND COALESCE(c.status, 'active') = 'active'
                LIMIT 1
            ) as cls_class_name,
            (
                SELECT c.cls_confidence
                FROM classifications c
                WHERE c.detection_id = d.detection_id
                  AND c.rank = 1
                  AND COALESCE(c.status, 'active') = 'active'
                LIMIT 1
            ) as cls_confidence,
            {species_sql} as species_key,
            {manual_species_sql} as manual_species_override,
            {species_source_sql} as species_source,
            {manual_bbox_sql} as manual_bbox_review,
            (
                SELECT COUNT(*)
                FROM detections ds
                WHERE ds.image_filename = d.image_filename
                  AND COALESCE(ds.status, 'active') = 'active'
                  AND (
                      COALESCE(ds.decision_state, '') NOT IN ('confirmed', 'rejected')
                      OR {ex_unclear_predicate_ds}
                  )
            ) as sibling_detection_count,
            COALESCE({is_favorite_sql}, 0) as is_favorite,
            COALESCE({is_gallery_eligible_sql}, 0) as is_gallery_eligible,
            {frame_width_sql} as frame_width,
            {frame_height_sql} as frame_height
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {detection_where_sql}
    ) review_items
    ORDER BY timestamp DESC, item_kind DESC, COALESCE(max_score, 0) DESC, CAST(COALESCE(active_detection_id, 0) AS INTEGER) ASC;
    """
    cur = conn.execute(query, [*orphan_params, *detection_params])
    return cur.fetchall()


def fetch_review_queue_summary_rows(
    conn: sqlite3.Connection,
    gallery_threshold: float = 0.7,
    exclude_deep_scanned: bool = False,
) -> list[sqlite3.Row]:
    """Lean review-queue projection for the cleanup preview/run path.

    Same membership as ``fetch_review_queue_images`` (shared WHERE via
    ``_queue_where_clauses``), but selects ONLY the columns the cleanup
    action and its event/export math need: item identity, the source
    filename + timestamp, the bbox quartet and species for BirdEvent
    clustering, and the favorite flag.

    It deliberately drops the per-row correlated subqueries the render
    query carries (cls_confidence, sibling_detection_count) — on a
    spinning disk those are the dominant random-seek cost. The single
    remaining correlated lookup is the species resolution, which is
    intrinsic to the queue's meaning. No ORDER BY: callers aggregate,
    they don't render a list.
    """
    detection_columns = table_columns(conn, "detections")
    species_sql = effective_species_sql_for_columns("d", detection_columns)
    is_favorite_sql = _detection_column_sql(detection_columns, "is_favorite", "0")

    orphan_where, orphan_params, detection_where, detection_params = (
        _queue_where_clauses(
            detection_columns, exclude_deep_scanned=exclude_deep_scanned
        )
    )
    detection_params = [gallery_threshold, *detection_params]

    orphan_where_sql = " AND ".join(orphan_where)
    detection_where_sql = " AND ".join(detection_where)

    query = f"""
        SELECT
            'image' AS item_kind,
            i.filename AS item_id,
            NULL AS detection_id,
            i.filename AS filename,
            i.timestamp AS timestamp,
            NULL AS bbox_x,
            NULL AS bbox_y,
            NULL AS bbox_w,
            NULL AS bbox_h,
            NULL AS species_key,
            0 AS is_favorite
        FROM images i
        WHERE {orphan_where_sql}

        UNION ALL

        SELECT
            'detection' AS item_kind,
            CAST(d.detection_id AS TEXT) AS item_id,
            d.detection_id AS detection_id,
            i.filename AS filename,
            i.timestamp AS timestamp,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            {species_sql} AS species_key,
            COALESCE({is_favorite_sql}, 0) AS is_favorite
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {detection_where_sql}
    """
    cur = conn.execute(query, [*orphan_params, *detection_params])
    return cur.fetchall()


def fetch_review_queue_summary(
    conn: sqlite3.Connection,
    gallery_threshold: float = 0.7,
    exclude_deep_scanned: bool = False,
) -> dict[str, int]:
    """Count-only review-queue summary for the cleanup preview.

    Returns ``{"images", "detections", "favorites"}`` using the shared
    queue predicate. Pure aggregation — no rows materialised, no Python
    loop over the queue. The favorites count covers detection items only
    (orphan images carry no detection to favourite).
    """
    detection_columns = table_columns(conn, "detections")
    is_favorite_sql = _detection_column_sql(detection_columns, "is_favorite", "0")

    orphan_where, orphan_params, detection_where, detection_params = (
        _queue_where_clauses(
            detection_columns, exclude_deep_scanned=exclude_deep_scanned
        )
    )
    detection_params = [gallery_threshold, *detection_params]

    orphan_where_sql = " AND ".join(orphan_where)
    detection_where_sql = " AND ".join(detection_where)

    image_count = conn.execute(
        f"SELECT COUNT(*) FROM images i WHERE {orphan_where_sql}",
        orphan_params,
    ).fetchone()[0]

    det_row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS n,
            COALESCE(SUM(CASE WHEN COALESCE({is_favorite_sql}, 0) THEN 1 ELSE 0 END), 0) AS fav
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE {detection_where_sql}
        """,
        detection_params,
    ).fetchone()

    return {
        "images": int(image_count or 0),
        "detections": int(det_row["n"] or 0),
        "favorites": int(det_row["fav"] or 0),
    }


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


def fetch_review_queue_item_by_identity(
    conn: sqlite3.Connection,
    item_kind: str,
    item_id: str,
    gallery_threshold: float = 0.7,
) -> sqlite3.Row | None:
    """Return a single review-queue row by (item_kind, item_id).

    For ``item_kind='image'`` this filters orphan images by filename.
    For ``item_kind='detection'`` this filters detections by detection_id.
    Returns ``None`` when the item is not in the queue (already resolved, etc.).
    """
    detection_columns = table_columns(conn, "detections")
    image_columns = table_columns(conn, "images")
    original_present_sql = (
        "COALESCE(i.original_present, 1)"
        if "original_present" in image_columns
        else "1"
    )
    species_sql = effective_species_sql_for_columns("d", detection_columns)
    is_favorite_sql = _detection_column_sql(detection_columns, "is_favorite", "0")
    is_gallery_eligible_sql = _detection_column_sql(
        detection_columns, "is_gallery_eligible", "0"
    )
    manual_species_sql = _detection_column_sql(
        detection_columns, "manual_species_override"
    )
    species_source_sql = _detection_column_sql(detection_columns, "species_source")
    manual_bbox_sql = _detection_column_sql(detection_columns, "manual_bbox_review")
    ex_unclear_predicate = _ex_unclear_predicate_sql(detection_columns)
    ex_unclear_predicate_ds = _ex_unclear_predicate_sql(detection_columns, "ds")
    ex_unclear_reason_case = _ex_unclear_reason_case_sql(detection_columns)

    if item_kind == "image":
        query = f"""
        SELECT
            'image' as item_kind,
            i.filename as item_id,
            i.filename,
            i.filename as source_image_filename,
            i.timestamp,
            i.review_status,
            {original_present_sql} as original_present,
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
            NULL as cls_class_name,
            NULL as cls_confidence,
            NULL as species_key,
            NULL as manual_species_override,
            NULL as species_source,
            NULL as manual_bbox_review,
            0 as sibling_detection_count,
            0 as is_favorite,
            0 as is_gallery_eligible
        FROM images i
        WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
          AND NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
          AND i.filename = ?
        LIMIT 1
        """
        row = conn.execute(query, (item_id,)).fetchone()
    else:
        query = f"""
        SELECT
            'detection' as item_kind,
            CAST(d.detection_id AS TEXT) as item_id,
            i.filename,
            i.filename as source_image_filename,
            i.timestamp,
            i.review_status,
            {original_present_sql} as original_present,
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
                {ex_unclear_reason_case}
                ELSE 'low_score'
            END as review_reason,
            d.decision_state,
            d.bbox_quality,
            d.unknown_score,
            d.decision_reasons,
            d.od_confidence,
            (
                SELECT c.cls_class_name
                FROM classifications c
                WHERE c.detection_id = d.detection_id
                  AND c.rank = 1
                  AND COALESCE(c.status, 'active') = 'active'
                LIMIT 1
            ) as cls_class_name,
            (
                SELECT c.cls_confidence
                FROM classifications c
                WHERE c.detection_id = d.detection_id
                  AND c.rank = 1
                  AND COALESCE(c.status, 'active') = 'active'
                LIMIT 1
            ) as cls_confidence,
            {species_sql} as species_key,
            {manual_species_sql} as manual_species_override,
            {species_source_sql} as species_source,
            {manual_bbox_sql} as manual_bbox_review,
            (
                SELECT COUNT(*)
                FROM detections ds
                WHERE ds.image_filename = d.image_filename
                  AND COALESCE(ds.status, 'active') = 'active'
                  AND (
                      COALESCE(ds.decision_state, '') NOT IN ('confirmed', 'rejected')
                      OR {ex_unclear_predicate_ds}
                  )
            ) as sibling_detection_count,
            COALESCE({is_favorite_sql}, 0) as is_favorite,
            COALESCE({is_gallery_eligible_sql}, 0) as is_gallery_eligible
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE COALESCE(d.status, 'active') = 'active'
          AND (i.review_status IS NULL OR i.review_status = 'untagged')
          AND (
              (
                  COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
                  AND (
                      COALESCE(d.score, 0.0) < ?
                      OR d.decision_state IN ('uncertain', 'unknown')
                  )
              )
              OR {ex_unclear_predicate}
          )
          AND d.detection_id = ?
        LIMIT 1
        """
        row = conn.execute(query, (gallery_threshold, int(item_id))).fetchone()

    return row


def _shift_review_timestamp(ts: str, *, minutes: int) -> str:
    """Shift a ``YYYYMMDD_HHMMSS`` Review timestamp by ``minutes`` (may be negative).

    Returns the shifted timestamp in the same string format. Falls back to
    the input string when parsing fails so callers never crash on
    malformed historical rows.
    """
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return ts
    return (dt + timedelta(minutes=minutes)).strftime("%Y%m%d_%H%M%S")


def fetch_review_cluster_context(
    conn: sqlite3.Connection,
    *,
    untagged_time_range: tuple[str, str] | None,
    context_window_minutes: int = 30,
    max_context_rows: int = 200,
) -> tuple[list[dict], bool]:
    """Return confirmed-bird detections that anchor a Review cluster.

    The Review desk needs to see ``confirmed_bird`` Gallery detections
    that fall inside the same biological independence window as the
    currently loaded untagged set so the BirdEvent clusterer can place
    new uncertain frames into an existing visit instead of inventing a
    fresh, isolated event.

    Parameters
    ----------
    conn:
        Open SQLite connection. The function never commits or mutates.
    untagged_time_range:
        ``(min_timestamp, max_timestamp)`` of the currently loaded
        untagged Review rows in ``YYYYMMDD_HHMMSS`` form, or ``None`` /
        empty when there are no untagged rows. With no anchor range
        there is no neighbourhood to widen, and the function returns
        ``([], False)``.
    context_window_minutes:
        Symmetric widening (default 30 min, matching the
        ``EVENT_GAP_MINUTES_DEFAULT`` independence rule).
    max_context_rows:
        Hard safety cap. Defaults to 200. When the cap is hit the
        second tuple element is ``True`` so the Review blueprint can
        surface an operator-visible truncation hint.

    Returns
    -------
    tuple[list[dict], bool]
        A list of plain dicts shaped like ``fetch_review_queue_images``
        rows (so they can be concatenated and fed straight into
        ``core.events.build_bird_events``) plus a ``context_truncated``
        flag.

    Notes
    -----
    Every returned row carries ``context_only=True`` so the BirdEvent
    layer and the Review blueprint can distinguish read-only Gallery
    anchors from actionable untagged frames.
    """
    if not untagged_time_range:
        return [], False
    raw_min, raw_max = untagged_time_range
    if not raw_min or not raw_max:
        return [], False

    window_min = _shift_review_timestamp(raw_min, minutes=-int(context_window_minutes))
    window_max = _shift_review_timestamp(raw_max, minutes=int(context_window_minutes))

    detection_columns = table_columns(conn, "detections")
    species_sql = effective_species_sql_for_columns("d", detection_columns)
    is_favorite_sql = _detection_column_sql(detection_columns, "is_favorite", "0")
    is_gallery_eligible_sql = _detection_column_sql(
        detection_columns, "is_gallery_eligible", "0"
    )
    manual_species_sql = _detection_column_sql(
        detection_columns, "manual_species_override"
    )
    species_source_sql = _detection_column_sql(detection_columns, "species_source")
    manual_bbox_sql = _detection_column_sql(detection_columns, "manual_bbox_review")
    # Pull one extra row so we can detect when we hit the cap exactly.
    fetch_limit = int(max_context_rows) + 1

    query = f"""
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
        'context' as review_reason,
        d.decision_state,
        d.bbox_quality,
        d.unknown_score,
        d.decision_reasons,
        d.od_confidence,
        (
            SELECT c.cls_class_name
            FROM classifications c
            WHERE c.detection_id = d.detection_id
              AND c.rank = 1
              AND COALESCE(c.status, 'active') = 'active'
            LIMIT 1
        ) as cls_class_name,
        (
            SELECT c.cls_confidence
            FROM classifications c
            WHERE c.detection_id = d.detection_id
              AND c.rank = 1
              AND COALESCE(c.status, 'active') = 'active'
            LIMIT 1
        ) as cls_confidence,
        {species_sql} as species_key,
        {manual_species_sql} as manual_species_override,
        {species_source_sql} as species_source,
        {manual_bbox_sql} as manual_bbox_review,
        0 as sibling_detection_count,
        COALESCE({is_favorite_sql}, 0) as is_favorite,
        COALESCE({is_gallery_eligible_sql}, 0) as is_gallery_eligible
    FROM detections d
    JOIN images i ON i.filename = d.image_filename
    WHERE COALESCE(d.status, 'active') = 'active'
      AND i.review_status = 'confirmed_bird'
      AND i.timestamp IS NOT NULL
      AND i.timestamp >= ?
      AND i.timestamp <= ?
    ORDER BY i.timestamp ASC, d.detection_id ASC
    LIMIT ?
    """
    cur = conn.execute(query, (window_min, window_max, fetch_limit))
    rows = cur.fetchall()

    truncated = len(rows) > int(max_context_rows)
    if truncated:
        rows = rows[: int(max_context_rows)]

    context_rows: list[dict] = []
    for row in rows:
        row_dict = dict(row)
        row_dict["context_only"] = True
        context_rows.append(row_dict)
    return context_rows, truncated


def fetch_recent_review_species(
    conn: sqlite3.Connection,
    limit: int = 8,
    lookback_days: int = 7,
) -> list[sqlite3.Row]:
    """Return recently common active species for review quick-picks."""
    detection_columns = table_columns(conn, "detections")
    species_sql = effective_species_sql_for_columns("d", detection_columns)
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
    detection_columns = table_columns(conn, "detections")
    ex_unclear_predicate = _ex_unclear_predicate_sql(detection_columns)
    query = f"""
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
              AND (i.review_status IS NULL OR i.review_status = 'untagged')
              AND (
                  (
                      COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
                      AND (
                          COALESCE(d.score, 0.0) < ?
                          OR d.decision_state IN ('uncertain', 'unknown')
                      )
                  )
                  OR {ex_unclear_predicate}
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
    conn: sqlite3.Connection,
    filenames: Iterable[str] | None = None,
    delete_all: bool = False,
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
    updated_at: str | None = None,
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
        updated_at = datetime.now(UTC).isoformat()

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
