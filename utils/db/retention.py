"""Artifact-retention DB queries (V1 — originals only).

Candidate selection and the per-image favourite signal. The export-
relevance predicate is NOT re-authored here — it lives in
``utils.db.user_groundtruth.is_export_relevant_any`` (the export's own
source of truth) and is consumed by the Planner via
``core.user_groundtruth_core``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

_QUERY_BATCH_SIZE = 500

# is_favorite means "any active detection on this image is a manual
# favourite". The present+cutoff pre-filters keep this O(candidates) rather
# than a full scan; the Planner re-checks exact age, so a day-prefix is safe.
_CANDIDATE_SQL = """
    SELECT
        i.filename AS filename,
        i.timestamp AS timestamp,
        i.review_status AS review_status,
        COALESCE(i.original_present, 1) AS original_present,
        EXISTS (
            SELECT 1 FROM detections d
            WHERE d.image_filename = i.filename
              AND d.status = 'active'
              AND d.is_favorite = 1
        ) AS is_favorite
    FROM images i
    WHERE COALESCE(i.original_present, 1) = 1
      AND (? IS NULL OR i.timestamp < ?)
    ORDER BY i.timestamp ASC, i.filename ASC
"""


def iter_candidate_images(
    conn: sqlite3.Connection,
    cutoff_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Images that could be retention-deletable.

    Pre-filters to present-on-disk originals captured before
    ``cutoff_prefix`` (a ``YYYYMMDD`` timestamp prefix). Age and
    derivative-presence are resolved by the Planner (they need "now" and
    the filesystem respectively); this query is pure DB.
    """
    return [
        row
        for batch in iter_candidate_image_batches(conn, cutoff_prefix=cutoff_prefix)
        for row in batch
    ]


def iter_candidate_image_batches(
    conn: sqlite3.Connection,
    cutoff_prefix: str | None = None,
    *,
    batch_size: int = _QUERY_BATCH_SIZE,
) -> Iterator[list[dict[str, Any]]]:
    """Yield candidate rows with bounded keyset-paginated queries."""
    cursor_timestamp: str | None = None
    cursor_filename: str | None = None
    while True:
        cursor_sql = ""
        params: list[Any] = [cutoff_prefix, cutoff_prefix]
        if cursor_timestamp is not None and cursor_filename is not None:
            cursor_sql = "AND (i.timestamp > ? OR (i.timestamp = ? AND i.filename > ?))"
            params.extend([cursor_timestamp, cursor_timestamp, cursor_filename])
        params.append(max(1, int(batch_size)))
        rows = conn.execute(
            _CANDIDATE_SQL.replace(
                "ORDER BY i.timestamp ASC, i.filename ASC",
                f"{cursor_sql} ORDER BY i.timestamp ASC, i.filename ASC LIMIT ?",
            ),
            params,
        ).fetchall()
        if not rows:
            return
        batch = [
            {
                "filename": r["filename"],
                "timestamp": r["timestamp"],
                "review_status": r["review_status"],
                "original_present": int(r["original_present"]),
                "is_favorite": bool(r["is_favorite"]),
            }
            for r in rows
        ]
        yield batch
        cursor_timestamp = str(rows[-1]["timestamp"])
        cursor_filename = str(rows[-1]["filename"])


def candidate_image(
    conn: sqlite3.Connection,
    filename: str,
) -> dict[str, Any] | None:
    """Return current retention facts for one image, including favourite state."""
    row = conn.execute(
        """
        SELECT i.filename, i.timestamp, i.review_status,
               COALESCE(i.original_present, 1) AS original_present,
               EXISTS (
                   SELECT 1 FROM detections d
                   WHERE d.image_filename = i.filename
                     AND d.status = 'active'
                     AND d.is_favorite = 1
               ) AS is_favorite
          FROM images i
         WHERE i.filename = ?
        """,
        (filename,),
    ).fetchone()
    if row is None:
        return None
    return {
        "filename": row["filename"],
        "timestamp": row["timestamp"],
        "review_status": row["review_status"],
        "original_present": int(row["original_present"]),
        "is_favorite": bool(row["is_favorite"]),
    }


def count_candidate_images(
    conn: sqlite3.Connection,
    cutoff_prefix: str | None = None,
) -> int:
    """Count present originals before the coarse retention cutoff."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
          FROM images i
         WHERE COALESCE(i.original_present, 1) = 1
           AND (? IS NULL OR i.timestamp < ?)
        """,
        (cutoff_prefix, cutoff_prefix),
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def count_present_images(conn: sqlite3.Connection) -> int:
    """Count every image whose original is still marked present."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM images WHERE COALESCE(original_present, 1) = 1"
    ).fetchone()
    return int(row["n"] if row is not None else 0)


def _variable_batch_size(
    conn: sqlite3.Connection,
    *,
    fixed_params: int = 0,
) -> int:
    """Stay below SQLite's per-statement variable limit with headroom."""
    try:
        limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    except AttributeError:
        limit = 999
    return max(1, min(_QUERY_BATCH_SIZE, int(limit) - fixed_params))


def thumbnail_names_for_images(
    conn: sqlite3.Connection,
    filenames: list[str],
) -> dict[str, list[str]]:
    """Canonical thumbnail filenames per image, for the given filenames.

    Mirrors the gallery's ``thumbnail_path_virtual`` coalesce
    (``d.thumbnail_path`` else ``<stem>_crop_1.webp``) so retention checks
    the same thumb names the app actually serves. Only active detections
    contribute. Images with no active detection are absent from the result
    (the Planner falls back to the preview thumb for those).
    """
    if not filenames:
        return {}
    result: dict[str, list[str]] = {}
    batch_size = _variable_batch_size(conn)
    for start in range(0, len(filenames), batch_size):
        batch = filenames[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT
                d.image_filename AS filename,
                COALESCE(
                    NULLIF(d.thumbnail_path, ''),
                    REPLACE(d.image_filename, '.jpg', '_crop_1.webp')
                ) AS thumb_name
            FROM detections d
            WHERE d.status = 'active'
              AND d.image_filename IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in rows:
            result.setdefault(row["filename"], []).append(row["thumb_name"])
    return result


def mark_original_deleted(
    conn: sqlite3.Connection,
    filename: str,
    deleted_at: str,
) -> None:
    """Record that an original was removed: presence 0 + timestamp.

    Metadata-only operation; never touches the file. Caller commits.
    """
    conn.execute(
        "UPDATE images SET original_present = 0, original_deleted_at = ? "
        "WHERE filename = ?",
        (deleted_at, filename),
    )
