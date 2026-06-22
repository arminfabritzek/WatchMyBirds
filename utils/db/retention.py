"""Artifact-retention DB queries (V1 — originals only).

Candidate selection and the per-image favourite signal. The export-
relevance predicate is NOT re-authored here — it lives in
``utils.db.user_groundtruth.is_export_relevant_any`` (the export's own
source of truth) and is consumed by the Planner via
``core.user_groundtruth_core``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

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
    ORDER BY i.timestamp ASC
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
    rows = conn.execute(_CANDIDATE_SQL, (cutoff_prefix, cutoff_prefix)).fetchall()
    return [
        {
            "filename": r["filename"],
            "timestamp": r["timestamp"],
            "review_status": r["review_status"],
            "original_present": int(r["original_present"]),
            "is_favorite": bool(r["is_favorite"]),
        }
        for r in rows
    ]


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
    placeholders = ",".join("?" for _ in filenames)
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
        filenames,
    ).fetchall()
    result: dict[str, list[str]] = {}
    for r in rows:
        result.setdefault(r["filename"], []).append(r["thumb_name"])
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
