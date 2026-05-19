"""
Unclear bucket — SQL access layer.

The "Unclear" bucket holds detections that the temporal smoother accepted
as a bird event (``decision_state='confirmed'``) but the classifier
rejected for lacking species- or genus-threshold confidence
(``decision_level='reject'``).

The Gallery visibility filter in :mod:`utils.db.detections` excludes any
detection where ``decision_level='reject'``, so these rows are otherwise
invisible to the user. This module exposes the queries and mutations
that back the dedicated "Unclear" surface, where the operator can
either:

- **Confirm a day** — promote every reject detection of that day to a
  species-confirmed row using the classifier's own ``raw_species_name``
  as the manual override. The rows then surface in Gallery as
  human-confirmed species sightings.
- **Discard a day** — reject (soft-delete) every reject detection of
  that day. The rows then surface in Trash for the regular hard-delete
  pipeline.

Both operations restrict themselves to ``status='active'`` rows so a
prior soft-delete is never silently undone.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime


def fetch_unclear_days(
    conn: sqlite3.Connection,
    sample_limit: int = 9,
) -> list[dict]:
    """Return Unclear detections grouped by day, newest day first.

    Each entry has::

        {
            "day": "2026-05-19",                # ISO date (YYYY-MM-DD)
            "total_count": 696,
            "species_breakdown": [
                {"raw_species_name": "Parus_major", "count": 312},
                ...
            ],
            "samples": [
                {
                    "detection_id": 27108,
                    "thumbnail_path_virtual": "20260519_..._crop_1.webp",
                    "raw_species_name": "Parus_major",
                },
                ...
            ],
        }

    ``sample_limit`` caps the number of preview thumbnails per day. The
    species_breakdown is unbounded — typical days have 2-5 entries.
    """
    # Day list with totals
    day_rows = conn.execute(
        """
        SELECT
            substr(d.image_filename, 1, 4) || '-'
                || substr(d.image_filename, 5, 2) || '-'
                || substr(d.image_filename, 7, 2)            AS day,
            COUNT(*)                                          AS total_count
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND lower(COALESCE(d.decision_level, '')) IN ('reject', 'species_review')
          AND (i.review_status IS NULL OR i.review_status = 'untagged')
        GROUP BY day
        ORDER BY day DESC
        """
    ).fetchall()

    if not day_rows:
        return []

    days: list[dict] = []
    for day_row in day_rows:
        day = day_row["day"]
        date_prefix = day.replace("-", "")  # YYYYMMDD

        # Species breakdown for this day
        species_rows = conn.execute(
            """
            SELECT
                COALESCE(d.raw_species_name, '') AS raw_species_name,
                COUNT(*)                          AS count
            FROM detections d
            JOIN images i ON i.filename = d.image_filename
            WHERE d.status = 'active'
              AND lower(COALESCE(d.decision_level, '')) IN ('reject', 'species_review')
              AND (i.review_status IS NULL OR i.review_status = 'untagged')
              AND substr(d.image_filename, 1, 8) = ?
            GROUP BY raw_species_name
            ORDER BY count DESC
            """,
            (date_prefix,),
        ).fetchall()

        # Sample thumbnails for this day. The thumb files live under
        # ``derivatives/thumbs/<YYYY-MM-DD>/<filename>`` on disk (date-
        # sharded). The /uploads/derivatives/thumbs/<path:filename>
        # static route accepts the date-prefixed form directly; that
        # avoids the slow ``regenerate_derivative`` fallback Trash
        # uses when only the bare filename is served.
        sample_rows = conn.execute(
            """
            SELECT
                d.detection_id,
                ? || '/' || COALESCE(
                    NULLIF(d.thumbnail_path, ''),
                    REPLACE(i.filename, '.jpg', '_crop_1.webp')
                ) AS thumbnail_path_virtual,
                COALESCE(d.raw_species_name, '') AS raw_species_name
            FROM detections d
            JOIN images i ON i.filename = d.image_filename
            WHERE d.status = 'active'
              AND lower(COALESCE(d.decision_level, '')) IN ('reject', 'species_review')
              AND (i.review_status IS NULL OR i.review_status = 'untagged')
              AND substr(d.image_filename, 1, 8) = ?
            ORDER BY d.detection_id DESC
            LIMIT ?
            """,
            (day, date_prefix, sample_limit),
        ).fetchall()

        days.append(
            {
                "day": day,
                "total_count": int(day_row["total_count"]),
                "species_breakdown": [
                    {
                        "raw_species_name": row["raw_species_name"] or "unknown",
                        "count": int(row["count"]),
                    }
                    for row in species_rows
                ],
                "samples": [
                    {
                        "detection_id": int(row["detection_id"]),
                        "thumbnail_path_virtual": row["thumbnail_path_virtual"],
                        "raw_species_name": row["raw_species_name"] or "unknown",
                    }
                    for row in sample_rows
                ],
            }
        )

    return days


def fetch_unclear_total(conn: sqlite3.Connection) -> int:
    """Total number of Unclear detections across all days."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND lower(COALESCE(d.decision_level, '')) IN ('reject', 'species_review')
          AND (i.review_status IS NULL OR i.review_status = 'untagged')
        """
    ).fetchone()
    return int(row["n"]) if row else 0


def fetch_unclear_detection_ids_for_day(
    conn: sqlite3.Connection,
    day: str,
) -> list[int]:
    """Return all Unclear detection IDs for one ISO date (YYYY-MM-DD).

    Used by the day-level bulk actions. Includes a defensive guard
    against malformed input so an attacker can't smuggle wildcards
    through the substring match.
    """
    if not _is_iso_day(day):
        return []
    date_prefix = day.replace("-", "")
    rows = conn.execute(
        """
        SELECT d.detection_id
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.status = 'active'
          AND lower(COALESCE(d.decision_level, '')) IN ('reject', 'species_review')
          AND (i.review_status IS NULL OR i.review_status = 'untagged')
          AND substr(d.image_filename, 1, 8) = ?
        """,
        (date_prefix,),
    ).fetchall()
    return [int(row["detection_id"]) for row in rows]


def confirm_unclear_detections(
    conn: sqlite3.Connection,
    detection_ids: Iterable[int],
    source: str = "manual_bulk_confirm",
) -> int:
    """Promote Unclear detections to species-confirmed Gallery rows.

    For each row the classifier's own ``raw_species_name`` is used as
    the ``manual_species_override``. The visibility flips happen in one
    UPDATE so a partial state (e.g. ``decision_level='species'`` but
    ``decision_state`` still ``reject``) cannot leak.

    Restricts to ``status='active'`` rows so already-trashed detections
    are not silently revived. Returns the number of rows updated.
    """
    ids = [int(d) for d in detection_ids]
    if not ids:
        return 0

    placeholders = ",".join("?" for _ in ids)
    now_iso = datetime.now(UTC).isoformat()
    cur = conn.execute(
        f"""
        UPDATE detections
        SET manual_species_override = COALESCE(
                NULLIF(raw_species_name, ''),
                manual_species_override
            ),
            species_source         = ?,
            species_updated_at     = ?,
            decision_state         = 'confirmed',
            decision_level         = 'species'
        WHERE detection_id IN ({placeholders})
          AND status = 'active'
          AND lower(COALESCE(decision_level, '')) IN ('reject', 'species_review')
        """,
        [source, now_iso, *ids],
    )
    conn.commit()
    return cur.rowcount or 0


def _is_iso_day(value: str) -> bool:
    """Return True iff value matches YYYY-MM-DD with plausible ranges."""
    if not isinstance(value, str) or len(value) != 10:
        return False
    if value[4] != "-" or value[7] != "-":
        return False
    try:
        year = int(value[0:4])
        month = int(value[5:7])
        day = int(value[8:10])
    except ValueError:
        return False
    return 2000 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31
