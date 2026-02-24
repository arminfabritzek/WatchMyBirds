"""
Inbox Ingest Events (Audit Log).

This table is intentionally separate from `images` so we can record why inbox
files were skipped without polluting the gallery/review data model.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


def insert_inbox_ingest_event(
    conn: sqlite3.Connection,
    *,
    inbox_filename: str,
    status: str,
    reason: str | None = None,
    content_hash: str | None = None,
    source_id: int | None = None,
    image_filename: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Insert an inbox ingest audit event.

    Args:
        conn: SQLite connection (schema must include `inbox_ingest_events`).
        inbox_filename: Original filename as seen in inbox/pending.
        status: 'ingested' | 'skipped' | 'error' (free-form but keep consistent).
        reason: Optional short reason (e.g. 'missing_exif_datetime', 'missing_exif_gps').
        content_hash: Optional SHA-256 of the original file.
        source_id: Optional source_id (usually "User Import").
        image_filename: Optional final stored filename in `images` (if ingested).
        details: Optional dict with additional structured context (stored as JSON).
    """
    details_json = json.dumps(details or {}, sort_keys=True)
    created_at = datetime.now(UTC).isoformat()

    conn.execute(
        """
        INSERT INTO inbox_ingest_events (
            created_at,
            inbox_filename,
            content_hash,
            status,
            reason,
            source_id,
            image_filename,
            details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            created_at,
            inbox_filename,
            content_hash,
            status,
            reason,
            source_id,
            image_filename,
            details_json,
        ),
    )
    conn.commit()
