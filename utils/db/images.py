"""
Image CRUD Operations.

This module handles image-related database operations.
"""

import sqlite3
from typing import Any


def insert_image(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images (
            filename,
            timestamp,
            coco_json,
            downloaded_timestamp,
            detector_model_id,
            classifier_model_id,
            source_id,
            content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("filename"),
            row.get("timestamp"),
            row.get("coco_json"),
            row.get("downloaded_timestamp", ""),
            row.get("detector_model_id", ""),
            row.get("classifier_model_id", ""),
            row.get("source_id"),
            row.get("content_hash"),
        ),
    )
    conn.commit()


def check_image_exists_by_hash(conn: sqlite3.Connection, content_hash: str) -> bool:
    """Checks if an image with the given SHA-256 hash already exists."""
    if not content_hash:
        return False
    row = conn.execute(
        "SELECT 1 FROM images WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row is not None


def update_downloaded_timestamp(
    conn: sqlite3.Connection, filenames: list[str], download_ts: str
) -> None:
    names = list(filenames)
    if not names:
        return
    placeholders = ",".join("?" for _ in names)
    params = [download_ts] + names
    conn.execute(
        f"""
        UPDATE images
        SET downloaded_timestamp = ?
        WHERE filename IN ({placeholders});
        """,
        params,
    )
    conn.commit()
