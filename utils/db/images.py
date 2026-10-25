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
            content_hash,
            ptz_origin,
            ptz_preset_token,
            ptz_zone,
            ptz_state,
            ptz_camera_id,
            ptz_pan,
            ptz_tilt,
            ptz_zoom,
            ptz_position_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            row.get("ptz_origin"),
            row.get("ptz_preset_token"),
            row.get("ptz_zone"),
            row.get("ptz_state"),
            row.get("ptz_camera_id"),
            row.get("ptz_pan"),
            row.get("ptz_tilt"),
            row.get("ptz_zoom"),
            row.get("ptz_position_at"),
        ),
    )
    conn.commit()


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
