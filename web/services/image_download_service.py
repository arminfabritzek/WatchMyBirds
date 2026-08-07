"""Build ZIP archives of original frames for a set of detections.

Shared by the date-scoped Gallery edit page and the date-independent
Species view, which downloads favorites spanning many days.

One archive entry per source image: several detections on the same frame
collapse to a single original. Missing files are skipped rather than
fatal, and metadata burn-in is best-effort — a burn-in failure falls back
to the raw original instead of losing the image.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime

from config import get_config
from logging_config import get_logger
from web.services import db_service
from web.services import metadata_export_service as mx

logger = get_logger(__name__)


def _date_folder(timestamp: str) -> str:
    """``YYYYMMDD_HHMMSS`` -> ``YYYY-MM-DD``; empty when unparseable."""
    return (
        f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        if len(timestamp) >= 8
        else ""
    )


def collect_source_images(conn, detection_ids: list[int]) -> list[tuple[str, str, str]]:
    """Resolve detection ids to ``(abs_path, original_name, timestamp)``.

    Deduplicated by original filename, so N detections on one frame yield
    one entry. Rows without a filename or timestamp are dropped.
    """
    if not detection_ids:
        return []

    placeholders = ",".join("?" for _ in detection_ids)
    rows = conn.execute(
        f"""
            SELECT d.detection_id, i.filename as original_name, i.timestamp
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            WHERE d.detection_id IN ({placeholders})
        """,
        detection_ids,
    ).fetchall()

    output_dir = get_config().get("OUTPUT_DIR", "detections")
    collected: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for row in rows:
        original_name, timestamp = row["original_name"], row["timestamp"]
        if not original_name or not timestamp:
            continue
        if original_name in seen:
            continue
        seen.add(original_name)

        abs_path = os.path.join(
            output_dir, "originals", _date_folder(timestamp), original_name
        )
        collected.append((abs_path, original_name, timestamp))

    return collected


def build_zip(files: list[tuple[str, str, str]]) -> io.BytesIO:
    """Zip the resolved originals, applying metadata burn-in when enabled."""
    burn_in = mx.burn_in_enabled()
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, original_name, timestamp in files:
            if not os.path.exists(abs_path):
                continue
            if burn_in:
                try:
                    zf.writestr(
                        mx.export_filename(original_name, timestamp),
                        mx.produce_copy_bytes(original_name),
                    )
                    continue
                except Exception:
                    logger.exception(
                        "metadata burn-in failed for %s; zipping raw original",
                        original_name,
                    )
            zf.write(abs_path, arcname=original_name)

    buffer.seek(0)
    return buffer


def build_download_archive(detection_ids: list[int]) -> io.BytesIO | None:
    """Resolve ids, stamp ``downloaded_timestamp``, return the archive.

    ``None`` when nothing resolved, so callers can redirect instead of
    serving an empty ZIP.
    """
    with db_service.closing_connection() as conn:
        files = collect_source_images(conn, detection_ids)
        if not files:
            return None

        db_service.update_downloaded_timestamp(
            conn,
            [name for _, name, _ in files],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    return build_zip(files)
