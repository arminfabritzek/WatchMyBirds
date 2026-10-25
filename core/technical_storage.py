"""Fixed retention for disposable diagnostics, independent of photo settings."""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from utils.path_manager import PathManager

logger = logging.getLogger(__name__)
SAMPLE_DAYS = 90
SAMPLE_LIMITS = {
    "station_runtime_samples": ("sample_id", "sampled_at", 129_600),
    "weather_logs": ("id", "timestamp", 4_320),
}
TEMPORARY_MAX_AGE_SECONDS = 2 * 86400
LOG_MAX_AGE_SECONDS = 14 * 86400
LOG_FILE_BYTES = 5 * 1024**2
LOG_BACKUPS = 3


def prune_samples(
    conn: sqlite3.Connection, table: str, now: datetime | None = None
) -> int:
    key, timestamp, maximum = SAMPLE_LIMITS[table]
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=SAMPLE_DAYS)).isoformat()
    cursor = conn.execute(
        f"DELETE FROM {table} WHERE julianday({timestamp}) < julianday(?) "
        f"OR {key} <= (SELECT COALESCE(MAX({key}),0)-? FROM {table})",
        (cutoff, maximum),
    )
    return cursor.rowcount


def _unlink_owned(paths: PathManager, path: Path) -> None:
    relative = path.relative_to(paths.base_dir)
    checked = paths.contained_path(path, paths.base_dir)
    if checked is None or checked != paths.base_dir.resolve() / relative:
        raise ValueError("Unsafe diagnostic path")
    descriptor = os.open(paths.base_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        info = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Not a regular diagnostic file")
        os.unlink(relative.name, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def prune_temporary_files(paths: PathManager, now: float | None = None) -> int:
    now = time.time() if now is None else now
    removed = 0
    for path, kind in paths.disposable_files():
        age = LOG_MAX_AGE_SECONDS if kind == "log" else TEMPORARY_MAX_AGE_SECONDS
        try:
            if now - path.stat().st_mtime >= age:
                _unlink_owned(paths, path)
                removed += 1
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("Could not remove stale diagnostic file %s", path)
    return removed
