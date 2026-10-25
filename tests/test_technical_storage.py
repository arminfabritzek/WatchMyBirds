from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.technical_storage import SAMPLE_LIMITS, prune_samples, prune_temporary_files
from utils.db.connection import _retire_reject_audit
from utils.path_manager import PathManager


@pytest.mark.parametrize("table", list(SAMPLE_LIMITS))
def test_samples_have_age_and_count_bounds(table: str) -> None:
    key, timestamp, maximum = SAMPLE_LIMITS[table]
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE TABLE {table} ({key} INTEGER PRIMARY KEY, {timestamp} TEXT)")
    now = datetime.now(UTC)
    conn.executemany(
        f"INSERT INTO {table} VALUES (?,?)",
        [
            (1, (now - timedelta(days=91)).isoformat()),
            (2, now.isoformat()),
            (maximum + 3, now.isoformat()),
        ],
    )
    assert prune_samples(conn, table, now) == 2
    assert conn.execute(f"SELECT {key} FROM {table}").fetchall() == [(maximum + 3,)]


def test_disposable_cleanup_keeps_recent_files_media_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    now = datetime.now(UTC).timestamp()
    old = root / "restore_tmp" / "old.zip"
    recent = root / "restore_tmp" / "recent.zip"
    old.parent.mkdir()
    for path in (old, recent):
        path.write_bytes(b"temporary")
    os.utime(old, (now - 3 * 86400,) * 2)
    media = root / "originals" / "photo.jpg"
    media.parent.mkdir()
    media.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    os.utime(outside, (now - 3 * 86400,) * 2)
    (old.parent / "link.zip").symlink_to(outside)
    assert prune_temporary_files(PathManager(str(root)), now) == 1
    assert not old.exists()
    assert recent.exists() and media.exists() and outside.read_bytes() == b"untouched"


def test_reject_notes_are_retired_once_without_dropping_table() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE _migration_state(key TEXT PRIMARY KEY, applied_at TEXT)")
    conn.execute("CREATE TABLE reject_audit(audit_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO reject_audit VALUES (1)")
    conn.commit()
    _retire_reject_audit(conn)
    assert conn.execute("SELECT COUNT(*) FROM reject_audit").fetchone()[0] == 0
    conn.execute("INSERT INTO reject_audit VALUES (2)")
    conn.commit()
    _retire_reject_audit(conn)
    assert conn.execute("SELECT COUNT(*) FROM reject_audit").fetchone()[0] == 1
