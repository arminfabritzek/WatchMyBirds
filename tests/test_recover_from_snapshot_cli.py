"""Integration tests for scripts/recover_from_snapshot.py.

Runs the CLI as a real subprocess against synthetic snapshot and
destination directories under tmp_path -- never real user data, no
mounted disks, no Pi/Docker contact.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "recover_from_snapshot.py"


def _make_snapshot(
    tmp_path: Path, name: str, filename: str, *, corrupt: bool = False
) -> Path:
    snapshot_dir = tmp_path / name
    data_dir = snapshot_dir / "data"
    output_dir = data_dir / "output"
    shard = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
    originals_dir = output_dir / "originals" / shard
    originals_dir.mkdir(parents=True)
    (originals_dir / filename).write_bytes(b"jpeg-bytes")

    db_path = data_dir / "images.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE images (filename TEXT PRIMARY KEY, timestamp TEXT, "
        "content_hash TEXT, original_present INTEGER DEFAULT 1)"
    )
    conn.execute(
        "CREATE TABLE detections (detection_id INTEGER PRIMARY KEY, "
        "image_filename TEXT, bbox_x REAL, bbox_y REAL)"
    )
    conn.execute("CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE sources (source_id INTEGER PRIMARY KEY, name TEXT, type TEXT, uri TEXT)"
    )
    conn.execute(
        "INSERT INTO images VALUES (?, ?, ?, 1)",
        (filename, "2026-09-01T09:00:00", f"hash-{filename}"),
    )
    conn.commit()
    conn.close()

    digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    if corrupt:
        digest = "0" * 64
    (data_dir / "images.db.sha256").write_text(f"{digest}  images.db\n")
    (snapshot_dir / "COMPLETED").write_text(datetime.now(UTC).isoformat())
    return snapshot_dir


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


def _image_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT filename FROM images").fetchall()
    finally:
        conn.close()


def test_migration_into_fresh_destination_preserves_records_and_media(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    destination = tmp_path / "fresh_dest"

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "images.db").is_file()
    assert _image_rows(destination / "images.db") == [(filename,)]
    shard = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
    assert (destination / "originals" / shard / filename).is_file()
    assert (destination / "originals" / shard / filename).read_bytes() == b"jpeg-bytes"


def test_migration_refuses_populated_destination(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    destination = tmp_path / "populated_dest"
    destination.mkdir()
    conn = sqlite3.connect(destination / "images.db")
    conn.execute(
        "CREATE TABLE images (filename TEXT PRIMARY KEY, timestamp TEXT, "
        "content_hash TEXT, original_present INTEGER DEFAULT 1)"
    )
    conn.execute("INSERT INTO images VALUES ('existing.jpg', 't', 'h', 1)")
    conn.commit()
    conn.close()

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode != 0
    assert "already has" in result.stderr
    # Untouched: still the original row, not the snapshot's.
    assert _image_rows(destination / "images.db") == [("existing.jpg",)]


def test_recovery_requires_force(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    destination = tmp_path / "populated_dest"
    destination.mkdir()
    conn = sqlite3.connect(destination / "images.db")
    conn.execute(
        "CREATE TABLE images (filename TEXT PRIMARY KEY, timestamp TEXT, "
        "content_hash TEXT, original_present INTEGER DEFAULT 1)"
    )
    conn.execute("INSERT INTO images VALUES ('existing.jpg', 't', 'h', 1)")
    conn.commit()
    conn.close()

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "recovery",
    )

    assert result.returncode != 0
    assert "requires --force" in result.stderr
    assert _image_rows(destination / "images.db") == [("existing.jpg",)]


def test_recovery_with_force_replaces_and_leaves_a_safety_checkpoint(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    destination = tmp_path / "populated_dest"
    destination.mkdir()
    conn = sqlite3.connect(destination / "images.db")
    conn.execute(
        "CREATE TABLE images (filename TEXT PRIMARY KEY, timestamp TEXT, "
        "content_hash TEXT, original_present INTEGER DEFAULT 1)"
    )
    conn.execute("INSERT INTO images VALUES ('existing.jpg', 't', 'h', 1)")
    conn.commit()
    conn.close()

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "recovery",
        "--force",
    )

    assert result.returncode == 0, result.stderr
    assert "Safety checkpoint written" in result.stdout
    assert _image_rows(destination / "images.db") == [(filename,)]

    rollback_dir = destination / "backup_before_restore"
    assert rollback_dir.is_dir()
    rollback_files = list(rollback_dir.iterdir())
    assert len(rollback_files) == 1
    rollback_rows = _image_rows(rollback_files[0])
    assert rollback_rows == [("existing.jpg",)]


def test_missing_completed_marker_is_refused(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    (snapshot / "COMPLETED").unlink()
    destination = tmp_path / "dest"

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode != 0
    assert "COMPLETED" in result.stderr
    assert not (destination / "images.db").exists()


def test_corrupt_checksum_is_refused(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename, corrupt=True)
    destination = tmp_path / "dest"

    result = _run_cli(
        "--snapshot",
        str(snapshot),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode != 0
    assert "Checksum mismatch" in result.stderr
    assert not (destination / "images.db").exists()


def test_missing_snapshot_directory_is_refused(tmp_path) -> None:
    destination = tmp_path / "dest"

    result = _run_cli(
        "--snapshot",
        str(tmp_path / "does_not_exist"),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode != 0
    assert "not found" in result.stderr


@pytest.mark.skipif(
    sys.platform != "darwin" and sys.platform != "linux", reason="unix symlinks only"
)
def test_latest_symlink_is_followed(tmp_path) -> None:
    filename = "20260901_090000_bird.jpg"
    snapshot = _make_snapshot(tmp_path, "snap1", filename)
    latest_link = tmp_path / "latest"
    latest_link.symlink_to(snapshot)
    destination = tmp_path / "dest"

    result = _run_cli(
        "--snapshot",
        str(latest_link),
        "--destination",
        str(destination),
        "--mode",
        "migration",
    )

    assert result.returncode == 0, result.stderr
    assert _image_rows(destination / "images.db") == [(filename,)]
