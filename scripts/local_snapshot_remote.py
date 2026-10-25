"""Private SQLite backup and media manifest for the local development pull.

Executed over SSH with python's stdin; never installs code or stops the appliance.
Only a generated /tmp directory is written. Credentials/settings are not exported.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(output: Path, destination: Path) -> dict:
    """Capture SQLite online, then inventory only bird-media directories."""
    database = destination / "images.db"
    with closing(
        sqlite3.connect((output / "images.db").as_uri() + "?mode=ro", uri=True)
    ) as src:
        with closing(sqlite3.connect(database)) as dst:
            src.backup(dst, pages=256, sleep=0.05)
    os.chmod(database, 0o600)
    captured_at = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("source database integrity check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("source database has foreign-key errors")
        counts = {
            name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            for name in ("images", "detections", "human_label_facts")
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
        }
        expected: set[str] = set()
        source_missing: list[str] = []
        columns = {r[1] for r in conn.execute("PRAGMA table_info(images)")}
        present = "original_present" if "original_present" in columns else "1"
        for name, original_present in conn.execute(
            f"SELECT filename, {present} FROM images"
        ):
            filename = Path(name)
            if filename.name != name or len(name) < 8 or not name[:8].isdigit():
                raise ValueError("unsafe or unsupported image filename")
            date = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
            if original_present:
                expected.add(f"originals/{date}/{name}")
            expected.add(f"derivatives/optimized/{date}/{filename.stem}.webp")
        for original, thumbnail in conn.execute(
            "SELECT image_filename, thumbnail_path FROM detections"
        ):
            thumbnail = thumbnail or original.replace(".jpg", "_crop_1.webp")
            # Same date-prefixed path contract used by the gallery queries.
            if "/" not in thumbnail:
                date = f"{original[:4]}-{original[4:6]}-{original[6:8]}"
                thumbnail = f"{date}/{thumbnail}"
            expected.add(f"derivatives/thumbs/{thumbnail}")
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
    files: dict[str, dict] = {}
    for relative_root in ("originals", "derivatives/thumbs", "derivatives/optimized"):
        base = output / relative_root
        if base.is_symlink():
            raise ValueError("media roots must not be symlinks")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ValueError("media symlinks are not exported")
            if not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            # Keep unrelated files and potentially private metadata out of the pull.
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            before = path.stat()
            sha256 = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("source media changed while being hashed; retry pull")
            files[relative] = {"size": after.st_size, "sha256": sha256}
    for name in sorted(expected):
        parts = Path(name).parts
        if ".." in parts or Path(name).is_absolute():
            raise ValueError("unsafe database media reference")
        if name not in files:
            source_missing.append(name)
    version = output.parent.parent / "APP_VERSION"
    return {
        "format": 1,
        "captured_at_utc": captured_at,
        "manifest_completed_at_utc": datetime.now(UTC).isoformat(),
        "source_output": str(output),
        "source_version": version.read_text().strip()
        if version.is_file()
        else "unknown",
        "schema_user_version": schema_version,
        "counts": counts,
        "database_sha256": digest(database),
        "files": files,
        "source_missing_media": source_missing,
    }


if __name__ == "__main__":
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix="wmb-local-pull-"))
    # Report ownership before work so the caller can clean up even on failure.
    print(root, flush=True)
    manifest = collect(Path(sys.argv[1]).resolve(), root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Snapshot and media manifest ready", flush=True)
