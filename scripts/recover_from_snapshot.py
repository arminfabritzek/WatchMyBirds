#!/usr/bin/env python3
"""Recover or migrate a WatchMyBirds installation from a USB snapshot directory.

Consumes the directory layout `rpi/backup.sh` writes under
`/mnt/wmb-backup/snapshots/<stamp>_<kind>/` (also usable straight from a
mounted or copied stick, no archive step required). Two modes:

  migration  (default) -- destination database has no rows yet. The
             common "SD card died, flash a new image, plug the stick,
             run this" path. Refuses if the destination already has data.

  recovery   -- explicit, deliberate replacement of a populated
             destination. Requires --force. Takes a pre-restore backup
             of the current database before touching anything, and
             that backup path is always printed so it can be found again.

This script never starts or stops any service, and never touches a
process. Stop the app (RPi: `sudo systemctl stop app.service`; Docker:
`docker compose stop app`) before running it in recovery mode, and
start it again afterward. Running it against a live, connected database
is unsupported and can corrupt data -- see docs/USB_BACKUP.md.

Usage:
    .venv/bin/python scripts/recover_from_snapshot.py \\
        --snapshot /mnt/wmb-backup/snapshots/20260901_030000_scheduled \\
        --destination /opt/app/data \\
        --mode migration

    .venv/bin/python scripts/recover_from_snapshot.py \\
        --snapshot /mnt/wmb-backup/latest \\
        --destination /opt/app/data \\
        --mode recovery --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_snapshot_dir(raw: Path) -> Path:
    """Follow a 'latest' symlink; otherwise use the path as given."""
    resolved = raw.resolve()
    if resolved.is_symlink() or raw.is_symlink():
        resolved = raw.resolve(strict=True)
    return resolved


def _verify_snapshot(snapshot_dir: Path) -> list[str]:
    """Pre-flight checks shared with core.usb_backup_core.verify_snapshot's
    logic, reimplemented standalone here since this script must run
    without a configured OUTPUT_DIR (it's about to set one).
    """
    problems: list[str] = []

    if not (snapshot_dir / "COMPLETED").is_file():
        problems.append(
            f"No COMPLETED marker in {snapshot_dir} -- refusing a partial/crashed snapshot."
        )
        return problems
    if (snapshot_dir / "CORRUPT").is_file():
        problems.append(
            f"{snapshot_dir} is flagged CORRUPT by backup.sh. Pick a different snapshot."
        )

    db_path = snapshot_dir / "data" / "images.db"
    if not db_path.is_file():
        problems.append(f"No database found at {db_path}.")
        return problems

    sha_path = db_path.with_suffix(".db.sha256")
    if sha_path.is_file():
        result = subprocess.run(
            ["sha256sum", "-c", "--quiet", sha_path.name],
            cwd=db_path.parent,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            problems.append(
                f"Checksum mismatch on {db_path}: "
                f"{(result.stderr or result.stdout).strip()}"
            )

    integrity = subprocess.run(
        ["sqlite3", str(db_path), "pragma integrity_check;"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (integrity.stdout or "").strip()
    if integrity.returncode != 0 or output != "ok":
        problems.append(
            f"Database integrity_check failed: {output or integrity.stderr}"
        )

    output_dir = snapshot_dir / "data" / "output"
    if not output_dir.is_dir():
        problems.append(f"No media directory found at {output_dir}.")

    return problems


def _destination_has_data(db_path: Path) -> int:
    """Return the row count in `images`, or 0 if the DB doesn't exist/is empty."""
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT COALESCE((SELECT COUNT(*) FROM images), 0)"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        # Unreadable/corrupt destination DB counts as "has data" -- never
        # silently overwrite something we can't prove is empty.
        return 1


def _copy_media(snapshot_output_dir: Path, destination_output_dir: Path) -> None:
    destination_output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("originals", "derivatives"):
        src = snapshot_output_dir / name
        if not src.is_dir():
            continue
        dst = destination_output_dir / name
        subprocess.run(
            ["rsync", "-a", f"{src}/", f"{dst}/"],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="USB snapshot directory to recover from.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Target OUTPUT_DIR (contains images.db, originals/, derivatives/, settings.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=("migration", "recovery"),
        default="migration",
        help="migration (default): refuse a populated destination. "
        "recovery: deliberately replace one (requires --force).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required for --mode recovery. Has no effect in migration mode.",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Restore the database only; skip copying originals/derivatives.",
    )
    args = parser.parse_args()

    snapshot_dir = _resolve_snapshot_dir(args.snapshot)
    if not snapshot_dir.is_dir():
        print(f"error: snapshot directory not found: {snapshot_dir}", file=sys.stderr)
        return 1

    problems = _verify_snapshot(snapshot_dir)
    if problems:
        print("error: snapshot failed verification:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    destination_db = destination / "images.db"

    existing_rows = _destination_has_data(destination_db)
    if args.mode == "migration" and existing_rows > 0:
        print(
            f"error: destination database at {destination_db} already has "
            f"data (>= {existing_rows} row(s) in images). Migration mode "
            "refuses to touch a populated destination. Use "
            "--mode recovery --force if this is deliberate.",
            file=sys.stderr,
        )
        return 3
    if args.mode == "recovery" and not args.force:
        print(
            "error: --mode recovery requires --force. This will replace "
            f"the database at {destination_db}. Make sure app.service / "
            "the app container is stopped first.",
            file=sys.stderr,
        )
        return 3

    # config._CONFIG loads from os.environ on first get_config() call in
    # this process, so set it before anything imports config.
    os.environ["OUTPUT_DIR"] = str(destination)

    from utils.restore import _validate_db_schema

    snapshot_db = snapshot_dir / "data" / "images.db"
    is_valid, issues = _validate_db_schema(snapshot_db)
    if not is_valid:
        print("error: snapshot database failed schema validation:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 4

    rollback_path: Path | None = None
    if args.mode == "recovery" and existing_rows > 0:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        rollback_dir = destination / "backup_before_restore"
        rollback_dir.mkdir(parents=True, exist_ok=True)
        rollback_path = rollback_dir / f"images_{timestamp}.db"
        source_conn = sqlite3.connect(str(destination_db))
        dest_conn = sqlite3.connect(str(rollback_path))
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            source_conn.close()
        print(f"Safety checkpoint written: {rollback_path}")

    print(f"Restoring database from {snapshot_db} to {destination_db} ...")
    destination_db.parent.mkdir(parents=True, exist_ok=True)
    temp_new = destination_db.with_suffix(".db.new")
    shutil.copy2(snapshot_db, temp_new)
    for suffix in ("-wal", "-shm"):
        stale = destination_db.with_name(destination_db.name + suffix)
        stale.unlink(missing_ok=True)
    temp_new.rename(destination_db)

    if not args.skip_media:
        print("Copying media (originals/derivatives) ...")
        _copy_media(snapshot_dir / "data" / "output", destination)

    settings_src = snapshot_dir / "data" / "output" / "settings.yaml"
    if settings_src.is_file():
        settings_dst = destination / "settings.yaml"
        if args.mode == "migration" or args.force:
            shutil.copy2(settings_src, settings_dst)
            print(f"Copied settings.yaml to {settings_dst}")
        else:
            print(
                "Skipped settings.yaml (pass --force to include it in recovery mode)."
            )

    print("Recovery complete.")
    print(f"  Database: {destination_db}")
    if rollback_path is not None:
        print(f"  Pre-restore backup kept at: {rollback_path}")
    print(
        "Restart the app (systemd: `sudo systemctl start app.service`; "
        "Docker: `docker compose start app`) to pick up the restored data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
