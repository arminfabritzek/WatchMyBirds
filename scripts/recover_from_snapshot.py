#!/usr/bin/env python3
"""Recover or migrate a WatchMyBirds installation from a USB snapshot directory.

Consumes the directory layout `rpi/backup.sh` writes under
`/mnt/wmb-backup/snapshots/<stamp>_<kind>/` (also usable straight from a
mounted or copied stick, no archive step required). Two modes:

  migration  (default) -- destination database has no rows yet. The
             common "SD card died, flash a new image, plug the stick,
             run this" path. Refuses if the destination already has data.

  recovery   -- explicit, deliberate replacement of a populated
             destination. Requires --force. Retains the complete previous
             output directory as a checkpoint before publishing verified data.

This script never starts or stops any service, and never touches a
process. Stop the app (RPi: `sudo systemctl stop app.service`; Docker:
`docker compose stop app`) before running it in either mode, and
start it again afterward. Running it against a live, connected database
is unsupported and can corrupt data -- see docs/USB_BACKUP.md.

Usage:
    .venv/bin/python scripts/recover_from_snapshot.py \\
        --snapshot /mnt/wmb-backup/snapshots/20260901_030000_scheduled \\
        --destination /opt/app/data/output \\
        --mode migration --app-stopped

    .venv/bin/python scripts/recover_from_snapshot.py \\
        --snapshot /mnt/wmb-backup/latest \\
        --destination /opt/app/data/output \\
        --mode recovery --force --app-stopped
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from itertools import chain
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
    from core.usb_backup_core import verify_snapshot_directory

    result = verify_snapshot_directory(snapshot_dir)
    if result["ok"]:
        return []
    return [
        str(value)
        for value in (result.get("error"), result.get("media_message"))
        if value
    ]


def _destination_has_data(db_path: Path) -> int:
    """Return the row count in `images`, or 0 if the DB doesn't exist/is empty."""
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return 0
    try:
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
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


def _restore_directory(snapshot_dir: Path, destination: Path) -> Path | None:
    """Stage all output state before swapping directories while the app is stopped.

    The previous directory is retained as a complete recovery checkpoint.
    If publication fails, put it back before propagating the error.
    """
    if destination.is_mount():
        raise ValueError(
            "Destination is a mount point; run recovery on its host directory"
        )
    if (
        destination == snapshot_dir
        or destination in snapshot_dir.parents
        or snapshot_dir in destination.parents
    ):
        raise ValueError("Snapshot and destination must not overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-restore-", dir=destination.parent)
    )
    checkpoint = None
    try:
        source = snapshot_dir / "data" / "output"
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Snapshot output contains symbolic links")
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            (staging / f"images.db{suffix}").unlink(missing_ok=True)
        shutil.copy2(snapshot_dir / "data" / "images.db", staging / "images.db")
        from core.usb_backup_core import _verify_media

        result = _verify_media(snapshot_dir, staging / "images.db", output_dir=staging)
        if not result["media_ok"]:
            raise ValueError(result["media_message"])
        if destination.exists():
            owner = destination.stat()
            shutil.copystat(destination, staging)
            if os.geteuid() == 0:
                for item in chain([staging], staging.rglob("*")):
                    os.chown(item, owner.st_uid, owner.st_gid)
            checkpoint = Path(
                tempfile.mkdtemp(
                    prefix=f"{destination.name}-before-restore-", dir=destination.parent
                )
            )
            checkpoint.rmdir()
            destination.rename(checkpoint)
            print(f"Safety checkpoint written: {checkpoint}", flush=True)
        try:
            staging.rename(destination)
        except BaseException:
            if checkpoint is not None:
                checkpoint.rename(destination)
            raise
        return checkpoint
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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
        "--app-stopped",
        action="store_true",
        help="Confirm all app processes using the destination are stopped (required in both modes).",
    )
    args = parser.parse_args()

    if not args.app_stopped:
        parser.error("Stop all app processes first, then pass --app-stopped")

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
    destination_db = destination / "images.db"

    existing_rows = _destination_has_data(destination_db)
    from utils.path_manager import PathManager

    originals = PathManager(str(destination)).originals_dir
    if (
        existing_rows == 0
        and originals.exists()
        and any(p.is_file() for p in originals.rglob("*"))
    ):
        existing_rows = 1
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

    try:
        rollback_path = _restore_directory(snapshot_dir, destination)
    except (OSError, ValueError) as exc:
        print(f"error: recovery failed: {exc}", file=sys.stderr)
        return 5

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
