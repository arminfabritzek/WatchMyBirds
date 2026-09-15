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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from core import recovery_core

# Compatibility aliases for callers/tests that imported the old private helpers.
# The implementation itself lives only in core.recovery_core.
shutil = recovery_core.shutil


def _resolve_snapshot_dir(raw: Path) -> Path:
    return recovery_core.resolve_snapshot_directory(raw)


def _restore_directory(snapshot_dir: Path, destination: Path) -> Path | None:
    result = recovery_core.recover_snapshot(
        snapshot_dir, destination, mode="recovery", force=True
    )
    if result.checkpoint is not None:
        print(f"Safety checkpoint written: {result.checkpoint}", flush=True)
    return result.checkpoint


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

    destination = args.destination.resolve()
    destination_db = destination / "images.db"

    # config._CONFIG loads from os.environ on first get_config() call in
    # this process, so set it before anything imports config.
    os.environ["OUTPUT_DIR"] = str(destination)

    try:
        result = recovery_core.recover_snapshot(
            args.snapshot,
            destination,
            mode=args.mode,
            force=args.force,
            progress=lambda _stage, _percent, message: print(message, flush=True),
        )
        rollback_path = result.checkpoint
    except recovery_core.RecoveryError as exc:
        print(f"error: recovery failed [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: recovery failed: {exc}", file=sys.stderr)
        return 5

    print("Recovery complete.")
    print(f"  Database: {destination_db}")
    if rollback_path is not None:
        print(f"Safety checkpoint written: {rollback_path}")
        print(f"  Pre-restore backup kept at: {rollback_path}")
    print(
        "Restart the app (systemd: `sudo systemctl start app.service`; "
        "Docker: `docker compose start app`) to pick up the restored data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
