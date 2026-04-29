"""
USB Backup Service — Web-layer wrapper around core.usb_backup_core.

Routes use this; the service translates between the dataclass-based
core API and JSON-friendly dicts, and owns the `manual` trigger:
spawning rpi/backup.sh as a detached subprocess so it survives the
web process dying mid-snapshot.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from core import usb_backup_core
from logging_config import get_logger

logger = get_logger(__name__)

# Path to the backup script. Mirrors the location used by
# rpi/systemd/wmb-backup.service (which we trust for the scheduled
# pathway) -- keep these in sync.
BACKUP_SCRIPT = Path("/opt/app/rpi/backup.sh")

# Lock to prevent concurrent manual triggers. The script itself is
# idempotent, but two snapshots starting in the same second would
# collide on directory names anyway.
_TRIGGER_LOCK = threading.Lock()
# Tracks the most recently spawned manual backup for status reporting.
_LAST_MANUAL_TRIGGER: dict[str, Any] | None = None


# ----------------------------------------------------------------------
# Read-side passthroughs (just dict-translation)
# ----------------------------------------------------------------------


def get_status() -> dict[str, Any]:
    """Return stick state + recent snapshots in one call.

    This is the endpoint the UI polls, so keep it fast: stat-only,
    no integrity checks.
    """
    return usb_backup_core.get_backup_summary()


def list_snapshots(limit: int | None = None) -> list[dict[str, Any]]:
    """Return all snapshots (newest first), each as a dict."""
    return [s.to_dict() for s in usb_backup_core.list_snapshots(limit=limit)]


def get_snapshot(name: str) -> dict[str, Any] | None:
    """Return a single snapshot by name, or None if missing."""
    snap = usb_backup_core.get_snapshot(name)
    return snap.to_dict() if snap else None


def delete_snapshot(name: str) -> tuple[bool, str]:
    """Delete a snapshot by name."""
    return usb_backup_core.delete_snapshot(name)


def verify_snapshot(name: str) -> dict[str, Any]:
    """Re-run sha + integrity_check on a snapshot."""
    return usb_backup_core.verify_snapshot(name)


# ----------------------------------------------------------------------
# Manual trigger
# ----------------------------------------------------------------------


def is_trigger_supported() -> bool:
    """Check whether we can plausibly invoke the backup script.

    On dev hosts (Mac, CI) the script is missing — we should refuse
    and surface a clean error rather than 500.
    """
    return BACKUP_SCRIPT.is_file() and os.access(BACKUP_SCRIPT, os.X_OK)


def trigger_manual_backup() -> tuple[bool, str, dict[str, Any] | None]:
    """Spawn rpi/backup.sh --kind manual as a detached subprocess.

    Returns (started, message, info). `info` carries the PID and the
    expected log file location so the UI can give the operator
    something to point at.

    The subprocess is detached via start_new_session=True (a.k.a.
    setsid) so that if Flask crashes or restarts mid-backup, the
    backup keeps running.
    """
    global _LAST_MANUAL_TRIGGER

    if not is_trigger_supported():
        return (
            False,
            f"Backup script not available at {BACKUP_SCRIPT}.",
            None,
        )

    stick = usb_backup_core.get_stick_status()
    if stick.state != "connected":
        return (
            False,
            f"USB stick not ready (state: {stick.state}). {stick.detail or ''}".strip(),
            {"stick": stick.to_dict()},
        )

    if not _TRIGGER_LOCK.acquire(blocking=False):
        return (
            False,
            "Another manual backup is already starting; try again in a moment.",
            None,
        )

    try:
        # stdout/stderr go to journald via systemd-cat when available,
        # otherwise to a per-pid file under /tmp. The script also
        # appends to /mnt/wmb-backup/BACKUP_LOG.txt itself.
        log_target = subprocess.DEVNULL
        try:
            proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [str(BACKUP_SCRIPT), "--kind", "manual"],
                stdin=subprocess.DEVNULL,
                stdout=log_target,
                stderr=log_target,
                start_new_session=True,  # detach: setsid()
                close_fds=True,
            )
        except OSError as exc:
            logger.error("Failed to spawn manual backup: %s", exc)
            return False, f"Failed to spawn backup: {exc}", None

        info = {
            "pid": proc.pid,
            "kind": "manual",
            "log_path": str(usb_backup_core.BACKUP_LOG),
            "script": str(BACKUP_SCRIPT),
        }
        _LAST_MANUAL_TRIGGER = info
        logger.info("Manual USB backup spawned (pid=%s)", proc.pid)
        return True, "Manual backup started.", info
    finally:
        _TRIGGER_LOCK.release()


def get_last_manual_trigger() -> dict[str, Any] | None:
    """Return info about the most recent manual trigger this process saw.

    Note: this is process-local, so a Flask reload forgets it. The
    snapshot list (list_snapshots) is the source of truth for what
    actually completed.
    """
    return _LAST_MANUAL_TRIGGER
