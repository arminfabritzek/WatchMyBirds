"""
USB Backup Core — Business logic for USB-stick snapshots.

This module is the read-side of the USB backup feature. It inspects
snapshots created by `rpi/backup.sh` on `/mnt/wmb-backup/snapshots/`
and surfaces structured state to the API/UI layer.

The write-side (creating snapshots) lives entirely in `rpi/backup.sh`
and the systemd timer/service. This module never invokes that script
directly — that's the service layer's job.

See `docs/USB_BACKUP.md` and the focus plan
`agent_handoff/workflow/plans/2026-04-27_INFRA_usb-data-backup.md`
for the design.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These are the canonical on-Pi paths. They mirror the constants at the
# top of rpi/backup.sh — keep them in sync. We don't share a config file
# because backup.sh runs without Python in scope.
MOUNT_POINT = Path("/mnt/wmb-backup")
BACKUP_DEVICE = Path("/dev/disk/by-label/WMB-BACKUP")
SNAPSHOTS_DIR = MOUNT_POINT / "snapshots"
LATEST_LINK = MOUNT_POINT / "latest"
BACKUP_LOG = MOUNT_POINT / "BACKUP_LOG.txt"


# ----------------------------------------------------------------------
# Stick-level status
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StickStatus:
    """High-level state of the USB backup volume."""

    state: str  # "connected" | "missing" | "wrong_fs" | "not_writable" | "error"
    fstype: str | None
    total_bytes: int | None
    free_bytes: int | None
    used_bytes: int | None
    free_pct: float | None
    detail: str | None  # Human-readable elaboration; None on happy path

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "fstype": self.fstype,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "free_pct": self.free_pct,
            "detail": self.detail,
        }


def _findmnt_fstype(mount_point: Path) -> str | None:
    """Resolve the filesystem type of a mount via findmnt.

    Returns None if `mount_point` is unmounted, OR if it's only an
    autofs trigger placeholder (systemd's .automount unit registers an
    autofs stub even when no stick is present; that stub reports as
    'autofs' to findmnt but is not a real filesystem). Treating autofs
    as "no FS yet" lets get_stick_status() fall through to the
    missing/wrong-fs branch via blkid on the labelled device.
    """
    if not shutil.which("findmnt"):
        return None
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE", str(mount_point)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        fstype = result.stdout.strip() or None
        # autofs is a trigger placeholder, not a real on-disk FS.
        if fstype == "autofs":
            return None
        return fstype
    except (subprocess.SubprocessError, OSError):
        return None


def _blkid_fstype(device: Path) -> str | None:
    """Resolve the filesystem type of the labelled backup device."""
    if not shutil.which("blkid"):
        return None
    try:
        result = subprocess.run(
            ["blkid", "-o", "value", "-s", "TYPE", str(device)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (result.stdout.strip() or None) if result.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def _is_mounted(mount_point: Path) -> bool:
    """True iff a real filesystem (not just an autofs stub) is mounted.

    systemd's .automount unit registers an autofs trigger at
    `mount_point` regardless of whether a stick is plugged in.
    `os.path.ismount` returns True for that stub, so on its own it
    cannot distinguish "autofs trigger waiting" from "real ext4
    mounted". We cross-check with findmnt, which returns 'autofs' for
    the stub and an actual FS name once a stick is mounted —
    `_findmnt_fstype` filters autofs to None, so a non-None result is
    the real-mount signal.
    """
    try:
        if not os.path.ismount(mount_point):
            # No mount and no automount trigger active.
            if not BACKUP_DEVICE.exists():
                return False
            # Labelled device exists -- touch the path to wake the
            # .automount unit, then re-check below.
            try:
                mount_point.exists()
            except OSError:
                pass
    except OSError:
        return False

    # At this point os.path.ismount may return True for an autofs stub.
    # _findmnt_fstype returns None for autofs, the real FS name when a
    # stick is actually mounted.
    return _findmnt_fstype(mount_point) is not None


def get_stick_status() -> StickStatus:
    """Probe the stick state without touching snapshot data.

    Cheap: callable from the public status bar at high frequency.
    """
    if not _is_mounted(MOUNT_POINT):
        label_fstype = _blkid_fstype(BACKUP_DEVICE)
        if label_fstype is not None and label_fstype != "ext4":
            return StickStatus(
                state="wrong_fs",
                fstype=label_fstype,
                total_bytes=None,
                free_bytes=None,
                used_bytes=None,
                free_pct=None,
                detail=(
                    f"Stick is formatted as '{label_fstype}' but ext4 is required "
                    "for snapshot dedup (rsync --link-dest hardlinks). "
                    "Reformat per docs/USB_BACKUP.md."
                ),
            )
        return StickStatus(
            state="missing",
            fstype=None,
            total_bytes=None,
            free_bytes=None,
            used_bytes=None,
            free_pct=None,
            detail="USB backup stick not detected. See docs/USB_BACKUP.md.",
        )

    fstype = _findmnt_fstype(MOUNT_POINT)
    if fstype is not None and fstype != "ext4":
        return StickStatus(
            state="wrong_fs",
            fstype=fstype,
            total_bytes=None,
            free_bytes=None,
            used_bytes=None,
            free_pct=None,
            detail=(
                f"Stick is formatted as '{fstype}' but ext4 is required for "
                "snapshot dedup (rsync --link-dest hardlinks). "
                "Reformat per docs/USB_BACKUP.md."
            ),
        )

    try:
        usage = shutil.disk_usage(MOUNT_POINT)
    except OSError as exc:
        return StickStatus(
            state="error",
            fstype=fstype,
            total_bytes=None,
            free_bytes=None,
            used_bytes=None,
            free_pct=None,
            detail=f"Cannot stat mount point: {exc}",
        )

    if not os.access(MOUNT_POINT, os.W_OK):
        return StickStatus(
            state="not_writable",
            fstype=fstype,
            total_bytes=usage.total,
            free_bytes=usage.free,
            used_bytes=usage.used,
            free_pct=(usage.free / usage.total * 100) if usage.total else None,
            detail=(
                "Mount point is not writable for the app user. "
                "Run once as root: chown -R watchmybirds:watchmybirds /mnt/wmb-backup"
            ),
        )

    return StickStatus(
        state="connected",
        fstype=fstype,
        total_bytes=usage.total,
        free_bytes=usage.free,
        used_bytes=usage.used,
        free_pct=(usage.free / usage.total * 100) if usage.total else None,
        detail=None,
    )


# ----------------------------------------------------------------------
# Per-snapshot inspection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata about a single snapshot directory on the stick."""

    name: str  # e.g. "20260429_030000_scheduled"
    kind: str  # "scheduled" | "manual" | "pre-ota" | "pre-restore" | "unknown"
    completed: bool
    corrupt: bool
    corrupt_reason: str | None
    started_at: str | None
    completed_at: str | None
    total_bytes: int | None
    db_bytes: int | None
    output_bytes: int | None
    app_bytes: int | None
    app_version: str | None
    is_latest: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "completed": self.completed,
            "corrupt": self.corrupt,
            "corrupt_reason": self.corrupt_reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_bytes": self.total_bytes,
            "db_bytes": self.db_bytes,
            "output_bytes": self.output_bytes,
            "app_bytes": self.app_bytes,
            "app_version": self.app_version,
            "is_latest": self.is_latest,
        }


def _read_json_safely(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_first_line(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as f:
            return f.readline().strip() or None
    except OSError:
        return None


def _resolve_latest_target() -> Path | None:
    """Return the snapshot directory the 'latest' symlink points at."""
    try:
        if LATEST_LINK.is_symlink():
            return LATEST_LINK.resolve(strict=False)
    except OSError:
        pass
    return None


def _inspect_snapshot(directory: Path, latest_target: Path | None) -> SnapshotInfo:
    name = directory.name
    completed = (directory / "COMPLETED").is_file()
    corrupt_marker = directory / "CORRUPT"
    corrupt = corrupt_marker.is_file()
    corrupt_reason = None
    if corrupt:
        # CORRUPT file is "reason: ...\nts: ..."
        text = _read_first_line(corrupt_marker) or ""
        corrupt_reason = text.removeprefix("reason: ").strip() or None

    manifest = _read_json_safely(directory / "manifest.json") or {}
    sizes = manifest.get("sizes") or {}

    kind_raw = (
        _read_first_line(directory / "kind")
        or manifest.get("kind")
        or "unknown"
    )
    kind = kind_raw.strip() or "unknown"

    return SnapshotInfo(
        name=name,
        kind=kind,
        completed=completed,
        corrupt=corrupt,
        corrupt_reason=corrupt_reason,
        started_at=manifest.get("started_at"),
        completed_at=manifest.get("completed_at") or (
            _read_first_line(directory / "COMPLETED") if completed else None
        ),
        total_bytes=sizes.get("total_bytes"),
        db_bytes=sizes.get("db_bytes"),
        output_bytes=sizes.get("output_bytes"),
        app_bytes=sizes.get("app_bytes"),
        app_version=manifest.get("app_version"),
        is_latest=(latest_target is not None and directory == latest_target),
    )


def list_snapshots(*, limit: int | None = None) -> list[SnapshotInfo]:
    """Return all snapshots in newest-first order.

    Empty list if the stick is missing or the snapshots dir doesn't
    exist yet (fresh stick before first run).
    """
    if not _is_mounted(MOUNT_POINT) or not SNAPSHOTS_DIR.is_dir():
        return []

    latest = _resolve_latest_target()
    try:
        entries = sorted(
            (p for p in SNAPSHOTS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return []

    out: list[SnapshotInfo] = []
    for entry in entries:
        if limit is not None and len(out) >= limit:
            break
        out.append(_inspect_snapshot(entry, latest))
    return out


def get_snapshot(name: str) -> SnapshotInfo | None:
    """Look up a snapshot by exact directory name."""
    # Defense-in-depth: guard against path traversal even though Flask
    # already URL-decodes once.
    if "/" in name or ".." in name or not name:
        return None
    directory = SNAPSHOTS_DIR / name
    if not directory.is_dir():
        return None
    return _inspect_snapshot(directory, _resolve_latest_target())


def delete_snapshot(name: str) -> tuple[bool, str]:
    """Permanently delete a snapshot directory.

    Returns (success, message). Refuses to touch anything outside
    SNAPSHOTS_DIR or anything that doesn't look like a snapshot dir.
    """
    if "/" in name or ".." in name or not name:
        return False, "Invalid snapshot name."
    directory = SNAPSHOTS_DIR / name
    if not directory.is_dir():
        return False, f"Snapshot {name!r} not found."
    # Sanity: it must live directly under SNAPSHOTS_DIR.
    try:
        directory.resolve(strict=True).relative_to(SNAPSHOTS_DIR.resolve(strict=True))
    except (OSError, ValueError):
        return False, "Snapshot path is outside the snapshots directory."

    try:
        shutil.rmtree(directory)
    except OSError as exc:
        return False, f"Delete failed: {exc}"
    return True, f"Snapshot {name} deleted."


# ----------------------------------------------------------------------
# Verification (re-check sha + sqlite integrity_check on demand)
# ----------------------------------------------------------------------


def verify_snapshot(name: str) -> dict[str, Any]:
    """Re-run integrity checks on a snapshot.

    Returns a structured result. Does NOT mutate the snapshot directory
    (no automatic CORRUPT-flag rewrite — operator decides what to do).
    """
    snap = get_snapshot(name)
    if snap is None:
        return {"ok": False, "name": name, "error": "Snapshot not found."}

    directory = SNAPSHOTS_DIR / name
    db_path = directory / "data" / "images.db"
    sha_path = directory / "data" / "images.db.sha256"

    sha_ok: bool | None
    sha_message: str | None
    if not db_path.is_file():
        sha_ok, sha_message = None, "No DB in snapshot."
    elif not sha_path.is_file():
        sha_ok, sha_message = None, "No sha256 file alongside DB."
    else:
        try:
            result = subprocess.run(
                ["sha256sum", "-c", "--quiet", str(sha_path.name)],
                cwd=db_path.parent,
                capture_output=True,
                text=True,
                timeout=300,
            )
            sha_ok = result.returncode == 0
            sha_message = (result.stderr or result.stdout or "").strip() or None
        except (subprocess.SubprocessError, OSError) as exc:
            sha_ok, sha_message = False, str(exc)

    integrity_ok: bool | None
    integrity_message: str | None
    if not db_path.is_file():
        integrity_ok, integrity_message = None, None
    else:
        try:
            result = subprocess.run(
                ["sqlite3", str(db_path), "pragma integrity_check;"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (result.stdout or "").strip()
            integrity_ok = result.returncode == 0 and output == "ok"
            integrity_message = output or None
        except (subprocess.SubprocessError, OSError) as exc:
            integrity_ok, integrity_message = False, str(exc)

    overall_ok = (sha_ok is not False) and (integrity_ok is not False)

    return {
        "ok": overall_ok,
        "name": name,
        "sha_ok": sha_ok,
        "sha_message": sha_message,
        "integrity_ok": integrity_ok,
        "integrity_message": integrity_message,
        "previously_marked_corrupt": snap.corrupt,
    }


# ----------------------------------------------------------------------
# Aggregate "summary" used by the Settings page card
# ----------------------------------------------------------------------


def get_backup_summary(*, recent_limit: int = 5) -> dict[str, Any]:
    """One-call snapshot-of-snapshots for the UI card."""
    stick = get_stick_status()
    snapshots = list_snapshots(limit=recent_limit) if stick.state == "connected" else []

    most_recent_completed: SnapshotInfo | None = None
    for s in snapshots:
        if s.completed and not s.corrupt:
            most_recent_completed = s
            break

    return {
        "stick": stick.to_dict(),
        "snapshots_recent": [s.to_dict() for s in snapshots],
        "most_recent_completed": (
            most_recent_completed.to_dict() if most_recent_completed else None
        ),
        "warn_almost_full": (
            stick.free_pct is not None and stick.free_pct < 20.0
        ),
    }
