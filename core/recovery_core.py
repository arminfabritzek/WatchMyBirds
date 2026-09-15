"""Shared snapshot recovery engine used by the CLI and Pi runner.

The caller is responsible for stopping every process that can use the
destination database. This module owns validation, staging, atomic directory
publication, complete checkpoints, settings policy, and interrupted-swap
reconciliation. It never manages services.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

import yaml

from core.usb_backup_core import verify_snapshot_directory
from utils.path_manager import PathManager
from utils.restore import _validate_db_schema

ProgressCallback = Callable[[str, int, str], None]

# These values describe the destination appliance, its local access, or its
# private integrations. Keeping them prevents a recovered snapshot from
# replacing the password the user just used, changing cameras, cloning an
# installation identity, or unexpectedly enabling credentials from another
# machine. All other runtime behavior comes from the snapshot.
PRESERVED_SETTING_KEYS = frozenset(
    {
        "EDIT_PASSWORD",
        "CAMERA_URL",
        "GO2RTC_API_BASE",
        "GO2RTC_CONFIG_PATH",
        "GO2RTC_STREAM_NAME",
        "STREAM_SOURCE_MODE",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_GROUP_ID",
        "telemetry_installation_id",
    }
)
PRESERVED_OUTPUT_FILES = ("cameras.yaml", "go2rtc.yaml")

JOURNAL_SCHEMA_VERSION = 1
SPACE_HEADROOM_RATIO = 1.10


class RecoveryError(RuntimeError):
    """A safe, operator-actionable recovery refusal or failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RecoveryResult:
    destination: Path
    checkpoint: Path | None
    preserved_settings: tuple[str, ...]


def _emit(
    callback: ProgressCallback | None, stage: str, percent: int, message: str
) -> None:
    if callback is not None:
        callback(stage, percent, message)


def resolve_snapshot_directory(raw: Path) -> Path:
    """Resolve a direct snapshot path or the stick's ``latest`` symlink."""
    try:
        return raw.resolve(strict=True)
    except OSError as exc:
        raise RecoveryError(
            "snapshot_missing",
            "The selected backup was not found or is no longer available.",
        ) from exc


def destination_has_data(destination: Path) -> bool:
    """Fail closed when destination contents cannot be proven empty."""
    db_path = destination / "images.db"
    if db_path.is_file() and db_path.stat().st_size:
        try:
            conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                row = conn.execute("SELECT COUNT(*) FROM images").fetchone()
                if row and int(row[0]) > 0:
                    return True
            finally:
                conn.close()
        except sqlite3.Error:
            return True
    originals = PathManager(str(destination)).originals_dir
    try:
        return originals.is_dir() and any(
            path.is_file() for path in originals.rglob("*")
        )
    except OSError:
        return True


def _tree_bytes(root: Path) -> int:
    total = 0
    if root.is_symlink():
        raise RecoveryError("unsafe_snapshot", "The backup contains symbolic links.")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RecoveryError(
                "unsafe_snapshot", "The backup contains symbolic links."
            )
        if path.is_file():
            total += path.stat().st_size
    return total


def _row_counts(db_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table in ("images", "detections", "classifications"):
            counts[table] = int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
    finally:
        conn.close()
    return counts


def _read_settings(path: Path, *, strict: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RecoveryError(
            "invalid_settings", "The backup settings file is unreadable."
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict) and not strict:
        return {}
    if not isinstance(data, dict):
        raise RecoveryError(
            "invalid_settings", "The backup settings file is not a mapping."
        )
    return data


def inspect_snapshot(snapshot_dir: Path, destination: Path) -> dict[str, Any]:
    """Return the authenticated preview payload for a selected snapshot."""
    snapshot_dir = resolve_snapshot_directory(snapshot_dir)
    verification = verify_snapshot_directory(snapshot_dir)
    db_path = snapshot_dir / "data" / "images.db"
    schema_ok, schema_issues = _validate_db_schema(db_path)
    output_dir = snapshot_dir / "data" / "output"
    required_bytes = _tree_bytes(output_dir) + (
        db_path.stat().st_size if db_path.is_file() else 0
    )
    space_target = destination if destination.exists() else destination.parent
    while not space_target.exists() and space_target != space_target.parent:
        space_target = space_target.parent
    available_bytes = shutil.disk_usage(space_target).free
    required_with_headroom = int(required_bytes * SPACE_HEADROOM_RATIO)
    manifest: dict[str, Any] = {}
    try:
        raw_manifest = json.loads(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if isinstance(raw_manifest, dict):
            manifest = raw_manifest
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass

    counts: dict[str, int] = {}
    if verification.get("integrity_ok") and schema_ok:
        try:
            counts = _row_counts(db_path)
        except sqlite3.Error:
            schema_ok = False
            schema_issues.append("Could not count snapshot records")

    snapshot_settings = (
        _read_settings(output_dir / "settings.yaml") if output_dir.is_dir() else {}
    )
    current_settings = (
        _read_settings(destination / "settings.yaml", strict=False)
        if (destination / "settings.yaml").is_file()
        else {}
    )
    preserved = sorted(PRESERVED_SETTING_KEYS.intersection(current_settings))
    excluded = sorted(
        PRESERVED_SETTING_KEYS.intersection(snapshot_settings).difference(
            current_settings
        )
    )
    restored = sorted(set(snapshot_settings).difference(PRESERVED_SETTING_KEYS))
    compatible = bool(verification.get("ok") and schema_ok)
    blockers = []
    if not verification.get("ok"):
        blockers.append(
            str(
                verification.get("error")
                or verification.get("media_message")
                or "Backup verification failed"
            )
        )
    blockers.extend(schema_issues)
    if available_bytes < required_with_headroom:
        blockers.append("Not enough free space to stage this recovery safely.")

    has_data = destination_has_data(destination)
    return {
        "snapshot_id": snapshot_dir.name,
        "completed_at": manifest.get("completed_at"),
        "source": manifest.get("host") or "Unknown device",
        "app_version": manifest.get("app_version") or "unknown",
        "counts": counts,
        "compatible": compatible,
        "required_bytes": required_with_headroom,
        "available_bytes": available_bytes,
        "space_ok": available_bytes >= required_with_headroom,
        "destination_has_data": has_data,
        "mode": "recovery" if has_data else "migration",
        "settings_restored": restored,
        "settings_preserved": preserved,
        "settings_excluded": excluded,
        "network_preserved": True,
        "blockers": blockers,
        "verification": verification,
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make staged regular files and directory entries durable before rename."""
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RecoveryError(
                "unsafe_snapshot", "The backup contains symbolic links."
            )
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _unlink_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _remove_tree_durable(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        _fsync_directory(path.parent)


def write_json_durable(
    path: Path, payload: dict[str, Any], *, mode: int = 0o600
) -> None:
    """Atomically replace JSON and flush both contents and directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        os.fchmod(descriptor, mode)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temp.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_json_durable(path, payload)


def _journal_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}-recovery-journal.json"


def reconcile_interrupted_recovery(destination: Path) -> str | None:
    """Repair or finalize a directory swap interrupted by process/power loss."""
    journal_path = _journal_path(destination)
    if not journal_path.is_file():
        return None
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported journal")
        if Path(payload["destination"]) != destination:
            raise ValueError("journal destination mismatch")
        staging = Path(payload["staging"])
        checkpoint = Path(payload["checkpoint"]) if payload.get("checkpoint") else None
        if (
            staging.parent != destination.parent
            or not staging.name.startswith(f".{destination.name}-restore-")
            or staging.is_symlink()
        ):
            raise ValueError("unsafe staging path")
        if checkpoint is not None and (
            checkpoint.parent != destination.parent
            or not checkpoint.name.startswith(f"{destination.name}-before-restore-")
            or checkpoint.is_symlink()
        ):
            raise ValueError("unsafe checkpoint path")
        phase = payload.get("phase")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RecoveryError(
            "invalid_journal",
            "Recovery state is damaged; automatic repair was refused.",
        ) from exc

    if phase in {"rollback_prepared", "rollback_current_moved", "rollback_published"}:
        if destination.exists() and checkpoint and not checkpoint.exists():
            _unlink_durable(journal_path)
            return "The interrupted checkpoint rollback had already completed."
        if destination.exists() and checkpoint and checkpoint.exists():
            if staging.exists():
                raise RecoveryError(
                    "interrupted_swap",
                    "Rollback state is ambiguous; automatic repair was refused.",
                )
            _unlink_durable(journal_path)
            return (
                "The interrupted checkpoint rollback had not changed the installation."
            )
        if not destination.exists() and checkpoint and checkpoint.exists():
            checkpoint.rename(destination)
            _fsync_directory(destination.parent)
            _unlink_durable(journal_path)
            return "The interrupted checkpoint rollback was completed."
        if not destination.exists() and staging.exists():
            staging.rename(destination)
            _fsync_directory(destination.parent)
            _unlink_durable(journal_path)
            return "The interrupted checkpoint rollback restored the last usable data."
        raise RecoveryError(
            "interrupted_swap",
            "Checkpoint rollback was interrupted and could not be repaired automatically.",
        )

    if (
        phase in {"checkpoint_prepared", "checkpoint_moved"}
        and not destination.exists()
        and checkpoint
        and checkpoint.exists()
    ):
        checkpoint.rename(destination)
        _fsync_directory(destination.parent)
        if staging.exists():
            _remove_tree_durable(staging)
        _unlink_durable(journal_path)
        return (
            "The interrupted publication was rolled back to the previous installation."
        )
    if (
        phase == "checkpoint_moved"
        and destination.is_dir()
        and checkpoint
        and checkpoint.is_dir()
    ):
        if staging.exists():
            _remove_tree_durable(staging)
        _unlink_durable(journal_path)
        return "The interrupted recovery had already published its restored data."
    if phase == "published" and destination.is_dir():
        if staging.exists():
            _remove_tree_durable(staging)
        _unlink_durable(journal_path)
        return "The interrupted recovery had already published its restored data."
    if (
        phase == "verified"
        and checkpoint is None
        and destination.is_dir()
        and not staging.exists()
    ):
        _unlink_durable(journal_path)
        return "The interrupted migration had already published its restored data."
    if phase in {"staging", "verified", "checkpoint_prepared"} and destination.exists():
        if staging.exists():
            _remove_tree_durable(staging)
        _unlink_durable(journal_path)
        return "Incomplete staged recovery data was removed; the installation was unchanged."
    raise RecoveryError(
        "interrupted_swap",
        "Recovery was interrupted during publication and could not be repaired automatically.",
    )


def _apply_settings_policy(staging: Path, destination: Path) -> tuple[str, ...]:
    current_path = destination / "settings.yaml"
    staged_path = staging / "settings.yaml"
    current = (
        _read_settings(current_path, strict=False) if current_path.is_file() else {}
    )
    restored = _read_settings(staged_path) if staged_path.is_file() else {}
    preserved = tuple(sorted(PRESERVED_SETTING_KEYS.intersection(current)))
    for key in PRESERVED_SETTING_KEYS:
        if key in current:
            restored[key] = current[key]
        else:
            restored.pop(key, None)
    if restored or staged_path.exists() or current_path.exists():
        staged_path.write_text(
            yaml.safe_dump(restored, sort_keys=True), encoding="utf-8"
        )
        staged_path.chmod(0o600)

    for name in PRESERVED_OUTPUT_FILES:
        current_file = destination / name
        staged_file = staging / name
        if staged_file.exists() and not staged_file.is_file():
            raise RecoveryError(
                "invalid_device_settings",
                f"The backup contains an invalid device settings entry: {name}.",
            )
        if current_file.exists():
            if current_file.is_symlink() or not current_file.is_file():
                raise RecoveryError(
                    "invalid_device_settings",
                    f"The current device settings entry is unsafe: {name}.",
                )
            shutil.copy2(current_file, staged_file)
        else:
            staged_file.unlink(missing_ok=True)
    return preserved


def recover_snapshot(
    snapshot_dir: Path,
    destination: Path,
    *,
    mode: str,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> RecoveryResult:
    """Validate, stage, and atomically publish one complete snapshot."""
    if mode not in {"migration", "recovery"}:
        raise RecoveryError(
            "invalid_mode", "Recovery mode must be migration or recovery."
        )
    destination = destination.resolve()
    snapshot_dir = resolve_snapshot_directory(snapshot_dir)
    if destination.is_mount():
        raise RecoveryError(
            "unsafe_destination", "The recovery destination cannot be a mount point."
        )
    if (
        destination == snapshot_dir
        or destination in snapshot_dir.parents
        or snapshot_dir in destination.parents
    ):
        raise RecoveryError("overlap", "The backup and recovery destination overlap.")

    reconcile_interrupted_recovery(destination)
    has_data = destination_has_data(destination)
    if mode == "migration" and has_data:
        raise RecoveryError(
            "populated_destination",
            "This installation already has data; choose replacement recovery instead of migration.",
        )
    if mode == "recovery" and not force:
        raise RecoveryError(
            "confirmation_required",
            "Recovery mode requires --force as explicit confirmation.",
        )

    _emit(
        progress, "validating", 5, "Validating backup database, media, and checksums…"
    )
    preview = inspect_snapshot(snapshot_dir, destination)
    if not preview["compatible"]:
        raise RecoveryError(
            "invalid_snapshot",
            "; ".join(preview["blockers"]) or "Backup validation failed.",
        )
    if not preview["space_ok"]:
        raise RecoveryError(
            "low_space", "Not enough free space to stage this recovery safely."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-restore-", dir=destination.parent)
    )
    checkpoint: Path | None = None
    publication_committed = False
    journal = _journal_path(destination)
    payload = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "phase": "staging",
        "destination": str(destination),
        "staging": str(staging),
        "checkpoint": None,
        "snapshot_id": snapshot_dir.name,
    }
    _write_json_atomic(journal, payload)
    try:
        source = snapshot_dir / "data" / "output"
        _emit(
            progress, "staging", 20, "Copying recovered data into a safe staging area…"
        )
        if any(path.is_symlink() for path in chain([source], source.rglob("*"))):
            raise RecoveryError(
                "unsafe_snapshot", "The backup contains symbolic links."
            )
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            (staging / f"images.db{suffix}").unlink(missing_ok=True)
        shutil.copy2(snapshot_dir / "data" / "images.db", staging / "images.db")
        preserved = _apply_settings_policy(staging, destination)
        if destination.exists():
            owner = destination.stat()
            shutil.copystat(destination, staging)
            if os.geteuid() == 0:
                for item in chain([staging], staging.rglob("*")):
                    os.chown(item, owner.st_uid, owner.st_gid)

        _emit(
            progress,
            "verifying_staging",
            55,
            "Checking the staged database and every retained original…",
        )
        from core.usb_backup_core import _verify_media

        media = _verify_media(snapshot_dir, staging / "images.db", output_dir=staging)
        schema_ok, schema_issues = _validate_db_schema(staging / "images.db")
        if not media["media_ok"] or not schema_ok:
            detail = media.get("media_message") or "; ".join(schema_issues)
            raise RecoveryError("staging_verification_failed", str(detail))
        _fsync_tree(staging)
        payload["phase"] = "verified"
        _write_json_atomic(journal, payload)

        if destination.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            checkpoint = (
                destination.parent / f"{destination.name}-before-restore-{stamp}"
            )
            serial = 1
            while checkpoint.exists():
                checkpoint = (
                    destination.parent
                    / f"{destination.name}-before-restore-{stamp}-{serial}"
                )
                serial += 1
            payload["checkpoint"] = str(checkpoint)
            payload["phase"] = "checkpoint_prepared"
            _fsync_tree(destination)
            _write_json_atomic(journal, payload)
            _emit(
                progress,
                "checkpoint",
                70,
                "Retaining a complete checkpoint of the current installation…",
            )
            destination.rename(checkpoint)
            _fsync_directory(destination.parent)
            payload["phase"] = "checkpoint_moved"
            _write_json_atomic(journal, payload)

        _emit(progress, "publishing", 82, "Publishing the verified recovered data…")
        try:
            staging.rename(destination)
            _fsync_directory(destination.parent)
        except BaseException:
            if (
                checkpoint is not None
                and checkpoint.exists()
                and not destination.exists()
            ):
                checkpoint.rename(destination)
                _fsync_directory(destination.parent)
            raise
        payload["phase"] = "published"
        _write_json_atomic(journal, payload)
        publication_committed = True
        try:
            _unlink_durable(journal)
        except OSError:
            pass
        _emit(
            progress,
            "published",
            90,
            "Recovered data published; the app can now restart.",
        )
        return RecoveryResult(destination, checkpoint, preserved)
    except BaseException:
        if not publication_committed and not staging.exists() and destination.exists():
            destination.rename(staging)
            _fsync_directory(destination.parent)
        if checkpoint is not None and checkpoint.exists() and not destination.exists():
            checkpoint.rename(destination)
            _fsync_directory(destination.parent)
        if staging.exists():
            _remove_tree_durable(staging)
        if payload.get("phase") in {"staging", "verified"} or destination.exists():
            _unlink_durable(journal)
        raise


def rollback_checkpoint(destination: Path, checkpoint: Path) -> Path:
    """Replace a failed restored destination with its retained checkpoint."""
    destination = destination.resolve()
    reconcile_interrupted_recovery(destination)
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != destination.parent or not checkpoint.name.startswith(
        f"{destination.name}-before-restore-"
    ):
        raise RecoveryError(
            "unsafe_checkpoint", "The recovery checkpoint identifier is invalid."
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    failed = destination.parent / f".{destination.name}-restore-failed-{stamp}"
    serial = 1
    while failed.exists():
        failed = (
            destination.parent / f".{destination.name}-restore-failed-{stamp}-{serial}"
        )
        serial += 1
    journal = _journal_path(destination)
    payload = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "phase": "rollback_prepared",
        "destination": str(destination),
        "staging": str(failed),
        "checkpoint": str(checkpoint),
    }
    if destination.exists():
        _fsync_tree(destination)
    _fsync_tree(checkpoint)
    _write_json_atomic(journal, payload)
    if destination.exists():
        destination.rename(failed)
        _fsync_directory(destination.parent)
        payload["phase"] = "rollback_current_moved"
        _write_json_atomic(journal, payload)
    try:
        checkpoint.rename(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if failed.exists() and not destination.exists():
            failed.rename(destination)
            _fsync_directory(destination.parent)
        _unlink_durable(journal)
        raise
    payload["phase"] = "rollback_published"
    _write_json_atomic(journal, payload)
    _unlink_durable(journal)
    return failed


def settings_policy_labels(keys: Iterable[str]) -> list[str]:
    """Stable display labels for a preview without exposing secret values."""
    labels = {
        "EDIT_PASSWORD": "Admin password",
        "CAMERA_URL": "Camera connection",
        "GO2RTC_API_BASE": "Video relay address",
        "GO2RTC_CONFIG_PATH": "Video relay file location",
        "GO2RTC_STREAM_NAME": "Video relay stream",
        "STREAM_SOURCE_MODE": "Camera routing mode",
        "TELEGRAM_BOT_TOKEN": "Telegram bot credential",
        "TELEGRAM_CHAT_ID": "Telegram chat destination",
        "TELEGRAM_GROUP_ID": "Telegram group destination",
        "telemetry_installation_id": "Device identity",
    }
    return [labels.get(key, key.replace("_", " ").title()) for key in keys]
