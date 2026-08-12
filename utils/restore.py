# ------------------------------------------------------------------------------
# Restore Utilities for WatchMyBirds
# utils/restore.py
# ------------------------------------------------------------------------------
"""
Restore/Import functionality for backup archives.
Implements safe tar handling, DB merge/replace, and settings import.
"""

import json
import logging
import os
import shutil
import sqlite3
import tarfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from utils.db import _get_db_path as get_db_path
from utils.ingest import calculate_sha256
from utils.path_manager import PathManager, get_path_manager
from utils.settings import get_settings_path

logger = logging.getLogger(__name__)

# Maximum archive size (10 GB)
MAX_ARCHIVE_SIZE_BYTES = 10 * 1024 * 1024 * 1024
# Maximum settings.yaml size (1 MB)
MAX_SETTINGS_SIZE_BYTES = 1 * 1024 * 1024
# Maximum number of files in archive
MAX_FILE_COUNT = 100000

# Allowed top-level paths in archive
ALLOWED_ARCHIVE_PATHS = frozenset(
    [
        "images.db",
        "settings.yaml",
        "backup_manifest.json",
        "originals/",
        "derivatives/",
    ]
)

# Secret keys to mask in settings preview
SECRET_KEYS = frozenset(
    [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_GROUP_ID",
        "WEB_PASSWORD",
        "WEB_SECRET_KEY",
    ]
)

# Global restore lock state
_restore_lock = {"active": False, "started_at": None, "stage": None}


def is_restore_active() -> bool:
    """Returns True if a restore operation is currently in progress."""
    return _restore_lock["active"]


def get_restore_status() -> dict:
    """Returns current restore status."""
    return _restore_lock.copy()


def _set_restore_lock(active: bool, stage: str | None = None) -> None:
    """Sets the global restore lock state."""
    global _restore_lock
    _restore_lock["active"] = active
    _restore_lock["stage"] = stage
    if active:
        _restore_lock["started_at"] = datetime.now(UTC).isoformat()
    else:
        _restore_lock["started_at"] = None


# Persistent restart-required marker functions
def set_restart_required(pm: PathManager) -> None:
    """Creates marker file indicating restart is required after restore."""
    marker = pm.get_restart_required_marker()
    marker.write_text(datetime.now(UTC).isoformat())
    logger.info(f"Restart required marker created: {marker}")


def clear_restart_required(pm: PathManager) -> None:
    """Removes restart marker (called on fresh app start)."""
    marker = pm.get_restart_required_marker()
    if marker.exists():
        marker.unlink()
        logger.info("Restart required marker cleared")


def is_restart_required(pm: PathManager) -> bool:
    """Returns True if restart is required after a previous restore."""
    return pm.get_restart_required_marker().exists()


def _is_safe_tar_path(member: tarfile.TarInfo) -> tuple[bool, str]:
    """
    Validates a tar archive member for safety.

    Checks:
    - No absolute paths
    - No path traversal (..)
    - No symlinks or hardlinks
    - Must be in allowed paths

    Returns:
        tuple: (is_safe, error_message)
    """
    name = member.name

    # Block absolute paths
    if name.startswith("/"):
        return False, f"Absolute path blocked: {name}"

    # Block path traversal
    if ".." in name:
        return False, f"Path traversal blocked: {name}"

    # Block symlinks and hardlinks
    if member.issym() or member.islnk():
        return False, f"Symlink/hardlink blocked: {name}"

    # Check allowed paths
    parts = name.split("/")
    top_level = parts[0]

    # Allow exact top-level files
    if name in ("images.db", "settings.yaml", "backup_manifest.json"):
        return True, ""

    # Allow directories under allowed prefixes
    if top_level in ("originals", "derivatives"):
        return True, ""

    return False, f"Unexpected path blocked: {name}"


def _validated_archive_path(archive_path: Path) -> Path | None:
    """Return restore_tmp / secure_filename(basename), or None on rejection."""
    from werkzeug.utils import secure_filename

    safe_basename = secure_filename(os.path.basename(str(archive_path)))
    if not safe_basename:
        return None
    name_lower = safe_basename.lower()
    if not (name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz")):
        return None
    try:
        root = get_path_manager().get_restore_tmp_dir().resolve()
    except Exception:
        return None
    return root / safe_basename


def analyze_backup_archive(archive_path: Path) -> dict:
    """
    Analyzes a tar.gz archive without extracting.

    Returns:
        dict: {
            "has_db": bool,
            "has_originals": bool,
            "has_derivatives": bool,
            "has_settings": bool,
            "has_manifest": bool,
            "originals_count": int,
            "derivatives_count": int,
            "db_size_bytes": int,
            "settings_size_bytes": int,
            "settings_preview": dict,  # Key list, secret keys masked
            "manifest": dict | None,
            "warnings": list[str],
            "blockers": list[str],
            "total_size_bytes": int,
            "file_count": int,
        }
    """
    result = {
        "has_db": False,
        "has_originals": False,
        "has_derivatives": False,
        "has_settings": False,
        "has_manifest": False,
        "originals_count": 0,
        "derivatives_count": 0,
        "db_size_bytes": 0,
        "settings_size_bytes": 0,
        "settings_preview": {},
        "manifest": None,
        "warnings": [],
        "blockers": [],
        "total_size_bytes": 0,
        "file_count": 0,
    }

    # Defense-in-depth: refuse paths outside restore_tmp even when
    # called from a future caller that forgot to validate.
    safe_path = _validated_archive_path(archive_path)
    if safe_path is None:
        result["blockers"].append("Archive path outside restore sandbox")
        return result
    archive_path = safe_path

    if not archive_path.exists():
        result["blockers"].append("Archive file does not exist")
        return result

    # Check archive size
    archive_size = archive_path.stat().st_size
    if archive_size > MAX_ARCHIVE_SIZE_BYTES:
        result["blockers"].append(
            f"Archive too large: {archive_size / (1024**3):.2f} GB "
            f"(max: {MAX_ARCHIVE_SIZE_BYTES / (1024**3):.0f} GB)"
        )
        return result

    # Check magic header
    try:
        with open(archive_path, "rb") as f:
            magic = f.read(2)
            if magic != b"\x1f\x8b":  # gzip magic
                result["blockers"].append("Invalid archive format (not gzip)")
                return result
    except Exception as exc:
        logger.warning("Cannot read archive [%s]", type(exc).__name__, exc_info=True)
        result["blockers"].append("Cannot read archive (see logs)")
        return result

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                result["file_count"] += 1

                # Check file count limit
                if result["file_count"] > MAX_FILE_COUNT:
                    result["blockers"].append(
                        f"Too many files in archive: >{MAX_FILE_COUNT}"
                    )
                    return result

                # Safety check
                is_safe, error_msg = _is_safe_tar_path(member)
                if not is_safe:
                    result["blockers"].append(error_msg)
                    continue

                result["total_size_bytes"] += member.size

                # Analyze contents
                name = member.name

                if name == "images.db":
                    result["has_db"] = True
                    result["db_size_bytes"] = member.size

                elif name == "settings.yaml":
                    result["has_settings"] = True
                    result["settings_size_bytes"] = member.size

                    if member.size > MAX_SETTINGS_SIZE_BYTES:
                        result["blockers"].append(
                            f"settings.yaml too large: {member.size} bytes "
                            f"(max: {MAX_SETTINGS_SIZE_BYTES})"
                        )
                    else:
                        # Extract and parse settings preview
                        try:
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode("utf-8")
                                settings_data = yaml.safe_load(content)
                                result["settings_preview"] = _mask_settings(
                                    settings_data
                                )
                        except Exception as exc:
                            logger.warning(
                                "Could not parse settings.yaml [%s]",
                                type(exc).__name__,
                                exc_info=True,
                            )
                            result["warnings"].append("Could not parse settings.yaml")

                elif name == "backup_manifest.json":
                    result["has_manifest"] = True
                    try:
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8")
                            result["manifest"] = json.loads(content)
                    except Exception as exc:
                        logger.warning(
                            "Could not parse backup_manifest.json [%s]",
                            type(exc).__name__,
                            exc_info=True,
                        )
                        result["warnings"].append(
                            "Could not parse backup_manifest.json"
                        )

                elif name.startswith("originals/"):
                    result["has_originals"] = True
                    if member.isfile():
                        result["originals_count"] += 1

                elif name.startswith("derivatives/"):
                    result["has_derivatives"] = True
                    if member.isfile():
                        result["derivatives_count"] += 1

    except tarfile.TarError as exc:
        logger.warning("Invalid tar archive [%s]", type(exc).__name__, exc_info=True)
        result["blockers"].append("Invalid tar archive")
    except Exception as exc:
        logger.warning(
            "Error analyzing archive [%s]", type(exc).__name__, exc_info=True
        )
        result["blockers"].append("Error analyzing archive")

    return result


def _mask_settings(settings: dict) -> dict:
    """
    Creates a preview of settings with secret keys masked.
    """
    if not settings:
        return {}

    masked = {}
    for key, value in settings.items():
        if key in SECRET_KEYS:
            if value:
                masked[key] = "********"
            else:
                masked[key] = "(not set)"
        else:
            masked[key] = value
    return masked


def _check_disk_space(required_bytes: int, target_dir: Path) -> bool:
    """
    Checks if there's enough disk space for the restore operation.

    Args:
        required_bytes: Estimated bytes needed
        target_dir: Directory to check

    Returns:
        bool: True if enough space available
    """
    try:
        stat = os.statvfs(target_dir)
        available = stat.f_frsize * stat.f_bavail
        # Require 20% extra buffer
        return available > required_bytes * 1.2
    except Exception:
        # If we can't check, assume OK
        return True


def _create_rollback_snapshot(pm: PathManager) -> tuple[Path | None, Path | None]:
    """
    Creates rollback snapshots of DB and settings before restore.

    Returns:
        tuple: (db_backup_path, settings_backup_path) - either can be None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = pm.get_backup_before_restore_dir()

    db_backup = None
    settings_backup = None

    # Backup DB
    db_path = Path(get_db_path())
    if db_path.exists():
        db_backup = backup_dir / f"images_{timestamp}.db"
        try:
            source_conn = sqlite3.connect(str(db_path))
            dest_conn = sqlite3.connect(str(db_backup))
            source_conn.backup(dest_conn)
            dest_conn.close()
            source_conn.close()
            logger.info(f"Created DB rollback snapshot: {db_backup}")
        except Exception as e:
            logger.error(f"Failed to create DB rollback snapshot: {e}")
            db_backup = None

    # Backup settings
    settings_path = Path(get_settings_path())
    if settings_path.exists():
        settings_backup = backup_dir / f"settings_{timestamp}.yaml"
        try:
            shutil.copy2(settings_path, settings_backup)
            logger.info(f"Created settings rollback snapshot: {settings_backup}")
        except Exception as e:
            logger.error(f"Failed to create settings rollback snapshot: {e}")
            settings_backup = None

    return db_backup, settings_backup


def _apply_rollback(db_backup: Path | None, settings_backup: Path | None) -> dict:
    """
    Applies rollback from saved snapshots.
    Called when restore fails after changes have been made.

    Returns:
        dict: {"db_restored": bool, "settings_restored": bool}
    """
    result = {"db_restored": False, "settings_restored": False}

    if db_backup and db_backup.exists():
        try:
            target = Path(get_db_path())
            shutil.copy2(db_backup, target)
            result["db_restored"] = True
            logger.warning(f"ROLLBACK: Restored DB from {db_backup}")
        except Exception as e:
            logger.error(f"ROLLBACK FAILED: Could not restore DB: {e}")

    if settings_backup and settings_backup.exists():
        try:
            target = Path(get_settings_path())
            shutil.copy2(settings_backup, target)
            result["settings_restored"] = True
            logger.warning(f"ROLLBACK: Restored settings from {settings_backup}")
        except Exception as e:
            logger.error(f"ROLLBACK FAILED: Could not restore settings: {e}")

    return result


def _validate_db_schema(db_path: Path) -> tuple[bool, list[str]]:
    """
    Validates that the DB has the expected schema.

    Returns:
        tuple: (is_valid, list_of_issues)
    """
    required_tables = {"images", "detections", "classifications", "sources"}
    issues = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        missing = required_tables - tables
        if missing:
            issues.append(f"Missing required tables: {', '.join(missing)}")

        conn.close()

    except Exception as e:
        issues.append(f"Cannot open DB for validation: {e}")

    return len(issues) == 0, issues


def _get_db_row_counts(db_path: Path) -> dict:
    """
    Gets row counts from DB tables for merge reference.
    """
    counts = {}
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        for table in ["images", "detections", "classifications", "sources"]:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0

        # Check content_hash coverage
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM images WHERE content_hash IS NOT NULL "
                "AND content_hash != ''"
            )
            counts["with_hash"] = cursor.fetchone()[0]
            counts["without_hash"] = counts.get("images", 0) - counts["with_hash"]
        except sqlite3.OperationalError:
            counts["with_hash"] = 0
            counts["without_hash"] = counts.get("images", 0)

        conn.close()
    except Exception as e:
        logger.warning(f"Could not get DB row counts: {e}")

    return counts


def restore_from_archive(
    archive_path: Path,
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = False,
    db_strategy: str = "merge",  # "merge" | "replace"
    on_progress: callable = None,
) -> Generator[dict, None, None]:
    """
    Streaming Restore with Progress-Updates.

    Args:
        archive_path: Path to the tar.gz archive
        include_db: Import database
        include_originals: Import original images
        include_derivatives: Import derivative images
        include_settings: Import settings
        db_strategy: "merge" (add to existing) or "replace" (full replace)
        on_progress: Optional callback for progress updates

    Yields:
        dict: {
            "stage": str,
            "progress": int,
            "total": int,
            "message": str,
            "warnings": list[str],
            "conflicts": list[dict],
            "completed": bool,
            "requires_restart": bool,
            "error": str | None,
        }
    """
    pm = get_path_manager()

    conflicts = []
    warnings = []
    requires_restart = False

    def emit(
        stage: str,
        progress: int,
        total: int,
        message: str,
        completed: bool = False,
        error: str | None = None,
    ) -> None:
        result = {
            "stage": stage,
            "progress": progress,
            "total": total,
            "message": message,
            "warnings": warnings.copy(),
            "conflicts": conflicts.copy(),
            "completed": completed,
            "requires_restart": requires_restart,
            "error": error,
        }
        _set_restore_lock(True, stage)
        return result

    try:
        # Stage 0: Pre-flight checks
        yield emit("preflight", 0, 5, "Checking archive...")

        safe_archive = _validated_archive_path(archive_path)
        if safe_archive is None:
            yield emit(
                "preflight",
                0,
                5,
                "Blocked",
                completed=True,
                error="Archive path outside restore sandbox",
            )
            return
        archive_path = safe_archive

        analysis = analyze_backup_archive(archive_path)
        if analysis["blockers"]:
            yield emit(
                "preflight",
                0,
                5,
                "Blocked",
                completed=True,
                error="; ".join(analysis["blockers"]),
            )
            return

        warnings.extend(analysis["warnings"])

        # Stage 1: Space check
        yield emit("preflight", 1, 5, "Checking disk space...")

        if not _check_disk_space(analysis["total_size_bytes"], pm.base_dir):
            yield emit(
                "preflight",
                1,
                5,
                "Insufficient disk space",
                completed=True,
                error="Not enough disk space for restore",
            )
            return

        # Stage 2: Create rollback snapshot
        yield emit("preflight", 2, 5, "Creating rollback backup...")

        db_rollback, settings_rollback = _create_rollback_snapshot(pm)

        # Stage 3: Extract to staging
        yield emit("extract", 0, analysis["file_count"], "Extracting archive...")

        staging_dir = (
            pm.get_restore_tmp_dir()
            / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            extracted_count = 0
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar:
                    is_safe, _ = _is_safe_tar_path(member)
                    if not is_safe:
                        continue

                    # filter="data" (PEP 706): also rejects device files
                    # and link targets that escape staging_dir.
                    tar.extract(member, staging_dir, filter="data")
                    extracted_count += 1

                    if extracted_count % 50 == 0:
                        yield emit(
                            "extract",
                            extracted_count,
                            analysis["file_count"],
                            f"Extracted {extracted_count} files...",
                        )

            yield emit(
                "extract",
                extracted_count,
                extracted_count,
                f"Extracted {extracted_count} files",
            )

            # Stage 4: Import Settings (if requested)
            if include_settings and analysis["has_settings"]:
                yield emit("settings", 0, 1, "Importing settings...")

                new_settings_path = staging_dir / "settings.yaml"
                if new_settings_path.exists():
                    result = _import_settings(new_settings_path)
                    if result["warnings"]:
                        warnings.extend(result["warnings"])
                    if result["requires_restart"]:
                        requires_restart = True
                        set_restart_required(pm)

                yield emit("settings", 1, 1, "Settings imported")

            # Stage 5: Import Originals (if requested)
            if include_originals and analysis["has_originals"]:
                originals_dir = staging_dir / "originals"
                if originals_dir.exists():
                    yield emit(
                        "originals",
                        0,
                        analysis["originals_count"],
                        "Importing original images...",
                    )

                    imported_count = 0
                    for filepath in originals_dir.rglob("*"):
                        if filepath.is_file():
                            result = _import_original_file(filepath, originals_dir, pm)
                            if result["conflict"]:
                                conflicts.append(result["conflict"])
                            if result["warning"]:
                                warnings.append(result["warning"])

                            imported_count += 1
                            if imported_count % 25 == 0:
                                yield emit(
                                    "originals",
                                    imported_count,
                                    analysis["originals_count"],
                                    f"Imported {imported_count} images...",
                                )

                    yield emit(
                        "originals",
                        imported_count,
                        imported_count,
                        f"Imported {imported_count} original images",
                    )

            # Stage 6: Import Derivatives (if requested)
            if include_derivatives and analysis["has_derivatives"]:
                derivatives_dir = staging_dir / "derivatives"
                if derivatives_dir.exists():
                    yield emit(
                        "derivatives",
                        0,
                        analysis["derivatives_count"],
                        "Importing derivative images...",
                    )

                    imported_count = 0
                    for filepath in derivatives_dir.rglob("*"):
                        if filepath.is_file():
                            _import_derivative_file(filepath, derivatives_dir, pm)
                            imported_count += 1

                    yield emit(
                        "derivatives",
                        imported_count,
                        imported_count,
                        f"Imported {imported_count} derivatives",
                    )

            # Stage 7: Import Database (if requested)
            if include_db and analysis["has_db"]:
                yield emit("database", 0, 1, "Importing database...")

                backup_db_path = staging_dir / "images.db"
                if backup_db_path.exists():
                    # Validate schema
                    is_valid, schema_issues = _validate_db_schema(backup_db_path)
                    if not is_valid:
                        for issue in schema_issues:
                            warnings.append(f"DB Schema: {issue}")
                    else:
                        if db_strategy == "replace":
                            result = _replace_database(backup_db_path, pm)
                            requires_restart = True
                            set_restart_required(pm)
                        else:
                            result = _merge_database(backup_db_path)

                        if result["warnings"]:
                            warnings.extend(result["warnings"])
                        if result.get("conflicts"):
                            conflicts.extend(result["conflicts"])

                yield emit("database", 1, 1, "Database imported")

            # Stage 8: Cleanup
            yield emit("cleanup", 0, 1, "Cleaning up...")

            shutil.rmtree(staging_dir, ignore_errors=True)

            yield emit("cleanup", 1, 1, "Cleanup complete")

            # Final
            summary = "Restore complete. "
            if conflicts:
                summary += f"{len(conflicts)} conflicts resolved. "
            if warnings:
                summary += f"{len(warnings)} warnings. "
            if requires_restart:
                summary += "Restart required."

            yield emit("complete", 1, 1, summary, completed=True)

        except Exception as e:
            logger.error(f"Restore failed during extraction/import: {e}", exc_info=True)

            # Apply rollback if we have snapshots
            rollback_result = _apply_rollback(db_rollback, settings_rollback)
            rollback_msg = ""
            if rollback_result["db_restored"] or rollback_result["settings_restored"]:
                restored_items = []
                if rollback_result["db_restored"]:
                    restored_items.append("DB")
                if rollback_result["settings_restored"]:
                    restored_items.append("settings")
                rollback_msg = (
                    f" Rollback applied: {', '.join(restored_items)} restored."
                )
                logger.info(f"Rollback completed: {rollback_result}")

            # Attempt cleanup
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

            error_msg = f"{str(e)}{rollback_msg}"
            yield emit("error", 0, 1, error_msg, completed=True, error=error_msg)
            return

    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        yield emit("error", 0, 1, str(e), completed=True, error=str(e))

    finally:
        _set_restore_lock(False)


def _import_original_file(filepath: Path, source_root: Path, pm: PathManager) -> dict:
    """
    Imports a single original file with hash-based dedup and conflict handling.

    Returns:
        dict: {"imported": bool, "conflict": dict|None, "warning": str|None}
    """
    result = {"imported": False, "conflict": None, "warning": None}

    relative_path = filepath.relative_to(source_root)
    target_path = pm.originals_dir / relative_path

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate hash of source file
    try:
        source_hash = calculate_sha256(str(filepath))
    except Exception as e:
        result["warning"] = f"Cannot hash {filepath.name}: {e}"
        return result

    # Check if target already exists
    if target_path.exists():
        # Calculate hash of existing file
        try:
            target_hash = calculate_sha256(str(target_path))
        except Exception as e:
            result["warning"] = f"Cannot hash existing {target_path.name}: {e}"
            return result

        if source_hash == target_hash:
            # Same file, skip
            result["imported"] = False
            return result
        else:
            # Conflict: same filename, different hash
            # Deterministic rename
            new_name = _generate_conflict_filename(filepath.name, source_hash)
            conflict_path = target_path.parent / new_name

            shutil.copy2(filepath, conflict_path)
            result["imported"] = True
            result["conflict"] = {
                "original": str(relative_path),
                "renamed_to": new_name,
                "hash": source_hash[:8],
            }
            return result
    else:
        # New file, copy directly
        shutil.copy2(filepath, target_path)
        result["imported"] = True
        return result


def _generate_conflict_filename(filename: str, content_hash: str) -> str:
    """
    Generates a deterministic conflict filename.
    Format: name__conflict_<short-hash>.ext
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    short_hash = content_hash[:8]
    return f"{stem}__conflict_{short_hash}{suffix}"


def _import_derivative_file(filepath: Path, source_root: Path, pm: PathManager) -> bool:
    """
    Imports a single derivative file.
    Derivatives can be overwritten since they're regeneratable.

    Returns:
        bool: True if imported
    """
    relative_path = filepath.relative_to(source_root)
    target_path = pm.derivatives_dir / relative_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        shutil.copy2(filepath, target_path)
        return True

    return False


def _import_settings(new_settings_path: Path) -> dict:
    """
    Imports settings with diff preview and ENV override awareness.

    Returns:
        dict: {"warnings": list, "requires_restart": bool}
    """
    result = {"warnings": [], "requires_restart": False}

    try:
        with open(new_settings_path) as f:
            new_settings = yaml.safe_load(f)

        if not new_settings:
            return result

        # Import via settings module (respects runtime/boot distinction)
        from config import update_runtime_settings

        # Separate runtime vs boot-only keys
        boot_only_keys = {
            "OUTPUT_DIR",
            "DETECTOR_MODEL_CHOICE",
            "WEB_HOST",
            "WEB_PORT",
        }

        env_overrides = os.environ.keys()

        runtime_updates = {}
        for key, value in new_settings.items():
            # Check if ENV override exists
            if key in env_overrides:
                result["warnings"].append(
                    f"{key}: ENV override exists, skipping import"
                )
                continue

            if key in boot_only_keys:
                result["requires_restart"] = True
                result["warnings"].append(f"{key}: Boot-only key, restart required")

            runtime_updates[key] = value

        # Apply updates
        if runtime_updates:
            update_runtime_settings(runtime_updates)
            logger.info(f"Imported {len(runtime_updates)} settings")

    except Exception as e:
        result["warnings"].append(f"Settings import error: {e}")

    return result


def _merge_human_label_tables(
    conn: sqlite3.Connection,
    *,
    image_mapping: dict[str, str],
    detection_mapping: dict[int, int],
    result: dict[str, Any],
) -> dict[str, int]:
    """Merge canonical label histories after parent IDs have been mapped.

    The attached source database must be named ``backup``. Conflicting current
    chains are reported and left untouched; restore never invents a semantic
    ordering between two independently edited histories.
    """
    source_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM backup.sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    destination_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    required = {"label_subjects", "human_label_facts"}
    if not required <= source_tables:
        return {"subjects": 0, "facts": 0, "conflicts": 0}
    if not required <= destination_tables:
        result.setdefault("warnings", []).append(
            "Current database has no canonical human-label schema; label facts skipped"
        )
        return {"subjects": 0, "facts": 0, "conflicts": 0}

    subjects = conn.execute(
        "SELECT * FROM backup.label_subjects ORDER BY subject_id"
    ).fetchall()
    subject_columns = [description[0] for description in conn.execute(
        "SELECT * FROM backup.label_subjects LIMIT 0"
    ).description]
    subject_mapping: dict[int, int] = {}
    imported_subjects = 0

    for raw_subject in subjects:
        subject = dict(zip(subject_columns, raw_subject, strict=True))
        old_subject_id = int(subject["subject_id"])
        source_filename = str(subject["image_filename"])
        target_filename = image_mapping.get(source_filename)
        if target_filename is None:
            result.setdefault("warnings", []).append(
                f"Label subject {old_subject_id} skipped: image was not mapped"
            )
            continue

        target_detection_id: int | None = None
        if subject["scope"] == "object":
            source_detection_id = int(subject["detection_id"])
            target_detection_id = detection_mapping.get(source_detection_id)
            if target_detection_id is None:
                result.setdefault("warnings", []).append(
                    f"Label subject {old_subject_id} skipped: detection was not mapped"
                )
                continue

        if target_detection_id is None:
            existing = conn.execute(
                """
                SELECT subject_id FROM label_subjects
                WHERE scope = 'image' AND image_filename = ?
                """,
                (target_filename,),
            ).fetchone()
        else:
            existing = conn.execute(
                """
                SELECT subject_id FROM label_subjects
                WHERE scope = 'object' AND detection_id = ?
                """,
                (target_detection_id,),
            ).fetchone()

        if existing is not None:
            subject_mapping[old_subject_id] = int(existing[0])
            continue

        cursor = conn.execute(
            """
            INSERT INTO label_subjects (
                scope, image_filename, detection_id,
                proposal_bbox_x, proposal_bbox_y,
                proposal_bbox_w, proposal_bbox_h,
                proposal_coordinate_space, proposal_species_key,
                detector_model_version, classifier_model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject["scope"],
                target_filename,
                target_detection_id,
                subject["proposal_bbox_x"],
                subject["proposal_bbox_y"],
                subject["proposal_bbox_w"],
                subject["proposal_bbox_h"],
                subject["proposal_coordinate_space"],
                subject["proposal_species_key"],
                subject["detector_model_version"],
                subject["classifier_model_version"],
                subject["created_at"],
            ),
        )
        subject_mapping[old_subject_id] = int(cursor.lastrowid)
        imported_subjects += 1

    facts = conn.execute(
        "SELECT * FROM backup.human_label_facts ORDER BY fact_id"
    ).fetchall()
    fact_columns = [description[0] for description in conn.execute(
        "SELECT * FROM backup.human_label_facts LIMIT 0"
    ).description]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for raw_fact in facts:
        fact = dict(zip(fact_columns, raw_fact, strict=True))
        old_subject_id = int(fact["subject_id"])
        if old_subject_id not in subject_mapping:
            continue
        grouped.setdefault((old_subject_id, str(fact["fact_type"])), []).append(fact)

    imported_facts = 0
    conflicts = 0
    for (old_subject_id, fact_type), axis_facts in grouped.items():
        target_subject_id = subject_mapping[old_subject_id]
        existing_axis = conn.execute(
            """
            SELECT 1 FROM human_label_facts
            WHERE subject_id = ? AND fact_type = ?
            LIMIT 1
            """,
            (target_subject_id, fact_type),
        ).fetchone()
        if existing_axis is not None:
            subject = conn.execute(
                """
                SELECT scope, image_filename, detection_id
                FROM label_subjects WHERE subject_id = ?
                """,
                (target_subject_id,),
            ).fetchone()
            result.setdefault("conflicts", []).append(
                {
                    "type": "human_label_fact_axis",
                    "scope": subject[0],
                    "image_filename": subject[1],
                    "detection_id": subject[2],
                    "fact_type": fact_type,
                    "message": "Destination and backup both contain label history",
                }
            )
            conflicts += 1
            continue

        fact_mapping: dict[int, int] = {}
        pending = list(axis_facts)
        while pending:
            progressed = False
            for fact in list(pending):
                old_predecessor = fact["supersedes_fact_id"]
                if old_predecessor is not None and int(old_predecessor) not in fact_mapping:
                    continue
                target_predecessor = (
                    fact_mapping[int(old_predecessor)]
                    if old_predecessor is not None
                    else None
                )
                cursor = conn.execute(
                    """
                    INSERT INTO human_label_facts (
                        schema_version, subject_id, fact_type, assertion_state,
                        answer_value, species_key, bbox_x, bbox_y, bbox_w, bbox_h,
                        coordinate_space, context, source_kind, source_ref,
                        installation_id, app_version, created_at,
                        supersedes_fact_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact["schema_version"],
                        target_subject_id,
                        fact["fact_type"],
                        fact["assertion_state"],
                        fact["answer_value"],
                        fact["species_key"],
                        fact["bbox_x"],
                        fact["bbox_y"],
                        fact["bbox_w"],
                        fact["bbox_h"],
                        fact["coordinate_space"],
                        fact["context"],
                        fact["source_kind"],
                        fact["source_ref"],
                        fact["installation_id"],
                        fact["app_version"],
                        fact["created_at"],
                        target_predecessor,
                    ),
                )
                fact_mapping[int(fact["fact_id"])] = int(cursor.lastrowid)
                imported_facts += 1
                pending.remove(fact)
                progressed = True
            if not progressed:
                raise sqlite3.IntegrityError(
                    "backup human-label supersession chain is incomplete or cyclic"
                )

    return {
        "subjects": imported_subjects,
        "facts": imported_facts,
        "conflicts": conflicts,
    }


def _merge_database(backup_db_path: Path) -> dict:
    """
    Merges backup DB into current DB.

    Strategy:
    - ATTACH backup as read-only
    - Map sources by (name, type, uri)
    - Import images with hash-based dedup
    - Import detections/classifications with ID remapping

    Returns:
        dict: {"warnings": list, "conflicts": list, "stats": dict}
    """
    result = {"warnings": [], "conflicts": [], "stats": {}}

    try:
        current_db_path = get_db_path()
        conn = sqlite3.connect(current_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Attach backup DB
        cursor.execute("ATTACH DATABASE ? AS backup", (str(backup_db_path),))

        # Get backup row counts for stats
        backup_counts = _get_db_row_counts(backup_db_path)
        result["stats"]["backup"] = backup_counts

        # 1. Map sources across both the old ``id`` and current ``source_id``
        # schemas. The merge path accepts backups from either generation.
        source_mapping: dict[int, int] = {}
        cursor.execute("PRAGMA backup.table_info(sources)")
        backup_source_cols = {row[1] for row in cursor.fetchall()}
        cursor.execute("PRAGMA main.table_info(sources)")
        destination_source_cols = {row[1] for row in cursor.fetchall()}
        backup_source_pk = (
            "source_id" if "source_id" in backup_source_cols else "id"
        )
        destination_source_pk = (
            "source_id" if "source_id" in destination_source_cols else "id"
        )

        if backup_source_pk in backup_source_cols:
            cursor.execute(
                f"SELECT {backup_source_pk}, name, type, uri FROM backup.sources"
            )
            for row in cursor.fetchall():
                old_id, name, source_type, uri = row
                cursor.execute(
                    f"SELECT {destination_source_pk} FROM sources "
                    "WHERE name = ? AND type = ? AND uri IS ?",
                    (name, source_type, uri),
                )
                existing = cursor.fetchone()
                if existing:
                    source_mapping[int(old_id)] = int(existing[0])
                else:
                    columns = ["name", "type", "uri"]
                    values: list[Any] = [name, source_type, uri]
                    if "created_at" in destination_source_cols:
                        columns.append("created_at")
                        values.append(datetime.now(UTC).isoformat())
                    placeholders = ", ".join("?" for _ in columns)
                    cursor.execute(
                        f"INSERT INTO sources ({', '.join(columns)}) "
                        f"VALUES ({placeholders})",
                        values,
                    )
                    source_mapping[int(old_id)] = int(cursor.lastrowid)
        else:
            logger.warning(
                "Backup sources table has no supported ID column; source mapping skipped"
            )

        # 2. Import images with hash-based dedup
        images_imported = 0
        images_skipped = 0
        image_mapping: dict[str, str] = {}
        cursor.execute("PRAGMA main.table_info(images)")
        destination_image_cols = {row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT * FROM backup.images")
        backup_images = cursor.fetchall()
        image_columns = [desc[0] for desc in cursor.description]

        for img_row in backup_images:
            img = dict(zip(image_columns, img_row, strict=False))
            content_hash = img.get("content_hash")

            # Hash-based dedup check
            if content_hash:
                cursor.execute(
                    "SELECT filename FROM images WHERE content_hash = ?",
                    (content_hash,),
                )
                hash_match = cursor.fetchone()
                if hash_match:
                    image_mapping[str(img["filename"])] = str(hash_match[0])
                    images_skipped += 1
                    continue

            # Check filename
            cursor.execute(
                "SELECT filename, content_hash FROM images WHERE filename = ?",
                (img["filename"],),
            )
            existing = cursor.fetchone()

            if existing:
                image_mapping[str(img["filename"])] = str(existing[0])
                if content_hash and existing[1] and content_hash != existing[1]:
                    # Same filename, different hash -> warning
                    result["conflicts"].append(
                        {
                            "type": "image",
                            "filename": img["filename"],
                            "message": "Different content hash, skipped",
                        }
                    )
                images_skipped += 1
                continue

            # Map source_id
            old_source_id = img.get("source_id")
            new_source_id = source_mapping.get(old_source_id, old_source_id)

            image_values = dict(img)
            image_values["source_id"] = new_source_id
            columns = [
                column
                for column in image_columns
                if column in destination_image_cols
            ]
            placeholders = ", ".join("?" for _ in columns)
            cursor.execute(
                f"INSERT INTO images ({', '.join(columns)}) VALUES ({placeholders})",
                [image_values.get(column) for column in columns],
            )
            image_mapping[str(img["filename"])] = str(img["filename"])
            images_imported += 1

        result["stats"]["images_imported"] = images_imported
        result["stats"]["images_skipped"] = images_skipped

        # 3. Import detections - need to match by image filename
        detections_imported = 0
        detection_mapping: dict[int, int] = {}
        cursor.execute("PRAGMA main.table_info(detections)")
        destination_detection_cols = {row[1] for row in cursor.fetchall()}
        cursor.execute("PRAGMA main.table_info(classifications)")
        destination_classification_cols = {row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT * FROM backup.detections")
        backup_detections = cursor.fetchall()
        det_columns = [desc[0] for desc in cursor.description]

        for det_row in backup_detections:
            det = dict(zip(det_columns, det_row, strict=False))
            old_det_id_value = det.get("detection_id", det.get("id"))
            if old_det_id_value is None:
                result["warnings"].append("Detection skipped: source ID is missing")
                continue
            old_det_id = int(old_det_id_value)
            target_filename = image_mapping.get(str(det["image_filename"]))

            # Check if image exists in current DB
            if target_filename is None:
                continue

            # Check for duplicate detection
            cursor.execute(
                """SELECT detection_id FROM detections
                   WHERE image_filename = ? AND bbox_x = ? AND bbox_y = ?""",
                (target_filename, det.get("bbox_x"), det.get("bbox_y")),
            )
            existing_detection = cursor.fetchone()
            if existing_detection:
                detection_mapping[old_det_id] = int(existing_detection[0])
                continue

            detection_values = dict(det)
            detection_values["image_filename"] = target_filename
            columns = [
                column
                for column in det_columns
                if column in destination_detection_cols
                and column not in {"detection_id", "id"}
            ]
            placeholders = ", ".join("?" for _ in columns)
            cursor.execute(
                f"INSERT INTO detections ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [detection_values.get(column) for column in columns],
            )
            new_det_id = int(cursor.lastrowid)
            detection_mapping[old_det_id] = new_det_id
            detections_imported += 1

            # Import associated classifications
            cursor.execute(
                "SELECT * FROM backup.classifications WHERE detection_id = ?",
                (old_det_id,),
            )
            for cls_row in cursor.fetchall():
                cls_columns = [desc[0] for desc in cursor.description]
                cls = dict(zip(cls_columns, cls_row, strict=False))
                classification_values = dict(cls)
                classification_values["detection_id"] = new_det_id
                columns = [
                    column
                    for column in cls_columns
                    if column in destination_classification_cols
                    and column not in {"classification_id", "id"}
                ]
                placeholders = ", ".join("?" for _ in columns)
                cursor.execute(
                    f"INSERT INTO classifications ({', '.join(columns)}) "
                    f"VALUES ({placeholders})",
                    [classification_values.get(column) for column in columns],
                )

        result["stats"]["detections_imported"] = detections_imported
        result["stats"]["human_labels"] = _merge_human_label_tables(
            conn,
            image_mapping=image_mapping,
            detection_mapping=detection_mapping,
            result=result,
        )

        # Commit and cleanup
        conn.commit()
        cursor.execute("DETACH DATABASE backup")
        conn.close()

        logger.info(
            f"DB merge complete: {images_imported} images, "
            f"{detections_imported} detections imported"
        )

    except Exception as e:
        logger.error(f"DB merge failed: {e}", exc_info=True)
        result["warnings"].append(f"DB merge error: {e}")

    return result


def _replace_database(backup_db_path: Path, pm: PathManager) -> dict:
    """
    Replaces the current DB with the backup DB.
    Requires restart after completion.

    Returns:
        dict: {"warnings": list, "requires_restart": bool}
    """
    result = {"warnings": [], "requires_restart": True}

    try:
        current_db_path = Path(get_db_path())

        # Validate backup DB first
        is_valid, issues = _validate_db_schema(backup_db_path)
        if not is_valid:
            result["warnings"].extend(issues)
            result["warnings"].append("DB replace aborted due to schema issues")
            return result

        # Close all connections (best effort)
        # This won't work for connections in other threads/processes
        # The restart requirement handles this

        # CRITICAL: Delete WAL files first!
        # SQLite in WAL mode keeps .db-shm and .db-wal files.
        # If we only replace .db, SQLite reads stale data from old WAL files.
        wal_file = current_db_path.with_suffix(".db-wal")
        shm_file = current_db_path.with_suffix(".db-shm")

        if wal_file.exists():
            wal_file.unlink()
            logger.info(f"Deleted WAL file: {wal_file}")
        if shm_file.exists():
            shm_file.unlink()
            logger.info(f"Deleted SHM file: {shm_file}")

        # Atomic swap: copy to temp, then rename
        temp_new = current_db_path.with_suffix(".db.new")
        shutil.copy2(backup_db_path, temp_new)

        # On Unix, rename is atomic
        temp_new.rename(current_db_path)

        logger.info("DB replaced successfully. Restart required.")
        result["warnings"].append("Database replaced. Application restart required.")

    except Exception as e:
        logger.error(f"DB replace failed: {e}", exc_info=True)
        result["warnings"].append(f"DB replace error: {e}")

    return result


def cleanup_restore_tmp(pm: PathManager | None = None) -> None:
    """
    Cleans up the restore temp directory.
    Should be called on application startup.
    """
    if pm is None:
        pm = get_path_manager()

    restore_tmp = pm.base_dir / "restore_tmp"
    if restore_tmp.exists():
        try:
            shutil.rmtree(restore_tmp)
            logger.info("Cleaned up restore temp directory")
        except Exception as e:
            logger.warning(f"Could not cleanup restore temp: {e}")
