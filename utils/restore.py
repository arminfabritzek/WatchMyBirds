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

import yaml

from utils.db import _get_db_path as get_db_path
from utils.ingest import calculate_sha256
from utils.path_manager import get_path_manager
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


def _set_restore_lock(active: bool, stage: str = None):
    """Sets the global restore lock state."""
    global _restore_lock
    _restore_lock["active"] = active
    _restore_lock["stage"] = stage
    if active:
        _restore_lock["started_at"] = datetime.now(UTC).isoformat()
    else:
        _restore_lock["started_at"] = None


# Persistent restart-required marker functions
def set_restart_required(pm) -> None:
    """Creates marker file indicating restart is required after restore."""
    marker = pm.get_restart_required_marker()
    marker.write_text(datetime.now(UTC).isoformat())
    logger.info(f"Restart required marker created: {marker}")


def clear_restart_required(pm) -> None:
    """Removes restart marker (called on fresh app start)."""
    marker = pm.get_restart_required_marker()
    if marker.exists():
        marker.unlink()
        logger.info("Restart required marker cleared")


def is_restart_required(pm) -> bool:
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
    except Exception as e:
        result["blockers"].append(f"Cannot read archive: {e}")
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
                        except Exception as e:
                            result["warnings"].append(
                                f"Could not parse settings.yaml: {e}"
                            )

                elif name == "backup_manifest.json":
                    result["has_manifest"] = True
                    try:
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8")
                            result["manifest"] = json.loads(content)
                    except Exception as e:
                        result["warnings"].append(
                            f"Could not parse backup_manifest.json: {e}"
                        )

                elif name.startswith("originals/"):
                    result["has_originals"] = True
                    if member.isfile():
                        result["originals_count"] += 1

                elif name.startswith("derivatives/"):
                    result["has_derivatives"] = True
                    if member.isfile():
                        result["derivatives_count"] += 1

    except tarfile.TarError as e:
        result["blockers"].append(f"Invalid tar archive: {e}")
    except Exception as e:
        result["blockers"].append(f"Error analyzing archive: {e}")

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


def _create_rollback_snapshot(pm) -> tuple[Path | None, Path | None]:
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
        error: str = None,
    ):
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

                    tar.extract(member, staging_dir)
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


def _import_original_file(filepath: Path, source_root: Path, pm) -> dict:
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


def _import_derivative_file(filepath: Path, source_root: Path, pm) -> bool:
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

        # 1. Map sources (handle old schemas without id column)
        source_mapping = {}  # old_id -> new_id

        # Check if backup has sources table with id column
        cursor.execute("PRAGMA backup.table_info(sources)")
        backup_source_cols = {row[1] for row in cursor.fetchall()}  # column names

        if "id" in backup_source_cols:
            cursor.execute("SELECT id, name, type, uri FROM backup.sources")
            for row in cursor.fetchall():
                old_id, name, source_type, uri = row

                # Check if source exists
                cursor.execute(
                    "SELECT id FROM sources WHERE name = ? AND type = ? AND uri = ?",
                    (name, source_type, uri),
                )
                existing = cursor.fetchone()

                if existing:
                    source_mapping[old_id] = existing[0]
                else:
                    # Create new source
                    cursor.execute(
                        "INSERT INTO sources (name, type, uri, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (name, source_type, uri, datetime.now(UTC).isoformat()),
                    )
                    source_mapping[old_id] = cursor.lastrowid
        else:
            logger.warning(
                "Backup sources table has no 'id' column - skipping source mapping"
            )

        # 2. Import images with hash-based dedup
        images_imported = 0
        images_skipped = 0

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
                if cursor.fetchone():
                    images_skipped += 1
                    continue

            # Check filename
            cursor.execute(
                "SELECT filename, content_hash FROM images WHERE filename = ?",
                (img["filename"],),
            )
            existing = cursor.fetchone()

            if existing:
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

            # Insert image
            cursor.execute(
                """INSERT INTO images (
                    filename, timestamp, original_name, optimized_name,
                    coco_json, source_id, content_hash, downloaded_at,
                    detector_model_id, classifier_model_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    img["filename"],
                    img.get("timestamp"),
                    img.get("original_name"),
                    img.get("optimized_name"),
                    img.get("coco_json", "{}"),
                    new_source_id,
                    content_hash,
                    img.get("downloaded_at"),
                    img.get("detector_model_id"),
                    img.get("classifier_model_id"),
                    img.get("created_at", datetime.now(UTC).isoformat()),
                ),
            )
            images_imported += 1

        result["stats"]["images_imported"] = images_imported
        result["stats"]["images_skipped"] = images_skipped

        # 3. Import detections - need to match by image filename
        detections_imported = 0

        cursor.execute("SELECT * FROM backup.detections")
        backup_detections = cursor.fetchall()
        det_columns = [desc[0] for desc in cursor.description]

        for det_row in backup_detections:
            det = dict(zip(det_columns, det_row, strict=False))

            # Check if image exists in current DB
            cursor.execute(
                "SELECT filename FROM images WHERE filename = ?",
                (det["image_filename"],),
            )
            img_exists = cursor.fetchone()
            if not img_exists:
                continue

            # Check for duplicate detection
            cursor.execute(
                """SELECT detection_id FROM detections
                   WHERE image_filename = ? AND bbox_x = ? AND bbox_y = ?""",
                (det["image_filename"], det.get("bbox_x"), det.get("bbox_y")),
            )
            if cursor.fetchone():
                continue

            # Insert detection
            cursor.execute(
                """INSERT INTO detections (
                    image_filename, image_timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                    od_class_name, od_confidence, od_model_id, score, agreement_score,
                    review_status, thumbnail_path, detector_model_name, detector_model_version,
                    classifier_model_name, classifier_model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    det["image_filename"],
                    det.get("image_timestamp"),
                    det.get("bbox_x"),
                    det.get("bbox_y"),
                    det.get("bbox_w"),
                    det.get("bbox_h"),
                    det.get("od_class_name"),
                    det.get("od_confidence"),
                    det.get("od_model_id"),
                    det.get("score"),
                    det.get("agreement_score"),
                    det.get("review_status", "pending"),
                    det.get("thumbnail_path"),
                    det.get("detector_model_name"),
                    det.get("detector_model_version"),
                    det.get("classifier_model_name"),
                    det.get("classifier_model_version"),
                    det.get("created_at", datetime.now(UTC).isoformat()),
                ),
            )
            new_det_id = cursor.lastrowid
            detections_imported += 1

            # Import associated classifications
            old_det_id = det["id"]
            cursor.execute(
                "SELECT * FROM backup.classifications WHERE detection_id = ?",
                (old_det_id,),
            )
            for cls_row in cursor.fetchall():
                cls_columns = [desc[0] for desc in cursor.description]
                cls = dict(zip(cls_columns, cls_row, strict=False))

                cursor.execute(
                    """INSERT INTO classifications (
                        detection_id, cls_class_name, cls_confidence,
                        cls_model_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        new_det_id,
                        cls.get("cls_class_name"),
                        cls.get("cls_confidence"),
                        cls.get("cls_model_id"),
                        cls.get("created_at"),
                    ),
                )

        result["stats"]["detections_imported"] = detections_imported

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


def _replace_database(backup_db_path: Path, pm) -> dict:
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


def cleanup_restore_tmp(pm=None):
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
