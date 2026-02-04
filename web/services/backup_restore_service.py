"""
Backup & Restore Service - Web Layer Service for Backup and Restore Operations.

Thin wrapper over core.backup_restore_core for web-specific concerns.
"""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from core import backup_restore_core

# --- Restore State Management ---


def is_restore_active() -> bool:
    """Check if a restore operation is in progress."""
    return backup_restore_core.check_restore_active()


def get_restore_status() -> dict[str, Any]:
    """Get current restore status."""
    return backup_restore_core.get_current_restore_status()


# --- Restart State ---


def is_restart_required(output_dir: str) -> bool:
    """Check if restart is required after restore."""
    return backup_restore_core.check_restart_required(output_dir)


def mark_restart_required(output_dir: str) -> None:
    """Mark that restart is required."""
    backup_restore_core.mark_restart_required(output_dir)


def clear_restart_marker(output_dir: str) -> None:
    """Clear restart marker."""
    backup_restore_core.clear_restart_marker(output_dir)


# --- Archive Analysis ---


def analyze_archive(archive_path: str | Path) -> dict[str, Any]:
    """Analyze a backup archive."""
    return backup_restore_core.analyze_archive(archive_path)


# --- Restore Operations ---


def restore_from_archive(
    archive_path: str | Path,
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = False,
    db_strategy: str = "merge",
    on_progress: Callable | None = None,
) -> dict[str, Any]:
    """Perform restore from archive."""
    return backup_restore_core.perform_restore(
        archive_path,
        include_db=include_db,
        include_originals=include_originals,
        include_derivatives=include_derivatives,
        include_settings=include_settings,
        db_strategy=db_strategy,
        on_progress=on_progress,
    )


# --- Backup Operations ---


def get_backup_stats() -> dict[str, Any]:
    """Get backup size statistics."""
    return backup_restore_core.get_backup_statistics()


def stream_backup(
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = True,
) -> Generator[bytes, None, None]:
    """Stream a backup archive."""
    return backup_restore_core.stream_backup(
        include_db=include_db,
        include_originals=include_originals,
        include_derivatives=include_derivatives,
        include_settings=include_settings,
    )


# --- Constants ---
MAX_ARCHIVE_SIZE_BYTES = backup_restore_core.MAX_ARCHIVE_SIZE


# --- Cleanup ---


def cleanup_temp_files() -> None:
    """Clean up temporary restore files."""
    backup_restore_core.cleanup_temp_files()
