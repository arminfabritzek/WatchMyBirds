"""
Backup & Restore Core - Business Logic for Backup and Restore Operations.

Provides a clean interface to backup and restore functionality,
abstracting away the infrastructure details.
"""

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from utils.backup import (
    get_backup_stats,
    stream_backup_archive,
)
from utils.path_manager import get_path_manager

# Import from infrastructure
from utils.restore import (
    MAX_ARCHIVE_SIZE_BYTES,
    analyze_backup_archive,
    cleanup_restore_tmp,
    clear_restart_required,
    get_restore_status,
    is_restart_required,
    is_restore_active,
    restore_from_archive,
    set_restart_required,
)

logger = logging.getLogger(__name__)

# Re-export constants
MAX_ARCHIVE_SIZE = MAX_ARCHIVE_SIZE_BYTES


# --- Restore State Management ---


def check_restore_active() -> bool:
    """
    Check if a restore operation is currently in progress.

    Returns:
        True if restore is active
    """
    return is_restore_active()


def get_current_restore_status() -> dict[str, Any]:
    """
    Get current restore operation status.

    Returns:
        Dictionary with restore status info
    """
    return get_restore_status()


# --- Restart Required State ---


def check_restart_required(output_dir: str) -> bool:
    """
    Check if a restart is required after restore.

    Args:
        output_dir: Output directory path

    Returns:
        True if restart is required
    """
    pm = get_path_manager(output_dir)
    return is_restart_required(pm)


def mark_restart_required(output_dir: str) -> None:
    """
    Mark that a restart is required.

    Args:
        output_dir: Output directory path
    """
    pm = get_path_manager(output_dir)
    set_restart_required(pm)


def clear_restart_marker(output_dir: str) -> None:
    """
    Clear the restart required marker.

    Args:
        output_dir: Output directory path
    """
    pm = get_path_manager(output_dir)
    clear_restart_required(pm)


# --- Backup Archive Analysis ---


def analyze_archive(archive_path: str | Path) -> dict[str, Any]:
    """
    Analyzes a backup archive without extracting.

    Args:
        archive_path: Path to the tar.gz archive

    Returns:
        Dictionary with archive analysis results
    """
    return analyze_backup_archive(Path(archive_path))


# --- Restore Operations ---


def perform_restore(
    archive_path: str | Path,
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = False,
    db_strategy: str = "merge",
    on_progress: Callable | None = None,
) -> dict[str, Any]:
    """
    Performs a restore operation from a backup archive.

    Args:
        archive_path: Path to the tar.gz archive
        include_db: Import database
        include_originals: Import original images
        include_derivatives: Import derivative images
        include_settings: Import settings
        db_strategy: "merge" or "replace"
        on_progress: Progress callback function

    Returns:
        Dictionary with restore results
    """
    return restore_from_archive(
        Path(archive_path),
        include_db=include_db,
        include_originals=include_originals,
        include_derivatives=include_derivatives,
        include_settings=include_settings,
        db_strategy=db_strategy,
        on_progress=on_progress,
    )


# --- Backup Operations ---


def get_backup_statistics() -> dict[str, Any]:
    """
    Get statistics about data available for backup.

    Returns:
        Dictionary with backup size statistics
    """
    return get_backup_stats()


def stream_backup(
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = True,
) -> Generator[bytes, None, None]:
    """
    Creates a streaming backup archive.

    Args:
        include_db: Include database
        include_originals: Include original images
        include_derivatives: Include derivative images
        include_settings: Include settings

    Returns:
        Generator yielding bytes chunks of the archive
    """
    return stream_backup_archive(
        include_db=include_db,
        include_originals=include_originals,
        include_derivatives=include_derivatives,
        include_settings=include_settings,
    )


# --- Cleanup ---


def cleanup_temp_files() -> None:
    """
    Cleans up temporary files from restore operations.
    """
    cleanup_restore_tmp()
