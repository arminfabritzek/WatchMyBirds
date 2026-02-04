"""
Path Service - Web Layer Service for Path Operations.

Thin wrapper over core.path_core for web-specific concerns.
"""

from pathlib import Path

from core import path_core


def get_path_manager(output_dir: str | None = None):
    """Get or create a PathManager instance."""
    return path_core.get_path_manager(output_dir)


def get_original_path(output_dir: str, filename: str) -> Path:
    """Get the path to an original image."""
    return path_core.get_original_path(output_dir, filename)


def get_derivative_path(output_dir: str, filename: str, derivative_type: str) -> Path:
    """Get the path to a derivative image."""
    return path_core.get_derivative_path(output_dir, filename, derivative_type)


def get_inbox_pending_path(output_dir: str) -> Path:
    """Get the path to the inbox/pending directory."""
    return path_core.get_inbox_pending_path(output_dir)


def get_restore_upload_path(filename: str) -> Path:
    """Get the path for a restore upload file."""
    return path_core.get_restore_upload_path(filename)
