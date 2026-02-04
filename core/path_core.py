"""
Path Core - Path Management Business Logic.

Provides path resolution and management abstracted from the web layer.
"""

from pathlib import Path

from utils.path_manager import get_path_manager as _get_path_manager


def get_path_manager(output_dir: str | None = None):
    """
    Get or create a PathManager instance.

    Args:
        output_dir: Optional output directory override

    Returns:
        PathManager instance
    """
    return _get_path_manager(output_dir)


def get_original_path(output_dir: str, filename: str) -> Path:
    """
    Get the path to an original image.

    Args:
        output_dir: Base output directory
        filename: Image filename

    Returns:
        Path to the original image
    """
    pm = _get_path_manager(output_dir)
    return pm.get_original_path(filename)


def get_derivative_path(output_dir: str, filename: str, derivative_type: str) -> Path:
    """
    Get the path to a derivative image.

    Args:
        output_dir: Base output directory
        filename: Image filename
        derivative_type: Type of derivative ('thumb', 'optimized', etc.)

    Returns:
        Path to the derivative image
    """
    pm = _get_path_manager(output_dir)
    return pm.get_derivative_path(filename, derivative_type)


def get_inbox_pending_path(output_dir: str) -> Path:
    """
    Get the path to the inbox/pending directory.

    Args:
        output_dir: Base output directory

    Returns:
        Path to inbox/pending directory
    """
    pm = _get_path_manager(output_dir)
    return pm.inbox_pending_dir


def get_restore_upload_path(filename: str) -> Path:
    """
    Get the path for a restore upload file.

    Args:
        filename: Upload filename

    Returns:
        Path for the upload
    """
    pm = _get_path_manager()
    return pm.get_restore_upload_path(filename)
