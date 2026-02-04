"""
Gallery Service - Web Layer Service for Gallery Operations.

Thin wrapper over core.gallery_core that handles web-specific concerns
like request context and response formatting.
"""

from pathlib import Path

from core import gallery_core


def get_detections_for_date(date_str_iso: str) -> list[dict]:
    """
    Fetch detections for a specific date.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_detections_for_date(date_str_iso)


def get_all_detections() -> list[dict]:
    """
    Fetch all active detections.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_all_detections()


def get_captured_detections() -> list[dict]:
    """
    Get captured detections with caching.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_captured_detections()


def get_captured_detections_by_date() -> dict[str, list]:
    """
    Get detections grouped by date.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_captured_detections_by_date()


def get_daily_covers(common_names: dict[str, str] | None = None) -> dict[str, dict]:
    """
    Get daily cover images for gallery overview.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_daily_covers(common_names)


def get_daily_species_summary(
    date_iso: str, common_names: dict[str, str] | None = None
) -> list[dict]:
    """
    Get per-species counts for a date.

    Delegates to core.gallery_core.
    """
    return gallery_core.get_daily_species_summary(date_iso, common_names)


def invalidate_cache() -> None:
    """Invalidate gallery cache."""
    gallery_core.invalidate_cache()


# --- Thumbnail Generation ---


def generate_preview_thumbnail(
    original_path: str | Path, preview_path: str | Path, size: int = 256
) -> bool:
    """Generate a preview thumbnail."""
    return gallery_core.generate_preview_thumbnail(original_path, preview_path, size)


def get_image_paths(output_dir: str, filename: str) -> dict[str, Path]:
    """Get resolved paths for an image file."""
    return gallery_core.get_image_paths(output_dir, filename)


# --- Sibling Detections ---


def get_sibling_detections(original_name: str) -> list[dict]:
    """Get sibling detections for an image."""
    return gallery_core.get_sibling_detections(original_name)
