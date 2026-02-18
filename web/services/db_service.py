"""
DB Service - Web Layer Service for Database Operations.

Thin wrapper over core.db_core for web-specific concerns.
"""

from core import db_core

# --- Connection Management ---


def get_connection():
    """Get a database connection."""
    return db_core.get_connection()


def closing_connection():
    """Context manager that creates and auto-closes a DB connection."""
    return db_core.closing_connection()


# --- Detection Operations ---


def fetch_detections_for_gallery(
    conn, date_iso: str = None, limit: int = None, order_by: str = None
) -> list:
    """Fetch detections for the gallery."""
    return db_core.fetch_detections_for_gallery(
        conn, date_iso, limit=limit, order_by=order_by
    )


def reject_detections(conn, detection_ids: list[int]) -> None:
    """Reject detections (move to trash)."""
    db_core.reject_detections(conn, detection_ids)


def restore_detections(conn, detection_ids: list[int]) -> None:
    """Restore detections from trash."""
    db_core.restore_detections(conn, detection_ids)


def update_review_status(conn, filenames, new_status: str) -> int:
    """Update review status for files."""
    return db_core.update_review_status(conn, filenames, new_status)


def update_downloaded_timestamp(conn, filenames, download_time) -> None:
    """Update download timestamp for files."""
    db_core.update_downloaded_timestamp(conn, filenames, download_time)


# --- Gallery Operations ---


def fetch_daily_covers(conn, min_score: float = 0.0) -> list:
    """Fetch daily cover images."""
    return db_core.fetch_daily_covers(conn, min_score)


def fetch_detection_species_summary(conn, date_iso: str) -> list:
    """Fetch species summary for a date."""
    return db_core.fetch_detection_species_summary(conn, date_iso)


# --- Trash Operations ---


def fetch_trash_items(conn, page: int = 1, limit: int = 50) -> tuple:
    """Fetch items in trash."""
    return db_core.fetch_trash_items(conn, page, limit)


def fetch_trash_count(conn) -> int:
    """Get count of items in trash."""
    return db_core.fetch_trash_count(conn)


def restore_no_bird_images(conn, image_filenames: list[str]) -> int:
    """Restore no_bird images from trash."""
    return db_core.restore_no_bird_images(conn, image_filenames)


# --- Analytics Operations ---


def fetch_analytics_summary(conn) -> dict:
    """Fetch analytics summary."""
    return db_core.fetch_analytics_summary(conn)


def fetch_all_detection_times(conn) -> list:
    """Fetch all detection timestamps."""
    return db_core.fetch_all_detection_times(conn)


def fetch_species_timestamps(conn) -> list:
    """Fetch timestamps grouped by species."""
    return db_core.fetch_species_timestamps(conn)


def fetch_day_count(conn, date_str_iso: str) -> int:
    """Fetch count of detections for a given date."""
    return db_core.fetch_day_count(conn, date_str_iso)


def fetch_review_queue_count(conn, save_threshold: float) -> int:
    """Fetch count of items in review queue."""
    return db_core.fetch_review_queue_count(conn, save_threshold)
