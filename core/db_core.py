"""
DB Core - Database Access Layer.

Provides a clean interface to database operations,
serving as an abstraction over utils.db.
"""

from utils.db import (
    fetch_all_detection_times as _fetch_all_detection_times,
)
from utils.db import (
    fetch_analytics_summary as _fetch_analytics_summary,
)
from utils.db import (
    fetch_daily_covers as _fetch_daily_covers,
)
from utils.db import (
    fetch_day_count as _fetch_day_count,
)
from utils.db import (
    fetch_detection_species_summary as _fetch_detection_species_summary,
)
from utils.db import (
    fetch_detections_for_gallery as _fetch_detections_for_gallery,
)
from utils.db import (
    fetch_review_queue_count as _fetch_review_queue_count,
)
from utils.db import (
    fetch_species_timestamps as _fetch_species_timestamps,
)
from utils.db import (
    fetch_trash_count as _fetch_trash_count,
)
from utils.db import (
    fetch_trash_items as _fetch_trash_items,
)
from utils.db import (
    get_connection as _get_connection,
)
from utils.db import (
    reject_detections as _reject_detections,
)
from utils.db import (
    restore_detections as _restore_detections,
)
from utils.db import (
    restore_no_bird_images as _restore_no_bird_images,
)
from utils.db import (
    update_downloaded_timestamp as _update_downloaded_timestamp,
)
from utils.db import (
    update_review_status as _update_review_status,
)

# --- Connection Management ---


def get_connection():
    """Get a database connection."""
    return _get_connection()


# --- Detection Operations ---


def fetch_detections_for_gallery(
    conn, date_iso: str = None, limit: int = None, order_by: str = None
) -> list:
    """Fetch detections for the gallery."""
    return _fetch_detections_for_gallery(conn, date_iso, limit=limit, order_by=order_by)


def reject_detections(conn, detection_ids: list[int]) -> None:
    """Reject detections (move to trash)."""
    _reject_detections(conn, detection_ids)


def restore_detections(conn, detection_ids: list[int]) -> None:
    """Restore detections from trash."""
    _restore_detections(conn, detection_ids)


def update_review_status(conn, filenames, new_status: str) -> int:
    """Update review status for files."""
    return _update_review_status(conn, filenames, new_status)


def update_downloaded_timestamp(conn, filenames, download_time) -> None:
    """Update download timestamp for files."""
    _update_downloaded_timestamp(conn, filenames, download_time)


# --- Gallery Operations ---


def fetch_daily_covers(conn, min_score: float = 0.0) -> list:
    """Fetch daily cover images."""
    return _fetch_daily_covers(conn, min_score)


def fetch_detection_species_summary(conn, date_iso: str) -> list:
    """Fetch species summary for a date."""
    return _fetch_detection_species_summary(conn, date_iso)


# --- Trash Operations ---


def fetch_trash_items(conn, page: int = 1, limit: int = 50) -> tuple:
    """Fetch items in trash."""
    return _fetch_trash_items(conn, page, limit)


def fetch_trash_count(conn) -> int:
    """Get count of items in trash."""
    return _fetch_trash_count(conn)


def restore_no_bird_images(conn, image_filenames: list[str]) -> int:
    """Restore no_bird images from trash."""
    return _restore_no_bird_images(conn, image_filenames)


# --- Analytics Operations ---


def fetch_analytics_summary(conn) -> dict:
    """Fetch analytics summary."""
    return _fetch_analytics_summary(conn)


def fetch_all_detection_times(conn) -> list:
    """Fetch all detection timestamps."""
    return _fetch_all_detection_times(conn)


def fetch_species_timestamps(conn) -> list:
    """Fetch timestamps grouped by species."""
    return _fetch_species_timestamps(conn)


def fetch_day_count(conn, date_str_iso: str) -> int:
    """Fetch count of detections for a given date."""
    return _fetch_day_count(conn, date_str_iso)


def fetch_review_queue_count(conn, save_threshold: float) -> int:
    """Fetch count of items in review queue."""
    return _fetch_review_queue_count(conn, save_threshold)
