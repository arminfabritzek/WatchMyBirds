"""
DB Core - Database Access Layer.

Provides a clean interface to database operations,
serving as an abstraction over utils.db.
"""

from utils.db import (
    apply_species_override as _apply_species_override,
)
from utils.db import (
    apply_species_override_many as _apply_species_override_many,
)
from utils.db import (
    closing_connection as _closing_connection,
)
from utils.db import (
    fetch_active_detection_ids_in_date_range as _fetch_active_detection_ids_in_date_range,
)
from utils.db import (
    fetch_active_detection_selection_in_date_range as _fetch_active_detection_selection_in_date_range,
)
from utils.db import (
    fetch_active_detection_selection_by_source_type as _fetch_active_detection_selection_by_source_type,
)
from utils.db import (
    fetch_trash_candidate_selection_in_date_range as _fetch_trash_candidate_selection_in_date_range,
)
from utils.db import (
    fetch_trash_candidate_selection_by_source_type as _fetch_trash_candidate_selection_by_source_type,
)
from utils.db import (
    fetch_all_detection_times as _fetch_all_detection_times,
)
from utils.db import (
    fetch_analytics_summary as _fetch_analytics_summary,
)
from utils.db import (
    fetch_count_last_24h as _fetch_count_last_24h,
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
    fetch_detections_last_24h as _fetch_detections_last_24h,
)
from utils.db import (
    fetch_random_favorites as _fetch_random_favorites,
)
from utils.db import (
    fetch_review_queue_count as _fetch_review_queue_count,
)
from utils.db import (
    fetch_review_queue_image as _fetch_review_queue_image,
)
from utils.db import (
    fetch_review_queue_item_by_identity as _fetch_review_queue_item_by_identity,
)
from utils.db import (
    fetch_review_cluster_context as _fetch_review_cluster_context,
)
from utils.db import (
    fetch_review_queue_images as _fetch_review_queue_images,
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
    set_manual_bbox_review as _set_manual_bbox_review,
)
from utils.db import (
    update_downloaded_timestamp as _update_downloaded_timestamp,
)
from utils.db import (
    fetch_recent_review_species as _fetch_recent_review_species,
)
from utils.db import (
    update_review_status as _update_review_status,
)

# --- Connection Management ---


def get_connection():
    return _get_connection()


def closing_connection():
    return _closing_connection()


# --- Detection Operations ---


def fetch_detections_for_gallery(
    conn, date_iso: str = None, limit: int = None, order_by: str = None
) -> list:
    return _fetch_detections_for_gallery(conn, date_iso, limit=limit, order_by=order_by)


def fetch_active_detection_ids_in_date_range(
    conn, from_date: str, to_date: str
) -> list[int]:
    return _fetch_active_detection_ids_in_date_range(conn, from_date, to_date)


def fetch_active_detection_selection_in_date_range(
    conn, from_date: str, to_date: str
) -> dict:
    return _fetch_active_detection_selection_in_date_range(conn, from_date, to_date)


def fetch_active_detection_selection_by_source_type(
    conn, source_type: str
) -> dict:
    return _fetch_active_detection_selection_by_source_type(conn, source_type)


def fetch_trash_candidate_selection_in_date_range(
    conn, from_date: str, to_date: str
) -> dict:
    return _fetch_trash_candidate_selection_in_date_range(conn, from_date, to_date)


def fetch_trash_candidate_selection_by_source_type(
    conn, source_type: str
) -> dict:
    return _fetch_trash_candidate_selection_by_source_type(conn, source_type)


def reject_detections(conn, detection_ids: list[int]) -> None:
    _reject_detections(conn, detection_ids)


def apply_species_override(conn, detection_id: int, species: str, source: str) -> None:
    _apply_species_override(conn, detection_id, species, source)


def apply_species_override_many(
    conn, detection_ids: list[int], species: str, source: str
) -> int:
    return _apply_species_override_many(conn, detection_ids, species, source)


def set_manual_bbox_review(
    conn, detection_id: int, review_state: str | None
) -> None:
    _set_manual_bbox_review(conn, detection_id, review_state)


def restore_detections(conn, detection_ids: list[int]) -> None:
    _restore_detections(conn, detection_ids)


def update_review_status(conn, filenames, new_status: str) -> int:
    return _update_review_status(conn, filenames, new_status)


def update_downloaded_timestamp(conn, filenames, download_time) -> None:
    _update_downloaded_timestamp(conn, filenames, download_time)


# --- Gallery Operations ---


def fetch_daily_covers(conn, min_score: float = 0.0) -> list:
    return _fetch_daily_covers(conn, min_score)


def fetch_random_favorites(conn, limit: int = 6) -> list:
    return _fetch_random_favorites(conn, limit=limit)


def fetch_detection_species_summary(conn, date_iso: str) -> list:
    return _fetch_detection_species_summary(conn, date_iso)


# --- Trash Operations ---


def fetch_trash_items(conn, page: int = 1, limit: int = 50) -> tuple:
    return _fetch_trash_items(conn, page, limit)


def fetch_trash_count(conn) -> int:
    return _fetch_trash_count(conn)


def restore_no_bird_images(conn, image_filenames: list[str]) -> int:
    return _restore_no_bird_images(conn, image_filenames)


# --- Analytics Operations ---


def fetch_analytics_summary(conn, min_score: float = 0.0) -> dict:
    return _fetch_analytics_summary(conn, min_score=min_score)


def fetch_all_detection_times(conn, min_score: float = 0.0) -> list:
    return _fetch_all_detection_times(conn, min_score=min_score)


def fetch_species_timestamps(conn, min_score: float = 0.0) -> list:
    return _fetch_species_timestamps(conn, min_score=min_score)


def fetch_day_count(conn, date_str_iso: str) -> int:
    return _fetch_day_count(conn, date_str_iso)


def fetch_review_queue_count(conn, gallery_threshold: float) -> int:
    return _fetch_review_queue_count(conn, gallery_threshold)


def fetch_review_queue_images(
    conn,
    gallery_threshold: float,
    exclude_deep_scanned: bool = False,
) -> list:
    return _fetch_review_queue_images(
        conn,
        gallery_threshold,
        exclude_deep_scanned=exclude_deep_scanned,
    )


def fetch_review_queue_image(
    conn,
    filename: str,
    gallery_threshold: float,
    exclude_deep_scanned: bool = False,
):
    return _fetch_review_queue_image(
        conn,
        filename,
        gallery_threshold=gallery_threshold,
        exclude_deep_scanned=exclude_deep_scanned,
    )


def fetch_review_cluster_context(
    conn,
    *,
    untagged_time_range,
    context_window_minutes: int = 30,
    max_context_rows: int = 200,
):
    return _fetch_review_cluster_context(
        conn,
        untagged_time_range=untagged_time_range,
        context_window_minutes=context_window_minutes,
        max_context_rows=max_context_rows,
    )


def fetch_review_queue_item_by_identity(
    conn,
    item_kind: str,
    item_id: str,
    gallery_threshold: float = 0.7,
):
    return _fetch_review_queue_item_by_identity(
        conn,
        item_kind,
        item_id,
        gallery_threshold=gallery_threshold,
    )


def fetch_recent_review_species(
    conn, limit: int = 8, lookback_days: int = 7
) -> list:
    return _fetch_recent_review_species(
        conn, limit=limit, lookback_days=lookback_days
    )


# --- 24h Rolling Window Operations ---


def fetch_count_last_24h(conn, threshold_timestamp: str) -> int:
    return _fetch_count_last_24h(conn, threshold_timestamp)


def fetch_detections_last_24h(
    conn, threshold_timestamp: str, limit: int | None = None, order_by: str = "time"
) -> list:
    return _fetch_detections_last_24h(
        conn, threshold_timestamp, limit=limit, order_by=order_by
    )
