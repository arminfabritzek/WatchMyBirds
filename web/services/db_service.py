"""
DB Service - Web Layer Service for Database Operations.

Thin wrapper over core.db_core for web-specific concerns.
"""

from core import db_core

# --- Connection Management ---


def get_connection():
    return db_core.get_connection()


def closing_connection():
    return db_core.closing_connection()


# --- Detection Operations ---


def fetch_detections_for_gallery(
    conn, date_iso: str = None, limit: int = None, order_by: str = None
) -> list:
    return db_core.fetch_detections_for_gallery(
        conn, date_iso, limit=limit, order_by=order_by
    )


def fetch_active_detection_ids_in_date_range(
    conn, from_date: str, to_date: str
) -> list[int]:
    return db_core.fetch_active_detection_ids_in_date_range(conn, from_date, to_date)


def fetch_active_detection_selection_in_date_range(
    conn, from_date: str, to_date: str
) -> dict:
    return db_core.fetch_active_detection_selection_in_date_range(
        conn, from_date, to_date
    )


def fetch_active_detection_selection_by_source_type(
    conn, source_type: str
) -> dict:
    return db_core.fetch_active_detection_selection_by_source_type(conn, source_type)


def fetch_trash_candidate_selection_in_date_range(
    conn, from_date: str, to_date: str
) -> dict:
    return db_core.fetch_trash_candidate_selection_in_date_range(
        conn, from_date, to_date
    )


def fetch_trash_candidate_selection_by_source_type(conn, source_type: str) -> dict:
    return db_core.fetch_trash_candidate_selection_by_source_type(conn, source_type)


def reject_detections(conn, detection_ids: list[int]) -> None:
    db_core.reject_detections(conn, detection_ids)


def apply_species_override(conn, detection_id: int, species: str, source: str) -> None:
    db_core.apply_species_override(conn, detection_id, species, source)


def apply_species_override_many(
    conn, detection_ids: list[int], species: str, source: str
) -> int:
    return db_core.apply_species_override_many(conn, detection_ids, species, source)


def set_manual_bbox_review(
    conn, detection_id: int, review_state: str | None
) -> None:
    db_core.set_manual_bbox_review(conn, detection_id, review_state)


def restore_detections(conn, detection_ids: list[int]) -> None:
    db_core.restore_detections(conn, detection_ids)


def update_review_status(conn, filenames, new_status: str) -> int:
    return db_core.update_review_status(conn, filenames, new_status)


def update_downloaded_timestamp(conn, filenames, download_time) -> None:
    db_core.update_downloaded_timestamp(conn, filenames, download_time)


# --- Gallery Operations ---


def fetch_daily_covers(conn, min_score: float = 0.0) -> list:
    return db_core.fetch_daily_covers(conn, min_score)


def fetch_random_favorites(conn, limit: int = 6) -> list:
    return db_core.fetch_random_favorites(conn, limit=limit)


def fetch_detection_species_summary(conn, date_iso: str) -> list:
    return db_core.fetch_detection_species_summary(conn, date_iso)


# --- Trash Operations ---


def fetch_trash_items(conn, page: int = 1, limit: int = 50) -> tuple:
    return db_core.fetch_trash_items(conn, page, limit)


def fetch_trash_count(conn) -> int:
    return db_core.fetch_trash_count(conn)


def restore_no_bird_images(conn, image_filenames: list[str]) -> int:
    return db_core.restore_no_bird_images(conn, image_filenames)


# --- Analytics Operations ---


def fetch_analytics_summary(conn, min_score: float = 0.0) -> dict:
    return db_core.fetch_analytics_summary(conn, min_score=min_score)


def fetch_all_detection_times(conn, min_score: float = 0.0) -> list:
    return db_core.fetch_all_detection_times(conn, min_score=min_score)


def fetch_species_timestamps(conn, min_score: float = 0.0) -> list:
    return db_core.fetch_species_timestamps(conn, min_score=min_score)


def fetch_day_count(conn, date_str_iso: str) -> int:
    return db_core.fetch_day_count(conn, date_str_iso)


def fetch_review_queue_count(conn, gallery_threshold: float) -> int:
    return db_core.fetch_review_queue_count(conn, gallery_threshold)


def fetch_review_queue_images(
    conn,
    gallery_threshold: float,
    exclude_deep_scanned: bool = False,
) -> list:
    return db_core.fetch_review_queue_images(
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
    return db_core.fetch_review_queue_image(
        conn,
        filename,
        gallery_threshold=gallery_threshold,
        exclude_deep_scanned=exclude_deep_scanned,
    )


def fetch_review_queue_item_by_identity(
    conn,
    item_kind: str,
    item_id: str,
    gallery_threshold: float,
):
    return db_core.fetch_review_queue_item_by_identity(
        conn,
        item_kind,
        item_id,
        gallery_threshold=gallery_threshold,
    )


def fetch_recent_review_species(
    conn, limit: int = 8, lookback_days: int = 7
) -> list:
    return db_core.fetch_recent_review_species(
        conn, limit=limit, lookback_days=lookback_days
    )


# --- 24h Rolling Window Operations ---


def fetch_count_last_24h(conn, threshold_timestamp: str) -> int:
    return db_core.fetch_count_last_24h(conn, threshold_timestamp)


def fetch_detections_last_24h(
    conn, threshold_timestamp: str, limit: int | None = None, order_by: str = "time"
) -> list:
    return db_core.fetch_detections_last_24h(
        conn, threshold_timestamp, limit=limit, order_by=order_by
    )
