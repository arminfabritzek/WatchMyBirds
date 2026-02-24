"""
BACKWARD COMPATIBILITY FACADE.

This file maintains backward compatibility for imports from `utils.db`.
All functionality has been moved to `utils.db/` package.

Usage remains unchanged:
    from utils.db import get_connection, insert_detection, fetch_trash_items

For new code, prefer direct imports from submodules:
    from utils.db.detections import insert_detection
    from utils.db.connection import get_connection
"""

# Re-export everything from the db package
from utils.db import (  # noqa: F401
    DB_FILENAME,
    _ensure_column,
    _ensure_column_on_table,
    _get_db_path,
    _init_schema,
    check_image_exists_by_hash,
    delete_no_bird_images,
    delete_orphan_images,
    fetch_all_detection_times,
    fetch_all_time_daily_counts,
    fetch_analytics_summary,
    fetch_count_last_24h,
    fetch_daily_covers,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    fetch_detections_last_24h,
    fetch_hourly_counts,
    fetch_orphan_count,
    fetch_orphan_images,
    fetch_review_queue_count,
    fetch_review_queue_images,
    fetch_sibling_detections,
    fetch_species_timestamps,
    fetch_trash_count,
    fetch_trash_items,
    get_connection,
    get_or_create_default_source,
    get_or_create_user_import_source,
    insert_classification,
    insert_detection,
    insert_image,
    purge_detections,
    reject_detections,
    restore_detections,
    restore_no_bird_images,
    update_downloaded_timestamp,
    update_review_status,
)
