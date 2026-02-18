"""
WatchMyBirds Database Module.

This package provides modular database access for the WatchMyBirds application.
All functions are re-exported here for backward compatibility with the original
monolithic utils/db.py structure.

Usage:
    from utils.db import get_connection, insert_detection, fetch_trash_items
    # or
    from utils.db.detections import insert_detection
"""

# Connection and Schema
# Analytics Operations
from utils.db.analytics import (
    fetch_all_detection_times,
    fetch_all_time_daily_counts,
    fetch_analytics_summary,
    fetch_species_timestamps,
)
from utils.db.connection import (
    DB_FILENAME,
    _ensure_column,
    _ensure_column_on_table,
    _get_db_path,
    _init_schema,
    closing_connection,
    get_connection,
    get_or_create_default_source,
    get_or_create_user_import_source,
)

# Detection Operations
from utils.db.detections import (
    fetch_daily_covers,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    fetch_hourly_counts,
    fetch_sibling_detections,
    insert_classification,
    insert_detection,
    purge_detections,
    reject_detections,
    restore_detections,
)

# Image Operations
from utils.db.images import (
    check_image_exists_by_hash,
    insert_image,
    update_downloaded_timestamp,
)

# Review Queue Operations
from utils.db.review_queue import (
    delete_no_bird_images,
    delete_orphan_images,
    fetch_orphan_count,
    fetch_orphan_images,
    fetch_review_queue_count,
    fetch_review_queue_images,
    restore_no_bird_images,
    update_review_status,
)

# Trash Operations
from utils.db.trash import (
    fetch_trash_count,
    fetch_trash_items,
)

__all__ = [
    # Connection
    "DB_FILENAME",
    "_get_db_path",
    "closing_connection",
    "get_connection",
    "_init_schema",
    "get_or_create_default_source",
    "get_or_create_user_import_source",
    "_ensure_column",
    "_ensure_column_on_table",
    # Images
    "insert_image",
    "check_image_exists_by_hash",
    "update_downloaded_timestamp",
    # Detections
    "insert_detection",
    "insert_classification",
    "fetch_detections_for_gallery",
    "fetch_sibling_detections",
    "fetch_day_count",
    "fetch_hourly_counts",
    "fetch_daily_covers",
    "fetch_detection_species_summary",
    "reject_detections",
    "restore_detections",
    "purge_detections",
    # Trash
    "fetch_trash_items",
    "fetch_trash_count",
    # Analytics
    "fetch_all_time_daily_counts",
    "fetch_all_detection_times",
    "fetch_species_timestamps",
    "fetch_analytics_summary",
    # Review Queue
    "fetch_orphan_images",
    "delete_orphan_images",
    "fetch_orphan_count",
    "fetch_review_queue_images",
    "fetch_review_queue_count",
    "restore_no_bird_images",
    "delete_no_bird_images",
    "update_review_status",
]
