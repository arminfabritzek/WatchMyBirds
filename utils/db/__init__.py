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
from utils.db.connection import (
    DB_FILENAME,
    _ensure_column,
    _ensure_column_on_table,
    _get_db_path,
    _init_schema,
    closing_connection,
    get_connection,
    get_or_create_default_source,
)

# Detection Operations
from utils.db.detections import (
    fetch_active_detection_ids_in_date_range,
    fetch_active_detection_selection_in_date_range,
    fetch_bbox_centers,
    fetch_count_last_24h,
    fetch_daily_covers,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    fetch_detections_last_24h,
    fetch_gallery_total_species_count,
    fetch_hourly_counts,
    fetch_random_favorites,
    fetch_sibling_detections,
    fetch_species_story_board_candidates,
    fetch_trash_candidate_selection_in_date_range,
    insert_classification,
    insert_detection,
    is_detection_visible_in_gallery,
    purge_detections,
    reject_detections,
    restore_detections,
)

# Image Operations
from utils.db.images import (
    insert_image,
    update_downloaded_timestamp,
)

# Reject-audit log (metadata-only, no image/crop/detection row)
# Review Queue Operations
from utils.db.review_queue import (
    delete_orphan_images,
    fetch_orphan_count,
    fetch_orphan_images,
    fetch_recent_review_species,
    fetch_review_cluster_context,
    fetch_review_queue_count,
    fetch_review_queue_image,
    fetch_review_queue_images,
    fetch_review_queue_item_by_identity,
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
    "_ensure_column",
    "_ensure_column_on_table",
    # Images
    "insert_image",
    "update_downloaded_timestamp",
    # Detections
    "insert_detection",
    "insert_classification",
    "is_detection_visible_in_gallery",
    "fetch_active_detection_ids_in_date_range",
    "fetch_active_detection_selection_in_date_range",
    "fetch_trash_candidate_selection_in_date_range",
    "fetch_bbox_centers",
    "fetch_detections_for_gallery",
    "fetch_gallery_total_species_count",
    "fetch_species_story_board_candidates",
    "fetch_sibling_detections",
    "fetch_day_count",
    "fetch_count_last_24h",
    "fetch_detections_last_24h",
    "fetch_hourly_counts",
    "fetch_random_favorites",
    "fetch_daily_covers",
    "fetch_detection_species_summary",
    "reject_detections",
    "restore_detections",
    "purge_detections",
    # Trash
    "fetch_trash_items",
    "fetch_trash_count",
    # Reject-audit log
    # Review Queue
    "fetch_orphan_images",
    "delete_orphan_images",
    "fetch_orphan_count",
    "fetch_review_cluster_context",
    "fetch_review_queue_image",
    "fetch_review_queue_images",
    "fetch_review_queue_item_by_identity",
    "fetch_recent_review_species",
    "fetch_review_queue_count",
]
