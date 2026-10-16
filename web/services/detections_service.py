"""
Detections Service - Web Layer Service for Detection Operations.

Thin wrapper over core.detections_core for web-specific concerns.
"""

from typing import Any

from core import detections_core


def move_to_trash(conn: object, data: dict) -> dict[str, int]:
    return detections_core.move_to_trash(conn, data)


def reject_detections(detection_ids: list[int]) -> bool:
    """
    Mark detections as rejected (move to trash).

    Delegates to core.detections_core.
    """
    return detections_core.reject_detections(detection_ids)


def restore_detections(detection_ids: list[int]) -> bool:
    """
    Restore detections from trash.

    Delegates to core.detections_core.
    """
    return detections_core.restore_detections(detection_ids)


def hard_delete_detections(detection_ids: list[int]) -> dict[str, Any]:
    """
    Permanently delete detections.

    Delegates to core.detections_core.
    """
    return detections_core.hard_delete_detections(detection_ids)


def get_trash_items() -> list[dict]:
    """
    Get all items in trash.

    Delegates to core.detections_core.
    """
    return detections_core.get_trash_items()


def get_trash_count() -> int:
    """
    Get count of items in trash.

    Delegates to core.detections_core.
    """
    return detections_core.get_trash_count()


def get_review_queue_count(gallery_threshold: float) -> int:
    """
    Get count of items needing review.

    Delegates to core.detections_core.
    """
    return detections_core.get_review_queue_count(gallery_threshold)


# --- Hard Delete Operations ---


def hard_delete_detections_with_conn(
    conn,
    detection_ids: list[int] | None = None,
    before_date: str | None = None,
) -> dict[str, Any]:
    """Permanently delete detections with existing connection."""
    return detections_core.hard_delete_detections_with_conn(
        conn, detection_ids=detection_ids, before_date=before_date
    )


# --- Review Queue ---


def get_review_queue_images(output_dir: str, gallery_threshold: float) -> list[dict]:
    """Get images for the review queue."""
    return detections_core.get_review_queue_images(output_dir, gallery_threshold)


def get_preview_path(output_dir: str, filename: str):
    """Get the preview thumbnail path for a filename."""
    return detections_core.get_preview_path(output_dir, filename)
