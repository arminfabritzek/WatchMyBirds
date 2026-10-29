"""Regenerate a detection thumbnail after its box was corrected by hand.

Thumbnails are written once at capture time from the model's proposal. A
human correction changes the box but not the derivative, so the crop keeps
showing the old region and the person cannot see their own edit.

The square-crop geometry here mirrors ``CropService.create_thumbnail_crop``
so a refreshed crop is indistinguishable from a captured one. It is
duplicated rather than imported because ``detectors`` is off limits to the
web layer, and this runs on the request path.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from pathlib import Path

import cv2

from utils.image_ops import resize_square_crop

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 512
EXPANSION_PERCENT = 0.50
WEBP_QUALITY = 80


def square_crop_box(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    expansion_percent: float = EXPANSION_PERCENT,
) -> tuple[int, int, int, int] | None:
    """Square region around bbox, shifted inside the frame instead of padded."""
    x1, y1, x2, y2 = bbox
    side = int(max(x2 - x1, y2 - y1) * (1 + expansion_percent))
    if side <= 0:
        return None

    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(center_x - side / 2), int(center_y - side / 2)
    right, bottom = left + side, top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width:
        left -= right - frame_width
        right = frame_width
    if bottom > frame_height:
        top -= bottom - frame_height
        bottom = frame_height

    left, top = max(0, left), max(0, top)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def refresh_detection_thumbnail(
    *,
    original_path: Path,
    thumbnail_path: Path,
    bbox_norm: tuple[float, float, float, float],
    size: int = THUMBNAIL_SIZE,
) -> bool:
    """Rewrite one thumbnail from the original using a normalised bbox.

    Returns False on any missing input or unreadable image: a stale crop is
    a cosmetic defect, and must never fail the label write that triggered it.
    """
    if not original_path.is_file():
        logger.debug("crop refresh: original missing (%s)", original_path.name)
        return False

    frame = cv2.imread(str(original_path))
    if frame is None:
        logger.debug("crop refresh: unreadable original (%s)", original_path.name)
        return False

    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = bbox_norm
    pixel_box = (
        int(x * frame_width),
        int(y * frame_height),
        int((x + w) * frame_width),
        int((y + h) * frame_height),
    )

    square = square_crop_box(pixel_box, frame_width, frame_height)
    if square is None:
        return False

    left, top, right, bottom = square
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return False

    resized = resize_square_crop(crop, size)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # Preserve the usable preview until its replacement is fully encoded.
        with tempfile.NamedTemporaryFile(
            dir=thumbnail_path.parent, suffix=".webp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        if not cv2.imwrite(
            str(temporary_path),
            resized,
            [int(cv2.IMWRITE_WEBP_QUALITY), WEBP_QUALITY],
        ):
            return False
        os.replace(temporary_path, thumbnail_path)
        return True
    except (OSError, cv2.error):
        logger.warning("crop refresh: replacement failed (%s)", thumbnail_path.name)
        return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def current_detection_bbox(conn: sqlite3.Connection, detection_id: int) -> tuple | None:
    from utils.db.detections import effective_bbox_projection_sql

    geometry, join = effective_bbox_projection_sql(conn)
    row = conn.execute(
        f"SELECT {', '.join(geometry)} FROM detections d {join} WHERE d.detection_id=?",
        (detection_id,),
    ).fetchone()
    return tuple(row) if row is not None else None
