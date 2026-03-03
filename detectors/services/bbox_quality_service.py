"""
BBox Quality Service - Heuristic quality estimation for bounding boxes.

Computes a quality score in [0.0, 1.0] from geometric signals:
- Minimum size (tiny boxes are unreliable)
- Border clipping (boxes at image edges are likely truncated)
- Aspect ratio (extreme ratios indicate bad crops)
- Area ratio (box vs. frame, too large is suspicious)
"""

import numpy as np

# Penalty weights (sum to ~1.0 for interpretability)
_W_SIZE = 0.40
_W_BORDER = 0.30
_W_ASPECT = 0.15
_W_AREA = 0.15


def compute_bbox_quality(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> float:
    """
    Computes a heuristic quality score for a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        frame_shape: Shape of the source frame (H, W, ...).

    Returns:
        Quality score in [0.0, 1.0]. Higher is better.
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = frame_shape[0], frame_shape[1]

    if img_h <= 0 or img_w <= 0:
        return 0.0

    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)

    if box_w <= 0 or box_h <= 0:
        return 0.0

    size_score = _score_size(box_w, box_h, img_w, img_h)
    border_score = _score_border(x1, y1, x2, y2, img_w, img_h)
    aspect_score = _score_aspect_ratio(box_w, box_h)
    area_score = _score_area_ratio(box_w, box_h, img_w, img_h)

    quality = (
        _W_SIZE * size_score
        + _W_BORDER * border_score
        + _W_ASPECT * aspect_score
        + _W_AREA * area_score
    )

    return float(np.clip(quality, 0.0, 1.0))


def _score_size(box_w: int, box_h: int, img_w: int, img_h: int) -> float:
    """
    Score based on minimum dimension relative to image.
    Boxes smaller than ~3% of image dimension score poorly.
    """
    min_dim = min(box_w, box_h)
    img_min = min(img_w, img_h)
    ratio = min_dim / img_min

    # Linear ramp: 0 at ratio<=0.02, 1.0 at ratio>=0.10
    if ratio >= 0.10:
        return 1.0
    if ratio <= 0.02:
        return 0.0
    return (ratio - 0.02) / 0.08


def _score_border(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> float:
    """
    Penalizes boxes touching or near the image border.
    Uses a margin of 1% of image dimension.
    """
    margin_x = max(1, int(img_w * 0.01))
    margin_y = max(1, int(img_h * 0.01))

    touching = 0
    if x1 <= margin_x:
        touching += 1
    if y1 <= margin_y:
        touching += 1
    if x2 >= img_w - margin_x:
        touching += 1
    if y2 >= img_h - margin_y:
        touching += 1

    # 0 edges touching = 1.0, 1 edge = 0.6, 2 edges = 0.25, 3+ = 0.0
    penalties = {0: 1.0, 1: 0.6, 2: 0.25, 3: 0.0, 4: 0.0}
    return penalties.get(touching, 0.0)


def _score_aspect_ratio(box_w: int, box_h: int) -> float:
    """
    Penalizes extreme aspect ratios.
    Birds are roughly 1:1 to 2:1. Ratios beyond 4:1 are suspect.
    """
    ratio = max(box_w, box_h) / max(1, min(box_w, box_h))

    if ratio <= 2.0:
        return 1.0
    if ratio >= 5.0:
        return 0.0
    # Linear decay from 2.0 to 5.0
    return 1.0 - (ratio - 2.0) / 3.0


def _score_area_ratio(box_w: int, box_h: int, img_w: int, img_h: int) -> float:
    """
    Penalizes boxes that are too large relative to the frame.
    A single bird rarely fills >50% of the frame.
    """
    box_area = box_w * box_h
    img_area = img_w * img_h
    ratio = box_area / max(1, img_area)

    if ratio <= 0.30:
        return 1.0
    if ratio >= 0.70:
        return 0.0
    # Linear decay from 0.30 to 0.70
    return 1.0 - (ratio - 0.30) / 0.40
