"""Tests for BBox Quality Service (P1-01)."""

import pytest

from detectors.services.bbox_quality_service import compute_bbox_quality


@pytest.fixture
def hd_frame_shape():
    """Standard 1080p frame shape (H, W, C)."""
    return (1080, 1920, 3)


def test_tiny_box_scores_low(hd_frame_shape):
    """
    A very small box (e.g. 5x5 on a 1920x1080 frame) should score low
    because it's too small for reliable classification.
    """
    bbox = (500, 500, 505, 505)  # 5x5 pixels
    quality = compute_bbox_quality(bbox, hd_frame_shape)

    assert quality < 0.65, f"Tiny box should score low, got {quality}"


def test_border_clipped_box_scores_low(hd_frame_shape):
    """
    A box touching two edges of the frame should be penalized
    because it's likely a truncated/clipped object.
    """
    # Box touching left and top edges
    bbox = (0, 0, 100, 100)
    quality = compute_bbox_quality(bbox, hd_frame_shape)

    assert quality < 0.8, f"Border-clipped box should score lower, got {quality}"

    # Box touching all 4 edges (frame-filling)
    bbox_full = (0, 0, 1920, 1080)
    quality_full = compute_bbox_quality(bbox_full, hd_frame_shape)

    assert quality_full < quality, (
        f"Full-frame box should score lower than 2-edge box: {quality_full} vs {quality}"
    )


def test_centered_reasonable_box_scores_high(hd_frame_shape):
    """
    A well-sized, centered box with reasonable aspect ratio
    should score high (close to 1.0).
    """
    # ~200x200 centered box on 1920x1080
    bbox = (860, 440, 1060, 640)
    quality = compute_bbox_quality(bbox, hd_frame_shape)

    assert quality > 0.8, f"Centered reasonable box should score high, got {quality}"


def test_extreme_aspect_ratio_penalized(hd_frame_shape):
    """
    A very long and thin box (e.g. 300x20) should be penalized
    for extreme aspect ratio.
    """
    bbox = (500, 500, 800, 520)  # 300x20 pixels (15:1 ratio)
    quality = compute_bbox_quality(bbox, hd_frame_shape)

    assert quality < 0.7, f"Extreme aspect ratio should be penalized, got {quality}"


def test_quality_is_bounded_zero_one(hd_frame_shape):
    """Quality scores must always be in [0.0, 1.0]."""
    test_cases = [
        (0, 0, 1, 1),  # tiny corner
        (0, 0, 1920, 1080),  # full frame
        (500, 300, 700, 500),  # normal
        (960, 540, 961, 541),  # 1x1 pixel
    ]
    for bbox in test_cases:
        q = compute_bbox_quality(bbox, hd_frame_shape)
        assert 0.0 <= q <= 1.0, f"Quality {q} out of bounds for bbox {bbox}"
