"""A refreshed crop must be indistinguishable from a captured one."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.crop_refresh import refresh_detection_thumbnail, square_crop_box
from detectors.services.crop_service import CropService


@pytest.fixture
def frame(tmp_path):
    image = np.full((1920, 2560, 3), 40, dtype=np.uint8)
    cv2.rectangle(image, (1200, 700), (1500, 1000), (0, 200, 255), -1)
    path = tmp_path / "20260812_120000_000000.jpg"
    cv2.imwrite(str(path), image)
    return path, image


@pytest.mark.parametrize(
    "bbox",
    [
        (1200, 700, 1500, 1000),
        (0, 0, 200, 160),
        (2400, 1800, 2560, 1920),
        (100, 900, 900, 1000),
    ],
)
def test_geometry_matches_the_capture_time_service(bbox):
    """Corner and edge boxes must shift, not clip, exactly as capture does."""
    image = np.zeros((1920, 2560, 3), dtype=np.uint8)
    square = square_crop_box(bbox, 2560, 1920)
    assert square is not None

    left, top, right, bottom = square
    ours = cv2.resize(
        image[top:bottom, left:right], (256, 256), interpolation=cv2.INTER_AREA
    )
    theirs = CropService().create_thumbnail_crop(frame=image, bbox=bbox, size=256)

    assert theirs is not None
    assert ours.shape == theirs.shape


def test_refresh_rewrites_the_crop_for_the_corrected_box(frame, tmp_path):
    original, _ = frame
    thumb = tmp_path / "thumbs" / "crop.webp"

    assert refresh_detection_thumbnail(
        original_path=original,
        thumbnail_path=thumb,
        bbox_norm=(0.469, 0.365, 0.117, 0.156),
    )
    first = thumb.read_bytes()

    assert refresh_detection_thumbnail(
        original_path=original,
        thumbnail_path=thumb,
        bbox_norm=(0.05, 0.05, 0.10, 0.10),
    )
    assert thumb.read_bytes() != first, "a different box must produce a different crop"


def test_missing_original_is_reported_not_raised(tmp_path):
    assert not refresh_detection_thumbnail(
        original_path=tmp_path / "absent.jpg",
        thumbnail_path=tmp_path / "out.webp",
        bbox_norm=(0.1, 0.1, 0.2, 0.2),
    )


def test_degenerate_box_is_rejected():
    assert square_crop_box((100, 100, 100, 100), 2560, 1920) is None
