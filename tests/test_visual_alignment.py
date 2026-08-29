from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.visual_alignment import FeatureAligner


def _textured_scene() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.full((480, 720, 3), 24, dtype=np.uint8)
    for index in range(160):
        center = tuple(int(value) for value in rng.integers([10, 10], [710, 470]))
        color = tuple(int(value) for value in rng.integers(60, 250, size=3))
        cv2.circle(image, center, int(rng.integers(3, 10)), color, -1)
        if index % 7 == 0:
            cv2.line(
                image,
                (center[0] - 10, center[1]),
                (center[0] + 10, center[1]),
                color,
                2,
            )
    cv2.rectangle(image, (220, 140), (510, 360), (130, 90, 40), 6)
    cv2.putText(
        image,
        "FEEDER",
        (265, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (240, 240, 240),
        3,
    )
    return image


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_feature_alignment_recovers_translation(method, tmp_path):
    if method == "sift" and not hasattr(cv2, "SIFT_create"):
        pytest.skip("OpenCV build has no SIFT")
    reference = _textured_scene()
    expected_dx, expected_dy = 62.0, -31.0
    transform = np.float32([[1, 0, expected_dx], [0, 1, expected_dy]])
    current = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    debug_path = tmp_path / f"{method}.jpg"

    result = FeatureAligner(method=method).align(
        reference, current, debug_path=debug_path
    )

    assert result.success is True
    assert result.inliers >= 8
    assert result.quality >= 0.45
    assert result.pixel_dx == pytest.approx(expected_dx, abs=4.0)
    assert result.pixel_dy == pytest.approx(expected_dy, abs=4.0)
    assert result.scale == pytest.approx(1.0, abs=0.02)
    assert debug_path.exists()


def test_feature_alignment_recovers_zoom_scale():
    reference = _textured_scene()
    transform = cv2.getRotationMatrix2D((360, 240), 0.0, 1.12)
    current = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    result = FeatureAligner(method="auto").align(reference, current)

    assert result.success is True
    assert result.scale == pytest.approx(1.12, abs=0.025)


def test_feature_alignment_projects_clicked_reference_anchor():
    reference = _textured_scene()
    transform = np.float32([[1, 0, 62], [0, 1, -31]])
    current = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    result = FeatureAligner(method="auto").align(
        reference,
        current,
        reference_anchor=(0.75, 0.25),
    )

    assert result.success is True
    assert result.mapped_x == pytest.approx((720 * 0.75 + 62) / 720, abs=0.01)
    assert result.mapped_y == pytest.approx((480 * 0.25 - 31) / 480, abs=0.01)
    assert result.error_x == pytest.approx(result.mapped_x - 0.5)
    assert result.error_y == pytest.approx(result.mapped_y - 0.5)


def test_feature_alignment_rejects_invalid_reference_anchor():
    with pytest.raises(ValueError, match="reference_anchor"):
        FeatureAligner(method="orb").align(
            _textured_scene(), _textured_scene(), reference_anchor=(1.1, 0.5)
        )


def test_feature_alignment_rejects_featureless_frames(tmp_path):
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    debug_path = tmp_path / "failure.jpg"

    result = FeatureAligner(method="orb").align(
        blank, blank.copy(), debug_path=debug_path
    )

    assert result.success is False
    assert result.quality == 0.0
    assert "features" in result.reason
    assert debug_path.exists()
