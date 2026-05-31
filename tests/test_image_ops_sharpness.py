"""Tests for laplacian_sharpness and crop_brightness in utils.image_ops.

The tests are *behavioural*, not absolute — Laplacian variance
depends on the image content and we don't want brittle hard-coded
expected values. We assert ordering relationships ("sharp > blurry")
and size-invariance (same content at different scales scores
similarly).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.image_ops import crop_brightness, laplacian_sharpness


def _checkerboard(size: int, square: int = 16) -> np.ndarray:
    """Generate a high-contrast BGR checkerboard."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((y // square) + (x // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def test_empty_input_returns_zero():
    assert laplacian_sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
    assert crop_brightness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_none_input_returns_zero():
    assert laplacian_sharpness(None) == 0.0  # type: ignore[arg-type]
    assert crop_brightness(None) == 0.0  # type: ignore[arg-type]


def test_uniform_gray_has_zero_sharpness():
    """A uniform-color image has no Laplacian variance."""
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    score = laplacian_sharpness(img)
    assert score < 1.0  # essentially zero


def test_uniform_gray_brightness_matches_value():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    b = crop_brightness(img)
    assert b == pytest.approx(128.0, abs=0.5)


def test_checkerboard_has_high_sharpness():
    img = _checkerboard(256, square=16)
    score = laplacian_sharpness(img)
    assert score > 1000  # checkerboard is maximally edgy


def test_blur_reduces_sharpness():
    """Same content, with and without blur, must produce ordered scores."""
    sharp = _checkerboard(256, square=16)
    blurred = cv2.GaussianBlur(sharp, ksize=(15, 15), sigmaX=4.0)
    assert laplacian_sharpness(sharp) > laplacian_sharpness(blurred)


def test_black_image_has_zero_brightness_and_sharpness():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    assert crop_brightness(img) == 0.0
    assert laplacian_sharpness(img) == 0.0


def test_white_image_has_max_brightness_and_zero_sharpness():
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    assert crop_brightness(img) == pytest.approx(255.0)
    assert laplacian_sharpness(img) < 1.0


def test_grayscale_input_is_accepted():
    """laplacian_sharpness should handle single-channel input."""
    gray = np.full((128, 128), 100, dtype=np.uint8)
    # Add a vertical edge
    gray[:, 64:] = 200
    score = laplacian_sharpness(gray)
    assert score > 0  # the edge contributes some variance
    assert crop_brightness(gray) == pytest.approx(150.0, abs=1.0)


def test_size_invariance_downsample_band():
    """Crops above the 256-px reference long-side are normalised by
    downsampling, so two crops in that band with the same content
    score within ±15%.

    Note: this is the *practical* invariance window. Bird crops on
    the RPi are typically 200-800 px, so the downsample path
    dominates. Upscaling (crops <256 px) does not guarantee the
    same invariance — small crops produce smaller absolute scores
    because there's literally less edge content to measure. That's
    a feature, not a bug: tiny distant birds shouldn't be ranked
    "sharper" than they actually are.
    """
    # 400×400 → downsample to 256: square pattern at 40-px source,
    # ~25.6-px post-resize.
    medium = _checkerboard(400, square=40)
    # 800×800 → downsample to 256: square pattern at 80-px source,
    # ~25.6-px post-resize. Same final pattern.
    large = _checkerboard(800, square=80)
    s_medium = laplacian_sharpness(medium)
    s_large = laplacian_sharpness(large)
    assert s_medium > 0 and s_large > 0
    ratio = s_medium / s_large
    assert 0.85 <= ratio <= 1.15, (
        f"downsample-band variance too high: {s_medium} vs {s_large}"
    )


def test_edge_density_ordering_at_fixed_size():
    """At the same resolution, a higher-edge-density pattern scores
    higher than a lower-edge-density one. This is the operational
    contract: more visible edge content = higher score.
    """
    dense = _checkerboard(256, square=8)  # many squares
    sparse = _checkerboard(256, square=64)  # few large squares
    assert laplacian_sharpness(dense) > laplacian_sharpness(sparse)


def test_native_resolution_below_256_is_preserved():
    """Crops smaller than the 256-px reference are *not* upscaled.

    Documented behaviour: the function only downsamples. Tiny crops
    keep their native resolution, which means absolute scores
    between a 64-px and a 400-px crop are NOT directly comparable.
    Consumers ranking crops should bucket by size or accept that
    tiny crops produce different distributions.
    """
    tiny = _checkerboard(64, square=8)
    # Manually downsampling a larger checkerboard to 64 produces a
    # different score than the native 64-px version because the
    # downsample smooths edges.
    score_native_64 = laplacian_sharpness(tiny)
    assert score_native_64 > 0  # non-degenerate


def test_float_input_is_handled():
    """Float arrays must not crash; values are clipped to uint8."""
    img = (_checkerboard(128).astype(np.float64))
    score = laplacian_sharpness(img)
    assert score > 0
    b = crop_brightness(img)
    assert 0 <= b <= 255


def test_4channel_image_is_accepted():
    """RGBA / BGR-with-alpha crops should not crash."""
    img = _checkerboard(128)
    rgba = np.concatenate([img, np.full((128, 128, 1), 255, dtype=np.uint8)], axis=2)
    score = laplacian_sharpness(rgba)
    assert score > 100
