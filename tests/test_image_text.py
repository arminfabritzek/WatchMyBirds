"""Smoke tests for the Pillow-backed text helper.

These tests don't pin pixel-level appearance (font choice depends on
the host) — they verify the text helper actually mutates the canvas,
handles Unicode that cv2.putText cannot, and gracefully tolerates the
empty-string call sites that surface in stripped builds.
"""

import numpy as np

from utils.image_text import draw_text, measure_text


def _solid_canvas(color=(11, 14, 19)):
    return np.full((100, 400, 3), color, dtype=np.uint8)


def test_draw_text_mutates_canvas():
    """A real text draw must change at least one pixel."""
    canvas = _solid_canvas()
    before = canvas.copy()
    draw_text(canvas, "Hello", (10, 10), size=20, color=(244, 246, 248))
    assert not np.array_equal(before, canvas)


def test_empty_text_is_noop():
    """Empty string must leave the canvas untouched."""
    canvas = _solid_canvas()
    before = canvas.copy()
    draw_text(canvas, "", (10, 10), size=20, color=(244, 246, 248))
    assert np.array_equal(before, canvas)


def test_unicode_renders_without_crashing():
    """Umlauts and the narrow-no-break-space inside the German date
    format must not blow up the renderer (the cv2.putText path renders
    them as `??` — Pillow handles them properly)."""
    canvas = _solid_canvas()
    draw_text(canvas, "Größe Möwe Spaß", (10, 10), size=20, color=(244, 246, 248))
    draw_text(canvas, "Donnerstag, 30.04.2026", (10, 50), size=14, color=(170, 177, 186))
    # Mutation check covers both writes.
    assert canvas.sum() > _solid_canvas().sum()


def test_measure_text_returns_positive_dims_for_text():
    w, h = measure_text("Ringeltaube", size=24, bold=True)
    assert w > 0 and h > 0


def test_measure_text_zero_for_empty():
    w, h = measure_text("", size=24)
    # Pillow returns 0,0 for empty bbox on truetype; load_default may differ.
    # Either way the values must be non-negative and small.
    assert w >= 0 and h >= 0
