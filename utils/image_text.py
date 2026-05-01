"""Pillow-backed text rendering for report images.

OpenCV's ``cv2.putText`` uses Hershey stroke fonts: no anti-aliasing
worth the name, no TrueType, no proper Unicode (umlauts and the
narrow-no-break-space in formatted dates render as ``??``). For a
hobbyist surface that the operator looks at on a phone several times a
day, that's the most "alt und nicht schoen" surface in the app.

This module wraps Pillow's ``ImageDraw.text`` so report builders can
draw cleanly anti-aliased TrueType text directly onto a numpy/cv2 BGR
image, with a font-resolution chain that works the same on macOS dev
boxes and the Raspberry Pi target.

Font resolution order:

1. Bundled fonts under ``assets/fonts/`` (when present — empty for now,
   drop a TTF here to override).
2. DejaVu Sans on the RPi (``/usr/share/fonts/truetype/dejavu``).
3. macOS Helvetica/Arial (``/System/Library/Fonts``).
4. Pillow's built-in default bitmap font (last-resort fallback).

The first three return a real TrueType ``ImageFont``; the last returns
the legacy ``load_default()`` raster font. Callers should not branch on
which one they got — everyone uses the unified ``draw_text`` helper.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_FONTS_DIR = REPO_ROOT / "assets" / "fonts"

# Searched in order. Each tuple is (regular_path, bold_path); the first
# pair where regular_path exists wins.
_FONT_CANDIDATES: list[tuple[str, str]] = [
    (
        str(_BUNDLED_FONTS_DIR / "DejaVuSans.ttf"),
        str(_BUNDLED_FONTS_DIR / "DejaVuSans-Bold.ttf"),
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
]


def _boost_alpha(layer: Image.Image, intensity: float) -> Image.Image:
    """Multiply *layer*'s alpha channel by *intensity*, clamped to 255.

    Pillow's ``Image.point`` accepts a callable returning ``int``; the
    closure-with-float-multiplication form would otherwise infer as
    ``ImagePointTransform`` which type-checkers reject. Wrapping in a
    named helper keeps the cast explicit.
    """
    alpha = layer.split()[-1]
    boosted = alpha.point(lambda v: int(min(255, v * intensity)))  # type: ignore[arg-type]
    layer.putalpha(boosted)
    return layer


@lru_cache(maxsize=64)
def _resolve_font(
    size: int, bold: bool = False
) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Return an ImageFont at the requested size, bold or regular."""
    for regular, bold_path in _FONT_CANDIDATES:
        path = bold_path if bold else regular
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except (OSError, ValueError):
                continue
    return ImageFont.load_default()


def measure_text(text: str, *, size: int, bold: bool = False) -> tuple[int, int]:
    """Return (width, height) of *text* rendered at the given size.

    Used by callers that need to right-align or centre text without
    drawing a sacrificial copy first.
    """
    font = _resolve_font(size, bold=bold)
    # Pillow >= 10 uses textbbox; fall back gracefully for older versions.
    try:
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except AttributeError:
        # Legacy Pillow path
        size_tuple = font.getsize(text)  # type: ignore[attr-defined]
        return int(size_tuple[0]), int(size_tuple[1])


def draw_text(
    bgr_image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
    anchor: str = "lt",
) -> np.ndarray:
    """Draw *text* onto a BGR numpy image and return the modified image.

    Args:
        bgr_image: numpy array in OpenCV BGR layout. Modified in place
            *and* returned for fluent-style call chains.
        text: Unicode-safe string to draw.
        xy: anchor point on the image, in pixels.
        size: font size in pixels (Pillow's TrueType size, not point).
        color: BGR tuple matching cv2 conventions. Internally converted
            to RGB for Pillow so callers don't need to think about it.
        bold: when True, use the bold font variant.
        anchor: Pillow anchor spec; ``"lt"`` (left-top) by default.
            Common values: ``"lt"``, ``"rt"``, ``"mt"`` (centre-top),
            ``"lb"`` (left-bottom), ``"rb"`` (right-bottom).

    Why pass an OpenCV-shaped array around: the rest of the report
    builders work with cv2.imread output (BGR uint8) and bake their own
    rectangles via cv2.rectangle. Mixing with PIL would mean two
    full-canvas conversions per tile; instead we do a localised
    PIL->draw->blit cycle per text call.
    """
    if not text:
        return bgr_image

    font = _resolve_font(size, bold=bold)
    # OpenCV BGR -> PIL RGB. Pillow's draw works on RGB Images.
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

    # cv2 colour tuples are BGR; PIL wants RGB. Flip the channel order.
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))

    try:
        draw.text(xy, text, font=font, fill=rgb_color, anchor=anchor)
    except (TypeError, ValueError):
        # load_default() and very old Pillows don't accept anchor.
        draw.text(xy, text, font=font, fill=rgb_color)

    # Push back into the original BGR buffer so callers see the change.
    np.copyto(bgr_image, cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR))
    return bgr_image


def draw_text_with_glow(
    bgr_image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    glow_color: tuple[int, int, int],
    glow_radius: int = 8,
    glow_intensity: float = 1.0,
    bold: bool = False,
    anchor: str = "lt",
) -> np.ndarray:
    """Draw text with a soft outer glow (Pillow GaussianBlur on a mask).

    Used for the score-badge digits and species banner on the
    achievement card. Drops the glow first, then the crisp text on top
    so the highlight stays sharp inside the bloom.
    """
    if not text:
        return bgr_image

    font = _resolve_font(size, bold=bold)
    img_h, img_w = bgr_image.shape[:2]

    # Build the glow on its own RGBA layer so blur doesn't leak into
    # already-drawn pixels of the underlying image.
    glow_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    rgb_glow = (int(glow_color[2]), int(glow_color[1]), int(glow_color[0]), 255)
    try:
        glow_draw.text(xy, text, font=font, fill=rgb_glow, anchor=anchor)
    except (TypeError, ValueError):
        glow_draw.text(xy, text, font=font, fill=rgb_glow)

    # Two-pass blur for a softer halo.
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    if glow_intensity != 1.0:
        glow_layer = _boost_alpha(glow_layer, glow_intensity)

    # Composite glow under the existing canvas.
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    base = Image.alpha_composite(base, glow_layer)

    # Draw the crisp text on top.
    draw = ImageDraw.Draw(base)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]), 255)
    try:
        draw.text(xy, text, font=font, fill=rgb_color, anchor=anchor)
    except (TypeError, ValueError):
        draw.text(xy, text, font=font, fill=rgb_color)

    rendered = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    np.copyto(bgr_image, rendered)
    return bgr_image


def draw_glow_rect(
    bgr_image: np.ndarray,
    xy1: tuple[int, int],
    xy2: tuple[int, int],
    *,
    color: tuple[int, int, int],
    thickness: int = 4,
    glow_radius: int = 18,
    glow_intensity: float = 1.4,
    radius: int = 0,
) -> np.ndarray:
    """Draw a neon-glow rectangle outline onto *bgr_image*.

    The visible result is two stacked passes:
      1. A wide, blurred copy of the rectangle in *color* — the glow.
      2. A crisp inner stroke in the same colour — the rim of the neon
         tube.

    *radius* enables rounded corners; pass 0 for sharp Cyber rectangles.
    """
    img_h, img_w = bgr_image.shape[:2]
    glow_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    rgba = (int(color[2]), int(color[1]), int(color[0]), 255)
    if radius > 0:
        glow_draw.rounded_rectangle(
            [xy1, xy2], radius=radius, outline=rgba, width=thickness
        )
    else:
        glow_draw.rectangle([xy1, xy2], outline=rgba, width=thickness)

    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    if glow_intensity != 1.0:
        glow_layer = _boost_alpha(glow_layer, glow_intensity)

    base = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base = Image.alpha_composite(base, glow_layer)

    # Crisp inner stroke on top of the bloom.
    draw = ImageDraw.Draw(base)
    if radius > 0:
        draw.rounded_rectangle(
            [xy1, xy2], radius=radius, outline=rgba, width=max(1, thickness // 2)
        )
    else:
        draw.rectangle([xy1, xy2], outline=rgba, width=max(1, thickness // 2))

    rendered = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    np.copyto(bgr_image, rendered)
    return bgr_image


def draw_glow_pill(
    bgr_image: np.ndarray,
    xy1: tuple[int, int],
    xy2: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    glow_color: tuple[int, int, int] | None = None,
    glow_radius: int = 22,
    glow_intensity: float = 1.5,
    radius: int | None = None,
) -> np.ndarray:
    """Draw a filled rounded pill with a soft outer glow.

    Used for the score badge — the pill itself is opaque so text inside
    stays legible; the glow is what gives it the cyber-trophy feel.
    """
    x1, y1 = xy1
    x2, y2 = xy2
    pill_h = y2 - y1
    if radius is None:
        radius = pill_h // 2

    img_h, img_w = bgr_image.shape[:2]
    glow_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    glow_rgba = (
        int((glow_color or fill)[2]),
        int((glow_color or fill)[1]),
        int((glow_color or fill)[0]),
        255,
    )
    glow_draw.rounded_rectangle([xy1, xy2], radius=radius, fill=glow_rgba)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    if glow_intensity != 1.0:
        glow_layer = _boost_alpha(glow_layer, glow_intensity)

    base = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base = Image.alpha_composite(base, glow_layer)

    # Solid pill body on top.
    draw = ImageDraw.Draw(base)
    fill_rgba = (int(fill[2]), int(fill[1]), int(fill[0]), 255)
    draw.rounded_rectangle([xy1, xy2], radius=radius, fill=fill_rgba)

    rendered = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    np.copyto(bgr_image, rendered)
    return bgr_image
