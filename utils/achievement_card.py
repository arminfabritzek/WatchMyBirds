"""Render a single 1080x1080 'achievement card' for one species.

Each card is a standalone, post-ready square: the species photo fills
the frame, a neon rim glow surrounds it, a score badge sits in the
top-right ('3x HEUTE'), and a banner along the bottom carries the
species common name. No app branding, no pagination — every card is
self-contained so the operator can save/forward any single image
without context loss.

The neon palette is keyed to the same 0-7 slot system used by the
Review / Gallery / Stream surfaces (``core.species_colours``), so
species identity stays visually consistent across the whole app:
Ringeltaube is always the same hue, whether it shows up on /review or
on a Telegram card. The hex values are different from the Wong palette
because the report card is dark-on-dark and needs higher saturation to
sing.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.species_colours import SPECIES_COLOUR_SLOTS, assign_species_colours
from utils.image_text import (
    draw_glow_pill,
    draw_glow_rect,
    draw_text,
    draw_text_with_glow,
    measure_text,
)

CARD_SIZE: int = 1080

# Hyper-saturated Blade Runner neon. Slot index i lines up with slot i
# in core.species_colours so a species always picks the same hue. Each
# tuple is (BGR, glow_BGR); the glow companion is a *brighter*, almost-
# white-hot version of the rim — that's what produces the "tube of
# light" bloom on dark cyber backgrounds.
_NEON_PALETTE: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((255, 200, 30),  (255, 230, 140)),   # 0 - electric blue
    ((20, 200, 255),  (140, 230, 255)),   # 1 - hot orange
    ((140, 255, 30),  (220, 255, 160)),   # 2 - acid green
    ((230, 80, 255),  (255, 180, 255)),   # 3 - magenta
    ((255, 230, 100), (255, 250, 200)),   # 4 - cyan-ice
    ((50, 100, 255),  (180, 200, 255)),   # 5 - hot vermilion
    ((120, 255, 255), (210, 255, 255)),   # 6 - electric yellow
    ((220, 130, 255), (240, 200, 255)),   # 7 - violet
]

# Default fallback when the slot map doesn't contain the species — uses
# slot 0's colours so the card still renders.
_DEFAULT_SLOT: int = 0


def neon_for_species(
    species_key: str, colour_map: dict[str, int] | None = None
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(rim_bgr, glow_bgr)`` for *species_key*.

    *colour_map* is the result of ``assign_species_colours(...)`` for
    the report's full species set. When None, the species is assigned
    the default slot — fine for single-species previews.
    """
    if colour_map is not None and species_key in colour_map:
        slot = colour_map[species_key] % SPECIES_COLOUR_SLOTS
    else:
        slot = _DEFAULT_SLOT
    return _NEON_PALETTE[slot]


def _draw_check_icon(
    canvas: np.ndarray,
    centre: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw a circled checkmark — used as the 'verified' glyph next to
    the headline. ``size`` is the diameter of the surrounding circle.
    Color is BGR, matching the rest of the canvas."""
    cx, cy = centre
    r = size // 2
    cv2.circle(canvas, (cx, cy), r, color, thickness=thickness, lineType=cv2.LINE_AA)
    # Checkmark — three points: bottom-left → bottom-centre → top-right.
    p1 = (cx - r // 2, cy + 1)
    p2 = (cx - r // 8, cy + r // 3)
    p3 = (cx + r // 2, cy - r // 3)
    cv2.line(canvas, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, p2, p3, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_calendar_icon(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a small calendar glyph at ``top_left`` with the given side
    ``size``. The glyph is a rounded rectangle with two vertical 'staple'
    legs at the top and a single horizontal divider for the header row."""
    x, y = top_left
    cv2.rectangle(canvas, (x, y + size // 5), (x + size, y + size), color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    # Staples on top
    leg_h = size // 4
    cv2.line(canvas, (x + size // 4, y), (x + size // 4, y + leg_h),
             color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x + 3 * size // 4, y), (x + 3 * size // 4, y + leg_h),
             color, thickness=thickness, lineType=cv2.LINE_AA)
    # Header divider
    cv2.line(canvas, (x, y + size // 5 + size // 4),
             (x + size, y + size // 5 + size // 4),
             color, thickness=max(1, thickness - 1), lineType=cv2.LINE_AA)


def _draw_pin_icon(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a location-pin glyph: a circle on top, triangle pointing down."""
    x, y = top_left
    cx = x + size // 2
    head_r = size // 3
    cv2.circle(canvas, (cx, y + head_r + 1), head_r, color,
               thickness=thickness, lineType=cv2.LINE_AA)
    # Triangle from circle bottom to a point at (cx, y + size).
    pts = np.array(
        [
            [cx - head_r + 1, y + head_r + thickness],
            [cx + head_r - 1, y + head_r + thickness],
            [cx, y + size],
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_camera_icon(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a camera glyph: rounded body with a smaller hump on top and
    a circular lens in the centre."""
    x, y = top_left
    body_top = y + size // 4
    cv2.rectangle(canvas, (x, body_top), (x + size, y + size), color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    # Top hump (the viewfinder bump)
    hump_w = size // 3
    hx = x + (size - hump_w) // 2
    cv2.rectangle(canvas, (hx, y), (hx + hump_w, body_top), color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    # Lens
    lens_r = (size - body_top + y) // 3
    cv2.circle(canvas, (x + size // 2, body_top + (size - body_top + y) // 2),
               lens_r, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_heart_icon(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a heart glyph — two top circles fused with a downward triangle."""
    x, y = top_left
    r = size // 4
    cv2.circle(canvas, (x + r, y + r), r, color,
               thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (x + size - r, y + r), r, color,
               thickness=thickness, lineType=cv2.LINE_AA)
    # Tip triangle
    pts = np.array(
        [
            [x + 1, y + r],
            [x + size - 1, y + r],
            [x + size // 2, y + size - 1],
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop *image* to fill *width* x *height* without distortion."""
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(
        image,
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y1 = max(0, (resized.shape[0] - height) // 2)
    x1 = max(0, (resized.shape[1] - width) // 2)
    return resized[y1 : y1 + height, x1 : x1 + width]


def _bbox_aspect_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    target_w: int,
    target_h: int,
    margin: float = 0.55,
) -> np.ndarray:
    """Crop *image* to a (target_w / target_h) aspect window centred on
    *bbox*, with a ``margin`` of bbox padding around the subject. The
    crop window is shifted (not padded) when it would run off the
    image edges, so the subject stays in frame and there are NO black
    bars even for bboxes near the camera-frame edge.

    Returns an image resized to target_w × target_h.

    The bbox-centred shift behaviour is what differentiates this from
    ``utils.image_ops.create_square_crop``: that helper uses
    BORDER_CONSTANT padding when the desired window goes off-frame,
    which produces visible black bands. Here we instead pull the
    window back into the image, accepting that the bbox may no longer
    sit exactly in the middle of the crop (it just stays inside it).
    """
    src_h, src_w = image.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bbox_w = max(1, bx2 - bx1)
    bbox_h = max(1, by2 - by1)
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    target_aspect = target_w / max(1, target_h)
    # Pick a desired window: at least the bbox + margin in both
    # dimensions, then expand whichever side is necessary to hit the
    # aspect ratio.
    desired_w = int(bbox_w * (1 + margin))
    desired_h = int(bbox_h * (1 + margin))
    # Match aspect by enlarging the smaller side.
    if desired_w / desired_h < target_aspect:
        desired_w = int(desired_h * target_aspect)
    else:
        desired_h = int(desired_w / target_aspect)
    # Don't ask for more than the image has.
    desired_w = min(desired_w, src_w)
    desired_h = min(desired_h, src_h)
    # If the image is too narrow/short for the aspect, downsize the
    # other side too — the resize at the end will scale up.
    if desired_w / desired_h > target_aspect:
        desired_w = int(desired_h * target_aspect)
    elif desired_w / desired_h < target_aspect:
        desired_h = int(desired_w / target_aspect)

    desired_w = max(2, desired_w)
    desired_h = max(2, desired_h)

    # Centre on bbox, then clamp into image bounds (shift instead of pad).
    x1 = cx - desired_w // 2
    y1 = cy - desired_h // 2
    x1 = max(0, min(src_w - desired_w, x1))
    y1 = max(0, min(src_h - desired_h, y1))
    x2 = x1 + desired_w
    y2 = y1 + desired_h

    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def render_achievement_card(
    photo: np.ndarray,
    *,
    common_name: str,
    count: int,
    rim_color: tuple[int, int, int],
    glow_color: tuple[int, int, int],
    device_label: str = "",
    date_label: str = "",
) -> np.ndarray:
    """Render one 1080x1080 achievement card and return it as a BGR array.

    Layout regions, top to bottom:

    * **Photo region** — full width, ~74% of the height. Covered by the
      species photo; cropped centre to fit without letterboxing.
    * **Neon rim** — drawn around the entire card edge. The thickness +
      glow radius are tuned for visibility on phone thumbnails without
      eating into the photo.
    * **Score badge** — pill-shaped, top-right, overlapping the rim:
      "3x HEUTE" in bold caps. The pill colour matches the rim so the
      badge feels like part of the frame, not a sticker.
    * **Bottom banner** — coloured strip ~15% of the height carrying
      the species common name in big bold type plus a small
      device + date subline.
    """
    canvas_w = CARD_SIZE
    canvas_h = CARD_SIZE

    # The photo gets the bulk of the canvas. The bottom 240 px is the
    # banner band — large enough for 92pt all-caps species name plus the
    # device/date sub-line without crowding.
    banner_h = 240
    photo_h = canvas_h - banner_h

    canvas = np.full((canvas_h, canvas_w, 3), (10, 12, 16), dtype=np.uint8)

    # 1. Photo, full-width, top-aligned, cover-cropped.
    photo_fitted = _resize_cover(photo, canvas_w, photo_h)
    canvas[:photo_h, :canvas_w] = photo_fitted

    # 2. Bottom banner (dark, neon-tinted gradient via single-colour fill
    #    plus a thin coloured edge along the top of the banner).
    canvas[photo_h:canvas_h, :canvas_w] = (14, 16, 22)

    # 3. Neon rim around the card edge — Blade Runner tube. Two glow
    #    passes (outer wide bloom + inner crisp stroke) give the
    #    "lit gas tube" feel. Pillow's GaussianBlur on a sacrificial
    #    layer is what produces the real bloom; cv2.rectangle can't.
    rim_inset = 16
    # Pass 1: wide soft halo, well outside the rim line.
    draw_glow_rect(
        canvas,
        (rim_inset, rim_inset),
        (canvas_w - rim_inset - 1, canvas_h - rim_inset - 1),
        color=glow_color,
        thickness=4,
        glow_radius=46,
        glow_intensity=1.6,
        radius=28,
    )
    # Pass 2: thick coloured rim with its own tighter bloom.
    draw_glow_rect(
        canvas,
        (rim_inset, rim_inset),
        (canvas_w - rim_inset - 1, canvas_h - rim_inset - 1),
        color=rim_color,
        thickness=16,
        glow_radius=22,
        glow_intensity=1.5,
        radius=28,
    )

    # 4. Coloured separator above the banner — a thicker neon strip so
    #    the banner band reads as part of the same lit frame.
    sep_y1 = photo_h - 6
    sep_y2 = photo_h + 2
    draw_glow_rect(
        canvas,
        (rim_inset + 28, sep_y1),
        (canvas_w - rim_inset - 28, sep_y2),
        color=rim_color,
        thickness=6,
        glow_radius=24,
        glow_intensity=1.7,
        radius=3,
    )

    # 5. Score badge (top-right, overlapping the rim). Heavier glow,
    #    bigger pill, fatter type — reads from across the room.
    badge_text = f"{count}× HEUTE"
    badge_text_size = 46
    badge_w_text, badge_h_text = measure_text(
        badge_text, size=badge_text_size, bold=True
    )
    badge_pad_x = 34
    badge_pad_y = 18
    badge_w = badge_w_text + badge_pad_x * 2
    badge_h = badge_h_text + badge_pad_y * 2
    badge_x2 = canvas_w - rim_inset - 32
    badge_x1 = badge_x2 - badge_w
    badge_y1 = rim_inset - 8
    badge_y2 = badge_y1 + badge_h

    draw_glow_pill(
        canvas,
        (badge_x1, badge_y1),
        (badge_x2, badge_y2),
        fill=rim_color,
        glow_color=glow_color,
        glow_radius=38,
        glow_intensity=2.0,
    )
    # Badge text — black on bright pill for max contrast.
    draw_text(
        canvas,
        badge_text,
        (badge_x1 + badge_w // 2, badge_y1 + badge_h // 2),
        size=badge_text_size,
        color=(15, 15, 18),
        bold=True,
        anchor="mm",
    )

    # 6. Species name banner — BLADE RUNNER ALL-CAPS, big, double glow.
    #    Two passes: a wide soft halo in the species hue + a tight
    #    crisp shadow in pure white. All-caps removes descenders so the
    #    type reads as a continuous ribbon of light.
    species_caps = common_name.upper()
    name_size = 92
    name_y = photo_h + 64
    # Wide hue halo — reads as the neon glow around the letters.
    draw_text_with_glow(
        canvas,
        species_caps,
        (canvas_w // 2, name_y),
        size=name_size,
        color=glow_color,
        glow_color=glow_color,
        glow_radius=24,
        glow_intensity=1.9,
        bold=True,
        anchor="mm",
    )
    # Crisp white core on top so the letterforms stay legible.
    draw_text_with_glow(
        canvas,
        species_caps,
        (canvas_w // 2, name_y),
        size=name_size,
        color=(250, 252, 255),
        glow_color=(250, 252, 255),
        glow_radius=4,
        glow_intensity=1.0,
        bold=True,
        anchor="mm",
    )

    # 7. Sub-line: device name + date, smaller, mono-tracking feel via
    #    em-spaced separators. Slightly tinted with the rim hue so it
    #    feels part of the lit frame.
    sub_parts = [p for p in (device_label, date_label) if p]
    if sub_parts:
        sub_text = "   ·   ".join(p.upper() for p in sub_parts)
        draw_text(
            canvas,
            sub_text,
            (canvas_w // 2, photo_h + 158),
            size=28,
            color=(190, 198, 210),
            bold=True,
            anchor="mm",
        )

    return canvas


def build_species_colour_map(species_keys: list[str]) -> dict[str, int]:
    """Public wrapper that re-exports ``assign_species_colours`` so
    ``utils.daily_report`` doesn't need to import the core layer."""
    return assign_species_colours(species_keys)


def render_collector_card(
    species: list[dict],
    *,
    colour_map: dict[str, int],
    device_label: str = "",
    date_label: str = "",
) -> np.ndarray:
    """Render the daily 'collector card' summarising every species seen.

    Layout (top to bottom on 1080x1080):

    * **Header band** — compact stats line ("4 CONFIRMED SPECIES · 37
      VISITS") with a wide neon glow halo. Date / device on a sub-line
      below.
    * **Roster grid** — up to 6 species vignettes in a near-square grid
      (1 col for n=1, 2 cols for n=2/4, 3 cols for n=3/5/6). Each
      vignette is a square photo crop with a slot-coloured neon border
      and the common name centred underneath.

    The whole card uses a thicker, double-stacked frame (white outer
    + species-aware inner halo) to differentiate it from the per-species
    cards that precede it in the Telegram album.

    Args:
        species: list of dicts with keys ``scientific``, ``common_name``,
            ``count``, and ``photo`` (a BGR numpy array). Already sorted
            by count descending; this function uses the order as-is.
        colour_map: result of ``build_species_colour_map`` covering
            every species in the list, so each vignette gets the same
            hue as its standalone card earlier in the album.
        device_label: optional device name shown in the header sub-line.
        date_label: optional pre-formatted date shown in the header.
    """
    canvas_w = CARD_SIZE
    canvas_h = CARD_SIZE
    canvas = np.full((canvas_h, canvas_w, 3), (8, 10, 14), dtype=np.uint8)

    species = species[:6]  # roster cap matches the per-species album cap
    n = len(species)

    # Hero hue: pick a "house" colour for the card frame. We use slot 6
    # (electric yellow / pale gold) — it doesn't compete with any
    # species hue and reads as the "this is the daily set" frame.
    hero_rim, hero_glow = _NEON_PALETTE[6]

    # 1. Outer double frame — tightened on HUMAN request so more of the
    #    canvas is photo. Outer hairline at 6px from the edge, coloured
    #    inner rim at +4. Total reclaimed: ~16px on every side vs. the
    #    previous 14+6+14 stack.
    rim_inset = 6
    draw_glow_rect(
        canvas,
        (rim_inset, rim_inset),
        (canvas_w - rim_inset - 1, canvas_h - rim_inset - 1),
        color=(245, 250, 255),
        thickness=2,
        glow_radius=36,
        glow_intensity=1.3,
        radius=22,
    )
    draw_glow_rect(
        canvas,
        (rim_inset + 4, rim_inset + 4),
        (canvas_w - rim_inset - 5, canvas_h - rim_inset - 5),
        color=hero_rim,
        thickness=8,
        glow_radius=20,
        glow_intensity=1.6,
        radius=18,
    )

    # 2. Header band — single centred headline "✓ N CONFIRMED SPECIES",
    #    with a sub-line below it carrying a calendar-icon date and a
    #    pin-icon device label, separated by a thin divider. The total
    #    visit count was removed on HUMAN request: aggregate numbers
    #    competed with the per-vignette visit counts and the headline
    #    became noisy.
    species_word = "SPECIES"  # singular/plural identical, like in EN news headlines
    headline = f"{n} CONFIRMED {species_word}"
    headline_size = 36
    headline_y = 56
    # Measure the headline so we can place a check-icon to its left.
    headline_w, _ = measure_text(headline, size=headline_size, bold=True)
    icon_size = 32
    icon_gap = 14
    block_w = icon_size + icon_gap + headline_w
    block_x = (canvas_w - block_w) // 2
    icon_cx = block_x + icon_size // 2
    icon_cy = headline_y
    _draw_check_icon(
        canvas,
        (icon_cx, icon_cy),
        size=icon_size,
        color=hero_rim,
        thickness=3,
    )
    draw_text_with_glow(
        canvas,
        headline,
        (block_x + icon_size + icon_gap, headline_y),
        size=headline_size,
        color=(248, 250, 255),
        glow_color=hero_glow,
        glow_radius=14,
        glow_intensity=1.5,
        bold=True,
        anchor="lm",  # left-middle: anchor at the start of the text
    )

    # Sub-line: optional calendar+date and pin+device with a vertical
    # divider between them when both are present.
    sub_y = headline_y + 38
    sub_text_size = 20
    sub_color = (180, 190, 205)
    sub_icon_size = 22
    sub_icon_gap = 10

    date_text = (date_label or "").upper()
    device_text = (device_label or "").upper()

    # Pre-measure each segment's width so we can centre-align the whole
    # sub-line. Each segment is "[icon] [icon_gap] [text]".
    segments: list[tuple[str, str]] = []  # (kind, text)
    if date_text:
        segments.append(("calendar", date_text))
    if device_text:
        segments.append(("pin", device_text))

    if segments:
        seg_widths = []
        for _kind, text in segments:
            tw, _ = measure_text(text, size=sub_text_size, bold=True)
            seg_widths.append(sub_icon_size + sub_icon_gap + tw)
        divider_w = 28  # space + thin vertical bar + space between segments
        total_w = sum(seg_widths) + divider_w * (len(segments) - 1)
        cursor_x = (canvas_w - total_w) // 2

        for idx, (kind, text) in enumerate(segments):
            icon_y = sub_y - sub_icon_size // 2
            if kind == "calendar":
                _draw_calendar_icon(
                    canvas,
                    (cursor_x, icon_y),
                    size=sub_icon_size,
                    color=hero_rim,
                    thickness=2,
                )
            else:
                _draw_pin_icon(
                    canvas,
                    (cursor_x, icon_y),
                    size=sub_icon_size,
                    color=hero_rim,
                    thickness=2,
                )
            text_x = cursor_x + sub_icon_size + sub_icon_gap
            draw_text(
                canvas,
                text,
                (text_x, sub_y),
                size=sub_text_size,
                color=sub_color,
                bold=True,
                anchor="lm",
            )
            cursor_x += seg_widths[idx]
            if idx < len(segments) - 1:
                # Thin vertical divider between segments.
                bar_x = cursor_x + divider_w // 2
                cv2.line(
                    canvas,
                    (bar_x, sub_y - 10),
                    (bar_x, sub_y + 10),
                    (90, 100, 115),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                cursor_x += divider_w

    # 3. Roster grid. We pick the column count to keep the grid as
    #    square as possible:
    #      n=1 → 1 col            n=4 → 2 cols (2×2)
    #      n=2 → 2 cols           n=5 → 3 cols (3+2)
    #      n=3 → 3 cols           n=6 → 3 cols (3×3 → 3×2)
    #    The 2×2 case for n=4 is symmetric and reads better than 3+1.
    #    Grid is bracketed top by the sub-line and bottom by the
    #    "Keep watching" banner.
    grid_top = 120
    grid_bottom = canvas_h - 100  # banner takes the bottom ~68px (32 rim + 36 banner)
    grid_left = 22
    grid_right = canvas_w - 22
    grid_w = grid_right - grid_left
    grid_h = grid_bottom - grid_top

    if n <= 0:
        cols = 1
    elif n == 4:
        cols = 2
    else:
        cols = min(3, n)
    rows = max(1, (n + cols - 1) // cols)
    cell_gap = 12
    # The name strip carries three lines: common name + scientific
    # name + visit count. 120px is comfortable.
    name_strip_h = 120

    # Photo width = full cell-width budget. Photo height = whatever's
    # left in the cell after the name strip. This intentionally stops
    # forcing photos into a square — real frames are 16:9 / 4:3, and
    # photos now spread out to fill the available width AND height
    # of each cell, eliminating the side-gutters that the previous
    # square-clamp logic produced (especially at n=4 where cells were
    # ~470 wide but photos clamped to ~272).
    cell_w_budget = (grid_w - (cols - 1) * cell_gap) // cols
    cell_h_budget = (grid_h - (rows - 1) * cell_gap) // rows
    photo_w_in_cell = cell_w_budget
    photo_h_in_cell = cell_h_budget - name_strip_h
    # Guard against pathological cases (n=1 fills the whole canvas, photo
    # could be obscenely tall). Cap the height at 1.4x the width so the
    # frame stays roughly photographic.
    max_photo_h = int(photo_w_in_cell * 1.4)
    if photo_h_in_cell > max_photo_h:
        photo_h_in_cell = max_photo_h
    cell_w = photo_w_in_cell
    cell_h = photo_h_in_cell + name_strip_h

    for idx, entry in enumerate(species):
        row = idx // cols
        col = idx % cols
        # When the last row is partial (e.g. 5 species over a 3-col
        # grid → row 1 has only 2), centre the trailing cells in the row.
        items_in_row = min(cols, n - row * cols)
        row_w = items_in_row * cell_w + (items_in_row - 1) * cell_gap
        row_x0 = (canvas_w - row_w) // 2
        x0_cell = row_x0 + col * (cell_w + cell_gap)
        y0 = grid_top + row * (cell_h + cell_gap)
        # Photo and cell share the same width now — no inner offset.
        x0 = x0_cell

        # Per-species hue.
        scientific = str(entry.get("scientific") or entry.get("common_name") or "")
        rim_color, glow_color = neon_for_species(scientific, colour_map)

        # 3a. Photo cropped to fill the cell's photo region. When the
        #     entry carries a pixel bbox, we do a bbox-aware crop at
        #     the cell's aspect ratio (4:3 / similar) so the bird stays
        #     centred AND the cell fills cleanly without black bars.
        #     Without a bbox we fall back to a centred resize-cover.
        photo = entry.get("photo")
        bbox_px = entry.get("bbox_px")
        fallback_photo = entry.get("fallback_photo")
        fitted = None
        if photo is not None and bbox_px is not None:
            try:
                fitted = _bbox_aspect_crop(
                    photo, bbox_px,
                    target_w=photo_w_in_cell, target_h=photo_h_in_cell,
                    margin=0.55,
                )
            except Exception:
                fitted = None
        if fitted is None and photo is not None:
            fitted = _resize_cover(photo, photo_w_in_cell, photo_h_in_cell)
        if fitted is None and fallback_photo is not None:
            fitted = _resize_cover(fallback_photo, photo_w_in_cell, photo_h_in_cell)
        if fitted is not None:
            canvas[y0 : y0 + photo_h_in_cell, x0 : x0 + photo_w_in_cell] = fitted

        # 3b. Vignette neon border (rim only, no glow inside the photo
        #     so the bird stays visible).
        draw_glow_rect(
            canvas,
            (x0, y0),
            (x0 + photo_w_in_cell - 1, y0 + photo_h_in_cell - 1),
            color=rim_color,
            thickness=6,
            glow_radius=14,
            glow_intensity=1.4,
            radius=12,
        )

        # 3c. Species name strip — three stacked lines below the photo:
        #       1. Common name (large, all-caps, glow-stroked). Italic
        #          when the resolved name is itself a Latin / genus
        #          fallback (e.g. "Phoenicurus (Art unklar)" — the
        #          ``common_is_latin`` flag from
        #          daily_report._resolve_common_name).
        #       2. Scientific name (smaller, dimmer, ITALIC — Latin
        #          binomial convention from field guides). Suppressed
        #          when the common name is itself Latin to avoid the
        #          "Phoenicurus (Art unklar) / Phoenicurus sp." doublet.
        #       3. Visit count (large coloured number + "VISITS" suffix).
        common = str(entry.get("common_name") or scientific.replace("_", " "))
        common_is_latin = bool(entry.get("common_is_latin", False))
        sci_label = scientific.replace("_", " ").strip()
        count = int(entry.get("count", 0) or 0)
        visits_word = "VISIT" if count == 1 else "VISITS"

        common_max = max(8, cell_w // 22)
        if len(common) > common_max:
            common = common[: common_max - 1].rstrip() + "…"
        sci_max = max(10, cell_w // 14)
        if len(sci_label) > sci_max:
            sci_label = sci_label[: sci_max - 1].rstrip() + "…"

        # Y positions: common name baseline 24px below photo, scientific
        # name 26px below that, count line 36px below scientific.
        # When the common name is already Latin we skip the scientific
        # subline entirely and shift the count up to fill the gap.
        common_y = y0 + photo_h_in_cell + 24
        if common_is_latin:
            sci_y = None
            count_y = common_y + 36
        else:
            sci_y = common_y + 26
            count_y = sci_y + 36

        cell_cx = x0_cell + cell_w // 2

        draw_text_with_glow(
            canvas,
            common.upper(),
            (cell_cx, common_y),
            size=28,
            color=(245, 250, 255),
            glow_color=glow_color,
            glow_radius=10,
            glow_intensity=1.3,
            bold=True,
            italic=common_is_latin,
            anchor="mm",
        )
        if sci_y is not None:
            draw_text(
                canvas,
                sci_label,
                (cell_cx, sci_y),
                size=18,
                color=(160, 170, 185),
                bold=False,
                italic=True,
                anchor="mm",
            )
        # Visit count: big tinted number + small uppercased "VISITS".
        count_text = str(count)
        count_w, _ = measure_text(count_text, size=30, bold=True)
        gap = 8
        word_w, _ = measure_text(visits_word, size=15, bold=True)
        line_w = count_w + gap + word_w
        line_x0 = cell_cx - line_w // 2
        draw_text(
            canvas,
            count_text,
            (line_x0, count_y),
            size=30,
            color=glow_color,
            bold=True,
            anchor="lm",
        )
        draw_text(
            canvas,
            visits_word,
            (line_x0 + count_w + gap, count_y + 2),
            size=15,
            color=(170, 180, 195),
            bold=True,
            anchor="lm",
        )

    # 4. Bottom-right pill — "[heart] Thanks to nature!" sign-off on a
    #    short translucent slab. The slab is sized to the content (icon
    #    + gap + text + horizontal padding) instead of running canvas-
    #    wide, so the empty left side from the removed "Keep watching"
    #    text doesn't draw a long ghost line.
    banner_y_top = canvas_h - 88
    banner_y_bot = canvas_h - 44
    banner_cy = (banner_y_top + banner_y_bot) // 2
    banner_icon_size = 22
    banner_text_size = 19
    pill_pad_x = 22
    icon_text_gap = 14
    right_canvas_margin = 60

    thanks_text = "Thanks to nature!"
    thanks_w, _ = measure_text(thanks_text, size=banner_text_size, bold=True)
    pill_inner_w = banner_icon_size + icon_text_gap + thanks_w
    pill_w = pill_inner_w + 2 * pill_pad_x
    pill_x2 = canvas_w - right_canvas_margin
    pill_x1 = pill_x2 - pill_w

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (pill_x1, banner_y_top),
        (pill_x2, banner_y_bot),
        (18, 22, 30),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, dst=canvas)
    cv2.rectangle(
        canvas,
        (pill_x1, banner_y_top),
        (pill_x2, banner_y_bot),
        (60, 70, 85),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    heart_x = pill_x1 + pill_pad_x
    _draw_heart_icon(
        canvas,
        (heart_x, banner_cy - banner_icon_size // 2),
        size=banner_icon_size,
        color=hero_rim,
        thickness=2,
    )
    draw_text(
        canvas,
        thanks_text,
        (heart_x + banner_icon_size + icon_text_gap, banner_cy),
        size=banner_text_size,
        color=(220, 228, 240),
        bold=True,
        anchor="lm",
    )

    return canvas
