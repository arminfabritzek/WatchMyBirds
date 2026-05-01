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
    set_code: str = "",
) -> np.ndarray:
    """Render the daily 'collector card' summarising every species seen.

    Layout (top to bottom on 1080x1080):

    * **Header band** — bold stats line ("6 ARTEN · 29 SICHTUNGEN") with
      a wide neon glow halo. Date / device on a sub-line below.
    * **Roster grid** — up to 6 species vignettes (2 columns × 3 rows
      max). Each vignette is a square photo crop with a slot-coloured
      neon border, the common name centred underneath in the species
      hue, and the count as a small pill in the vignette's top-right.
    * **Footer set-code** — discreet bottom strip with a "set code"
      string ("30.04.2026 · 6/6") so the card reads as a unique daily
      collectible, like a TCG card.

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
        set_code: short identifier shown in the footer (e.g.
            "30.04.2026 · 6 SPECIES"). Falls back to the date if empty.
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

    # 1. Outer double frame — wider than per-species cards so the
    #    collector card feels like the "boss" of the album.
    rim_inset = 14
    # Outer halo (white, very wide bloom).
    draw_glow_rect(
        canvas,
        (rim_inset, rim_inset),
        (canvas_w - rim_inset - 1, canvas_h - rim_inset - 1),
        color=(245, 250, 255),
        thickness=3,
        glow_radius=58,
        glow_intensity=1.4,
        radius=32,
    )
    # Coloured inner rim (hero hue).
    draw_glow_rect(
        canvas,
        (rim_inset + 6, rim_inset + 6),
        (canvas_w - rim_inset - 7, canvas_h - rim_inset - 7),
        color=hero_rim,
        thickness=14,
        glow_radius=28,
        glow_intensity=1.7,
        radius=26,
    )

    # 2. Header band: "X ARTEN · Y SICHTUNGEN".
    total_obs = sum(int(s.get("count", 0) or 0) for s in species)
    art_label = "ART" if n == 1 else "ARTEN"
    obs_label = "SICHTUNG" if total_obs == 1 else "SICHTUNGEN"
    headline = f"{n} {art_label}  ·  {total_obs} {obs_label}"
    header_y = 96
    draw_text_with_glow(
        canvas,
        headline,
        (canvas_w // 2, header_y),
        size=58,
        color=(248, 250, 255),
        glow_color=hero_glow,
        glow_radius=22,
        glow_intensity=1.8,
        bold=True,
        anchor="mm",
    )

    # Sub-line: date + device, all-caps.
    sub_parts = [p for p in (date_label, device_label) if p]
    if sub_parts:
        sub_text = "   ·   ".join(p.upper() for p in sub_parts)
        draw_text(
            canvas,
            sub_text,
            (canvas_w // 2, header_y + 50),
            size=24,
            color=(180, 190, 205),
            bold=True,
            anchor="mm",
        )

    # 3. Roster grid. 2 columns × up to 3 rows. Each vignette is a
    #    square photo crop with a slot-coloured neon border and a
    #    species-name strip underneath. Precise grid math depends on n
    #    so a 3-species card centres nicely in the available area.
    grid_top = 200
    grid_bottom = canvas_h - 110
    grid_left = 64
    grid_right = canvas_w - 64
    grid_w = grid_right - grid_left
    grid_h = grid_bottom - grid_top

    cols = 2 if n > 1 else 1
    rows = max(1, (n + cols - 1) // cols)
    cell_gap = 28
    cell_w = (grid_w - (cols - 1) * cell_gap) // cols
    cell_h = (grid_h - (rows - 1) * cell_gap) // rows
    # Reserve the bottom 64px of each cell for the species name strip.
    photo_h_in_cell = cell_h - 64

    for idx, entry in enumerate(species):
        row = idx // cols
        col = idx % cols
        # When the last row is partial (3 species → row 1 has only 1),
        # centre the trailing cells in the row.
        items_in_row = min(cols, n - row * cols)
        row_w = items_in_row * cell_w + (items_in_row - 1) * cell_gap
        row_x0 = grid_left + (grid_w - row_w) // 2
        x0 = row_x0 + col * (cell_w + cell_gap)
        y0 = grid_top + row * (cell_h + cell_gap)

        # Per-species hue.
        scientific = str(entry.get("scientific") or entry.get("common_name") or "")
        rim_color, glow_color = neon_for_species(scientific, colour_map)

        # 3a. Photo cropped to fill the cell's photo region.
        photo = entry.get("photo")
        if photo is not None:
            fitted = _resize_cover(photo, cell_w, photo_h_in_cell)
            canvas[y0 : y0 + photo_h_in_cell, x0 : x0 + cell_w] = fitted

        # 3b. Vignette neon border (rim only, no glow inside the photo
        #     so the bird stays visible).
        draw_glow_rect(
            canvas,
            (x0, y0),
            (x0 + cell_w - 1, y0 + photo_h_in_cell - 1),
            color=rim_color,
            thickness=6,
            glow_radius=14,
            glow_intensity=1.4,
            radius=12,
        )

        # 3c. Count pill in the cell's top-right corner.
        count = int(entry.get("count", 0) or 0)
        count_text = f"{count}×"
        count_size = 26
        cw, ch = measure_text(count_text, size=count_size, bold=True)
        pill_w = cw + 22
        pill_h = ch + 14
        pill_x2 = x0 + cell_w - 12
        pill_x1 = pill_x2 - pill_w
        pill_y1 = y0 + 12
        pill_y2 = pill_y1 + pill_h
        draw_glow_pill(
            canvas,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            fill=rim_color,
            glow_color=glow_color,
            glow_radius=18,
            glow_intensity=1.6,
        )
        draw_text(
            canvas,
            count_text,
            (pill_x1 + pill_w // 2, pill_y1 + pill_h // 2),
            size=count_size,
            color=(15, 15, 18),
            bold=True,
            anchor="mm",
        )

        # 3d. Species name strip underneath the photo.
        name_y = y0 + photo_h_in_cell + 30
        common = str(entry.get("common_name") or scientific.replace("_", " "))
        # Clip names that are too long for narrow cells.
        max_chars = max(8, cell_w // 22)
        if len(common) > max_chars:
            common = common[: max_chars - 1].rstrip() + "…"
        draw_text_with_glow(
            canvas,
            common.upper(),
            (x0 + cell_w // 2, name_y),
            size=30,
            color=(245, 250, 255),
            glow_color=glow_color,
            glow_radius=10,
            glow_intensity=1.3,
            bold=True,
            anchor="mm",
        )

    # 4. Footer "set code" — bottom strip below the grid.
    footer_text = (set_code or date_label or "").upper()
    if footer_text:
        # Tag styled like "30.04.2026  ·  6 / 6  COLLECTED"
        if "·" not in footer_text:
            footer_text = f"{footer_text}   ·   {n} / {n}   COLLECTED"
        draw_text(
            canvas,
            footer_text,
            (canvas_w // 2, canvas_h - 56),
            size=22,
            color=(170, 178, 190),
            bold=True,
            anchor="mm",
        )

    # 5. Tiny watermark "SAMMELKARTE" on the very bottom edge — nothing
    #    overlapping the rim, just a small label so the operator sees
    #    this card differs from the per-species cards.
    draw_text(
        canvas,
        "· SAMMELKARTE ·",
        (canvas_w // 2, canvas_h - 30),
        size=16,
        color=(120, 130, 145),
        bold=True,
        anchor="mm",
    )

    return canvas
