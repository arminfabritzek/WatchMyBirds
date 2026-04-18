#!/usr/bin/env python3
"""Generate a simple comic-style species icon from a scientific or common name.

Used as a deterministic fallback when no photo exists in ``assets/review_species/``.
The same name always produces the same icon; different species look different
because each genus has a distinct silhouette template (body, head, beak shape).

Usage:
    python scripts/generate_species_icon.py Columba_palumbus
    python scripts/generate_species_icon.py "Blåmeis" --out /tmp/bluetit.png
    python scripts/generate_species_icon.py --missing  # fill every species without an icon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
REVIEW_DIR = ASSETS_DIR / "review_species"
SPECIES_JSON = ASSETS_DIR / "common_names_NO.json"

CANVAS = 512
BG = (255, 255, 255, 255)
BODY_COLOR = (58, 76, 102, 255)        # dark slate-blue, same across species
OUTLINE = (28, 36, 50, 255)             # darker outline
BEAK_COLOR = (245, 188, 72, 255)        # warm yellow
EYE_COLOR = (28, 36, 50, 255)
EYE_HILITE = (255, 255, 255, 255)
BRANCH_COLOR = (118, 86, 58, 255)


@dataclass
class Silhouette:
    """Parameters that define a bird's comic silhouette.

    All sizes are fractions of the canvas (0..1) so everything scales together.
    """
    body_w: float = 0.42         # body ellipse width
    body_h: float = 0.40         # body ellipse height
    head_r: float = 0.16         # head radius
    head_offset_x: float = 0.22  # head center relative to body center
    head_offset_y: float = -0.22
    beak_len: float = 0.09       # beak length
    beak_thick: float = 0.04     # beak thickness at base
    beak_shape: str = "cone"     # "cone" | "hook" | "chisel" | "long"
    tail_len: float = 0.22
    tail_angle: float = 25.0     # degrees downward from horizontal
    tail_fan: float = 18.0       # fan spread
    crest: float = 0.0           # 0 no crest, >0 = crest height
    cap: bool = False            # head cap (tits, caps)
    cap_color: tuple[int, int, int, int] = (20, 26, 38, 255)  # default dark cap
    cheek_patch: bool = False    # white cheek (great tit, blue tit)
    chin_bib: bool = False       # dark throat bib (great tit, sparrow)
    accent: tuple[int, int, int, int] = (245, 188, 72, 255)  # breast/belly accent
    throat_patch: bool = False   # robin-style red throat + upper breast
    wing_bar: bool = False       # small white wing bar
    forked_tail: bool = False    # swallow-style fork (drops central feather)


# Genus defaults — a reader should be able to tell a tit from a pigeon
# from a woodpecker at a glance.  Species-level tweaks happen in SPECIES_OVERRIDES.
GENUS_TEMPLATES: dict[str, Silhouette] = {
    # --- tits (round, big head, short beak) ---
    "Parus": Silhouette(
        body_w=0.36, body_h=0.36, head_r=0.18, head_offset_x=0.20, head_offset_y=-0.20,
        beak_len=0.06, beak_thick=0.035, beak_shape="cone",
        tail_len=0.20, cap=True, cheek_patch=True, chin_bib=True,
        accent=(236, 196, 64, 255),  # yellow belly
    ),
    "Cyanistes": Silhouette(
        body_w=0.34, body_h=0.34, head_r=0.17, head_offset_x=0.19, head_offset_y=-0.20,
        beak_len=0.05, beak_thick=0.03, beak_shape="cone",
        tail_len=0.18, cap=True, cap_color=(64, 124, 204, 255),
        cheek_patch=True, accent=(236, 196, 64, 255),  # yellow belly
    ),
    "Periparus": Silhouette(
        body_w=0.32, body_h=0.32, head_r=0.17, head_offset_x=0.19, head_offset_y=-0.19,
        beak_len=0.05, beak_thick=0.03, beak_shape="cone",
        tail_len=0.16, cap=True, cheek_patch=True,
        accent=(210, 200, 184, 255),
    ),
    "Poecile": Silhouette(
        body_w=0.34, body_h=0.33, head_r=0.17, head_offset_x=0.19, head_offset_y=-0.19,
        beak_len=0.05, beak_thick=0.03, beak_shape="cone",
        tail_len=0.18, cap=True, chin_bib=True,
        accent=(220, 212, 196, 255),
    ),
    "Lophophanes": Silhouette(
        body_w=0.34, body_h=0.33, head_r=0.17, head_offset_x=0.19, head_offset_y=-0.19,
        beak_len=0.05, beak_thick=0.03, beak_shape="cone",
        tail_len=0.18, crest=0.11, chin_bib=True,
        accent=(220, 210, 194, 255),
    ),
    "Aegithalos": Silhouette(
        body_w=0.30, body_h=0.30, head_r=0.16, head_offset_x=0.16, head_offset_y=-0.18,
        beak_len=0.04, beak_thick=0.025, beak_shape="cone",
        tail_len=0.38, tail_fan=10.0,
        accent=(232, 210, 214, 255),
    ),
    # --- finches (round, short conical beak) ---
    "Fringilla": Silhouette(
        body_w=0.38, body_h=0.36, head_r=0.16, beak_len=0.07, beak_thick=0.045,
        beak_shape="cone", accent=(196, 108, 76, 255), wing_bar=True,
    ),
    "Chloris": Silhouette(
        body_w=0.38, body_h=0.36, head_r=0.16, beak_len=0.08, beak_thick=0.050,
        beak_shape="cone", accent=(182, 196, 92, 255),
    ),
    "Carduelis": Silhouette(
        body_w=0.36, body_h=0.34, head_r=0.15, beak_len=0.08, beak_thick=0.035,
        beak_shape="long", accent=(220, 84, 60, 255),
    ),
    "Spinus": Silhouette(
        body_w=0.34, body_h=0.32, head_r=0.15, beak_len=0.07, beak_thick=0.035,
        beak_shape="cone", cap=True, accent=(222, 200, 72, 255),
    ),
    "Pyrrhula": Silhouette(
        body_w=0.40, body_h=0.38, head_r=0.18, beak_len=0.06, beak_thick=0.055,
        beak_shape="cone", cap=True, accent=(210, 86, 86, 255),
    ),
    "Coccothraustes": Silhouette(
        body_w=0.42, body_h=0.40, head_r=0.20, beak_len=0.10, beak_thick=0.075,
        beak_shape="cone", accent=(200, 148, 96, 255),
    ),
    # --- sparrows / buntings ---
    "Passer": Silhouette(
        body_w=0.38, body_h=0.36, head_r=0.16, beak_len=0.06, beak_thick=0.045,
        beak_shape="cone", chin_bib=True, accent=(180, 138, 90, 255),
    ),
    "Emberiza": Silhouette(
        body_w=0.36, body_h=0.34, head_r=0.15, beak_len=0.06, beak_thick=0.040,
        beak_shape="cone", accent=(232, 196, 76, 255),
    ),
    # --- thrushes / chats ---
    "Turdus": Silhouette(
        body_w=0.42, body_h=0.40, head_r=0.17, beak_len=0.09, beak_thick=0.035,
        beak_shape="long", accent=(245, 188, 72, 255),
    ),
    "Erithacus": Silhouette(
        body_w=0.34, body_h=0.34, head_r=0.18, beak_len=0.05, beak_thick=0.035,
        beak_shape="cone", throat_patch=True, accent=(224, 92, 72, 255),
    ),
    "Phoenicurus": Silhouette(
        body_w=0.34, body_h=0.33, head_r=0.16, beak_len=0.06, beak_thick=0.035,
        beak_shape="cone", throat_patch=True, accent=(220, 108, 80, 255),
    ),
    "Luscinia": Silhouette(
        body_w=0.36, body_h=0.34, head_r=0.16, beak_len=0.07, beak_thick=0.035,
        beak_shape="cone", accent=(188, 150, 120, 255),
    ),
    # --- tiny insectivores ---
    "Troglodytes": Silhouette(
        body_w=0.30, body_h=0.28, head_r=0.14, beak_len=0.07, beak_thick=0.030,
        beak_shape="long", tail_len=0.12, tail_angle=-35.0,
        accent=(176, 136, 96, 255),
    ),
    "Phylloscopus": Silhouette(
        body_w=0.30, body_h=0.28, head_r=0.14, beak_len=0.06, beak_thick=0.028,
        beak_shape="long", accent=(196, 200, 148, 255),
    ),
    "Regulus": Silhouette(
        body_w=0.26, body_h=0.24, head_r=0.13, beak_len=0.05, beak_thick=0.025,
        beak_shape="cone", crest=0.10, accent=(222, 196, 92, 255),
    ),
    "Sylvia": Silhouette(
        body_w=0.34, body_h=0.32, head_r=0.15, beak_len=0.06, beak_thick=0.030,
        beak_shape="cone", cap=True, accent=(200, 194, 178, 255),
    ),
    "Prunella": Silhouette(
        body_w=0.34, body_h=0.32, head_r=0.15, beak_len=0.06, beak_thick=0.030,
        beak_shape="cone", accent=(148, 138, 124, 255),
    ),
    "Bombycilla": Silhouette(
        body_w=0.38, body_h=0.34, head_r=0.16, beak_len=0.05, beak_thick=0.035,
        beak_shape="cone", crest=0.14, accent=(196, 156, 112, 255),
    ),
    "Motacilla": Silhouette(
        body_w=0.30, body_h=0.28, head_r=0.15, beak_len=0.06, beak_thick=0.028,
        beak_shape="long", tail_len=0.40, tail_fan=6.0, cap=True,
        accent=(220, 220, 220, 255),
    ),
    # --- swallows / swifts (sleek, forked tail) ---
    "Hirundo": Silhouette(
        body_w=0.40, body_h=0.24, head_r=0.14, beak_len=0.04, beak_thick=0.025,
        beak_shape="cone", tail_len=0.36, tail_fan=30.0, tail_angle=5.0,
        accent=(220, 148, 110, 255), forked_tail=True,
    ),
    "Delichon": Silhouette(
        body_w=0.38, body_h=0.24, head_r=0.14, beak_len=0.04, beak_thick=0.025,
        beak_shape="cone", tail_len=0.28, tail_fan=22.0, tail_angle=8.0,
        accent=(236, 236, 236, 255), forked_tail=True,
    ),
    "Apus": Silhouette(
        body_w=0.44, body_h=0.22, head_r=0.13, beak_len=0.04, beak_thick=0.022,
        beak_shape="cone", tail_len=0.30, tail_fan=20.0, tail_angle=8.0,
        accent=(92, 90, 86, 255),
    ),
    # --- pigeons / doves ---
    "Columba": Silhouette(
        body_w=0.52, body_h=0.46, head_r=0.16, head_offset_x=0.26, head_offset_y=-0.24,
        beak_len=0.07, beak_thick=0.035, beak_shape="cone",
        tail_len=0.26, accent=(196, 180, 178, 255),
    ),
    "Streptopelia": Silhouette(
        body_w=0.48, body_h=0.42, head_r=0.15, head_offset_x=0.24, head_offset_y=-0.22,
        beak_len=0.06, beak_thick=0.030, beak_shape="cone",
        tail_len=0.28, accent=(216, 204, 192, 255),
    ),
    # --- woodpeckers (elongated, chisel beak) ---
    "Dendrocopos": Silhouette(
        body_w=0.34, body_h=0.44, head_r=0.16, head_offset_x=0.18, head_offset_y=-0.28,
        beak_len=0.14, beak_thick=0.050, beak_shape="chisel",
        tail_len=0.22, tail_fan=6.0, cap=True,
        cap_color=(196, 60, 60, 255), accent=(232, 232, 232, 255),
    ),
    "Dryobates": Silhouette(
        body_w=0.28, body_h=0.38, head_r=0.14, head_offset_x=0.14, head_offset_y=-0.24,
        beak_len=0.11, beak_thick=0.040, beak_shape="chisel",
        tail_len=0.18, tail_fan=6.0, cap=True,
        cap_color=(196, 60, 60, 255), accent=(232, 232, 232, 255),
    ),
    "Picus": Silhouette(
        body_w=0.38, body_h=0.46, head_r=0.17, head_offset_x=0.20, head_offset_y=-0.28,
        beak_len=0.15, beak_thick=0.054, beak_shape="chisel",
        tail_len=0.24, tail_fan=6.0, cap=True,
        cap_color=(196, 60, 60, 255), accent=(176, 196, 120, 255),
    ),
    # --- corvids (large, strong beak) ---
    "Corvus": Silhouette(
        body_w=0.48, body_h=0.44, head_r=0.18, beak_len=0.14, beak_thick=0.070,
        beak_shape="hook", tail_len=0.28,
        accent=(90, 96, 108, 255),
    ),
    "Pica": Silhouette(
        body_w=0.40, body_h=0.36, head_r=0.16, beak_len=0.10, beak_thick=0.050,
        beak_shape="hook", tail_len=0.48, tail_fan=8.0, tail_angle=10.0,
        accent=(236, 236, 236, 255),
    ),
    "Garrulus": Silhouette(
        body_w=0.42, body_h=0.40, head_r=0.17, beak_len=0.10, beak_thick=0.052,
        beak_shape="hook", tail_len=0.26, crest=0.06,
        accent=(200, 156, 120, 255),
    ),
    # --- climbers ---
    "Sitta": Silhouette(
        body_w=0.38, body_h=0.34, head_r=0.17, beak_len=0.12, beak_thick=0.038,
        beak_shape="long", tail_len=0.14, cap=True,
        accent=(196, 148, 108, 255),
    ),
    "Certhia": Silhouette(
        body_w=0.32, body_h=0.28, head_r=0.14, beak_len=0.13, beak_thick=0.028,
        beak_shape="hook", tail_len=0.26,
        accent=(224, 224, 224, 255),
    ),
    # --- misc ---
    "Sturnus": Silhouette(
        body_w=0.40, body_h=0.36, head_r=0.15, beak_len=0.10, beak_thick=0.035,
        beak_shape="long", tail_len=0.16,
        accent=(108, 92, 120, 255),
    ),
    "Cuculus": Silhouette(
        body_w=0.44, body_h=0.34, head_r=0.15, beak_len=0.08, beak_thick=0.035,
        beak_shape="hook", tail_len=0.34,
        accent=(208, 208, 208, 255),
    ),
}


SPECIES_OVERRIDES: dict[str, dict] = {
    # species-level nudges where the genus default would lose information
    "Cyanistes_caeruleus": {"accent": (92, 148, 216, 255)},        # blue crown
    "Parus_major":        {"accent": (232, 196, 64, 255)},         # bright yellow
    "Pyrrhula_pyrrhula":  {"accent": (212, 88, 88, 255)},          # red breast
    "Erithacus_rubecula": {"accent": (224, 92, 72, 255)},
    "Carduelis_carduelis":{"accent": (220, 84, 60, 255)},
    "Sturnus_vulgaris":   {"accent": (102, 82, 128, 255)},
    "Corvus_corone":      {"body_color": (32, 34, 40, 255)},
    "Corvus_monedula":    {"accent": (148, 160, 180, 255)},
    "Turdus_merula":      {"body_color": (34, 34, 38, 255), "accent": (232, 176, 40, 255)},
    "Emberiza_citrinella":{"accent": (244, 210, 72, 255)},
    "Chloris_chloris":    {"accent": (186, 204, 96, 255)},
}


def digest(name: str) -> bytes:
    return hashlib.md5(name.encode("utf-8")).digest()


def hash_float(name: str, index: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """Deterministic [lo, hi) float derived from ``name`` at byte ``index``."""
    b = digest(name)[index % len(digest(name))]
    return lo + (b / 256.0) * (hi - lo)


def canonical_key(user_input: str) -> str:
    """Map a common name or scientific name to the Genus_species key."""
    s = user_input.strip().replace(" ", "_")
    # Try direct key match first.
    try:
        mapping = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return s
    if s in mapping:
        return s
    low = s.lower()
    for key, common in mapping.items():
        if common.lower() == low or key.lower() == low:
            return key
    return s


def genus_of(species_key: str) -> str:
    return species_key.split("_", 1)[0]


def build_silhouette(species_key: str) -> Silhouette:
    genus = genus_of(species_key)
    base = GENUS_TEMPLATES.get(genus, Silhouette())
    # Copy so we never mutate the shared template.
    sil = Silhouette(**{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()})  # type: ignore[attr-defined]
    for k, v in SPECIES_OVERRIDES.get(species_key, {}).items():
        if k == "body_color":
            continue  # handled in draw()
        setattr(sil, k, v)
    # A tiny hash-driven wobble so congenerics (e.g. two Turdus) differ slightly.
    sil.body_w *= 0.96 + 0.08 * hash_float(species_key, 0)
    sil.body_h *= 0.96 + 0.08 * hash_float(species_key, 1)
    sil.tail_len *= 0.94 + 0.12 * hash_float(species_key, 2)
    return sil


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------

def _px(v: float) -> int:
    return int(round(v * CANVAS))


Point = tuple[float, float]
OUTLINE_W = 6


def _polygon(draw: ImageDraw.ImageDraw, pts: Sequence[Point],
             fill, outline=OUTLINE, width: int = OUTLINE_W) -> None:
    draw.polygon(list(pts), fill=fill, outline=outline, width=width)


def _ellipse(draw: ImageDraw.ImageDraw, cx: float, cy: float,
             rx: float, ry: float, fill, outline=OUTLINE, width: int = OUTLINE_W) -> None:
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry),
                 fill=fill, outline=outline, width=width)


def draw_branch(draw: ImageDraw.ImageDraw) -> None:
    y = _px(0.84)
    draw.line((_px(0.02), y, _px(0.98), y + _px(0.03)),
              fill=BRANCH_COLOR, width=_px(0.045))


def _rotate(points: Sequence[Point], cx: float, cy: float,
            angle_deg: float) -> list[Point]:
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [((x - cx) * ca - (y - cy) * sa + cx,
             (x - cx) * sa + (y - cy) * ca + cy) for x, y in points]


def _draw_tail(draw: ImageDraw.ImageDraw, sil: Silhouette,
               root_x: float, root_y: float, body_color) -> None:
    """Fan of overlapping feather triangles pointing left, rotated by tail_angle.

    Narrow fans (small ``tail_fan``) give a slender tail base — right for
    magpies.  Wide, forked fans (swallows) drop the middle feather.
    """
    tl = _px(sil.tail_len)
    # Base width scales with the fan angle: wide fan → wide base.
    # A 25° fan produces roughly the old width; an 8° fan is much slimmer.
    half_base = max(_px(0.020), int(_px(0.055) * (sil.tail_fan / 25.0)))
    fan_rad = math.radians(sil.tail_fan)
    n = 5
    forked = sil.forked_tail
    for i in range(n):
        t = (i - (n - 1) / 2.0) / ((n - 1) / 2.0)  # -1..+1
        if forked and abs(t) < 0.2:
            continue
        angle = t * fan_rad
        tip_x = root_x - tl * math.cos(angle)
        tip_y = root_y + tl * math.sin(angle)
        base_top = (root_x, root_y - half_base)
        base_bot = (root_x, root_y + half_base)
        feather = [base_top, (tip_x, tip_y), base_bot]
        feather = _rotate(feather, root_x, root_y, sil.tail_angle)
        _polygon(draw, feather, fill=body_color, width=OUTLINE_W - 2)


def draw_bird(img: Image.Image, sil: Silhouette, species_key: str) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    draw_branch(draw)

    body_color = SPECIES_OVERRIDES.get(species_key, {}).get("body_color", BODY_COLOR)

    # Body center — biased slightly right so the tail has room on the left.
    cx = _px(0.56)
    cy = _px(0.56)
    bw = _px(sil.body_w)
    bh = _px(sil.body_h)

    # --- tail (before body so its root tucks behind the body outline) ---
    tail_root_x = cx - int(bw * 0.85)
    tail_root_y = cy + int(bh * 0.15)
    _draw_tail(draw, sil, tail_root_x, tail_root_y, body_color)

    # --- body ---
    _ellipse(draw, cx, cy, bw, bh, fill=body_color)

    # --- belly accent: a chord clipped inside the body ---
    belly_inset = OUTLINE_W // 2
    belly_box = (cx - bw + belly_inset, cy - bh + belly_inset,
                 cx + bw - belly_inset, cy + bh - belly_inset)
    draw.chord(belly_box, start=35, end=145, fill=sil.accent, outline=None)

    # --- wing: closed chord on the upper back ---
    # chord() clips the arc with a straight secant, so it stays flush with
    # the body outline without a V-kerbe.  The wing sits on top of the body.
    wing_box = (cx - int(bw * 0.80), cy - int(bh * 0.55),
                cx + int(bw * 0.45), cy + int(bh * 0.45))
    draw.chord(wing_box, start=170, end=350,
               fill=_darken(body_color, 0.18), outline=OUTLINE, width=OUTLINE_W - 2)
    if sil.wing_bar:
        draw.arc(wing_box, start=220, end=320,
                 fill=(240, 240, 240, 255), width=max(3, _px(0.010)))

    # --- head ---
    hr = _px(sil.head_r)
    hcx = cx + _px(sil.head_offset_x)
    hcy = cy + _px(sil.head_offset_y)
    _ellipse(draw, hcx, hcy, hr, hr, fill=body_color)

    # --- cap (top half of head) in species-specific colour ---
    if sil.cap:
        cap_box = (hcx - hr, hcy - hr, hcx + hr, hcy + int(hr * 0.30))
        draw.chord(cap_box, start=180, end=360,
                   fill=sil.cap_color, outline=None)

    # --- white cheek patch ---
    if sil.cheek_patch:
        ccx = hcx - int(hr * 0.15)
        ccy = hcy + int(hr * 0.25)
        _ellipse(draw, ccx, ccy, int(hr * 0.55), int(hr * 0.38),
                 fill=(248, 248, 248, 255), outline=None, width=0)

    # --- dark chin / bib ---
    if sil.chin_bib:
        bib_box = (hcx - int(hr * 0.45), hcy + int(hr * 0.30),
                   hcx + int(hr * 0.75), hcy + int(hr * 1.25))
        draw.chord(bib_box, start=200, end=340,
                   fill=(24, 28, 40, 255), outline=None)

    # --- throat / upper-breast patch (robin, redstart) ---
    if sil.throat_patch:
        tp_box = (hcx - int(hr * 0.60), hcy + int(hr * 0.20),
                  hcx + int(hr * 0.85), hcy + int(hr * 1.80))
        draw.chord(tp_box, start=200, end=340, fill=sil.accent, outline=None)

    # --- crest ---
    if sil.crest > 0:
        ch = _px(sil.crest)
        cpts: list[Point] = [
            (hcx - int(hr * 0.30), hcy - int(hr * 0.90)),
            (hcx + int(hr * 0.25), hcy - int(hr * 0.90) - ch),
            (hcx + int(hr * 0.45), hcy - int(hr * 0.75)),
        ]
        _polygon(draw, cpts, fill=body_color)

    # --- beak ---
    _draw_beak(draw, sil, hcx, hcy, hr)

    # --- eye ---
    eye_r = max(6, int(hr * 0.18))
    ex = hcx + int(hr * 0.35)
    ey = hcy - int(hr * 0.10)
    _ellipse(draw, ex, ey, eye_r, eye_r, fill=EYE_COLOR, outline=None, width=0)
    _ellipse(draw, ex - int(eye_r * 0.35), ey - int(eye_r * 0.35),
             int(eye_r * 0.40), int(eye_r * 0.40),
             fill=EYE_HILITE, outline=None, width=0)


def _darken(rgba: tuple[int, int, int, int], amount: float) -> tuple[int, int, int, int]:
    r, g, b, a = rgba
    f = 1.0 - amount
    return (int(r * f), int(g * f), int(b * f), a)


def _draw_beak(draw: ImageDraw.ImageDraw, sil: Silhouette,
               hcx: float, hcy: float, hr: float) -> None:
    tip_x = hcx + hr + _px(sil.beak_len)
    tip_y = hcy + int(hr * 0.18)
    root_x = hcx + int(hr * 0.85)
    bt = _px(sil.beak_thick)

    if sil.beak_shape == "chisel":
        # Strong, near-horizontal blade: thicker at base, narrow chisel tip.
        pts: list[Point] = [
            (root_x - int(bt * 0.2), hcy - int(bt * 0.6)),
            (root_x - int(bt * 0.2), hcy + int(bt * 0.9)),
            (tip_x, tip_y + int(bt * 0.3)),
            (tip_x, tip_y - int(bt * 0.4)),
        ]
    elif sil.beak_shape == "hook":
        pts = [
            (root_x, hcy),
            (root_x, hcy + bt),
            (tip_x - int(bt * 0.25), tip_y + int(bt * 0.3)),
            (tip_x, tip_y),
            (tip_x - int(bt * 0.15), tip_y - int(bt * 0.25)),
        ]
    elif sil.beak_shape == "long":
        pts = [
            (root_x, hcy + int(hr * 0.04)),
            (root_x, hcy + int(hr * 0.04) + int(bt * 0.7)),
            (tip_x, tip_y),
        ]
    else:  # cone
        pts = [(root_x, hcy), (root_x, hcy + bt), (tip_x, tip_y)]
    _polygon(draw, pts, fill=BEAK_COLOR, width=4)


def round_corners(img: Image.Image, radius: float = 0.10) -> Image.Image:
    r = _px(radius)
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=r, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def _load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _stamp(img: Image.Image, species_key: str) -> None:
    """Add a small corner label so generated icons are visibly machine-made."""
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(16)
    text = f"generate_species_icon.py · {species_key}"
    pad = 10
    # Bottom-left translucent badge.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # very old Pillow fallback
        tw, th = draw.textsize(text, font=font)  # type: ignore[attr-defined]
    x0, y0 = pad, CANVAS - th - pad * 2
    draw.rounded_rectangle(
        (x0, y0, x0 + tw + pad * 2, y0 + th + pad),
        radius=6, fill=(30, 30, 30, 180),
    )
    draw.text((x0 + pad, y0 + pad // 2), text,
              fill=(240, 240, 240, 255), font=font)


def render_icon(species_key: str, stamp: bool = False) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS, CANVAS), BG)
    sil = build_silhouette(species_key)
    draw_bird(canvas, sil, species_key)
    if stamp:
        _stamp(canvas, species_key)
    return round_corners(canvas)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output(species_key: str) -> Path:
    return REVIEW_DIR / f"{species_key}.png"


def _species_with_icon() -> set[str]:
    if not REVIEW_DIR.exists():
        return set()
    return {p.stem for p in REVIEW_DIR.iterdir() if p.is_file()}


def _iter_missing() -> list[str]:
    mapping = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))
    have = _species_with_icon()
    return [k for k in mapping
            if k[:1].isupper() and k != "Unknown_species" and k not in have]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?",
                        help="Scientific key (Parus_major) or common name (Kjøttmeis)")
    parser.add_argument("--out", type=Path,
                        help="Output path (defaults to assets/review_species/<Key>.png)")
    parser.add_argument("--missing", action="store_true",
                        help="Render one icon for every Norwegian species without one")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing icons")
    parser.add_argument("--stamp", action="store_true",
                        help="Burn a small 'generate_species_icon.py' label into the PNG")
    args = parser.parse_args(argv)

    if args.missing:
        targets = _iter_missing()
        if not targets:
            print("Every species already has an icon.")
            return 0
        for key in targets:
            out = args.out / f"{key}.png" if args.out and args.out.is_dir() else _default_output(key)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not args.force:
                print(f"skip  {key} (exists)")
                continue
            render_icon(key, stamp=args.stamp).save(out, format="PNG")
            print(f"wrote {out.relative_to(REPO_ROOT)}")
        return 0

    if not args.name:
        parser.error("provide a species name or use --missing")

    species_key = canonical_key(args.name)
    out = args.out or _default_output(species_key)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    render_icon(species_key, stamp=args.stamp).save(out, format="PNG")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
