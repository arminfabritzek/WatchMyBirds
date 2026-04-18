#!/usr/bin/env python3
"""
generate_species_icon_vector.py
================================
Bold flat-art bird icons using pycairo. No ML, no downloads.
All birds face LEFT. 512x512 PNG, bold black outline, saturated flat colours.

Usage:
    python generate_species_icon_vector.py                      # render all 49
    python generate_species_icon_vector.py Parus_major          # single species
    python generate_species_icon_vector.py --out /tmp/x.png Parus_major
    python generate_species_icon_vector.py --outdir /tmp/birds/
"""

import math
import os
import sys

# ---------------------------------------------------------------------------
# Cairo import — support the Homebrew site-packages path
# ---------------------------------------------------------------------------
_CAIRO_SITE = "/opt/homebrew/lib/python3.14/site-packages"
if _CAIRO_SITE not in sys.path:
    sys.path.insert(0, _CAIRO_SITE)

import cairo  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W, H = 512, 512
OUTLINE = (0.0, 0.0, 0.0)
BRANCH_COLOR = (0.45, 0.28, 0.10)
BRANCH_Y = 405
BRANCH_X0, BRANCH_X1 = 20, 490
DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__),
    "..", "assets", "review_species", "_generated_samples",
)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def rgb(r: int, g: int, b: int) -> tuple:
    return r / 255.0, g / 255.0, b / 255.0


# ---------------------------------------------------------------------------
# Surface / context helpers
# ---------------------------------------------------------------------------
def new_surface():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    return surface, ctx


def _set_color(ctx, color):
    if len(color) == 4:
        ctx.set_source_rgba(*color)
    else:
        ctx.set_source_rgb(*color)


def fill(ctx, color):
    _set_color(ctx, color)
    ctx.fill_preserve()


def stroke(ctx, color, width=8.0):
    _set_color(ctx, color)
    ctx.set_line_width(width)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()


def circle(ctx, cx, cy, r, fill_color, stroke_color=None, sw=8.0):
    if stroke_color is None:
        stroke_color = OUTLINE
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    fill(ctx, fill_color)
    stroke(ctx, stroke_color, sw)


def ellipse(ctx, cx, cy, rx, ry, fill_color, stroke_color=None, sw=8.0, angle=0.0):
    if stroke_color is None:
        stroke_color = OUTLINE
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.scale(rx, ry)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    fill(ctx, fill_color)
    stroke(ctx, stroke_color, sw)


def poly(ctx, pts, fill_color, stroke_color=None, sw=7.0):
    if stroke_color is None:
        stroke_color = OUTLINE
    if not pts:
        return
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    fill(ctx, fill_color)
    stroke(ctx, stroke_color, sw)


def curve(ctx, moves, fill_color, stroke_color=None, sw=7.0):
    """
    moves format: [('M',x,y), ('L',x,y), ('C',x1,y1,x2,y2,x,y), ('Z',)]
    Pass fill_color=None for stroke-only paths.
    """
    if stroke_color is None:
        stroke_color = OUTLINE
    for m in moves:
        op = m[0]
        if op == "M":
            ctx.move_to(m[1], m[2])
        elif op == "L":
            ctx.line_to(m[1], m[2])
        elif op == "C":
            ctx.curve_to(m[1], m[2], m[3], m[4], m[5], m[6])
        elif op == "Z":
            ctx.close_path()
    if fill_color is not None:
        fill(ctx, fill_color)
    stroke(ctx, stroke_color, sw)


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------
def draw_branch(ctx):
    ctx.move_to(BRANCH_X0, BRANCH_Y)
    ctx.line_to(BRANCH_X1, BRANCH_Y)
    _set_color(ctx, BRANCH_COLOR)
    ctx.set_line_width(12)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.stroke()
    # shadow
    ctx.move_to(BRANCH_X0, BRANCH_Y + 4)
    ctx.line_to(BRANCH_X1, BRANCH_Y + 4)
    ctx.set_source_rgba(0, 0, 0, 0.15)
    ctx.set_line_width(4)
    ctx.stroke()


# ---------------------------------------------------------------------------
# Feet helper
# ---------------------------------------------------------------------------
def draw_feet(ctx, x, y, color):
    ctx.set_line_width(4)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    _set_color(ctx, color)
    for side, sign in [(-1, -1), (1, 1)]:
        fx = x + side * 10
        ctx.move_to(fx, y)
        ctx.line_to(fx, y + 20)
        ctx.move_to(fx, y + 20)
        ctx.line_to(fx - 12, y + 26)
        ctx.move_to(fx, y + 20)
        ctx.line_to(fx, y + 28)
        ctx.move_to(fx, y + 20)
        ctx.line_to(fx + 12, y + 26)
    ctx.stroke()


# ===========================================================================
# TEMPLATE 1 — TIT (Parus family)
# ===========================================================================
def draw_tit(ctx, colors):
    c = colors
    bx, by = 285, 308
    hx, hy = 215, 260

    # tail
    tail_col = c.get("tail", c["wing"])
    if c.get("long_tail"):
        curve(ctx, [
            ("M", bx + 38, by - 8),
            ("C", bx + 95, by - 45, bx + 168, by - 55, bx + 168, by - 2),
            ("C", bx + 168, by + 48, bx + 98, by + 42, bx + 38, by + 18),
            ("Z",),
        ], tail_col)
        # inner light stripe
        curve(ctx, [
            ("M", bx + 55, by + 5),
            ("C", bx + 112, by - 8, bx + 152, by - 5, bx + 152, by + 5),
        ], None, c.get("tail_stripe", (1, 1, 1)), 3)
    else:
        curve(ctx, [
            ("M", bx + 30, by - 10),
            ("C", bx + 82, by - 32, bx + 122, by - 5, bx + 118, by + 22),
            ("C", bx + 112, by + 48, bx + 62, by + 30, bx + 30, by + 15),
            ("Z",),
        ], tail_col)

    # body
    ellipse(ctx, bx, by, 72, 55, c["body"])

    # wing panel
    curve(ctx, [
        ("M", bx - 20, by - 30),
        ("C", bx + 40, by - 52, bx + 82, by - 14, bx + 62, by + 32),
        ("C", bx + 32, by + 58, bx - 20, by + 36, bx - 20, by - 30),
        ("Z",),
    ], c["wing"])

    # belly
    belly_col = c.get("belly", c["body"])
    ellipse(ctx, bx - 12, by + 18, 36, 28, belly_col, belly_col, 0)

    # breast stripe (e.g. Parus major)
    if "stripe" in c:
        curve(ctx, [
            ("M", bx - 40, by - 22),
            ("C", bx - 30, by - 2, bx - 34, by + 30, bx - 42, by + 52),
            ("L", bx - 28, by + 53),
            ("C", bx - 20, by + 30, bx - 16, by, bx - 26, by - 22),
            ("Z",),
        ], c["stripe"])

    # head
    circle(ctx, hx, hy, 42, c["head"])

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 36, hy - 8),
            ("C", hx - 30, by - 100, hx + 30, by - 100, hx + 36, hy - 8),
            ("Z",),
        ], c["cap"])

    # cheek patch
    if "cheek" in c:
        ellipse(ctx, hx + 10, hy + 8, 20, 16, c["cheek"], c["cheek"], 0)

    # crest
    if "crest_color" in c:
        curve(ctx, [
            ("M", hx - 6, hy - 40),
            ("C", hx - 16, hy - 82, hx + 6, hy - 98, hx + 10, hy - 76),
            ("C", hx + 14, hy - 56, hx + 6, hy - 40, hx + 6, hy - 40),
            ("Z",),
        ], c["crest_color"])

    # beak
    poly(ctx, [
        (hx - 38, hy + 1),
        (hx - 70, hy + 8),
        (hx - 38, hy + 16),
    ], c["beak"])

    # eye
    circle(ctx, hx - 8, hy - 5, 8, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 6, hy - 7, 3, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 50, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 2 — FINCH
# ===========================================================================
def draw_finch(ctx, colors):
    c = colors
    bx, by = 290, 315
    hx, hy = 218, 270

    # tail
    tail_col = c.get("tail", c["wing"])
    curve(ctx, [
        ("M", bx + 25, by - 8),
        ("C", bx + 70, by - 26, bx + 112, by - 10, bx + 106, by + 18),
        ("C", bx + 100, by + 42, bx + 55, by + 30, bx + 25, by + 15),
        ("Z",),
    ], tail_col)

    # body
    ellipse(ctx, bx, by, 68, 52, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 15, by - 28),
        ("C", bx + 36, by - 50, bx + 76, by - 12, bx + 56, by + 28),
        ("C", bx + 26, by + 52, bx - 15, by + 32, bx - 15, by - 28),
        ("Z",),
    ], wing_col)

    # wing bar
    if "bar" in c:
        curve(ctx, [
            ("M", bx - 8, by - 10),
            ("C", bx + 22, by - 22, bx + 52, by - 8, bx + 47, by + 5),
            ("L", bx + 37, by + 8),
            ("C", bx + 40, by - 3, bx + 14, by - 14, bx - 2, by - 3),
            ("Z",),
        ], c["bar"])

    # yellow wing flash
    if "flash" in c:
        curve(ctx, [
            ("M", bx + 10, by - 30),
            ("C", bx + 30, by - 44, bx + 56, by - 27, bx + 51, by - 10),
            ("L", bx + 40, by - 8),
            ("C", bx + 44, by - 22, bx + 24, by - 35, bx + 8, by - 23),
            ("Z",),
        ], c["flash"])

    # belly
    belly_col = c.get("belly", c["body"])
    if belly_col != c["body"]:
        ellipse(ctx, bx - 18, by + 15, 32, 26, belly_col, belly_col, 0)

    # breast
    if "breast" in c:
        curve(ctx, [
            ("M", hx + 20, hy + 28),
            ("C", hx + 40, hy + 52, bx - 30, by + 40, bx - 38, by + 20),
            ("C", bx - 44, by, bx - 20, by - 20, hx + 18, hy + 20),
            ("Z",),
        ], c["breast"])

    # rump patch
    if "rump" in c:
        ellipse(ctx, bx + 55, by - 5, 22, 18, c["rump"], c["rump"], 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 40, head_col)

    # face patch (red/orange)
    if "face" in c:
        circle(ctx, hx + 5, hy + 5, 22, c["face"], c["face"], 0)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 34, hy - 7),
            ("C", hx - 28, hy - 50, hx + 28, hy - 50, hx + 34, hy - 7),
            ("Z",),
        ], c["cap"])

    # bib
    if "bib" in c:
        curve(ctx, [
            ("M", hx - 8, hy + 20),
            ("C", hx + 12, hy + 28, hx + 28, hy + 46, hx + 5, hy + 52),
            ("C", hx - 18, hy + 56, hx - 30, hy + 36, hx - 10, hy + 20),
            ("Z",),
        ], c["bib"])

    # conical beak (or massive)
    if c.get("massive_beak"):
        poly(ctx, [
            (hx - 30, hy - 8),
            (hx - 80, hy + 10),
            (hx - 30, hy + 28),
        ], c["beak"], sw=6)
    else:
        poly(ctx, [
            (hx - 35, hy + 2),
            (hx - 68, hy + 10),
            (hx - 35, hy + 20),
        ], c["beak"])

    # eye
    circle(ctx, hx - 6, hy - 4, 8, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 4, hy - 6, 3, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 48, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 3 — THRUSH
# ===========================================================================
def draw_thrush(ctx, colors):
    c = colors
    bx, by = 285, 305
    hx, hy = 208, 260

    # tail
    tail_col = c.get("tail", c["wing"])
    curve(ctx, [
        ("M", bx + 35, by - 10),
        ("C", bx + 82, by - 34, bx + 122, by - 8, bx + 116, by + 24),
        ("C", bx + 110, by + 50, bx + 62, by + 34, bx + 35, by + 18),
        ("Z",),
    ], tail_col)

    # body
    ellipse(ctx, bx, by, 80, 60, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 18, by - 32),
        ("C", bx + 46, by - 56, bx + 88, by - 12, bx + 64, by + 36),
        ("C", bx + 32, by + 60, bx - 18, by + 40, bx - 18, by - 32),
        ("Z",),
    ], wing_col)

    # breast
    breast_col = c.get("breast", c.get("belly", c["body"]))
    curve(ctx, [
        ("M", hx + 25, hy + 30),
        ("C", hx + 46, hy + 56, bx - 32, by + 54, bx - 40, by + 26),
        ("C", bx - 46, by + 2, bx - 18, by - 28, hx + 22, hy + 22),
        ("Z",),
    ], breast_col)

    # spots
    if c.get("spotted"):
        spot_col = c.get("spot_color", rgb(80, 55, 30))
        for sx, sy in [
            (hx + 30, hy + 52), (hx + 45, hy + 65), (hx + 28, hy + 72),
            (bx - 30, by + 35), (bx - 45, by + 46), (bx - 28, by + 52),
        ]:
            circle(ctx, sx, sy, 6, spot_col, spot_col, 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 44, head_col)

    # face patch
    if "face" in c:
        circle(ctx, hx + 8, hy + 8, 25, c["face"], c["face"], 0)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 38, hy - 8),
            ("C", hx - 32, hy - 54, hx + 30, hy - 54, hx + 38, hy - 8),
            ("Z",),
        ], c["cap"])

    # eye ring
    if "eye_ring" in c:
        circle(ctx, hx - 6, hy - 4, 13, c["eye_ring"], OUTLINE, 3)

    # beak
    poly(ctx, [
        (hx - 38, hy + 2),
        (hx - 74, hy + 10),
        (hx - 38, hy + 20),
    ], c["beak"])

    # eye
    circle(ctx, hx - 6, hy - 4, 9, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 4, hy - 6, 3, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 55, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 4 — WARBLER (slim, small)
# ===========================================================================
def draw_warbler(ctx, colors):
    c = colors
    bx, by = 286, 318
    hx, hy = 222, 278

    # tail
    tail_col = c.get("tail", c["body"])
    if c.get("tail_up"):
        curve(ctx, [
            ("M", bx + 28, by - 5),
            ("C", bx + 55, by - 32, bx + 74, by - 72, bx + 66, by - 92),
            ("C", bx + 58, by - 110, bx + 42, by - 96, bx + 38, by - 74),
            ("C", bx + 34, by - 52, bx + 40, by - 22, bx + 28, by + 10),
            ("Z",),
        ], tail_col)
    else:
        curve(ctx, [
            ("M", bx + 20, by - 6),
            ("C", bx + 62, by - 24, bx + 96, by - 6, bx + 92, by + 18),
            ("C", bx + 86, by + 40, bx + 48, by + 26, bx + 20, by + 12),
            ("Z",),
        ], tail_col)

    # body
    ellipse(ctx, bx, by, 58, 42, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 12, by - 22),
        ("C", bx + 32, by - 42, bx + 70, by - 8, bx + 52, by + 24),
        ("C", bx + 22, by + 46, bx - 12, by + 28, bx - 12, by - 22),
        ("Z",),
    ], wing_col)

    # supercilium
    if "supercilium" in c:
        curve(ctx, [
            ("M", hx - 26, hy - 8),
            ("C", hx + 4, hy - 24, hx + 30, hy - 18, hx + 36, hy - 10),
            ("L", hx + 28, hy - 4),
            ("C", hx + 22, hy - 12, hx + 2, hy - 16, hx - 18, hy - 2),
            ("Z",),
        ], c["supercilium"])

    # crown stripe (Regulus)
    if "crown" in c:
        curve(ctx, [
            ("M", hx - 12, hy - 28),
            ("C", hx - 8, hy - 52, hx + 8, hy - 52, hx + 12, hy - 28),
            ("L", hx + 5, hy - 26),
            ("C", hx + 2, hy - 44, hx - 2, hy - 44, hx - 5, hy - 26),
            ("Z",),
        ], c["crown"])

    # belly
    belly_col = c.get("belly", c["body"])
    if belly_col != c["body"]:
        ellipse(ctx, bx - 14, by + 12, 28, 22, belly_col, belly_col, 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 32, head_col)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 28, hy - 6),
            ("C", hx - 24, hy - 44, hx + 24, hy - 44, hx + 28, hy - 6),
            ("Z",),
        ], c["cap"])

    # thin pointed beak
    poly(ctx, [
        (hx - 28, hy + 2),
        (hx - 60, hy + 8),
        (hx - 28, hy + 15),
    ], c["beak"])

    # eye
    circle(ctx, hx - 5, hy - 3, 7, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 3, hy - 5, 2.5, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 40, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 5 — WOODPECKER
# ===========================================================================
def draw_woodpecker(ctx, colors):
    c = colors
    bx, by = 275, 295
    hx, hy = 200, 245

    # stiff tail propped down
    tail_col = c.get("tail", rgb(22, 22, 22))
    poly(ctx, [
        (bx + 32, by + 18),
        (bx + 112, by + 112),
        (bx + 96, by + 120),
        (bx + 18, by + 38),
    ], tail_col)

    # body (upright)
    ellipse(ctx, bx, by, 60, 78, c["body"], angle=math.radians(-8))

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 15, by - 42),
        ("C", bx + 38, by - 66, bx + 74, by - 15, bx + 57, by + 40),
        ("C", bx + 28, by + 66, bx - 15, by + 46, bx - 15, by - 42),
        ("Z",),
    ], wing_col)

    # scapular / cheek patch (white)
    if "patch" in c:
        ellipse(ctx, bx - 5, by - 18, 24, 18, c["patch"], c["patch"], 0)

    # barred belly
    if c.get("barred"):
        bar_col = c.get("bar_color", rgb(225, 218, 200))
        for i, bar_y in enumerate(range(by - 12, by + 52, 14)):
            if i % 2 == 0:
                curve(ctx, [
                    ("M", bx - 36, bar_y),
                    ("L", bx - 36, bar_y + 10),
                    ("L", bx + 12, bar_y + 10),
                    ("L", bx + 12, bar_y),
                    ("Z",),
                ], bar_col, bar_col, 0)

    # red nape
    if "nape" in c:
        circle(ctx, hx + 30, hy - 14, 16, c["nape"], c["nape"], 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 42, head_col)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 36, hy - 10),
            ("C", hx - 30, hy - 54, hx + 30, hy - 54, hx + 36, hy - 10),
            ("Z",),
        ], c["cap"])

    # moustache
    if "moustache" in c:
        curve(ctx, [
            ("M", hx - 15, hy + 14),
            ("C", hx + 10, hy + 12, hx + 28, hy + 22, hx + 30, hy + 33),
            ("L", hx + 22, hy + 36),
            ("C", hx + 20, hy + 26, hx + 5, hy + 18, hx - 18, hy + 22),
            ("Z",),
        ], c["moustache"])

    # long chisel beak
    poly(ctx, [
        (hx - 35, hy - 4),
        (hx - 88, hy + 8),
        (hx - 35, hy + 22),
    ], c["beak"])

    # eye
    circle(ctx, hx - 6, hy - 4, 9, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 4, hy - 6, 3, (1, 1, 1), (1, 1, 1), 0)

    # feet (gripping)
    ctx.set_line_width(5)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    _set_color(ctx, c["feet"])
    for fx in [bx - 8, bx + 12]:
        ctx.move_to(fx, by + 68)
        ctx.line_to(fx, by + 88)
        ctx.move_to(fx, by + 88)
        ctx.line_to(fx - 14, by + 96)
        ctx.move_to(fx, by + 88)
        ctx.line_to(fx, by + 97)
        ctx.move_to(fx, by + 88)
        ctx.line_to(fx + 14, by + 96)
    ctx.stroke()

    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 6 — CORVID
# ===========================================================================
def draw_corvid(ctx, colors):
    c = colors
    bx, by = 278, 305
    hx, hy = 195, 255

    # tail
    tail_col = c.get("tail", c["body"])
    if c.get("very_long_tail"):
        curve(ctx, [
            ("M", bx + 42, by - 12),
            ("C", bx + 100, by - 56, bx + 202, by - 88, bx + 218, by - 56),
            ("C", bx + 232, by - 24, bx + 166, by + 16, bx + 98, by + 26),
            ("C", bx + 60, by + 32, bx + 42, by + 18, bx + 42, by - 12),
            ("Z",),
        ], tail_col)
        if "tail_patch" in c:
            curve(ctx, [
                ("M", bx + 52, by + 1),
                ("C", bx + 96, by - 24, bx + 152, by - 47, bx + 160, by - 30),
                ("C", bx + 168, by - 14, bx + 122, by + 5, bx + 70, by + 16),
                ("Z",),
            ], c["tail_patch"], c["tail_patch"], 0)
    else:
        curve(ctx, [
            ("M", bx + 40, by - 12),
            ("C", bx + 90, by - 36, bx + 134, by - 10, bx + 126, by + 23),
            ("C", bx + 118, by + 52, bx + 66, by + 36, bx + 40, by + 18),
            ("Z",),
        ], tail_col)

    # body
    ellipse(ctx, bx, by, 88, 65, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 20, by - 40),
        ("C", bx + 50, by - 66, bx + 94, by - 12, bx + 70, by + 40),
        ("C", bx + 36, by + 66, bx - 20, by + 46, bx - 20, by - 40),
        ("Z",),
    ], wing_col)

    # belly
    belly_col = c.get("belly", c["body"])
    if belly_col != c["body"]:
        ellipse(ctx, bx - 22, by + 20, 42, 32, belly_col, belly_col, 0)

    # nape
    if "nape" in c:
        curve(ctx, [
            ("M", hx + 28, hy - 15),
            ("C", hx + 50, hy - 20, hx + 60, hy + 5, hx + 52, hy + 26),
            ("C", hx + 42, hy + 42, hx + 22, hy + 36, hx + 18, hy + 15),
            ("Z",),
        ], c["nape"])

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 50, head_col)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 44, hy - 10),
            ("C", hx - 36, hy - 60, hx + 36, hy - 60, hx + 44, hy - 10),
            ("Z",),
        ], c["cap"])

    # moustache
    if "moustache" in c:
        curve(ctx, [
            ("M", hx - 18, hy + 18),
            ("C", hx + 5, hy + 14, hx + 30, hy + 25, hx + 32, hy + 38),
            ("L", hx + 22, hy + 40),
            ("C", hx + 20, hy + 28, hx, hy + 20, hx - 14, hy + 25),
            ("Z",),
        ], c["moustache"])

    # blue wing patch (Garrulus)
    if "wing_patch" in c:
        curve(ctx, [
            ("M", bx + 15, by - 38),
            ("C", bx + 42, by - 52, bx + 72, by - 28, bx + 64, by - 8),
            ("L", bx + 52, by - 5),
            ("C", bx + 56, by - 22, bx + 32, by - 42, bx + 10, by - 30),
            ("Z",),
        ], c["wing_patch"])
        bar_c = c.get("wing_bar", rgb(40, 60, 140))
        for bi in range(3):
            bpx = bx + 20 + bi * 14
            curve(ctx, [
                ("M", bpx, by - 38 + bi * 4),
                ("L", bpx + 4, by - 38 + bi * 4),
                ("L", bpx + 4, by - 18 + bi * 4),
                ("L", bpx, by - 18 + bi * 4),
                ("Z",),
            ], bar_c, bar_c, 0)

    # strong beak
    poly(ctx, [
        (hx - 42, hy - 4),
        (hx - 84, hy + 8),
        (hx - 42, hy + 26),
    ], c["beak"])

    # eye
    circle(ctx, hx - 8, hy - 5, 10, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 6, hy - 7, 3.5, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 60, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 7 — PIGEON
# ===========================================================================
def draw_pigeon(ctx, colors):
    c = colors
    bx, by = 282, 308
    hx, hy = 205, 262

    # tail
    tail_col = c.get("tail", c["wing"])
    curve(ctx, [
        ("M", bx + 38, by - 5),
        ("C", bx + 86, by - 26, bx + 128, by - 2, bx + 120, by + 28),
        ("C", bx + 112, by + 56, bx + 68, by + 40, bx + 38, by + 22),
        ("Z",),
    ], tail_col)

    # body
    ellipse(ctx, bx, by, 90, 68, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 22, by - 38),
        ("C", bx + 48, by - 64, bx + 92, by - 10, bx + 70, by + 42),
        ("C", bx + 36, by + 70, bx - 22, by + 50, bx - 22, by - 38),
        ("Z",),
    ], wing_col)

    # neck ring
    if "neck_ring" in c:
        curve(ctx, [
            ("M", hx + 22, hy + 12),
            ("C", hx + 46, hy + 6, hx + 58, hy + 28, hx + 44, hy + 44),
            ("C", hx + 28, hy + 58, hx, hy + 54, hx - 5, hy + 40),
            ("C", hx - 10, hy + 26, hx + 2, hy + 18, hx + 22, hy + 12),
            ("Z",),
        ], c["neck_ring"])

    # white neck patch (Columba palumbus)
    if "neck_patch" in c:
        ellipse(ctx, hx + 30, hy + 20, 20, 14, c["neck_patch"], c["neck_patch"], 0)

    # iridescent throat
    if "throat" in c:
        ellipse(ctx, hx + 10, hy + 14, 16, 12, c["throat"], c["throat"], 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 36, head_col)

    # short beak
    poly(ctx, [
        (hx - 30, hy + 0),
        (hx - 64, hy + 8),
        (hx - 30, hy + 18),
    ], c["beak"])
    # cere
    ellipse(ctx, hx - 32, hy + 5, 7, 5, rgb(210, 200, 190), rgb(180, 170, 160), 2)

    # eye
    eye_col = c.get("eye_color", rgb(200, 160, 50))
    circle(ctx, hx - 5, hy - 4, 9, eye_col, OUTLINE, 3)
    circle(ctx, hx - 5, hy - 4, 5, (0.05, 0.05, 0.05), (0.05, 0.05, 0.05), 0)
    circle(ctx, hx - 3, hy - 6, 2.5, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 62, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 8 — SWIFT / SWALLOW
# ===========================================================================
def draw_swift_swallow(ctx, colors):
    c = colors
    bx, by = 272, 318
    hx, hy = 215, 295

    tail_col = c.get("tail", c["body"])
    fork_depth = c.get("fork_depth", 45)

    # upper fork arm
    curve(ctx, [
        ("M", bx + 25, by - 5),
        ("C", bx + 70, by - 15, bx + 116, by - 36, bx + 132, by - 57),
        ("C", bx + 142, by - 72, bx + 137, by - 80, bx + 122, by - 74),
        ("C", bx + 107, by - 68, bx + 80, by - 47, bx + 62, by - 27),
        ("Z",),
    ], tail_col)
    if fork_depth > 25:
        # lower fork arm
        curve(ctx, [
            ("M", bx + 25, by + 8),
            ("C", bx + 68, by + 18, bx + 112, by + 40, bx + 128, by + 60),
            ("C", bx + 138, by + 75, bx + 133, by + 82, bx + 118, by + 76),
            ("C", bx + 103, by + 69, bx + 76, by + 52, bx + 58, by + 32),
            ("Z",),
        ], tail_col)

    # body
    ellipse(ctx, bx, by, 68, 32, c["body"])

    # upper sickle wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 20, by - 8),
        ("C", bx + 10, by - 56, bx + 70, by - 60, bx + 77, by - 38),
        ("C", bx + 82, by - 22, bx + 52, by - 12, bx + 30, by - 5),
        ("Z",),
    ], wing_col)
    # lower sickle wing
    curve(ctx, [
        ("M", bx - 20, by + 8),
        ("C", bx + 10, by + 54, bx + 70, by + 57, bx + 77, by + 36),
        ("C", bx + 82, by + 20, bx + 52, by + 10, bx + 30, by + 5),
        ("Z",),
    ], wing_col)

    # belly / rump patches
    if "belly" in c:
        ellipse(ctx, bx - 10, by, 28, 14, c["belly"], c["belly"], 0)
    if "rump" in c:
        ellipse(ctx, bx + 30, by, 20, 12, c["rump"], c["rump"], 0)
    if "throat" in c:
        ellipse(ctx, hx + 2, hy + 2, 14, 10, c["throat"], c["throat"], 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 24, head_col)

    # tiny beak
    poly(ctx, [
        (hx - 20, hy + 0),
        (hx - 44, hy + 5),
        (hx - 20, hy + 12),
    ], c["beak"])

    # eye
    circle(ctx, hx - 4, hy - 3, 6, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 2, hy - 5, 2, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx - 10, by + 30, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 9 — NUTHATCH / CREEPER
# ===========================================================================
def draw_nuthatch_creeper(ctx, colors):
    c = colors
    bx, by = 285, 315
    hx, hy = 212, 272

    # tail
    tail_col = c.get("tail", c["wing"])
    if c.get("tail_down"):
        poly(ctx, [
            (bx + 30, by + 10),
            (bx + 88, by + 75),
            (bx + 74, by + 82),
            (bx + 16, by + 26),
        ], tail_col)
    else:
        curve(ctx, [
            ("M", bx + 28, by - 5),
            ("C", bx + 66, by - 20, bx + 96, by - 4, bx + 92, by + 20),
            ("C", bx + 86, by + 38, bx + 54, by + 28, bx + 28, by + 14),
            ("Z",),
        ], tail_col)

    # body
    ellipse(ctx, bx, by, 64, 48, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 14, by - 28),
        ("C", bx + 36, by - 50, bx + 74, by - 10, bx + 56, by + 30),
        ("C", bx + 26, by + 54, bx - 14, by + 36, bx - 14, by - 28),
        ("Z",),
    ], wing_col)

    # belly
    belly_col = c.get("belly", c["body"])
    if belly_col != c["body"]:
        curve(ctx, [
            ("M", hx + 22, hy + 30),
            ("C", hx + 42, hy + 54, bx - 28, by + 52, bx - 36, by + 28),
            ("C", bx - 42, by + 5, bx - 15, by - 18, hx + 18, hy + 22),
            ("Z",),
        ], belly_col)

    # eye stripe
    if "eye_stripe" in c:
        curve(ctx, [
            ("M", hx - 30, hy + 5),
            ("C", hx, hy - 5, hx + 30, hy - 2, hx + 40, hy + 5),
            ("L", hx + 40, hy + 15),
            ("C", hx + 28, hy + 8, hx, hy + 5, hx - 28, hy + 15),
            ("Z",),
        ], c["eye_stripe"])

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 36, head_col)

    # beak (curved or straight, long)
    beak_col = c["beak"]
    if c.get("curved_beak"):
        curve(ctx, [
            ("M", hx - 30, hy + 2),
            ("C", hx - 55, hy + 5, hx - 80, hy + 24, hx - 84, hy + 34),
            ("L", hx - 74, hy + 38),
            ("C", hx - 70, hy + 28, hx - 48, hy + 14, hx - 26, hy + 15),
            ("Z",),
        ], beak_col)
    else:
        poly(ctx, [
            (hx - 30, hy + 0),
            (hx - 80, hy + 8),
            (hx - 30, hy + 18),
        ], beak_col)

    # eye
    circle(ctx, hx - 6, hy - 3, 8, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 4, hy - 5, 3, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 46, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 10 — WAGTAIL / STARLING / WAXWING
# ===========================================================================
def draw_wagtail_starling(ctx, colors):
    c = colors
    bx, by = 280, 312
    hx, hy = 210, 268

    tail_col = c.get("tail", c["wing"])
    long_tail = c.get("long_tail", False)

    if long_tail:
        curve(ctx, [
            ("M", bx + 30, by - 8),
            ("C", bx + 76, by - 36, bx + 150, by - 54, bx + 160, by - 25),
            ("C", bx + 168, by + 2, bx + 122, by + 20, bx + 66, by + 16),
            ("C", bx + 46, by + 12, bx + 30, by + 5, bx + 30, by - 8),
            ("Z",),
        ], tail_col)
        if "tail_outer" in c:
            curve(ctx, [
                ("M", bx + 30, by - 8),
                ("L", bx + 160, by - 25),
                ("L", bx + 157, by - 18),
                ("L", bx + 30, by),
            ], None, c["tail_outer"], 3)
    else:
        curve(ctx, [
            ("M", bx + 25, by - 6),
            ("C", bx + 66, by - 24, bx + 102, by - 5, bx + 96, by + 18),
            ("C", bx + 90, by + 40, bx + 52, by + 26, bx + 25, by + 12),
            ("Z",),
        ], tail_col)

    # body
    body_rx = 58 if long_tail else 62
    ellipse(ctx, bx, by, body_rx, 46, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 14, by - 26),
        ("C", bx + 36, by - 48, bx + 76, by - 8, bx + 58, by + 26),
        ("C", bx + 28, by + 50, bx - 14, by + 32, bx - 14, by - 26),
        ("Z",),
    ], wing_col)

    # iridescent spots (Sturnus)
    if c.get("spotted"):
        spot_col = c.get("spot_color", rgb(215, 208, 188))
        for sx, sy in [
            (bx, by - 20), (bx + 25, by - 30), (bx - 10, by),
            (bx + 15, by + 5), (bx + 36, by - 8), (bx - 5, by + 20),
        ]:
            circle(ctx, sx, sy, 5, spot_col, spot_col, 0)

    # crest (Bombycilla)
    if "crest_color" in c:
        curve(ctx, [
            ("M", hx - 5, hy - 32),
            ("C", hx - 12, hy - 60, hx + 18, hy - 74, hx + 22, hy - 52),
            ("C", hx + 26, hy - 36, hx + 10, hy - 28, hx + 5, hy - 28),
            ("Z",),
        ], c["crest_color"])

    # wing tip (Bombycilla yellow)
    if "wing_tip" in c:
        curve(ctx, [
            ("M", bx + 45, by + 18),
            ("C", bx + 66, by + 12, bx + 82, by + 18, bx + 80, by + 30),
            ("L", bx + 70, by + 33),
            ("C", bx + 70, by + 22, bx + 56, by + 22, bx + 42, by + 27),
            ("Z",),
        ], c["wing_tip"])

    # belly
    belly_col = c.get("belly", c["body"])
    if belly_col != c["body"]:
        ellipse(ctx, bx - 16, by + 12, 32, 25, belly_col, belly_col, 0)

    # breast band (Motacilla)
    if "breast_band" in c:
        curve(ctx, [
            ("M", hx + 18, hy + 22),
            ("C", hx + 36, hy + 18, bx - 20, by - 10, bx - 28, by + 5),
            ("C", bx - 36, by + 20, bx - 22, by + 32, hx + 12, hy + 30),
            ("Z",),
        ], c["breast_band"])

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 36, head_col)

    # cap
    if "cap" in c:
        curve(ctx, [
            ("M", hx - 30, hy - 6),
            ("C", hx - 25, hy - 46, hx + 25, hy - 46, hx + 30, hy - 6),
            ("Z",),
        ], c["cap"])

    # beak
    poly(ctx, [
        (hx - 30, hy + 1),
        (hx - 64, hy + 9),
        (hx - 30, hy + 18),
    ], c["beak"])

    # eye
    circle(ctx, hx - 5, hy - 3, 8, (0.05, 0.05, 0.05), OUTLINE, 3)
    circle(ctx, hx - 3, hy - 5, 2.5, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 44, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# TEMPLATE 11 — CUCKOO
# ===========================================================================
def draw_cuckoo(ctx, colors):
    c = colors
    bx, by = 270, 308
    hx, hy = 200, 264

    # long graduated tail
    tail_col = c.get("tail", c["body"])
    curve(ctx, [
        ("M", bx + 32, by - 10),
        ("C", bx + 86, by - 42, bx + 150, by - 40, bx + 154, by - 10),
        ("C", bx + 158, by + 16, bx + 104, by + 32, bx + 52, by + 24),
        ("C", bx + 38, by + 18, bx + 32, by + 8, bx + 32, by - 10),
        ("Z",),
    ], tail_col)
    # tail barring
    if c.get("barred_tail"):
        for ti in range(4):
            tx = bx + 58 + ti * 22
            curve(ctx, [
                ("M", tx, by - 26 + ti * 2),
                ("L", tx + 5, by - 26 + ti * 2),
                ("L", tx + 5, by + 16 + ti * 2),
                ("L", tx, by + 16 + ti * 2),
                ("Z",),
            ], rgb(80, 80, 80), rgb(80, 80, 80), 0)

    # body
    ellipse(ctx, bx, by, 75, 48, c["body"])

    # wing
    wing_col = c.get("wing", c["body"])
    curve(ctx, [
        ("M", bx - 16, by - 28),
        ("C", bx + 38, by - 52, bx + 80, by - 10, bx + 60, by + 28),
        ("C", bx + 28, by + 54, bx - 16, by + 36, bx - 16, by - 28),
        ("Z",),
    ], wing_col)

    # barred belly
    if c.get("barred"):
        bar_col = c.get("bar_color", rgb(85, 85, 85))
        bar_bg = c.get("bar_bg", rgb(232, 228, 220))
        curve(ctx, [
            ("M", hx + 20, hy + 30),
            ("C", hx + 40, hy + 54, bx - 30, by + 52, bx - 38, by + 25),
            ("C", bx - 44, by + 3, bx - 18, by - 26, hx + 18, hy + 22),
            ("Z",),
        ], bar_bg)
        for bi in range(6):
            bar_y = hy + 32 + bi * 10
            if bar_y > by + 50:
                break
            curve(ctx, [
                ("M", hx + 18 + bi * 2, bar_y),
                ("L", bx - 32 + bi, bar_y),
                ("L", bx - 32 + bi, bar_y + 5),
                ("L", hx + 18 + bi * 2, bar_y + 5),
                ("Z",),
            ], bar_col, bar_col, 0)

    # head
    head_col = c.get("head", c["body"])
    circle(ctx, hx, hy, 40, head_col)

    # hooked beak
    curve(ctx, [
        ("M", hx - 34, hy + 0),
        ("C", hx - 60, hy - 2, hx - 80, hy + 4, hx - 82, hy + 10),
        ("L", hx - 72, hy + 16),
        ("C", hx - 70, hy + 10, hx - 52, hy + 8, hx - 34, hy + 14),
        ("Z",),
    ], c["beak"])

    # eye
    eye_col = c.get("eye_color", rgb(200, 165, 30))
    circle(ctx, hx - 6, hy - 3, 9, eye_col, OUTLINE, 3)
    circle(ctx, hx - 6, hy - 3, 5, (0.05, 0.05, 0.05), (0.05, 0.05, 0.05), 0)
    circle(ctx, hx - 4, hy - 5, 2.5, (1, 1, 1), (1, 1, 1), 0)

    draw_feet(ctx, bx, by + 48, c["feet"])
    draw_branch(ctx)


# ===========================================================================
# SPECIES REGISTRY
# ===========================================================================
SPECIES = {
    "Aegithalos_caudatus": {
        "name_no": "Stjertmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(248, 240, 235),
            "wing": rgb(88, 58, 48),
            "belly": rgb(248, 200, 180),
            "head": rgb(248, 240, 235),
            "beak": rgb(30, 28, 28),
            "feet": rgb(80, 65, 55),
            "cap": rgb(22, 18, 18),
            "long_tail": True,
            "tail": rgb(22, 18, 18),
            "tail2": rgb(248, 240, 235),
            "tail_stripe": rgb(248, 240, 235),
        },
    },
    "Apus_apus": {
        "name_no": "Tårnseiler",
        "template": draw_swift_swallow,
        "colors": {
            "body": rgb(35, 30, 28),
            "wing": rgb(28, 24, 22),
            "head": rgb(35, 30, 28),
            "beak": rgb(22, 18, 18),
            "feet": rgb(40, 35, 30),
            "tail": rgb(28, 24, 22),
            "belly": rgb(55, 50, 45),
            "fork_depth": 55,
        },
    },
    "Bombycilla_garrulus": {
        "name_no": "Sidensvans",
        "template": draw_wagtail_starling,
        "colors": {
            "body": rgb(160, 130, 112),
            "wing": rgb(78, 68, 62),
            "belly": rgb(148, 115, 95),
            "head": rgb(160, 130, 112),
            "beak": rgb(28, 22, 18),
            "feet": rgb(60, 50, 42),
            "cap": rgb(170, 138, 118),
            "crest_color": rgb(175, 145, 122),
            "wing_tip": rgb(235, 210, 30),
            "tail": rgb(55, 48, 42),
            "breast_band": rgb(55, 48, 42),
        },
    },
    "Carduelis_carduelis": {
        "name_no": "Stillits",
        "template": draw_finch,
        "colors": {
            "body": rgb(148, 118, 82),
            "wing": rgb(22, 22, 22),
            "belly": rgb(235, 228, 210),
            "head": rgb(235, 228, 210),
            "beak": rgb(220, 200, 165),
            "feet": rgb(80, 68, 55),
            "face": rgb(195, 40, 30),
            "cap": rgb(22, 22, 22),
            "flash": rgb(230, 195, 25),
        },
    },
    "Certhia_brachydactyla": {
        "name_no": "Trekryper",
        "template": draw_nuthatch_creeper,
        "colors": {
            "body": rgb(118, 88, 58),
            "wing": rgb(88, 65, 40),
            "belly": rgb(235, 228, 215),
            "head": rgb(105, 78, 52),
            "beak": rgb(68, 55, 40),
            "feet": rgb(72, 58, 42),
            "eye_stripe": rgb(235, 228, 215),
            "curved_beak": True,
            "tail_down": True,
            "tail": rgb(88, 65, 40),
        },
    },
    "Chloris_chloris": {
        "name_no": "Grønnfink",
        "template": draw_finch,
        "colors": {
            "body": rgb(108, 148, 62),
            "wing": rgb(78, 112, 45),
            "belly": rgb(118, 162, 72),
            "head": rgb(105, 145, 60),
            "beak": rgb(210, 185, 155),
            "feet": rgb(80, 70, 55),
            "flash": rgb(215, 198, 28),
            "tail": rgb(55, 82, 30),
        },
    },
    "Coccothraustes_coccothraustes": {
        "name_no": "Kjernebiter",
        "template": draw_finch,
        "colors": {
            "body": rgb(175, 128, 82),
            "wing": rgb(32, 28, 25),
            "belly": rgb(190, 148, 98),
            "head": rgb(195, 125, 65),
            "beak": rgb(148, 185, 195),
            "feet": rgb(80, 68, 55),
            "cap": rgb(22, 18, 18),
            "bar": rgb(255, 255, 255),
            "massive_beak": True,
        },
    },
    "Columba_palumbus": {
        "name_no": "Ringdue",
        "template": draw_pigeon,
        "colors": {
            "body": rgb(118, 128, 148),
            "wing": rgb(105, 115, 132),
            "head": rgb(105, 118, 138),
            "beak": rgb(195, 175, 148),
            "feet": rgb(195, 128, 110),
            "neck_patch": rgb(235, 235, 228),
            "throat": rgb(128, 148, 118),
            "eye_color": rgb(230, 215, 55),
            "tail": rgb(82, 88, 98),
        },
    },
    "Corvus_corone": {
        "name_no": "Kråke",
        "template": draw_corvid,
        "colors": {
            "body": rgb(32, 30, 30),
            "wing": rgb(25, 22, 22),
            "head": rgb(32, 30, 30),
            "beak": rgb(22, 18, 18),
            "feet": rgb(38, 34, 32),
            "tail": rgb(22, 18, 18),
        },
    },
    "Corvus_monedula": {
        "name_no": "Kaie",
        "template": draw_corvid,
        "colors": {
            "body": rgb(65, 65, 68),
            "wing": rgb(35, 32, 32),
            "head": rgb(32, 30, 30),
            "beak": rgb(22, 18, 18),
            "feet": rgb(45, 40, 38),
            "cap": rgb(22, 18, 18),
            "nape": rgb(115, 118, 125),
            "tail": rgb(42, 38, 38),
        },
    },
    "Cuculus_canorus": {
        "name_no": "Gjøk",
        "template": draw_cuckoo,
        "colors": {
            "body": rgb(118, 122, 132),
            "wing": rgb(90, 92, 100),
            "head": rgb(108, 112, 122),
            "beak": rgb(55, 52, 45),
            "feet": rgb(198, 168, 55),
            "tail": rgb(82, 85, 92),
            "barred": True,
            "barred_tail": True,
        },
    },
    "Cyanistes_caeruleus": {
        "name_no": "Blåmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(68, 122, 175),
            "wing": rgb(50, 92, 148),
            "belly": rgb(218, 215, 55),
            "head": rgb(255, 255, 255),
            "beak": rgb(28, 24, 22),
            "feet": rgb(75, 65, 52),
            "cap": rgb(45, 95, 168),
            "cheek": rgb(255, 255, 255),
            "stripe": rgb(22, 18, 18),
        },
    },
    "Delichon_urbicum": {
        "name_no": "Taksvale",
        "template": draw_swift_swallow,
        "colors": {
            "body": rgb(28, 28, 32),
            "wing": rgb(22, 22, 28),
            "head": rgb(28, 28, 32),
            "beak": rgb(22, 18, 18),
            "feet": rgb(245, 242, 238),
            "belly": rgb(245, 242, 238),
            "rump": rgb(245, 242, 238),
            "tail": rgb(22, 22, 28),
            "fork_depth": 18,
        },
    },
    "Dendrocopos_major": {
        "name_no": "Flaggspett",
        "template": draw_woodpecker,
        "colors": {
            "body": rgb(22, 22, 22),
            "wing": rgb(22, 22, 22),
            "head": rgb(22, 22, 22),
            "beak": rgb(48, 45, 40),
            "feet": rgb(58, 52, 42),
            "cap": rgb(22, 22, 22),
            "nape": rgb(198, 35, 28),
            "patch": rgb(238, 232, 218),
            "barred": True,
            "bar_color": rgb(238, 232, 218),
        },
    },
    "Dryobates_minor": {
        "name_no": "Dvergspett",
        "template": draw_woodpecker,
        "colors": {
            "body": rgb(28, 25, 22),
            "wing": rgb(28, 25, 22),
            "head": rgb(235, 228, 212),
            "beak": rgb(55, 50, 42),
            "feet": rgb(58, 52, 42),
            "cap": rgb(188, 35, 28),
            "patch": rgb(235, 228, 212),
            "barred": True,
            "bar_color": rgb(235, 228, 212),
        },
    },
    "Emberiza_citrinella": {
        "name_no": "Gulspurv",
        "template": draw_finch,
        "colors": {
            "body": rgb(148, 118, 60),
            "wing": rgb(118, 88, 42),
            "belly": rgb(228, 218, 28),
            "head": rgb(225, 215, 25),
            "beak": rgb(165, 148, 118),
            "feet": rgb(188, 155, 120),
            "breast": rgb(228, 218, 28),
            "tail": rgb(95, 72, 38),
        },
    },
    "Erithacus_rubecula": {
        "name_no": "Rødstrupe",
        "template": draw_thrush,
        "colors": {
            "body": rgb(88, 78, 62),
            "wing": rgb(78, 68, 54),
            "belly": rgb(235, 228, 210),
            "head": rgb(88, 78, 62),
            "beak": rgb(48, 42, 35),
            "feet": rgb(72, 60, 48),
            "face": rgb(215, 95, 32),
            "breast": rgb(218, 98, 35),
        },
    },
    "Fringilla_coelebs": {
        "name_no": "Bokfink",
        "template": draw_finch,
        "colors": {
            "body": rgb(78, 92, 125),
            "wing": rgb(35, 32, 28),
            "belly": rgb(185, 138, 108),
            "head": rgb(72, 88, 120),
            "beak": rgb(148, 162, 175),
            "feet": rgb(75, 65, 52),
            "breast": rgb(188, 142, 112),
            "bar": rgb(235, 230, 218),
            "tail": rgb(35, 32, 28),
        },
    },
    "Fringilla_montifringilla": {
        "name_no": "Bjørkefink",
        "template": draw_finch,
        "colors": {
            "body": rgb(35, 30, 28),
            "wing": rgb(28, 25, 22),
            "belly": rgb(222, 118, 38),
            "head": rgb(22, 18, 18),
            "beak": rgb(215, 195, 45),
            "feet": rgb(72, 62, 50),
            "breast": rgb(218, 115, 35),
            "rump": rgb(245, 242, 235),
            "bar": rgb(215, 115, 28),
        },
    },
    "Garrulus_glandarius": {
        "name_no": "Nøtteskrike",
        "template": draw_corvid,
        "colors": {
            "body": rgb(185, 148, 125),
            "wing": rgb(22, 18, 18),
            "head": rgb(185, 148, 125),
            "beak": rgb(42, 38, 34),
            "feet": rgb(148, 122, 95),
            "moustache": rgb(22, 18, 18),
            "wing_patch": rgb(72, 112, 195),
            "wing_bar": rgb(35, 58, 142),
            "belly": rgb(195, 158, 132),
            "tail": rgb(22, 18, 18),
        },
    },
    "Hirundo_rustica": {
        "name_no": "Låvesvale",
        "template": draw_swift_swallow,
        "colors": {
            "body": rgb(28, 45, 88),
            "wing": rgb(22, 38, 78),
            "head": rgb(28, 45, 88),
            "beak": rgb(22, 18, 18),
            "feet": rgb(55, 48, 40),
            "throat": rgb(188, 75, 28),
            "belly": rgb(232, 222, 195),
            "tail": rgb(22, 38, 78),
            "fork_depth": 65,
        },
    },
    "Lophophanes_cristatus": {
        "name_no": "Toppmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(98, 88, 75),
            "wing": rgb(85, 75, 62),
            "belly": rgb(230, 218, 198),
            "head": rgb(235, 228, 215),
            "beak": rgb(32, 28, 22),
            "feet": rgb(75, 62, 50),
            "cap": rgb(32, 28, 22),
            "cheek": rgb(235, 228, 215),
            "crest_color": rgb(32, 28, 22),
        },
    },
    "Luscinia_megarhynchos": {
        "name_no": "Sørnattergal",
        "template": draw_thrush,
        "colors": {
            "body": rgb(138, 105, 72),
            "wing": rgb(125, 95, 65),
            "belly": rgb(218, 205, 185),
            "head": rgb(130, 98, 68),
            "beak": rgb(48, 42, 35),
            "feet": rgb(108, 88, 65),
            "tail": rgb(185, 80, 45),
            "breast": rgb(218, 205, 185),
        },
    },
    "Motacilla_alba": {
        "name_no": "Linerle",
        "template": draw_wagtail_starling,
        "colors": {
            "body": rgb(235, 230, 220),
            "wing": rgb(55, 52, 50),
            "belly": rgb(235, 230, 220),
            "head": rgb(235, 230, 220),
            "beak": rgb(28, 22, 18),
            "feet": rgb(42, 38, 34),
            "cap": rgb(22, 18, 18),
            "breast_band": rgb(22, 18, 18),
            "tail": rgb(22, 18, 18),
            "tail_outer": rgb(235, 230, 220),
            "long_tail": True,
        },
    },
    "Parus_major": {
        "name_no": "Kjøttmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(68, 112, 52),
            "wing": rgb(52, 88, 40),
            "belly": rgb(228, 215, 48),
            "head": rgb(22, 18, 18),
            "beak": rgb(22, 18, 18),
            "feet": rgb(72, 60, 48),
            "cap": rgb(22, 18, 18),
            "cheek": rgb(242, 238, 228),
            "stripe": rgb(22, 18, 18),
        },
    },
    "Passer_domesticus": {
        "name_no": "Gråspurv",
        "template": draw_finch,
        "colors": {
            "body": rgb(138, 112, 82),
            "wing": rgb(118, 88, 55),
            "belly": rgb(195, 185, 172),
            "head": rgb(115, 128, 148),
            "beak": rgb(42, 38, 32),
            "feet": rgb(138, 112, 82),
            "cap": rgb(88, 75, 58),
            "bib": rgb(28, 22, 18),
            "bar": rgb(228, 215, 195),
        },
    },
    "Passer_montanus": {
        "name_no": "Pilfink",
        "template": draw_finch,
        "colors": {
            "body": rgb(148, 112, 72),
            "wing": rgb(118, 88, 52),
            "belly": rgb(215, 205, 188),
            "head": rgb(235, 228, 215),
            "beak": rgb(38, 34, 28),
            "feet": rgb(128, 102, 72),
            "cap": rgb(155, 78, 35),
            "bib": rgb(28, 22, 18),
        },
    },
    "Periparus_ater": {
        "name_no": "Svartmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(110, 118, 128),
            "wing": rgb(88, 95, 105),
            "belly": rgb(215, 208, 195),
            "head": rgb(22, 18, 18),
            "beak": rgb(22, 18, 18),
            "feet": rgb(72, 62, 50),
            "cap": rgb(22, 18, 18),
            "cheek": rgb(242, 238, 228),
        },
    },
    "Phoenicurus_ochruros": {
        "name_no": "Svartrødstjert",
        "template": draw_thrush,
        "colors": {
            "body": rgb(42, 40, 40),
            "wing": rgb(38, 35, 35),
            "belly": rgb(185, 82, 38),
            "head": rgb(35, 32, 32),
            "beak": rgb(28, 22, 18),
            "feet": rgb(32, 28, 25),
            "tail": rgb(195, 88, 42),
            "breast": rgb(188, 85, 40),
        },
    },
    "Phoenicurus_phoenicurus": {
        "name_no": "Rødstjert",
        "template": draw_thrush,
        "colors": {
            "body": rgb(115, 128, 148),
            "wing": rgb(95, 108, 128),
            "belly": rgb(205, 118, 55),
            "head": rgb(108, 120, 142),
            "beak": rgb(28, 22, 18),
            "feet": rgb(42, 38, 32),
            "cap": rgb(28, 25, 22),
            "face": rgb(28, 25, 22),
            "breast": rgb(205, 118, 55),
            "tail": rgb(195, 105, 45),
        },
    },
    "Phylloscopus_collybita": {
        "name_no": "Gransanger",
        "template": draw_warbler,
        "colors": {
            "body": rgb(112, 105, 72),
            "wing": rgb(95, 88, 60),
            "belly": rgb(215, 208, 182),
            "head": rgb(108, 100, 68),
            "beak": rgb(42, 38, 30),
            "feet": rgb(28, 22, 18),
            "tail": rgb(95, 88, 60),
        },
    },
    "Phylloscopus_trochilus": {
        "name_no": "Løvsanger",
        "template": draw_warbler,
        "colors": {
            "body": rgb(118, 128, 72),
            "wing": rgb(100, 108, 60),
            "belly": rgb(228, 222, 175),
            "head": rgb(112, 122, 68),
            "beak": rgb(48, 42, 32),
            "feet": rgb(105, 88, 65),
            "supercilium": rgb(232, 225, 182),
            "tail": rgb(95, 105, 55),
        },
    },
    "Pica_pica": {
        "name_no": "Skjære",
        "template": draw_corvid,
        "colors": {
            "body": rgb(22, 18, 18),
            "wing": rgb(22, 18, 18),
            "head": rgb(22, 18, 18),
            "beak": rgb(22, 18, 18),
            "feet": rgb(42, 38, 34),
            "belly": rgb(245, 242, 238),
            "tail": rgb(38, 88, 55),
            "tail_patch": rgb(38, 88, 55),
            "very_long_tail": True,
        },
    },
    "Picus_viridis": {
        "name_no": "Grønnspett",
        "template": draw_woodpecker,
        "colors": {
            "body": rgb(68, 118, 52),
            "wing": rgb(55, 98, 42),
            "head": rgb(68, 118, 52),
            "beak": rgb(48, 45, 40),
            "feet": rgb(58, 52, 42),
            "cap": rgb(195, 32, 28),
            "moustache": rgb(195, 32, 28),
            "patch": rgb(228, 222, 195),
        },
    },
    "Poecile_montanus": {
        "name_no": "Granmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(148, 118, 88),
            "wing": rgb(125, 98, 72),
            "belly": rgb(222, 210, 188),
            "head": rgb(22, 18, 18),
            "beak": rgb(22, 18, 18),
            "feet": rgb(72, 62, 50),
            "cap": rgb(22, 18, 18),
            "cheek": rgb(242, 238, 228),
            "stripe": rgb(22, 18, 18),
        },
    },
    "Poecile_palustris": {
        "name_no": "Løvmeis",
        "template": draw_tit,
        "colors": {
            "body": rgb(138, 115, 88),
            "wing": rgb(118, 95, 72),
            "belly": rgb(228, 215, 195),
            "head": rgb(22, 18, 18),
            "beak": rgb(22, 18, 18),
            "feet": rgb(72, 62, 50),
            "cap": rgb(22, 18, 18),
            "cheek": rgb(242, 238, 228),
        },
    },
    "Prunella_modularis": {
        "name_no": "Jernspurv",
        "template": draw_finch,
        "colors": {
            "body": rgb(105, 88, 68),
            "wing": rgb(88, 72, 55),
            "belly": rgb(115, 125, 148),
            "head": rgb(108, 118, 142),
            "beak": rgb(42, 38, 30),
            "feet": rgb(145, 118, 88),
            "breast": rgb(112, 122, 145),
        },
    },
    "Pyrrhula_pyrrhula": {
        "name_no": "Dompap",
        "template": draw_finch,
        "colors": {
            "body": rgb(115, 118, 128),
            "wing": rgb(28, 25, 22),
            "belly": rgb(218, 55, 42),
            "head": rgb(22, 18, 18),
            "beak": rgb(32, 28, 24),
            "feet": rgb(95, 78, 62),
            "cap": rgb(22, 18, 18),
            "breast": rgb(215, 52, 38),
            "bar": rgb(235, 228, 215),
        },
    },
    "Regulus_ignicapilla": {
        "name_no": "Rødtoppfuglekonge",
        "template": draw_warbler,
        "colors": {
            "body": rgb(98, 115, 62),
            "wing": rgb(80, 95, 50),
            "belly": rgb(215, 210, 185),
            "head": rgb(92, 108, 58),
            "beak": rgb(38, 32, 25),
            "feet": rgb(68, 55, 42),
            "supercilium": rgb(228, 222, 195),
            "crown": rgb(205, 92, 28),
            "tail": rgb(78, 92, 48),
        },
    },
    "Regulus_regulus": {
        "name_no": "Fuglekonge",
        "template": draw_warbler,
        "colors": {
            "body": rgb(95, 112, 60),
            "wing": rgb(78, 95, 48),
            "belly": rgb(218, 212, 188),
            "head": rgb(90, 105, 58),
            "beak": rgb(38, 32, 25),
            "feet": rgb(68, 55, 42),
            "supercilium": rgb(228, 222, 195),
            "crown": rgb(228, 208, 30),
            "tail": rgb(75, 90, 46),
        },
    },
    "Sitta_europaea": {
        "name_no": "Spettmeis",
        "template": draw_nuthatch_creeper,
        "colors": {
            "body": rgb(78, 108, 148),
            "wing": rgb(65, 92, 128),
            "belly": rgb(198, 128, 72),
            "head": rgb(75, 105, 145),
            "beak": rgb(52, 48, 40),
            "feet": rgb(148, 108, 72),
            "eye_stripe": rgb(22, 18, 18),
            "tail": rgb(58, 80, 112),
        },
    },
    "Spinus_spinus": {
        "name_no": "Grønnsisik",
        "template": draw_finch,
        "colors": {
            "body": rgb(108, 145, 58),
            "wing": rgb(42, 38, 32),
            "belly": rgb(205, 215, 125),
            "head": rgb(100, 138, 52),
            "beak": rgb(148, 165, 178),
            "feet": rgb(78, 68, 55),
            "cap": rgb(22, 18, 18),
            "bib": rgb(22, 18, 18),
            "flash": rgb(215, 205, 28),
        },
    },
    "Streptopelia_decaocto": {
        "name_no": "Tyrkerdue",
        "template": draw_pigeon,
        "colors": {
            "body": rgb(205, 178, 155),
            "wing": rgb(185, 158, 135),
            "head": rgb(198, 172, 148),
            "beak": rgb(38, 34, 30),
            "feet": rgb(185, 118, 105),
            "neck_ring": rgb(28, 22, 18),
            "eye_color": rgb(195, 45, 32),
            "tail": rgb(158, 138, 118),
        },
    },
    "Sturnus_vulgaris": {
        "name_no": "Stær",
        "template": draw_wagtail_starling,
        "colors": {
            "body": rgb(42, 50, 55),
            "wing": rgb(35, 42, 48),
            "belly": rgb(48, 58, 62),
            "head": rgb(40, 48, 52),
            "beak": rgb(215, 195, 45),
            "feet": rgb(188, 148, 105),
            "spotted": True,
            "spot_color": rgb(215, 208, 188),
            "tail": rgb(35, 42, 48),
        },
    },
    "Sylvia_atricapilla": {
        "name_no": "Munk",
        "template": draw_warbler,
        "colors": {
            "body": rgb(118, 118, 122),
            "wing": rgb(95, 95, 98),
            "belly": rgb(225, 218, 205),
            "head": rgb(112, 112, 118),
            "beak": rgb(48, 42, 35),
            "feet": rgb(78, 68, 55),
            "cap": rgb(22, 18, 18),
            "tail": rgb(88, 88, 92),
        },
    },
    "Sylvia_borin": {
        "name_no": "Hagesanger",
        "template": draw_warbler,
        "colors": {
            "body": rgb(138, 128, 108),
            "wing": rgb(118, 108, 90),
            "belly": rgb(225, 215, 195),
            "head": rgb(130, 120, 102),
            "beak": rgb(52, 46, 38),
            "feet": rgb(92, 78, 62),
            "tail": rgb(110, 100, 82),
        },
    },
    "Troglodytes_troglodytes": {
        "name_no": "Gjerdesmett",
        "template": draw_warbler,
        "colors": {
            "body": rgb(145, 108, 68),
            "wing": rgb(122, 88, 52),
            "belly": rgb(185, 148, 98),
            "head": rgb(138, 102, 65),
            "beak": rgb(48, 42, 32),
            "feet": rgb(115, 90, 62),
            "tail_up": True,
            "tail": rgb(128, 95, 58),
        },
    },
    "Turdus_merula": {
        "name_no": "Svarttrost",
        "template": draw_thrush,
        "colors": {
            "body": rgb(28, 25, 22),
            "wing": rgb(22, 20, 18),
            "belly": rgb(28, 25, 22),
            "head": rgb(28, 25, 22),
            "beak": rgb(228, 162, 28),
            "feet": rgb(65, 55, 42),
            "eye_ring": rgb(215, 148, 22),
        },
    },
    "Turdus_philomelos": {
        "name_no": "Måltrost",
        "template": draw_thrush,
        "colors": {
            "body": rgb(118, 88, 58),
            "wing": rgb(100, 72, 45),
            "belly": rgb(235, 225, 195),
            "head": rgb(112, 82, 52),
            "beak": rgb(55, 48, 38),
            "feet": rgb(155, 118, 82),
            "breast": rgb(235, 218, 178),
            "spotted": True,
            "spot_color": rgb(90, 62, 35),
        },
    },
}


# ===========================================================================
# Render
# ===========================================================================
def render_species(key: str, out_path: str) -> None:
    entry = SPECIES[key]
    surface, ctx = new_surface()
    entry["template"](ctx, entry["colors"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    surface.write_to_png(out_path)


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate flat-art bird icons.")
    parser.add_argument("species", nargs="?", help="Species key, e.g. Parus_major")
    parser.add_argument("--out", help="Output path (single species)")
    parser.add_argument("--outdir", help="Output directory (overrides default)")
    args = parser.parse_args()

    outdir = os.path.realpath(args.outdir or DEFAULT_OUT)

    if args.species:
        key = args.species
        if key not in SPECIES:
            print(f"Unknown species '{key}'. Available keys:")
            for k in sorted(SPECIES):
                print(f"  {k}")
            sys.exit(1)
        out_path = args.out or os.path.join(outdir, f"{key}_vector.png")
        render_species(key, out_path)
        print(f"Wrote {out_path}")
    else:
        os.makedirs(outdir, exist_ok=True)
        for key in sorted(SPECIES):
            out_path = os.path.join(outdir, f"{key}_vector.png")
            render_species(key, out_path)
            print(f"  {key:42s} -> {out_path}")
        print(f"\nDone - {len(SPECIES)} icons written to {outdir}")


if __name__ == "__main__":
    main()
