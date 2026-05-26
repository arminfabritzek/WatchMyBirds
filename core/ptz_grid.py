"""
PTZ grid-mode helpers — cell routing, hysteresis, preset naming.

Pure functions over normalized [0, 1] frame coordinates. No camera I/O,
no controller state — the AutoPtzController is the only caller that
should hold state. Keeping these isolated lets the grid-mode logic be
unit-tested without an ONVIF stack or threading setup.

Forward-compatible: grid mode is one of three Auto-Cam modes (preset,
hybrid, grid). The controller branches on `ptz.mode == "grid"`; this
module owns the math for that branch.


"""

from __future__ import annotations


# Default grid shape choices the setup wizard exposes. Operator picks
# one once per camera; stored as ``ptz.grid_shape: [rows, cols]``.
ALLOWED_GRID_SHAPES: tuple[tuple[int, int], ...] = (
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 4),
)


def normalize_grid_shape(raw: object) -> tuple[int, int]:
    """Validate a grid-shape value and fall back to 3×3 on garbage.

    Accepts a 2-tuple/list of ints. Anything outside ALLOWED_GRID_SHAPES
    falls back to (3, 3) — the default chosen in the Meta plan, balanced
    between coverage and per-cell setup effort.
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            shape = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError):
            shape = (3, 3)
    else:
        shape = (3, 3)
    if shape not in ALLOWED_GRID_SHAPES:
        shape = (3, 3)
    return shape


def cell_for_center(
    center_x: float,
    center_y: float,
    rows: int,
    cols: int,
    current_cell: tuple[int, int] | None = None,
    hysteresis: float = 0.05,
) -> tuple[int, int]:
    """Map a bbox center in [0, 1]² to a (row, col) grid cell.

    Hysteresis: if `current_cell` is set and the bbox center is within
    `hysteresis` (in normalized frame units) of the current cell's
    nearest boundary while still inside that cell's expanded box, the
    current cell wins. This prevents flap when a bird sits right on the
    line between two cells.

    Clamps out-of-range inputs to the nearest valid cell — the grid
    covers the full frame by construction, so a detection at (1.0, 1.0)
    must map to the bottom-right cell.

    Returns (row, col), zero-indexed. row grows downward.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cx = max(0.0, min(1.0, float(center_x)))
    cy = max(0.0, min(1.0, float(center_y)))

    # Bare cell lookup: which (row, col) box contains the center.
    col = min(cols - 1, int(cx * cols))
    row = min(rows - 1, int(cy * rows))

    if current_cell is None:
        return (row, col)

    cur_row, cur_col = current_cell
    if not (0 <= cur_row < rows and 0 <= cur_col < cols):
        # Stale current_cell from a different grid shape — discard it.
        return (row, col)

    if (row, col) == (cur_row, cur_col):
        return (row, col)

    # Different cell — check if the move is "decisive" or within
    # hysteresis margin of the current cell's expanded box.
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    cur_x_min = cur_col * cell_w - hysteresis
    cur_x_max = (cur_col + 1) * cell_w + hysteresis
    cur_y_min = cur_row * cell_h - hysteresis
    cur_y_max = (cur_row + 1) * cell_h + hysteresis

    if cur_x_min <= cx < cur_x_max and cur_y_min <= cy < cur_y_max:
        # Still within the hysteresis-expanded current cell — stay put.
        return (cur_row, cur_col)

    return (row, col)


def cell_preset_name(row: int, col: int) -> str:
    """Canonical name for a grid-cell preset: ``grid_r{row}_c{col}``.

    Zero-indexed. The ONVIF preset token stays whatever the camera
    assigns; this name carries the cell identity in the
    ``preset.name`` field and in ``ptz.grid_cells`` map keys.
    """
    return f"grid_r{int(row)}_c{int(col)}"


def parse_cell_preset_name(name: str) -> tuple[int, int] | None:
    """Inverse of `cell_preset_name`. Returns None on non-grid names.

    Used by the setup wizard to repopulate the "already-set cells"
    state when the operator returns to redo individual cells.
    """
    if not isinstance(name, str) or not name.startswith("grid_r"):
        return None
    try:
        rest = name[len("grid_r"):]
        row_str, _, col_str = rest.partition("_c")
        if not row_str or not col_str:
            return None
        return (int(row_str), int(col_str))
    except (TypeError, ValueError):
        return None


def required_cell_count(shape: tuple[int, int]) -> int:
    """Number of presets the operator must place for this grid shape."""
    rows, cols = shape
    return int(rows) * int(cols)
