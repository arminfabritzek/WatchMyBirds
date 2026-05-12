"""Tiny shared helpers for timestamp + bbox geometry.

Lives in its own module to break the ``core.gallery_core`` <->
``core.events`` import cycle: both consumers import from here instead
of from each other.
"""

from __future__ import annotations

import math
from datetime import datetime


def ts_to_epoch(ts: str) -> float:
    """Convert ``YYYYMMDD_HHMMSS`` (or with ``_ffffff`` suffix) to epoch seconds.

    Returns 0.0 on failure. Truncates to second-level precision; the
    hand-rolled slice parser is ~5x faster than ``datetime.strptime``
    and this function is hot in the ``/`` render path (~4 k calls).
    """
    if not ts or len(ts) < 15:
        return 0.0
    try:
        return datetime(
            int(ts[0:4]),
            int(ts[4:6]),
            int(ts[6:8]),
            int(ts[9:11]),
            int(ts[11:13]),
            int(ts[13:15]),
        ).timestamp()
    except (ValueError, TypeError):
        return 0.0


def bbox_dist(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Euclidean distance between bbox centres (normalised coords)."""
    cx_a = (ax or 0) + (aw or 0) / 2.0
    cy_a = (ay or 0) + (ah or 0) / 2.0
    cx_b = (bx or 0) + (bw or 0) / 2.0
    cy_b = (by or 0) + (bh or 0) / 2.0
    return math.hypot(cx_a - cx_b, cy_a - cy_b)
