"""
User-Groundtruth Core — re-export layer for the export service.

Architecture rule (HARD invariant H-01): ``web/services/*`` may not
import directly from ``utils/*``. This module provides the bridge:
``utils.db.user_groundtruth`` is the implementation, this module
re-exports the public functions so the export service in
``web/services/`` can import from ``core.user_groundtruth_core``
instead of crossing the boundary.

Zero business logic lives here — pure re-export. Tests for the
underlying queries live in ``tests/test_user_groundtruth_queries.py``.
"""

from __future__ import annotations

from utils.db.user_groundtruth import (
    count_pending_by_bucket,
    fetch_confirmed_positives,
    fetch_favorites,
    fetch_hard_negatives,
    fetch_species_relabels,
)

__all__ = [
    "count_pending_by_bucket",
    "fetch_confirmed_positives",
    "fetch_favorites",
    "fetch_hard_negatives",
    "fetch_species_relabels",
]
