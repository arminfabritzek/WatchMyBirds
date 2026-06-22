"""Retention Service — thin web-layer wrapper over core.retention_core.

Imports only core (H-01 enforcement test forbids utils.* here); the DB
connection is opened inside core.retention_core.
"""

from typing import Any

from core import retention_core


def preview() -> dict[str, Any]:
    """Dry-run preview: deletable counts, bytes, protection breakdown."""
    return retention_core.preview()


def run() -> dict[str, int]:
    """Execute retention (deletes deletable originals)."""
    return retention_core.run()


def is_original_retention_deleted(filename: str) -> bool:
    """True iff the original for `filename` was removed by retention."""
    return retention_core.is_original_retention_deleted(filename)
