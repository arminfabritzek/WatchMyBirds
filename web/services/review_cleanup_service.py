"""Review-cleanup Service — thin web-layer wrapper over core.review_cleanup_core.

Imports only core (H-01 enforcement test forbids utils.* here); the DB
connection is opened inside core.review_cleanup_core.
"""

from typing import Any

from core import review_cleanup_core


def preview() -> dict[str, Any]:
    """Dry-run preview: queue counts + favorite/export-relevant disclosure."""
    return review_cleanup_core.preview()


def run() -> dict[str, int]:
    """Move the review queue to Trash (reversible, no file deletion)."""
    return review_cleanup_core.run()
