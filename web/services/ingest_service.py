"""
Ingest Service - Web Layer Service for Image Ingestion.

Thin wrapper over core.ingest_core for web-specific concerns.
"""

from typing import Any

from core import ingest_core


def process_inbox(pending_dir: str, file_snapshot: list[str]) -> dict[str, Any]:
    """
    Process inbox files from a pre-determined snapshot.

    Delegates to core.ingest_core.
    """
    return ingest_core.process_inbox(pending_dir, file_snapshot)
