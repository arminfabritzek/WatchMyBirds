"""
Ingest Core - Business Logic for Image Ingestion.

Provides a clean interface to image ingestion functionality,
abstracting away the infrastructure details.
"""

import logging
from typing import Any

from utils.ingest import ingest_inbox_folder as _ingest_inbox_folder

logger = logging.getLogger(__name__)


def process_inbox(pending_dir: str, file_snapshot: list[str]) -> dict[str, Any]:
    """
    Process inbox files from a pre-determined snapshot.

    This function processes images from inbox/pending and moves them to:
    - inbox/processed/YYYYMMDD/ for successfully ingested files
    - inbox/skipped/YYYYMMDD/ for duplicates
    - inbox/error/YYYYMMDD/ for files that failed processing

    Args:
        pending_dir: Path to inbox/pending directory
        file_snapshot: List of absolute file paths to process

    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Starting inbox ingest for {len(file_snapshot)} files")
        _ingest_inbox_folder(pending_dir, file_snapshot)
        logger.info("Inbox ingest completed")
        return {"status": "success", "processed": len(file_snapshot)}
    except Exception as e:
        logger.error(f"Inbox ingest error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
