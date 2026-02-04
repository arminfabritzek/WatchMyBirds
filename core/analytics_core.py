"""
Analytics Core - Analytics Aggregation Logic.

Provides all analytics-related operations and data aggregation.
"""

import logging
from typing import Any

from config import get_config
from utils.db import (
    fetch_all_detection_times,
    fetch_analytics_summary,
    fetch_day_count,
    fetch_species_timestamps,
    get_connection,
)

logger = logging.getLogger(__name__)
config = get_config()

# In-Memory Cache for species summary
_species_summary_cache: dict[str, Any] = {"timestamp": 0, "payload": None}
_CACHE_TIMEOUT = 60  # seconds


def get_analytics_summary() -> dict[str, Any]:
    """
    Fetches the analytics summary from the database.

    Returns:
        Dictionary with analytics summary data
    """
    with get_connection() as conn:
        return fetch_analytics_summary(conn)


def get_detection_times(date_iso: str | None = None) -> list[str]:
    """
    Fetches all detection timestamps, optionally filtered by date.

    Args:
        date_iso: Optional date filter in YYYY-MM-DD format

    Returns:
        List of timestamp strings
    """
    with get_connection() as conn:
        return fetch_all_detection_times(conn, date_iso)


def get_species_timestamps(species: str) -> list[dict]:
    """
    Fetches timestamps for a specific species.

    Args:
        species: Species name to filter by

    Returns:
        List of timestamp dictionaries
    """
    with get_connection() as conn:
        rows = fetch_species_timestamps(conn, species)
        return [dict(row) for row in rows]


def get_day_count() -> int:
    """
    Returns the number of days with detections.

    Returns:
        Count of unique detection days
    """
    with get_connection() as conn:
        return fetch_day_count(conn)


def get_species_summary_cached(force_refresh: bool = False) -> dict[str, Any] | None:
    """
    Returns cached species summary or None if cache expired.

    Args:
        force_refresh: If True, returns None to force cache refresh

    Returns:
        Cached payload or None
    """
    import time

    if force_refresh:
        _species_summary_cache["timestamp"] = 0
        _species_summary_cache["payload"] = None
        return None

    now = time.time()
    if now - _species_summary_cache["timestamp"] < _CACHE_TIMEOUT:
        return _species_summary_cache["payload"]
    return None


def set_species_summary_cache(payload: dict[str, Any]) -> None:
    """
    Updates the species summary cache.

    Args:
        payload: The data to cache
    """
    import time

    _species_summary_cache["timestamp"] = time.time()
    _species_summary_cache["payload"] = payload


def invalidate_analytics_cache() -> None:
    """Invalidates all analytics caches."""
    global _species_summary_cache
    _species_summary_cache = {"timestamp": 0, "payload": None}
