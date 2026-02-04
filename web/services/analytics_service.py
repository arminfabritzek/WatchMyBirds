"""
Analytics Service - Web Layer Service for Analytics Operations.

Thin wrapper over core.analytics_core for web-specific concerns.
"""

from typing import Any

from core import analytics_core


def get_analytics_summary() -> dict[str, Any]:
    """
    Get analytics summary data.

    Delegates to core.analytics_core.
    """
    return analytics_core.get_analytics_summary()


def get_detection_times(date_iso: str | None = None) -> list[str]:
    """
    Get all detection timestamps.

    Delegates to core.analytics_core.
    """
    return analytics_core.get_detection_times(date_iso)


def get_species_timestamps(species: str) -> list[dict]:
    """
    Get timestamps for a specific species.

    Delegates to core.analytics_core.
    """
    return analytics_core.get_species_timestamps(species)


def get_day_count() -> int:
    """
    Get count of days with detections.

    Delegates to core.analytics_core.
    """
    return analytics_core.get_day_count()


def get_species_summary_cached(force_refresh: bool = False) -> dict[str, Any] | None:
    """
    Get cached species summary.

    Delegates to core.analytics_core.
    """
    return analytics_core.get_species_summary_cached(force_refresh)


def set_species_summary_cache(payload: dict[str, Any]) -> None:
    """
    Update species summary cache.

    Delegates to core.analytics_core.
    """
    analytics_core.set_species_summary_cache(payload)


def invalidate_cache() -> None:
    """Invalidate analytics cache."""
    analytics_core.invalidate_analytics_cache()
