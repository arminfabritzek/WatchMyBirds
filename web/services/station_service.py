"""Ordinary station browsing, separate from diagnostic event assessment."""

from core.automatic_station_report import StationReportCache
from core.station_observations import (
    grouped_review,
    media_paths,
    review_page,
    species_selection_ids,
    validate_selection,
)

__all__ = [
    "grouped_review",
    "media_paths",
    "review_page",
    "species_selection_ids",
    "validate_selection",
    "create_report_cache",
]


def create_report_cache() -> StationReportCache:
    return StationReportCache()
