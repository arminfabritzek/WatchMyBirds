"""Shared event aggregation + effort calculation for Analytics and Insights.

Both `/analytics` and `/insights` consume events through this module so the
two dashboards never drift. Event grouping itself stays in `core.events` —
this layer is the single DB-bound entry point that calls it.

Public surface:
    fetch_events_for_analysis(conn, *, min_score, profile_overrides) -> list[BirdEvent]
    calculate_effort(conn) -> EffortStats
    get_events_cached(conn, *, min_score) -> list[BirdEvent]

The cache is keyed on `(min_score, max(images.timestamp))` so newly arrived
images automatically invalidate the cache. A short TTL fallback handles the
edge case where the timestamp does not change (e.g. a re-run on the same
snapshot during testing).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from core.event_intelligence import EventGroupingProfile
from core.events import BirdEvent, build_bird_events
from utils.db.analytics import _fetch_event_intelligence_rows


@dataclass(frozen=True)
class EffortStats:
    """Standardised observation effort over the visible detection window.

    Used as the denominator for activity rates (RAI) and as context on the
    Insights station-report header.
    """

    first_ts: str | None
    last_ts: str | None
    total_days: int
    active_days: int


_CACHE_TTL_SEC = 60.0
_cache_state: dict[tuple, tuple[float, list[BirdEvent]]] = {}


def fetch_events_for_analysis(
    conn: sqlite3.Connection,
    *,
    min_score: float = 0.0,
    profile_overrides: Mapping[str, EventGroupingProfile] | None = None,
) -> list[BirdEvent]:
    """Fetch detections and group them into BirdEvents in one call.

    This is the sole entry point both dashboards should use for event-based
    metrics. It reuses the visibility predicate and species resolution logic
    that the rest of the app depends on, so the same detection that is
    hidden in the Gallery is also absent from Analytics and Insights.
    """
    rows = _fetch_event_intelligence_rows(conn, min_score=min_score)
    if not rows:
        return []
    return build_bird_events(rows, profile_overrides=profile_overrides)


def calculate_effort(conn: sqlite3.Connection) -> EffortStats:
    """Compute observation effort from the images table.

    `total_days` is the calendar-day span between first and last image
    (inclusive). `active_days` is the count of distinct calendar days that
    contain at least one image — a proxy for "camera-on" days until real
    health logs land.
    """
    row = conn.execute(
        """
        SELECT
            MIN(timestamp) AS first_ts,
            MAX(timestamp) AS last_ts,
            COUNT(DISTINCT substr(timestamp, 1, 8)) AS active_days
        FROM images
        WHERE timestamp IS NOT NULL AND TRIM(timestamp) != ''
        """
    ).fetchone()

    if row is None or row["first_ts"] is None:
        return EffortStats(first_ts=None, last_ts=None, total_days=0, active_days=0)

    first_ts = row["first_ts"]
    last_ts = row["last_ts"]
    active_days = int(row["active_days"] or 0)
    total_days = _calendar_day_span(first_ts, last_ts)

    return EffortStats(
        first_ts=first_ts,
        last_ts=last_ts,
        total_days=total_days,
        active_days=active_days,
    )


def get_events_cached(
    conn: sqlite3.Connection,
    *,
    min_score: float = 0.0,
) -> list[BirdEvent]:
    """Return events with a TTL+timestamp cache.

    Cache key includes `max(images.timestamp)` so the result invalidates
    automatically when a new image arrives. A 60-second TTL covers the
    same-timestamp re-run case (tests, repeated dashboard refreshes).
    """
    max_ts_row = conn.execute(
        "SELECT MAX(timestamp) FROM images WHERE timestamp IS NOT NULL"
    ).fetchone()
    max_ts = max_ts_row[0] if max_ts_row else None
    cache_key = (round(min_score, 4), max_ts)

    now = time.monotonic()
    hit = _cache_state.get(cache_key)
    if hit is not None and (now - hit[0]) < _CACHE_TTL_SEC:
        return hit[1]

    events = fetch_events_for_analysis(conn, min_score=min_score)
    _cache_state[cache_key] = (now, events)
    return events


def clear_events_cache() -> None:
    """Drop all cached event lists. Test utility; rarely needed in prod."""
    _cache_state.clear()


def _calendar_day_span(first_ts: str | None, last_ts: str | None) -> int:
    if not first_ts or not last_ts:
        return 0
    try:
        first_day = datetime.strptime(first_ts[:8], "%Y%m%d").date()
        last_day = datetime.strptime(last_ts[:8], "%Y%m%d").date()
    except ValueError:
        return 0
    return (last_day - first_day).days + 1


__all__ = [
    "EffortStats",
    "calculate_effort",
    "clear_events_cache",
    "fetch_events_for_analysis",
    "get_events_cached",
]
