"""Shared event aggregation + effort calculation for Analytics and Insights.

Both `/analytics` and `/insights` consume events through this module so the
two dashboards never drift. Event grouping itself stays in `core.events` —
this layer is the single DB-bound entry point that calls it.

Public surface:
    fetch_events_for_analysis(conn, *, min_score, profile_overrides) -> list[BirdEvent]
    calculate_effort(conn) -> EffortStats
    get_events_cached(conn, *, min_score) -> list[BirdEvent]

The cache retains only the latest version per database and exact threshold,
with a global entry bound and TTL. Read-only SQLite revision monitors invalidate
file-backed snapshots, including corrections that leave timestamps unchanged.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.event_intelligence import EventGroupingProfile
from core.events import BirdEvent, build_bird_events
from core.read_cache import ReadCache
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


class _EventCache(ReadCache):
    def __init__(self) -> None:
        super().__init__(max_entries=8)
        self._monitors: OrderedDict[
            str, tuple[tuple[int, int], sqlite3.Connection, object]
        ] = OrderedDict()

    def database_version(self, path: Path) -> tuple[object, int]:
        # data_version is connection-local; a stable read-only connection lets
        # short-lived request connections share revisions across WAL checkpoints.
        with self._condition:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            key = str(path)
            monitor = self._monitors.get(key)
            if monitor is not None and monitor[0] != identity:
                self._monitors.pop(key)[1].close()
                monitor = None
            if monitor is None:
                reader = sqlite3.connect(
                    path.as_uri() + "?mode=ro", uri=True, check_same_thread=False
                )
                monitor = (identity, reader, object())
                self._monitors[key] = monitor
            self._monitors.move_to_end(key)
            while len(self._monitors) > self.max_entries:
                self._monitors.popitem(last=False)[1][1].close()
            return monitor[2], int(
                monitor[1].execute("PRAGMA data_version").fetchone()[0]
            )

    def clear(self) -> None:
        with self._condition:
            super().clear()
            for _, reader, _ in self._monitors.values():
                reader.close()
            self._monitors.clear()


_cache_state = _EventCache()


def fetch_events_for_analysis(
    conn: sqlite3.Connection,
    *,
    min_score: float = 0.0,
    profile_overrides: Mapping[str, EventGroupingProfile] | None = None,
) -> list[BirdEvent]:
    """Fetch detections and group them into BirdEvents in one call.

    This is the sole entry point both dashboards should use for event-based
    metrics. It reuses the visibility predicate and species resolution logic
    that the rest of the app depends on. Moderation exclusions apply here,
    while aesthetic thumbnail quality deliberately does not erase a valid
    confirmed event from Analytics or Insights.
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
    """Cache the latest snapshot per database and exact score threshold."""
    database = next(
        row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"
    )
    if database and conn.in_transaction:
        # Uncommitted reads must never escape their connection through shared state.
        return fetch_events_for_analysis(conn, min_score=min_score)
    max_ts = conn.execute("SELECT MAX(timestamp) FROM images").fetchone()[0]
    if database:
        path = Path(database).resolve()
        identity = str(path)
        try:
            version = (max_ts, *_cache_state.database_version(path))
        except (OSError, sqlite3.Error):
            return fetch_events_for_analysis(conn, min_score=min_score)
    else:
        identity = conn
        version = (
            max_ts,
            conn.total_changes,
            conn.execute("PRAGMA data_version").fetchone()[0],
        )
    return _cache_state.get(
        (identity, min_score),
        _CACHE_TTL_SEC,
        lambda: fetch_events_for_analysis(conn, min_score=min_score),
        version=version,
    )


def clear_events_cache() -> None:
    """Drop retained snapshots and cancel publication of in-flight results."""
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
