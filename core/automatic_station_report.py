"""Station history snapshots, refreshed on opening or by explicit request."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from utils.cover_selection import cover_sort_key
from utils.db.detections import fetch_species_covers
from utils.db.observations import selection_sql, species_summaries

logger = logging.getLogger(__name__)


def build_report(conn: sqlite3.Connection, selection: dict[str, str]) -> dict:
    where, params = selection_sql(selection)
    species = [
        row for row in species_summaries(conn, selection) if row["observation_count"]
    ]
    summary = dict(
        conn.execute(
            f"""SELECT COUNT(*) AS objects,
        COUNT(CASE WHEN species_key != 'Unknown_species' THEN 1 END) AS identified_objects,
        COUNT(CASE WHEN unknown_bird THEN 1 END) AS unknown_birds,
        COUNT(CASE WHEN active AND NOT admitted AND NOT answered THEN 1 END) AS open_objects,
        COUNT(CASE WHEN active AND original_present = 0 THEN 1 END) AS missing_originals,
        MIN(timestamp) AS first_capture, MAX(timestamp) AS last_capture
        FROM current_observations WHERE {where}""",
            params,
        ).fetchone()
    )
    summary["human_species"] = sum(row["human_count"] > 0 for row in species)
    summary["automatic_species"] = sum(row["automatic_count"] > 0 for row in species)
    summary["species_count"] = len(species)
    day = "substr(REPLACE(timestamp, '-', ''), 1, 8)"
    daily = [
        dict(row)
        for row in conn.execute(
            f"""SELECT species_key,
        {day} AS day,
        MAX(origin = 'human') AS human, MAX(origin = 'automatic') AS automatic,
        MAX(origin = 'legacy') AS legacy,
        MAX(original_present = 0) AS missing_original
        FROM current_observations WHERE {where} AND species_key != 'Unknown_species'
        GROUP BY species_key, day ORDER BY day""",
            params,
        )
    ]
    last = daily[-1]["day"] if daily else ""
    calendar_days: list[str] = []
    if last:
        end = datetime.strptime(last, "%Y%m%d").date()
        first = max(
            datetime.strptime(daily[0]["day"], "%Y%m%d").date(),
            end - timedelta(days=365),
        )
        while first <= end:
            calendar_days.append(first.isoformat())
            first += timedelta(days=1)
    cells = {
        (
            row["species_key"],
            datetime.strptime(row["day"], "%Y%m%d").date().isoformat(),
        ): row
        for row in daily
    }
    for row in species:
        row["calendar"] = [
            cells.get((row["species_key"], day)) for day in calendar_days
        ]
    covers = fetch_species_covers(conn, selection=selection)
    covers = [
        row
        for row in covers
        if row["species_key"] != "Unknown_species" and row.get("observation_count")
    ]
    covers.sort(key=cover_sort_key, reverse=True)
    highlights = covers[:6]
    models = [
        dict(row)
        for row in conn.execute(
            f"""SELECT COALESCE(classifier_model_id, 'Unknown model') AS model,
        MIN(timestamp) AS first_capture, MAX(timestamp) AS last_capture, COUNT(*) AS objects
        FROM current_observations WHERE {where} GROUP BY classifier_model_id ORDER BY first_capture""",
            params,
        )
    ]
    runtime = dict(
        conn.execute("""SELECT COUNT(*) AS samples, MIN(sampled_at) AS first_sample,
        MAX(sampled_at) AS last_sample FROM station_runtime_samples""").fetchone()
    )
    gaps = [
        dict(row)
        for row in conn.execute("""WITH samples AS (
        SELECT sampled_at, LAG(sampled_at) OVER (ORDER BY sampled_at) AS previous,
            LAG(sample_interval_seconds) OVER (ORDER BY sampled_at) AS expected
        FROM station_runtime_samples
        ), gaps AS (SELECT previous, sampled_at,
            (julianday(sampled_at)-julianday(previous))*24 AS hours
            FROM samples WHERE (julianday(sampled_at)-julianday(previous))*86400 > expected * 2)
        SELECT *, COUNT(*) OVER () AS total_gaps FROM gaps ORDER BY hours DESC LIMIT 8""")
    ]
    return {
        "selection": selection,
        "summary": summary,
        "species": species,
        "highlights": highlights,
        "calendar_days": calendar_days,
        "runtime": runtime,
        "gaps": gaps,
        "models": models,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


class StationReportCache:
    """Bounded snapshots with one background worker and no capture watchers."""

    MAX_AGE_SECONDS = 600

    def __init__(
        self,
        connection_factory: Callable[[], sqlite3.Connection] | None = None,
        *,
        builder: Callable[[sqlite3.Connection, dict[str, str]], dict] = build_report,
        max_entries: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if connection_factory is None:
            from utils.db import get_connection

            connection_factory = get_connection
        self._factory = connection_factory
        self._builder = builder
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._entries: OrderedDict[tuple, dict] = OrderedDict()
        self._max_entries = max_entries
        self._revision = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    def status(
        self, selection: dict[str, str], *, opening: bool = False, refresh: bool = False
    ) -> dict:
        key = tuple(sorted(selection.items()))
        with self._condition:
            if key not in self._entries:
                self._entries[key] = {
                    "state": "building",
                    "payload": None,
                    "revision": None,
                    "completed_at": None,
                    "requested": 0,
                }
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            entry = self._entries[key]
            self._entries.move_to_end(key)
            expired = entry["completed_at"] is None or (
                self._clock() - entry["completed_at"] > self.MAX_AGE_SECONDS
            )
            if refresh:
                entry["requested"] += 1
                entry["state"] = "updating" if entry["payload"] else "building"
            elif entry["state"] not in {"building", "updating"} and opening and expired:
                entry["state"] = "updating" if entry["payload"] else "building"
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="station-report", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
            return {
                "state": entry["state"],
                "revision": entry["revision"],
                "report": entry["payload"],
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                pending = next(
                    (
                        (key, entry)
                        for key, entry in self._entries.items()
                        if entry["state"] in {"building", "updating"}
                    ),
                    None,
                )
                if pending is None:
                    self._condition.wait()
                    continue
                key, entry = pending
                requested = entry["requested"]
            try:
                conn = self._factory()
                try:
                    conn.execute("BEGIN")
                    payload = self._builder(conn, dict(key))
                finally:
                    conn.close()
            except Exception:
                logger.exception("Station report calculation failed")
                with self._condition:
                    if entry["requested"] == requested:
                        entry["state"] = "error"
                continue
            with self._condition:
                if self._closed:
                    return
                if self._entries.get(key) is entry:
                    if entry["requested"] != requested:
                        continue
                    self._revision += 1
                    entry.update(
                        state="ready",
                        payload=payload,
                        revision=self._revision,
                        completed_at=self._clock(),
                    )
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=5)
