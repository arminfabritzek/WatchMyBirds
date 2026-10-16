"""Derived everyday station history; diagnostic event contracts remain separate."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from utils.db.detections import fetch_species_covers
from utils.db.observations import observations_cte, selection_sql, species_summaries

logger = logging.getLogger(__name__)


def build_report(conn: sqlite3.Connection, selection: dict[str, str]) -> dict:
    where, params = selection_sql(selection)
    cte = observations_cte(conn)
    species = [
        row for row in species_summaries(conn, selection) if row["observation_count"]
    ]
    summary = dict(
        conn.execute(
            cte
            + f"""SELECT COUNT(*) AS objects,
        COUNT(CASE WHEN species_key != 'Unknown_species' THEN 1 END) AS identified_objects,
        COUNT(CASE WHEN unknown_bird THEN 1 END) AS unknown_birds,
        COUNT(CASE WHEN COALESCE(conflict, 0) THEN 1 END) AS conflicts,
        COUNT(CASE WHEN active AND NOT admitted AND NOT answered THEN 1 END) AS open_objects,
        COUNT(CASE WHEN active AND original_present = 0 THEN 1 END) AS missing_originals,
        MIN(timestamp) AS first_capture, MAX(timestamp) AS last_capture
        FROM observations WHERE {where}""",
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
            cte
            + f"""SELECT species_key,
        {day} AS day,
        MAX(origin = 'human') AS human, MAX(origin = 'automatic') AS automatic,
        MAX(origin = 'legacy') AS legacy,
        MAX(original_present = 0) AS missing_original
        FROM observations WHERE {where} AND species_key != 'Unknown_species'
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
    # Favor sparse human records, then representative photo quality; novelty
    # and geography never turn an unreviewed model name into a sensation.
    covers = [
        row
        for row in covers
        if row["species_key"] != "Unknown_species" and row.get("observation_count")
    ]
    covers.sort(
        key=lambda row: (
            0 if row.get("human_count") else 1,
            row.get("human_days") or 999,
            -int(row.get("quality_gallery_ok") != 0),
            -float(row.get("score") or 0),
            row["species_key"],
        )
    )
    highlights = covers[:6]
    models = [
        dict(row)
        for row in conn.execute(
            cte
            + f"""SELECT COALESCE(classifier_model_id, 'Unknown model') AS model,
        MIN(timestamp) AS first_capture, MAX(timestamp) AS last_capture, COUNT(*) AS objects
        FROM observations WHERE {where} GROUP BY classifier_model_id ORDER BY first_capture""",
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
    """Bounded per-application cache with one coalescing background worker.

    data_version is read from one long-lived observer connection. It sees
    commits by every writer, including another process and media-flag updates.
    Neither stale payloads nor late worker results are returned as current.
    """

    def __init__(
        self,
        connection_factory: Callable[[], sqlite3.Connection] | None = None,
        *,
        builder: Callable[[sqlite3.Connection, dict[str, str]], dict] = build_report,
        media_version: Callable[[], tuple] | None = None,
        max_entries: int = 8,
        debounce: float = 0.25,
        poll_interval: float = 2.0,
    ) -> None:
        if connection_factory is None:
            from config import get_config
            from core.station_observations import media_revision
            from utils.db import get_connection

            connection_factory = get_connection
            output_dir = str(get_config()["OUTPUT_DIR"])
            if media_version is None:
                media_version = lambda: media_revision(output_dir)
        self._media_version = media_version
        self._factory = connection_factory
        self._builder = builder
        self._observer = connection_factory()
        self._condition = threading.Condition(threading.RLock())
        self._entries: OrderedDict[tuple, dict] = OrderedDict()
        self._max_entries = max_entries
        self._debounce = debounce
        self._poll_interval = poll_interval
        self._source_version = self._version()
        self._revision = 0
        self._instance = uuid.uuid4().hex
        self._closed = False
        self._thread: threading.Thread | None = None

    def _version(self) -> tuple:
        return (
            int(self._observer.execute("PRAGMA data_version").fetchone()[0]),
            self._media_version() if self._media_version else (),
        )

    def _refresh(self) -> None:
        version = self._version()
        if version != self._source_version:
            self._source_version = version
            self._revision += 1
            for entry in self._entries.values():
                entry.update(state="updating", due=time.monotonic() + self._debounce)

    def status(self, selection: dict[str, str]) -> dict:
        key = tuple(sorted(selection.items()))
        with self._condition:
            try:
                self._refresh()
            except sqlite3.Error:
                self._condition.notify_all()
                return {
                    "state": "error",
                    "revision": f"{self._instance}:{self._revision}",
                    "report": None,
                }
            if key not in self._entries:
                self._entries[key] = {
                    "state": "building",
                    "due": time.monotonic() + self._debounce,
                    "payload": None,
                    "revision": None,
                }
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            self._entries.move_to_end(key)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="station-report", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
            entry = self._entries[key]
            return {
                "state": entry["state"],
                "revision": f"{self._instance}:{self._revision}",
                "report": entry["payload"]
                if entry["state"] == "ready" and entry["revision"] == self._revision
                else None,
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                try:
                    self._refresh()
                except sqlite3.Error:
                    for entry in self._entries.values():
                        entry["state"] = "error"
                    self._condition.wait(timeout=self._poll_interval)
                    continue
                due = next(
                    (
                        (key, entry)
                        for key, entry in self._entries.items()
                        if entry["state"] != "ready"
                        and entry["due"] <= time.monotonic()
                    ),
                    None,
                )
                if due is None:
                    self._condition.wait(
                        timeout=min(
                            self._poll_interval,
                            max(
                                0.01,
                                min(
                                    (
                                        e["due"] - time.monotonic()
                                        for e in self._entries.values()
                                        if e["state"] != "ready"
                                    ),
                                    default=self._poll_interval,
                                ),
                            ),
                        )
                    )
                    continue
                key, entry = due
                revision = self._revision
                entry["state"] = "updating" if entry["payload"] else "building"
            payload = None
            error = False
            try:
                conn = self._factory()
                try:
                    conn.execute("BEGIN")
                    payload = self._builder(conn, dict(key))
                finally:
                    conn.close()
            except Exception:
                logger.exception("Automatic station report update failed")
                error = True
            with self._condition:
                if self._closed:
                    return
                try:
                    self._refresh()
                except sqlite3.Error:
                    entry.update(state="error", due=time.monotonic() + 5)
                    continue
                if self._entries.get(key) is not entry:
                    continue
                if revision != self._revision:
                    entry.update(
                        state="updating", due=time.monotonic() + self._debounce
                    )
                elif error:
                    entry.update(state="error", due=time.monotonic() + 5)
                else:
                    entry.update(state="ready", payload=payload, revision=revision)
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=5)
        self._observer.close()
