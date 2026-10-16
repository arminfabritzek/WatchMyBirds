from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.automatic_station_report import StationReportCache, build_report
from core.station_observations import validate_selection
from tests.test_station_observations import answer, seed
from utils.db.connection import _init_schema
from utils.db.detections import fetch_species_covers, fetch_species_page


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Callable[[], sqlite3.Connection]]:
    path = tmp_path / "images.db"

    def factory() -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    conn = factory()
    _init_schema(conn)
    conn.execute("PRAGMA journal_mode=WAL")
    seed(conn, identifier=1, species="Parus_major", quality=1)
    seed(conn, identifier=2, species="Parus_major", quality=1)
    answer(conn, 1, "20261013_082211_1.jpg", species="Parus_major")
    conn.commit()
    yield factory
    conn.close()


def await_ready(
    cache: StationReportCache, selection: dict[str, str] | None = None
) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = cache.status(selection or validate_selection())
        if status["state"] == "ready":
            return status["report"]
        time.sleep(0.01)
    pytest.fail("Report did not become ready")


def test_origins_union_and_automatic_use_without_event_approval(
    database: Callable[[], sqlite3.Connection],
) -> None:
    conn = database()
    report = build_report(conn, validate_selection())
    row = report["species"][0]
    assert (row["human_days"], row["automatic_days"], row["days"]) == (1, 1, 1)
    assert report["summary"]["human_species"] == 1
    assert report["summary"]["automatic_species"] == 1
    assert report["summary"]["species_count"] == 1
    conn.execute("DELETE FROM human_label_facts")
    conn.execute(
        "UPDATE detections SET manual_species_override=NULL, species_source=NULL"
    )
    automatic = build_report(conn, validate_selection())
    assert automatic["summary"]["human_species"] == 0
    assert automatic["summary"]["automatic_species"] == 1
    conn.close()


def test_capture_commits_and_new_folders_leave_snapshot_until_refresh(
    database: Callable[[], sqlite3.Connection], tmp_path: Path
) -> None:
    elapsed = [0.0]
    cache = StationReportCache(database, clock=lambda: elapsed[0])
    try:
        before = await_ready(cache)
        revision = cache.status(validate_selection())["revision"]
        for minute in range(1, 11):
            conn = database()
            seed(conn, identifier=minute + 2, species="Picus_viridis", quality=1)
            conn.commit()
            conn.close()
            folder = tmp_path / "originals" / f"2026-10-{minute:02}"
            folder.mkdir(parents=True)
            (folder / "capture.jpg").write_bytes(b"synthetic capture")
            elapsed[0] = minute * 60
            status = cache.status(validate_selection())
            assert status["report"] == before
            assert status["revision"] == revision
        elapsed[0] = 601
        assert cache.status(validate_selection())["report"] == before
        current = cache.status(validate_selection(), refresh=True)
        assert current["report"] == before
        assert await_ready(cache)["summary"]["objects"] == 12
        assert cache.status(validate_selection())["revision"] != revision
    finally:
        cache.close()


def test_age_limit_is_checked_only_on_opening(
    database: Callable[[], sqlite3.Connection],
) -> None:
    elapsed = [0.0]
    cache = StationReportCache(database, clock=lambda: elapsed[0])
    try:
        before = await_ready(cache)
        elapsed[0] = 600
        assert cache.status(validate_selection(), opening=True)["state"] == "ready"
        elapsed[0] = 601
        assert cache.status(validate_selection())["state"] == "ready"
        status = cache.status(validate_selection(), opening=True)
        assert status["state"] == "updating" and status["report"] == before
        await_ready(cache)
    finally:
        cache.close()


@pytest.mark.parametrize("refresh_during_build", [False, True])
def test_capture_during_build_stays_snapshot_until_explicit_refresh(
    database: Callable[[], sqlite3.Connection],
    refresh_during_build: bool,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = []

    def builder(conn: sqlite3.Connection, selection: dict[str, str]) -> dict:
        result = build_report(conn, selection)
        calls.append(result)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return result

    cache = StationReportCache(database, builder=builder)
    try:
        assert cache.status(validate_selection())["report"] is None
        assert started.wait(5)
        conn = database()
        seed(conn, identifier=3, species="Picus_viridis", quality=1)
        conn.commit()
        conn.close()
        if refresh_during_build:
            cache.status(validate_selection(), refresh=True)
        release.set()
        assert await_ready(cache)["summary"]["objects"] == (
            3 if refresh_during_build else 2
        )
        assert len(calls) == (2 if refresh_during_build else 1)
        cache.status(validate_selection(), refresh=True)
        assert await_ready(cache)["summary"]["objects"] == 3
    finally:
        release.set()
        cache.close()


def test_failed_refresh_keeps_snapshot_and_requires_explicit_retry(
    database: Callable[[], sqlite3.Connection],
) -> None:
    attempts = []

    def builder(conn: sqlite3.Connection, selection: dict[str, str]) -> dict:
        attempts.append(1)
        if len(attempts) == 2:
            raise RuntimeError("controlled failure")
        return build_report(conn, selection)

    cache = StationReportCache(database, builder=builder)
    try:
        before = await_ready(cache)
        cache.status(validate_selection(), refresh=True)
        deadline = time.monotonic() + 2
        while (
            cache.status(validate_selection())["state"] != "error"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert cache.status(validate_selection())["state"] == "error"
        assert cache.status(validate_selection())["report"] == before
        assert len(attempts) == 2
        cache.status(validate_selection(), refresh=True)
        assert await_ready(cache)["summary"]["objects"] == 2
        assert len(attempts) == 3
    finally:
        cache.close()


@pytest.mark.parametrize(
    "selection",
    [
        dict(start="2026-02-30"),
        dict(start="2026-10-15", end="2026-10-14"),
    ],
)
def test_invalid_selection_is_rejected(selection: dict[str, str] | None) -> None:
    with pytest.raises(ValueError):
        validate_selection(**selection)


def test_period_keeps_imported_records_without_inventing_profile(
    database: Callable[[], sqlite3.Connection],
) -> None:
    conn = database()
    conn.execute(
        "INSERT INTO sources(source_id,name,type) VALUES (2, 'Imported', 'folder_upload')"
    )
    conn.execute("UPDATE images SET source_id=2")
    report = build_report(conn, validate_selection())
    assert report["summary"]["species_count"] == 1
    assert report["summary"]["objects"] == 2
    assert report["highlights"]
    selection = validate_selection(end="2026-10-13")
    assert fetch_species_covers(conn, selection=selection)[0]["observation_count"] == 2
    rows, total, _ = fetch_species_page(conn, "Parus_major", selection=selection)
    assert total == 2 and {r["detection_id"] for r in rows} == {1, 2}
    assert report["runtime"]["samples"] == 0 and report["gaps"] == []
    report = build_report(conn, validate_selection(start="2026-10-14"))
    assert not report["species"]
    conn.close()
