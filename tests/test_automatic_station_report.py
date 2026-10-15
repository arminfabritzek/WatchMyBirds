from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.automatic_station_report import StationReportCache, build_report
from core.human_label_core import (
    HumanAnswer,
    record_human_answer,
    retract_species_identity,
)
from core.station_observations import validate_selection
from tests.test_station_observations import answer, provenance, seed
from utils.db.connection import _init_schema


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


def test_correction_retraction_and_new_capture_invalidate_cached_results(
    database: Callable[[], sqlite3.Connection],
) -> None:
    cache = StationReportCache(database, debounce=0, poll_interval=0.01)
    try:
        assert await_ready(cache)["summary"]["species_count"] == 1
        conn = database()
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename="20261013_082211_1.jpg",
                detection_id=1,
                species_identity="corrected",
                species_key="Pica_pica",
            ),
            provenance(),
        )
        conn.commit()
        assert cache.status(validate_selection())["report"] is None
        report = await_ready(cache)
        assert {r["species_key"] for r in report["species"]} == {
            "Parus_major",
            "Pica_pica",
        }
        retract_species_identity(
            conn,
            image_filename="20261013_082211_1.jpg",
            detection_id=1,
            provenance=provenance(),
        )
        conn.commit()
        assert cache.status(validate_selection())["report"] is None
        assert await_ready(cache)["summary"]["human_species"] == 0
        seed(conn, identifier=3, species="Picus_viridis", quality=1)
        conn.commit()
        assert cache.status(validate_selection())["report"] is None
        assert await_ready(cache)["summary"]["species_count"] == 2
        conn.execute("UPDATE images SET original_present = 0")
        conn.commit()
        assert cache.status(validate_selection())["report"] is None
        assert await_ready(cache)["summary"]["missing_originals"] == 3
        conn.close()
    finally:
        cache.close()


def test_late_worker_result_is_never_current(
    database: Callable[[], sqlite3.Connection],
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = []

    def builder(conn: sqlite3.Connection, selection: dict[str, str] | None) -> dict:
        result = build_report(conn, selection)
        calls.append(result)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return result

    cache = StationReportCache(
        database, builder=builder, debounce=0, poll_interval=0.01
    )
    try:
        assert cache.status(validate_selection())["report"] is None
        assert started.wait(5)
        conn = database()
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename="20261013_082211_1.jpg",
                detection_id=1,
                species_identity="corrected",
                species_key="Pica_pica",
            ),
            provenance(),
        )
        conn.commit()
        conn.close()
        assert cache.status(validate_selection())["report"] is None
        release.set()
        report = await_ready(cache)
        assert len(calls) >= 2
        assert any(r["species_key"] == "Pica_pica" for r in report["species"])
    finally:
        release.set()
        cache.close()


def test_failure_hides_old_payload_and_retries_automatically(
    database: Callable[[], sqlite3.Connection],
) -> None:
    attempts = []

    def builder(conn: sqlite3.Connection, selection: dict[str, str] | None) -> dict:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("controlled failure")
        return build_report(conn, selection)

    cache = StationReportCache(
        database, builder=builder, debounce=0, poll_interval=0.01
    )
    try:
        cache.status(validate_selection())
        deadline = time.monotonic() + 2
        while (
            cache.status(validate_selection())["state"] != "error"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert cache.status(validate_selection())["report"] is None
        # A new data revision triggers retry without a manual completion step.
        conn = database()
        conn.execute("UPDATE images SET original_present=0")
        conn.commit()
        conn.close()
        assert await_ready(cache)["summary"]["missing_originals"] == 2
    finally:
        cache.close()


@pytest.mark.parametrize(
    "selection",
    [
        dict(start="2026-02-30"),
        dict(start="2026-10-15", end="2026-10-14"),
        dict(source="camera' OR 1=1"),
    ],
)
def test_invalid_selection_is_rejected(selection: dict[str, str] | None) -> None:
    with pytest.raises(ValueError):
        validate_selection(**selection)


def test_period_source_and_missing_profile_do_not_invent_claims(
    database: Callable[[], sqlite3.Connection],
) -> None:
    conn = database()
    conn.execute(
        "INSERT INTO sources(source_id,name,type) VALUES (2, 'Imported', 'folder_upload')"
    )
    conn.execute("UPDATE images SET source_id=2")
    report = build_report(conn, validate_selection(source="camera"))
    assert report["summary"]["species_count"] == 0 and not report["highlights"]
    report = build_report(conn, validate_selection(source="import"))
    assert report["summary"]["species_count"] == 1
    assert report["summary"]["imported_objects"] == 2
    assert report["runtime"]["samples"] == 0 and report["gaps"] == []
    report = build_report(conn, validate_selection(start="2026-10-14"))
    assert not report["species"]
    conn.close()


def test_filesystem_media_loss_invalidates_without_database_write(
    database: Callable[[], sqlite3.Connection], tmp_path: Path
) -> None:
    from core.station_observations import media_revision

    folder = tmp_path / "originals" / "2026-10-13"
    folder.mkdir(parents=True)
    image = folder / "one.jpg"
    image.write_bytes(b"local fixture")
    cache = StationReportCache(
        database,
        media_version=lambda: media_revision(str(tmp_path)),
        debounce=0,
        poll_interval=0.01,
    )
    try:
        await_ready(cache)
        before = cache.status(validate_selection())["revision"]
        image.unlink()
        current = cache.status(validate_selection())
        assert current["report"] is None and current["revision"] != before
        assert await_ready(cache)["summary"]["species_count"] == 1
    finally:
        cache.close()
