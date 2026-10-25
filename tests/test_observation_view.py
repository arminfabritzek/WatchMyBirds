from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.automatic_station_report import build_report
from utils.db.connection import _init_schema
from utils.db.detections import fetch_detections_for_gallery, fetch_species_page
from utils.db.observations import species_summaries


def test_migrated_fixture_preserves_every_visible_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import reset_config_in_place

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    reset_config_in_place()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        (Path(__file__).parent / "fixtures/human_answers_0_5_9.sql").read_text()
    )
    conn.executescript(
        (
            Path(__file__).parent / "fixtures/historical_automatic_decisions.sql"
        ).read_text()
    )
    _init_schema(conn, existing_installation=True)
    try:
        expected = {
            "detection:1": "Cyanistes_caeruleus",
            "detection:11": "Cyanistes_caeruleus",
            **{f"detection:{did}": "Parus_major" for did in (3, 4, 5, 14)},
            "detection:301": "Parus_major",
            "manual:1": "Parus_major",
            "manual:2": "Parus_major",
        }
        assert {
            r["object_key"]: r["species_key"]
            for r in fetch_detections_for_gallery(conn)
        } == expected
        assert {
            r["object_key"]: r["species_key"]
            for s in species_summaries(conn)
            for r in fetch_species_page(conn, s["species_key"])[0]
        } == expected
        assert [
            (r["species_key"], r["observation_count"])
            for r in build_report(conn, {})["species"]
        ] == [("Cyanistes_caeruleus", 2), ("Parus_major", 7)]
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok=0"
            ).fetchone()[0]
            == 0
        )
        conn.execute("UPDATE detections SET quality_gallery_ok=0 WHERE detection_id=14")
        assert {
            r["object_key"]: r["species_key"]
            for r in fetch_detections_for_gallery(conn)
        } == expected
    finally:
        conn.close()


def test_startup_recreates_view_without_reassessing_decisions() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    try:
        from tests.test_station_observations import seed

        seed(conn, identifier=1, species="Parus_major")
        conn.execute("DROP VIEW current_observations")
        conn.execute(
            "CREATE VIEW current_observations AS SELECT 'stale' AS species_key"
        )
        _init_schema(conn, existing_installation=True)
        assert (
            conn.execute("SELECT species_key FROM current_observations").fetchone()[0]
            == "Parus_major"
        )
        conn.execute("UPDATE detections SET decision_level=NULL")
        assert (
            conn.execute("SELECT admitted FROM current_observations").fetchone()[0] == 1
        )
    finally:
        conn.close()


@pytest.mark.parametrize("level", [None, ""])
@pytest.mark.parametrize("state", ["confirmed", None, "uncertain"])
def test_historical_empty_level_preserves_the_stored_decision(
    state: str | None,
    level: str | None,
) -> None:
    from unittest.mock import patch

    from tests.test_station_observations import seed
    from utils.daily_report import _fetch_species_best_photos
    from utils.db.detections import fetch_observation_page, fetch_species_covers

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    try:
        seed(conn, identifier=301, species="Parus_major", state=state, level=level)
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 0
        expected = ["Parus_major"] if state == "confirmed" else []
        for rows in (
            fetch_detections_for_gallery(conn),
            fetch_species_covers(conn),
            fetch_species_page(conn, "Parus_major")[0],
            build_report(conn, {})["species"],
        ):
            assert [row["species_key"] for row in rows] == expected
        with patch("utils.daily_report.os.path.isfile", return_value=True):
            assert [
                row["species"] for row in _fetch_species_best_photos(conn, "2026-10-13")
            ] == expected
        assert fetch_observation_page(conn, answered=False)[1] == int(
            state != "confirmed"
        )
        row = conn.execute(
            "SELECT decision_state, decision_level FROM detections"
        ).fetchone()
        assert tuple(row) == (state, level)
    finally:
        conn.close()
