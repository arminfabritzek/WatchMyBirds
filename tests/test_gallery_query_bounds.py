"""SQL selection preserves gallery semantics with bounded detail hydration."""

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.human_label_core import (
    HumanAnswer,
    LabelProvenance,
    record_human_answer,
    retract_bbox_quality,
)
from tests.labeling_helpers import _reset_test_config, _seed
from tests.test_gallery_blueprint import _seed_detection
from utils.db import connection
from utils.db.detections import (
    fetch_detections_for_gallery,
    fetch_gallery_days,
    fetch_sibling_detections_batch,
    fetch_species_covers,
    fetch_species_page,
)
from web.services.human_label_service import (
    fetch_detection_review_states,
    summarize_detection_review_progress,
)


@pytest.fixture
def conn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[sqlite3.Connection]:
    _reset_test_config(monkeypatch, tmp_path)
    with connection.closing_connection() as db:
        yield db


def seed(
    conn: sqlite3.Connection,
    index: int,
    *,
    day: int = 0,
    species: str = "Parus_major",
    **kwargs,
) -> int:
    date = (datetime(2026, 1, 1) + timedelta(days=day)).strftime("%Y%m%d")
    return _seed_detection(
        conn,
        filename=f"{date}_120000_{index}.jpg",
        timestamp=f"{date}_120000",
        species=species,
        decision_state="confirmed",
        **kwargs,
    )


def test_species_pages_ties_filters_and_empty_selection(
    conn: sqlite3.Connection,
) -> None:
    ids = [seed(conn, i) for i in range(105)]
    other = seed(conn, 106, species="Turdus_merula")
    conn.execute(
        "UPDATE detections SET is_favorite = 1 WHERE detection_id IN (?, ?)",
        (ids[0], ids[-1]),
    )
    conn.execute("UPDATE detections SET score = 0.1 WHERE detection_id = ?", (ids[-1],))
    conn.commit()
    with patch(
        "utils.db.detections.fetch_detections_for_gallery",
        wraps=fetch_detections_for_gallery,
    ) as hydrate:
        pages = [fetch_species_page(conn, "Parus_major", page=p) for p in (1, 2, 3)]
    assert [p[1] for p in pages] == [105] * 3
    assert [len(p[0]) for p in pages] == [50, 50, 5]
    assert [row["detection_id"] for p in pages for row in p[0]] == ids[::-1]
    assert all(
        len(call.kwargs["detection_ids"]) <= 50 for call in hydrate.call_args_list
    )
    filtered, count, page = fetch_species_page(
        conn, "Parus_major", min_score=0.5, favorites_only=True, page=999
    )
    assert count == 1 and page == 1 and filtered[0]["detection_id"] == ids[0]
    assert fetch_species_page(conn, "missing", page=-10) == ([], 0, 1)
    assert fetch_detections_for_gallery(conn, detection_ids=[]) == []
    assert fetch_species_page(conn, "Turdus_merula")[0][0]["detection_id"] == other


def test_species_correction_unknown_visibility_and_companions(
    conn: sqlite3.Connection,
) -> None:
    selected = seed(conn, 1)
    other = seed(conn, 2, species="Turdus_merula")
    filename = conn.execute(
        "SELECT image_filename FROM detections WHERE detection_id = ?", (selected,)
    ).fetchone()[0]
    companion = _seed(
        conn,
        filename=filename,
        timestamp="20260101_120000",
        species="Cyanistes_caeruleus",
    )
    conn.execute(
        "UPDATE detections SET decision_state = 'unknown' WHERE detection_id = ?",
        (companion,),
    )
    conn.execute(
        "UPDATE detections SET manual_species_override = 'Cyanistes_caeruleus', species_source = 'manual' WHERE detection_id = ?",
        (selected,),
    )
    conn.execute(
        "UPDATE detections SET species_source = 'manual_unknown' WHERE detection_id = ?",
        (other,),
    )
    conn.commit()
    assert fetch_species_page(conn, "Parus_major")[1] == 0
    rows, total, _ = fetch_species_page(conn, "Cyanistes_caeruleus")
    assert total == 1 and rows[0]["detection_id"] == selected
    assert fetch_species_page(conn, "Unknown_species")[1] == 1
    siblings = fetch_sibling_detections_batch(conn, [filename])
    assert {r["detection_id"] for r in siblings[filename]} == {selected, companion}
    conn.execute(
        "UPDATE images SET review_status = 'no_bird' WHERE filename = ?", (filename,)
    )
    conn.commit()
    assert fetch_species_page(conn, "Cyanistes_caeruleus")[1] == 0


def test_species_covers_preserve_priority_and_hydrate_only_covers(
    conn: sqlite3.Connection,
) -> None:
    ids = [seed(conn, i, score=0.5 + i / 100) for i in range(10)]
    conn.execute(
        "UPDATE detections SET is_favorite = 1 WHERE detection_id = ?", (ids[0],)
    )
    conn.execute(
        "UPDATE detections SET is_gallery_eligible = 1 WHERE detection_id = ?",
        (ids[1],),
    )
    conn.commit()
    with patch(
        "utils.db.detections.fetch_detections_for_gallery",
        wraps=fetch_detections_for_gallery,
    ) as hydrate:
        assert fetch_species_covers(conn)[0]["detection_id"] == ids[0]
        assert hydrate.call_args.kwargs["detection_ids"] == [ids[0]]
    assert fetch_species_covers(conn, min_score=0.51)[0]["detection_id"] == ids[1]
    assert fetch_species_covers(conn, min_score=0.51, favorites_only=True) == []
    conn.execute(
        "UPDATE images SET ptz_origin = 'preset' WHERE filename = (SELECT image_filename FROM detections WHERE detection_id = ?)",
        (ids[2],),
    )
    assert fetch_species_covers(conn, min_score=0.51)[0]["detection_id"] == ids[2]
    conn.execute(
        "UPDATE detections SET is_gallery_eligible = 1 WHERE detection_id = ?",
        (ids[2],),
    )
    assert fetch_species_covers(conn, min_score=0.51)[0]["detection_id"] == ids[2]


def test_days_are_paged_with_current_review_aggregates(
    conn: sqlite3.Connection,
) -> None:
    ids = [seed(conn, i, day=i) for i in range(35)]
    provenance = LabelProvenance(
        "test-installation", "test", "normal_correction", "watchmybirds_ui"
    )
    filename = conn.execute(
        "SELECT image_filename FROM detections WHERE detection_id = ?", (ids[-1],)
    ).fetchone()[0]
    record_human_answer(
        conn,
        HumanAnswer(
            filename,
            ids[-1],
            species_identity="confirmed",
            species_key="Parus_major",
            bbox_quality="suitable",
        ),
        provenance,
    )
    first, total, page = fetch_gallery_days(conn)
    second, _, _ = fetch_gallery_days(conn, page=2)
    assert total == 35 and page == 1 and len(first) == 30 and len(second) == 5
    assert not {row["date_key"] for row in first} & {row["date_key"] for row in second}
    states = fetch_detection_review_states(conn, [ids[-1]])
    assert first[0]["review_progress"] == summarize_detection_review_progress(
        states, [ids[-1]]
    )
    retract_bbox_quality(
        conn, image_filename=filename, detection_id=ids[-1], provenance=provenance
    )
    after = fetch_gallery_days(conn)[0][0]["review_progress"]
    assert after["reviewed"] == 1 and after["boxes_reviewed"] == 0 and after["complete"]
    record_human_answer(
        conn, HumanAnswer(filename, ids[-1], species_identity="wrong"), provenance
    )
    assert fetch_gallery_days(conn)[1] == 34
    assert fetch_gallery_days(conn, page=999)[2] == 2
    _seed_detection(
        conn,
        filename="import.jpg",
        timestamp="20251231_120000",
        species="Parus_major",
        decision_state="confirmed",
    )
    assert fetch_gallery_days(conn, page=2)[0][-1]["date_key"] == "2025-12-31"
