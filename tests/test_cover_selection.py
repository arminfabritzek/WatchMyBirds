"""The same photo wins in SQL, Python, reports and event covers."""

import random
import sqlite3
from collections.abc import Iterator
from itertools import product
from pathlib import Path

import pytest

from core.automatic_station_report import build_report
from core.events import build_bird_events
from core.gallery_core import _choose_story_board_frames, pick_cover_for_group
from utils.cover_selection import cover_order_sql, cover_sort_key
from utils.db.connection import _init_schema
from utils.db.detections import (
    fetch_daily_covers,
    fetch_detections_for_gallery,
    fetch_random_favorites,
    fetch_species_covers,
    fetch_species_story_board_candidates,
)


@pytest.fixture
def conn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[sqlite3.Connection]:
    from tests.labeling_helpers import _reset_test_config

    _reset_test_config(monkeypatch, tmp_path)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    _init_schema(db)
    yield db
    db.close()


def seed(conn: sqlite3.Connection, index: int, **quality: object) -> int:
    stamp = f"20260101_1200{index:02d}"
    filename = stamp + ".jpg"
    conn.execute(
        "INSERT INTO images(filename, timestamp, ptz_origin) VALUES (?, ?, ?)",
        (filename, stamp, quality.pop("ptz_origin", None)),
    )
    values = {
        "image_filename": filename,
        "created_at": stamp,
        "od_class_name": "bird",
        "decision_state": "confirmed",
        "decision_level": "species",
        "status": "active",
        "bbox_x": 0.2,
        "bbox_y": 0.2,
        "bbox_w": 0.2,
        "bbox_h": 0.2,
        "score": 0.7,
        "bbox_quality": 0.5,
        **quality,
    }
    did = conn.execute(
        f"INSERT INTO detections({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
        list(values.values()),
    ).lastrowid
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank) "
        "VALUES (?, 'Parus_major', 0.95, 1)",
        (did,),
    )
    conn.commit()
    return int(did)


@pytest.mark.parametrize(
    "winner,loser",
    [
        ({"is_favorite": 1, "quality_gallery_ok": 0}, {"quality_gallery_ok": 1}),
        ({"quality_gallery_ok": 1}, {"quality_gallery_ok": 0, "ptz_origin": "preset"}),
        ({"quality_gallery_ok": None, "score": 0.9}, {"quality_gallery_ok": 1}),
        ({"quality_gallery_ok": -1}, {"quality_gallery_ok": 0}),
        ({"ptz_origin": "manual_drive"}, {"is_gallery_eligible": 1}),
        ({"is_gallery_eligible": 1}, {"aesthetic_score": 0.99}),
        ({"aesthetic_score": 0.8}, {"score": 0.99}),
        ({"score": 0.8}, {"bbox_quality": 0.99}),
        ({"bbox_quality": 0.8}, {}),
    ],
)
def test_all_selectors_choose_same_photo(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    winner: dict,
    loser: dict,
) -> None:
    from utils.daily_report import _fetch_species_best_photos

    expected = seed(conn, 1, **winner)
    seed(conn, 2, **loser)
    rows = fetch_detections_for_gallery(conn)
    assert max(rows, key=cover_sort_key)["detection_id"] == expected
    assert pick_cover_for_group(rows, rng=random.Random(1))["detection_id"] == expected
    assert _choose_story_board_frames(rows)[0]["detection_id"] == expected
    assert build_bird_events(rows)[0].cover_detection_id == expected
    assert fetch_daily_covers(conn)[0]["detection_id"] == expected
    assert fetch_species_covers(conn)[0]["detection_id"] == expected
    assert fetch_species_story_board_candidates(conn)[0]["detection_id"] == expected
    assert build_report(conn, {})["highlights"][0]["detection_id"] == expected
    monkeypatch.setattr("utils.daily_report.os.path.isfile", lambda _: True)
    best = _fetch_species_best_photos(conn, "2026-01-01")[0]
    assert (
        best["image_filename"]
        == rows[[r["detection_id"] for r in rows].index(expected)]["image_filename"]
    )


def test_sql_and_python_complete_order_match(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE candidates (
        id INTEGER, is_favorite INTEGER, quality_gallery_ok INTEGER, ptz_origin TEXT,
        is_gallery_eligible INTEGER, aesthetic_score REAL, score REAL,
        bbox_quality REAL, timestamp TEXT)""")
    rows = [
        (i, *values)
        for i, values in enumerate(
            product(
                (None, 0, 2),
                (None, 0, 1, -1),
                (None, "overview", "preset", "manual_drive"),
                (None, 0, 1),
                (None, 0.0, 0.8),
                (None, 0.0, 0.9),
                (None, 0.0, 0.7),
                (None, "20260101_120000", "20260101_120001"),
            )
        )
    ]
    conn.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    candidates = [dict(row) for row in conn.execute("SELECT * FROM candidates")]
    sql_ids = [
        row[0]
        for row in conn.execute(
            f"SELECT id FROM candidates ORDER BY {cover_order_sql()}, id DESC"
        )
    ]
    python_ids = [
        row["id"]
        for row in sorted(
            candidates, key=lambda row: (cover_sort_key(row), row["id"]), reverse=True
        )
    ]
    assert sql_ids == python_ids


def test_unrated_and_sharp_are_equal_and_timestamp_alias_matches() -> None:
    assert cover_sort_key(
        {"quality_gallery_ok": None, "image_timestamp": "20260101_120000"}
    ) == cover_sort_key({"quality_gallery_ok": 1, "timestamp": "20260101_120000"})


def test_random_favorites_rank_photos_before_sampling_species(
    conn: sqlite3.Connection,
) -> None:
    expected = seed(conn, 1, is_favorite=1, quality_gallery_ok=0)
    seed(conn, 2, is_gallery_eligible=1, quality_gallery_ok=1)
    assert fetch_random_favorites(conn)[0]["detection_id"] == expected


def test_recency_wins_without_using_detection_id(conn: sqlite3.Connection) -> None:
    expected = seed(conn, 2)
    seed(conn, 1)
    assert fetch_daily_covers(conn)[0]["detection_id"] == expected
    assert fetch_species_covers(conn)[0]["detection_id"] == expected
    assert (
        pick_cover_for_group(fetch_detections_for_gallery(conn))["detection_id"]
        == expected
    )


def test_event_context_cannot_replace_actionable_photo() -> None:
    rows = [
        {
            "detection_id": 1,
            "timestamp": "20260101_120000",
            "species_key": "Parus_major",
            "score": 0.5,
        },
        {
            "detection_id": 2,
            "timestamp": "20260101_120001",
            "species_key": "Parus_major",
            "is_favorite": 1,
            "context_only": True,
        },
    ]
    assert build_bird_events(rows)[0].cover_detection_id == 1


def test_live_alert_uses_saved_cover_metadata(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext

    from detectors.services.notification_service import NotificationService

    favorite = seed(conn, 1, is_favorite=1, quality_gallery_ok=0, score=0.4)
    sharp = seed(conn, 2, quality_gallery_ok=1, score=0.99)
    monkeypatch.setattr(
        "detectors.services.notification_service.get_config",
        lambda: {"TELEGRAM_MODE": "live"},
    )
    monkeypatch.setattr(
        "detectors.services.notification_service.closing_connection",
        lambda: nullcontext(conn),
    )
    service = NotificationService()
    for did, score in ((favorite, 0.4), (sharp, 0.99)):
        info = service.create_species_info(
            "Parus_major", score, f"/photo-{did}.jpg", detection_id=did
        )
        service.queue_detection(info)
    assert (
        service._pending_species["Parus_major"]["image_path"]
        == f"/photo-{favorite}.jpg"
    )
