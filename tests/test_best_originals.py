"""Which originals every species keeps."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from core.best_originals import best_original_keys
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from tests.labeling_helpers import _reset_test_config
from tests.test_gallery_blueprint import _seed_detection
from utils.db import connection, insert_classification, insert_detection

PROVENANCE = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")


@pytest.fixture
def conn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[sqlite3.Connection]:
    _reset_test_config(monkeypatch, tmp_path)
    with connection.closing_connection() as db:
        yield db


def name(index: int) -> str:
    return f"20260101_1200{index:02d}_{index}.jpg"


def bird(
    conn: sqlite3.Connection,
    index: int,
    *,
    species: str = "Parus_major",
    score: float = 0.5,
    decision_state: str = "confirmed",
) -> str:
    identifier = _seed_detection(
        conn,
        filename=name(index),
        timestamp=name(index)[:15],
        species=species,
        decision_state=decision_state,
        score=score,
    )
    return f"detection:{identifier}"


def detection_id(key: str) -> int:
    return int(key.split(":")[1])


def test_ranks_per_species_with_the_cover_order(conn: sqlite3.Connection) -> None:
    best, second, _third = (
        bird(conn, index, score=score) for index, score in enumerate((0.9, 0.8, 0.7))
    )
    only = bird(conn, 9, species="Turdus_merula", score=0.1)
    assert best_original_keys(conn, 2) == {best, second, only}
    assert best_original_keys(conn, 1) == {best, only}


def test_favorite_ranks_first_and_counts(conn: sqlite3.Connection) -> None:
    favorite = bird(conn, 0, score=0.1)
    high = bird(conn, 1, score=0.9)
    bird(conn, 2, score=0.8)
    conn.execute(
        "UPDATE detections SET is_favorite=1 WHERE detection_id=?",
        (detection_id(favorite),),
    )
    conn.commit()
    assert best_original_keys(conn, 2) == {favorite, high}


def test_only_photos_with_an_original_compete(conn: sqlite3.Connection) -> None:
    bird(conn, 0, score=0.9)
    second = bird(conn, 1, score=0.8)
    third = bird(conn, 2, score=0.7)
    conn.execute("UPDATE images SET original_present=0 WHERE filename=?", (name(0),))
    conn.commit()
    assert best_original_keys(conn, 2) == {second, third}


def test_unknown_open_and_trashed_objects_never_compete(
    conn: sqlite3.Connection,
) -> None:
    kept = bird(conn, 0, score=0.2)
    bird(conn, 1, score=0.9, decision_state="uncertain")
    trashed = bird(conn, 2, score=0.95)
    conn.execute(
        "UPDATE detections SET status='rejected' WHERE detection_id=?",
        (detection_id(trashed),),
    )
    conn.commit()
    assert best_original_keys(conn, 5) == {kept}


def test_two_birds_on_one_photo_use_one_place(conn: sqlite3.Connection) -> None:
    first = bird(conn, 0, score=0.9)
    companion = insert_detection(
        conn,
        {
            "image_filename": name(0),
            "bbox_x": 0.5,
            "bbox_y": 0.5,
            "bbox_w": 0.2,
            "bbox_h": 0.2,
            "od_class_name": "bird",
            "od_confidence": 0.9,
            "created_at": name(0)[:15],
            "score": 0.85,
            "decision_state": "confirmed",
            "decision_level": "species",
        },
    )
    insert_classification(
        conn,
        {
            "detection_id": companion,
            "cls_class_name": "Parus_major",
            "cls_confidence": 0.9,
            "cls_model_id": "cls-test",
            "rank": 1,
            "created_at": name(0)[:15],
        },
    )
    other = bird(conn, 1, score=0.8)
    conn.commit()
    assert best_original_keys(conn, 2) == {first, other}


def test_zero_limit_returns_nothing_without_a_query(conn: sqlite3.Connection) -> None:
    bird(conn, 0)
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        assert best_original_keys(conn, 0) == frozenset()
    finally:
        conn.set_trace_callback(None)
    assert statements == []


def test_species_correction_moves_the_photo(conn: sqlite3.Connection) -> None:
    moved = bird(conn, 0, score=0.9)
    stays = bird(conn, 1, score=0.8)
    assert best_original_keys(conn, 1) == {moved}
    record_human_answer(
        conn,
        HumanAnswer(
            name(0),
            detection_id(moved),
            species_identity="corrected",
            species_key="Cyanistes_caeruleus",
        ),
        PROVENANCE,
    )
    conn.commit()
    assert best_original_keys(conn, 1) == {moved, stays}


def test_missing_file_gives_its_place_to_the_next_photo(
    conn: sqlite3.Connection,
) -> None:
    bird(conn, 0, score=0.9)
    second = bird(conn, 1, score=0.8)
    third = bird(conn, 2, score=0.7)
    keys = best_original_keys(conn, 2, media_exists=lambda photo: photo != name(0))
    assert keys == {second, third}
