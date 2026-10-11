from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from scripts.aesthetic_tag_nightly import (
    apply_auto_favorites,
    fetch_unscored_detections,
)
from utils.db.connection import _init_schema


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    for did, filename in enumerate(
        ("empty.jpg", "empty.jpg", "mixed.jpg", "mixed.jpg", "rare.jpg"), 1
    ):
        conn.execute("INSERT OR IGNORE INTO images(filename) VALUES (?)", (filename,))
        conn.execute(
            """INSERT INTO detections(detection_id,image_filename,thumbnail_path,
               od_class_name,decision_state,decision_level,created_at,is_favorite,
               bbox_x,bbox_y,bbox_w,bbox_h,rating,rating_source)
               VALUES (?,?,?,'bird','confirmed','species','2026-10-11T12:00:00',1,
                       0.1,0.1,0.5,0.5,5,'manual')""",
            (did, filename, f"{did}.webp"),
        )
        conn.execute(
            """INSERT INTO classifications(detection_id,cls_class_name,rank)
               VALUES (?,'Nycticorax_nycticorax',1)""",
            (did,),
        )
    provenance = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")
    record_human_answer(
        conn,
        HumanAnswer(image_filename="empty.jpg", image_bird_presence="absent"),
        provenance,
    )
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="mixed.jpg", detection_id=3, object_bird_presence="absent"
        ),
        provenance,
    )
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="rare.jpg", detection_id=5, object_bird_presence="present"
        ),
        provenance,
    )
    # Exercise canonical authority even when legacy status is inconsistent.
    conn.execute("UPDATE detections SET status='active'")
    conn.execute("UPDATE classifications SET status='active'")
    yield conn
    conn.close()


@pytest.mark.parametrize("per_species_cap", [None, 10])
@pytest.mark.parametrize("rescore", [False, True])
def test_scoring_excludes_human_absence_at_both_scopes(
    connection: sqlite3.Connection, per_species_cap: int | None, rescore: bool
) -> None:
    rows = fetch_unscored_detections(
        connection, "2026-01-01", per_species_cap=per_species_cap, rescore=rescore
    )
    assert {row["detection_id"] for row in rows} == {4, 5}


def test_auto_picks_respect_human_absence_and_preserve_manual_favorites(
    connection: sqlite3.Connection,
) -> None:
    connection.execute("UPDATE detections SET aesthetic_score=0.99")
    assert apply_auto_favorites(connection, "2026-01-01", True)["total_tagged"] == 2
    assert (
        connection.execute(
            "SELECT SUM(is_gallery_eligible) FROM detections"
        ).fetchone()[0]
        == 0
    )
    assert apply_auto_favorites(connection, "2026-01-01", False)["total_tagged"] == 2
    rows = connection.execute(
        "SELECT detection_id,is_gallery_eligible,is_favorite,rating,rating_source FROM detections"
    ).fetchall()
    assert {row[0] for row in rows if row[1]} == {4, 5}
    assert all(row[2] == 1 for row in rows)
    assert all((row[3], row[4]) == (5, "manual") for row in rows)
    assert apply_auto_favorites(connection, "2026-01-01", False)["total_tagged"] == 0


def test_superseded_image_absence_no_longer_blocks_scoring(
    connection: sqlite3.Connection,
) -> None:
    record_human_answer(
        connection,
        HumanAnswer(image_filename="empty.jpg", image_bird_presence="present"),
        LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui"),
    )
    rows = fetch_unscored_detections(connection, "2026-01-01")
    assert {row["detection_id"] for row in rows} == {1, 2, 4, 5}
