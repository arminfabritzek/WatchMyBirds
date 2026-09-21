"""A human "species unknown" box stays visible as a companion box.

UI_STANDARD.md §0c is binding: every active box on a photo is drawn in
the detail modal. A person who answered "there is a bird here, I cannot
name the species" has asserted something *stronger* than any model
confidence, so their box must not be filtered out by the gallery's
temporal-smoother gate.
"""

import sqlite3

import pytest

from utils.db.detections import (
    fetch_sibling_detections,
    fetch_sibling_detections_batch,
)

UNKNOWN_IMAGE = "20260920_094825_673647.jpg"


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged'
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            score REAL,
            thumbnail_path TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            is_favorite INTEGER DEFAULT 0,
            decision_state TEXT,
            decision_level TEXT,
            quality_gallery_ok INTEGER
        );

        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active'
        );
        """
    )
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO images(filename, timestamp, source_id, review_status)"
        " VALUES (?, ?, ?, ?)",
        (UNKNOWN_IMAGE, "20260920_094825", 1, "untagged"),
    )
    conn.executemany(
        """
        INSERT INTO detections(
            detection_id, image_filename, status, created_at,
            bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, od_confidence, score,
            manual_species_override, species_source,
            decision_state, decision_level, quality_gallery_ok
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            # The human-withdrawn species: override cleared, state 'unknown'.
            (
                74759,
                UNKNOWN_IMAGE,
                "active",
                "2026-09-20T09:48:25",
                10.0,
                10.0,
                40.0,
                40.0,
                "bird",
                0.93,
                0.93,
                None,
                "manual_unknown",
                "unknown",
                None,
                1,
            ),
            # An ordinary confirmed companion on the same photo.
            (
                74760,
                UNKNOWN_IMAGE,
                "active",
                "2026-09-20T09:48:25",
                80.0,
                20.0,
                30.0,
                30.0,
                "bird",
                0.88,
                0.88,
                "Parus_major",
                "manual",
                "confirmed",
                "species",
                1,
            ),
            # A genuine model-uncertain row: must stay hidden.
            (
                74761,
                UNKNOWN_IMAGE,
                "active",
                "2026-09-20T09:48:25",
                5.0,
                60.0,
                20.0,
                20.0,
                "bird",
                0.30,
                0.30,
                None,
                None,
                "unknown",
                None,
                1,
            ),
        ],
    )
    conn.executemany(
        "INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank)"
        " VALUES (?, ?, ?, 1)",
        [
            (74759, "Sitta_europaea", 0.93),
            (74760, "Parus_major", 0.88),
            (74761, "Sitta_europaea", 0.30),
        ],
    )
    conn.commit()


def test_human_unknown_box_is_returned_as_sibling():
    conn = _build_conn()
    _seed(conn)

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert 74759 in ids, (
        "human 'species unknown' box must stay visible (UI_STANDARD §0c)"
    )
    assert 74760 in ids, "ordinary confirmed companion must stay visible"


def test_model_uncertain_box_stays_hidden():
    """The exception is for human answers only, not for model uncertainty."""
    conn = _build_conn()
    _seed(conn)

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert 74761 not in ids, "model-uncertain rows must not leak into the gallery"


def test_human_unknown_box_survives_the_batch_variant():
    conn = _build_conn()
    _seed(conn)

    grouped = fetch_sibling_detections_batch(conn, [UNKNOWN_IMAGE])
    ids = {row["detection_id"] for row in grouped.get(UNKNOWN_IMAGE, [])}

    assert 74759 in ids
    assert 74760 in ids
    assert 74761 not in ids


@pytest.mark.parametrize("source", ["manual_unknown", "manual_wrong"])
def test_both_human_withdrawal_sources_stay_visible(source: str):
    conn = _build_conn()
    _seed(conn)
    conn.execute(
        "UPDATE detections SET species_source = ? WHERE detection_id = 74759",
        (source,),
    )
    conn.commit()

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert 74759 in ids


def test_rejected_human_unknown_box_stays_hidden():
    """Visibility follows the human answer, not the species_source alone."""
    conn = _build_conn()
    _seed(conn)
    conn.execute("UPDATE detections SET status = 'rejected' WHERE detection_id = 74759")
    conn.commit()

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert 74759 not in ids


def test_a_trashed_frame_hides_the_human_unknown_box_too():
    """Trash outranks the human answer: the whole frame is gone."""
    conn = _build_conn()
    _seed(conn)
    conn.execute(
        "UPDATE images SET review_status = 'no_bird' WHERE filename = ?",
        (UNKNOWN_IMAGE,),
    )
    conn.commit()

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert ids == set(), "a frame marked 'no bird' shows no companion boxes"


def test_a_quality_vetoed_human_unknown_box_stays_hidden():
    """The nightly sharpness veto still applies to a human-answered box."""
    conn = _build_conn()
    _seed(conn)
    conn.execute(
        "UPDATE detections SET quality_gallery_ok = 0 WHERE detection_id = 74759"
    )
    conn.commit()

    ids = {row["detection_id"] for row in fetch_sibling_detections(conn, UNKNOWN_IMAGE)}

    assert 74759 not in ids


def test_a_frame_with_no_rows_at_all_returns_empty():
    """A missing / orphaned frame must not raise, just come back empty."""
    conn = _build_conn()
    _seed(conn)

    assert fetch_sibling_detections(conn, "20260101_000000_absent.jpg") == []
    assert fetch_sibling_detections_batch(conn, []) == {}
