"""Regression tests for fetch_trash_items / fetch_trash_count.

Covers the 2026-05-25 change: ``review_status='no_bird'`` images are
the training-export hard-negative corpus, not trash. They must not
appear in the trash grid or be counted toward the trash badge —
otherwise a routine "Empty Trash" sweep silently wipes verified FP
crops needed by the user-groundtruth export pipeline.
"""

import sqlite3

import pytest

from utils.db.trash import fetch_trash_count, fetch_trash_items


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory DB with the minimum schema for trash queries."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            review_updated_at TEXT,
            source_id INTEGER
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            status TEXT,
            created_at TEXT,
            thumbnail_path TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            FOREIGN KEY(image_filename) REFERENCES images(filename)
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active',
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id)
        );
    """)
    return conn


@pytest.fixture
def conn():
    c = _make_conn()
    yield c
    c.close()


def test_fetch_trash_items_returns_rejected_detections_only(conn):
    """Trash grid surfaces rejected detections only; no_bird images are hidden."""
    # Rejected detection — belongs in trash
    conn.execute(
        "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
        ("20260301120000_cam1.jpg", "20260301120000"),
    )
    conn.execute(
        """INSERT INTO detections
           (image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, od_confidence, status, created_at)
           VALUES (?, 0.1, 0.2, 0.3, 0.4, 'bird', 0.9, 'rejected', '2026-03-01')""",
        ("20260301120000_cam1.jpg",),
    )
    conn.execute(
        """INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank)
           VALUES (1, 'Parus_major', 0.85, 1)""",
    )

    # no_bird image — explicitly NOT trash (training hard-negative)
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, review_updated_at) "
        "VALUES (?, ?, 'no_bird', '2026-03-02')",
        ("20260302080000_cam1.jpg", "20260302080000"),
    )
    conn.commit()

    items, total = fetch_trash_items(conn)

    assert total == 1
    assert len(items) == 1
    assert items[0]["trash_type"] == "detection"
    assert items[0]["cls_class_name"] == "Parus_major"
    assert items[0]["species_key"] is not None

    types = {it["trash_type"] for it in items}
    assert "image" not in types, (
        "no_bird images must not appear in trash — they are the "
        "training-export hard-negative corpus"
    )


def test_fetch_trash_count_excludes_no_bird_images(conn):
    """Badge count covers rejected detections only — no_bird is preserved corpus."""
    # Two rejected detections
    conn.execute(
        "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
        ("20260301120000_cam1.jpg", "20260301120000"),
    )
    for _ in range(2):
        conn.execute(
            """INSERT INTO detections
               (image_filename, status, created_at)
               VALUES (?, 'rejected', '2026-03-01')""",
            ("20260301120000_cam1.jpg",),
        )

    # Five no_bird images — must not bump the badge
    for i in range(5):
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, review_updated_at) "
            "VALUES (?, ?, 'no_bird', '2026-03-02')",
            (f"20260302_0800{i:02d}_cam1.jpg", f"20260302080{i:02d}0"),
        )
    conn.commit()

    assert fetch_trash_count(conn) == 2
