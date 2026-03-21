"""Regression test: fetch_trash_items UNION ALL must work with mixed content."""

import sqlite3

import pytest

from utils.db.trash import fetch_trash_items


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


def test_fetch_trash_items_mixed_content(conn):
    """UNION ALL must succeed when both rejected detections and no_bird images exist."""
    # Insert an image with a rejected detection
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

    # Insert a no_bird image
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, review_updated_at) "
        "VALUES (?, ?, 'no_bird', '2026-03-02')",
        ("20260302080000_cam1.jpg", "20260302080000"),
    )
    conn.commit()

    # This must not raise "SELECTs ... do not have the same number of result columns"
    items, total = fetch_trash_items(conn)

    assert total == 2
    assert len(items) == 2

    types = {it["trash_type"] for it in items}
    assert types == {"detection", "image"}

    # Detection item should carry species info
    det = next(it for it in items if it["trash_type"] == "detection")
    assert det["cls_class_name"] == "Parus_major"
    assert det["species_key"] is not None

    # Image item should have NULL species fields
    img = next(it for it in items if it["trash_type"] == "image")
    assert img["manual_species_override"] is None
    assert img["species_source"] is None
    assert img["species_key"] is None
