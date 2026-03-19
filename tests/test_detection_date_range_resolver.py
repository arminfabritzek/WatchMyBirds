import sqlite3

from utils.db.detections import fetch_active_detection_ids_in_date_range


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        );
        """
    )
    return conn


def test_fetch_active_detection_ids_in_date_range_is_inclusive_and_stable():
    conn = _build_conn()
    conn.executemany(
        "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
        [
            ("a.jpg", "20260301_080000"),
            ("b.jpg", "20260302_090000"),
            ("c.jpg", "20260303_100000"),
        ],
    )
    conn.executemany(
        "INSERT INTO detections(detection_id, image_filename, status) VALUES (?, ?, ?)",
        [
            (11, "b.jpg", "active"),
            (10, "a.jpg", "active"),
            (12, "c.jpg", "active"),
        ],
    )

    ids = fetch_active_detection_ids_in_date_range(conn, "2026-03-01", "2026-03-02")

    assert ids == [10, 11]


def test_fetch_active_detection_ids_in_date_range_excludes_rejected_and_out_of_range():
    conn = _build_conn()
    conn.executemany(
        "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
        [
            ("in.jpg", "20260304_120000"),
            ("out.jpg", "20260306_120000"),
        ],
    )
    conn.executemany(
        "INSERT INTO detections(detection_id, image_filename, status) VALUES (?, ?, ?)",
        [
            (21, "in.jpg", "rejected"),
            (22, "in.jpg", "active"),
            (23, "out.jpg", "active"),
        ],
    )

    ids = fetch_active_detection_ids_in_date_range(conn, "2026-03-04", "2026-03-05")

    assert ids == [22]
