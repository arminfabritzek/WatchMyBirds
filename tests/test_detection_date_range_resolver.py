import sqlite3

from utils.db.detections import (
    fetch_active_detection_ids_in_date_range,
    fetch_active_detection_selection_in_date_range,
    fetch_trash_candidate_selection_in_date_range,
)


def _build_conn() -> sqlite3.Connection:
    from utils.db.connection import _init_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
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
        """INSERT INTO detections(decision_level, od_class_name, detection_id, image_filename, status, decision_state) VALUES ('species', 'Parus_major', ?, ?, ?, ?)""",
        [
            (11, "b.jpg", "active", "confirmed"),
            (10, "a.jpg", "active", "confirmed"),
            (12, "c.jpg", "active", "confirmed"),
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
        """INSERT INTO detections(decision_level, od_class_name, detection_id, image_filename, status, decision_state) VALUES ('species', 'Parus_major', ?, ?, ?, ?)""",
        [
            (21, "in.jpg", "rejected", "confirmed"),
            (22, "in.jpg", "active", "confirmed"),
            (23, "out.jpg", "active", "confirmed"),
        ],
    )

    ids = fetch_active_detection_ids_in_date_range(conn, "2026-03-04", "2026-03-05")

    assert ids == [22]


def test_fetch_active_detection_selection_in_date_range_returns_distinct_images():
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
        """INSERT INTO detections(decision_level, od_class_name, detection_id, image_filename, status, decision_state) VALUES ('species', 'Parus_major', ?, ?, ?, ?)""",
        [
            (10, "a.jpg", "active", "confirmed"),
            (11, "a.jpg", "active", "confirmed"),
            (12, "b.jpg", "active", "confirmed"),
            (13, "c.jpg", "rejected", "confirmed"),
        ],
    )

    selection = fetch_active_detection_selection_in_date_range(
        conn, "2026-03-01", "2026-03-02"
    )

    assert selection["detection_ids"] == [10, 11, 12]
    assert selection["image_filenames"] == ["a.jpg", "b.jpg"]
    assert selection["image_count"] == 2


def test_fetch_trash_candidate_selection_in_date_range_includes_review_only_and_orphans():
    conn = _build_conn()
    conn.executemany(
        "INSERT INTO images(filename, timestamp, review_status) VALUES (?, ?, ?)",
        [
            ("gallery.jpg", "20260301_080000", "untagged"),
            ("review-only.jpg", "20260301_090000", "untagged"),
            ("orphan.jpg", "20260302_090000", "untagged"),
            ("confirmed.jpg", "20260302_100000", "confirmed_bird"),
            ("trashed.jpg", "20260302_110000", "no_bird"),
        ],
    )
    conn.executemany(
        """INSERT INTO detections(decision_level, od_class_name, detection_id, image_filename, status, decision_state) VALUES ('species', 'Parus_major', ?, ?, ?, ?)""",
        [
            (51, "gallery.jpg", "active", None),
            (52, "review-only.jpg", "active", "unknown"),
            (53, "confirmed.jpg", "active", None),
            (54, "trashed.jpg", "active", None),
        ],
    )

    selection = fetch_trash_candidate_selection_in_date_range(
        conn, "2026-03-01", "2026-03-02"
    )

    assert selection["detection_ids"] == [51, 52, 53, 54]
    assert selection["image_filenames"] == [
        "gallery.jpg",
        "review-only.jpg",
        "confirmed.jpg",
        "trashed.jpg",
        "orphan.jpg",
    ]
    assert selection["orphan_image_filenames"] == ["orphan.jpg"]
    assert selection["orphan_count"] == 1
    assert selection["image_count"] == 5
