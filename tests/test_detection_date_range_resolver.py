import sqlite3

from utils.db.detections import (
    fetch_active_detection_ids_in_date_range,
    fetch_active_detection_selection_in_date_range,
    fetch_active_detection_selection_by_source_type,
    fetch_trash_candidate_selection_in_date_range,
    fetch_trash_candidate_selection_by_source_type,
)


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged'
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            decision_state TEXT,
            decision_level TEXT
        );

        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL
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
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
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
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
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
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
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


def test_fetch_active_detection_selection_by_source_type_filters_imported_images():
    conn = _build_conn()
    conn.executemany(
        "INSERT INTO sources(source_id, name, type) VALUES (?, ?, ?)",
        [
            (1, "Default Camera", "ipcam"),
            (2, "User Import", "folder_upload"),
        ],
    )
    conn.executemany(
        "INSERT INTO images(filename, timestamp, source_id) VALUES (?, ?, ?)",
        [
            ("cam.jpg", "20260304_080000", 1),
            ("import-a.jpg", "20260304_090000", 2),
            ("import-b.jpg", "20260304_100000", 2),
        ],
    )
    conn.executemany(
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
        [
            (31, "cam.jpg", "active", "confirmed"),
            (32, "import-a.jpg", "active", "confirmed"),
            (33, "import-a.jpg", "active", "confirmed"),
            (34, "import-b.jpg", "rejected", "confirmed"),
        ],
    )

    selection = fetch_active_detection_selection_by_source_type(conn, "folder_upload")

    assert selection["detection_ids"] == [32, 33]
    assert selection["image_filenames"] == ["import-a.jpg"]
    assert selection["image_count"] == 1


def test_fetch_active_detection_selection_by_source_type_returns_zero_for_no_matches():
    conn = _build_conn()
    conn.execute(
        "INSERT INTO sources(source_id, name, type) VALUES (?, ?, ?)",
        (1, "Default Camera", "ipcam"),
    )
    conn.execute(
        "INSERT INTO images(filename, timestamp, source_id) VALUES (?, ?, ?)",
        ("cam.jpg", "20260304_080000", 1),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, status) VALUES (?, ?, ?)",
        (41, "cam.jpg", "active"),
    )

    selection = fetch_active_detection_selection_by_source_type(conn, "folder_upload")

    assert selection["detection_ids"] == []
    assert selection["image_filenames"] == []
    assert selection["image_count"] == 0


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
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
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

    assert selection["detection_ids"] == [51, 52]
    assert selection["image_filenames"] == ["gallery.jpg", "review-only.jpg", "orphan.jpg"]
    assert selection["orphan_image_filenames"] == ["orphan.jpg"]
    assert selection["orphan_count"] == 1
    assert selection["image_count"] == 3


def test_fetch_trash_candidate_selection_by_source_type_includes_review_only_and_orphans():
    conn = _build_conn()
    conn.executemany(
        "INSERT INTO sources(source_id, name, type) VALUES (?, ?, ?)",
        [
            (1, "Default Camera", "ipcam"),
            (2, "User Import", "folder_upload"),
        ],
    )
    conn.executemany(
        "INSERT INTO images(filename, timestamp, source_id, review_status) VALUES (?, ?, ?, ?)",
        [
            ("cam.jpg", "20260304_080000", 1, "untagged"),
            ("import-gallery.jpg", "20260304_090000", 2, "untagged"),
            ("import-review.jpg", "20260304_100000", 2, "untagged"),
            ("import-orphan.jpg", "20260304_110000", 2, "untagged"),
            ("import-confirmed.jpg", "20260304_120000", 2, "confirmed_bird"),
            ("import-trash.jpg", "20260304_130000", 2, "no_bird"),
        ],
    )
    conn.executemany(
        "INSERT INTO detections(detection_id, image_filename, status, decision_state) VALUES (?, ?, ?, ?)",
        [
            (61, "cam.jpg", "active", None),
            (62, "import-gallery.jpg", "active", None),
            (63, "import-review.jpg", "active", "uncertain"),
            (64, "import-confirmed.jpg", "active", None),
            (65, "import-trash.jpg", "active", None),
        ],
    )

    selection = fetch_trash_candidate_selection_by_source_type(conn, "folder_upload")

    assert selection["detection_ids"] == [62, 63]
    assert selection["image_filenames"] == [
        "import-gallery.jpg",
        "import-review.jpg",
        "import-orphan.jpg",
    ]
    assert selection["orphan_image_filenames"] == ["import-orphan.jpg"]
    assert selection["orphan_count"] == 1
    assert selection["image_count"] == 3
