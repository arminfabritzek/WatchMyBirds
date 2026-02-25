import sqlite3

from utils.db.analytics import fetch_bird_visits


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            od_class_name TEXT,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            status TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )
    return conn


def _insert_det(
    conn: sqlite3.Connection,
    det_id: int,
    filename: str,
    ts: str,
    species: str,
    bbox: tuple[float, float, float, float],
) -> None:
    conn.execute("INSERT INTO images(filename, timestamp) VALUES (?, ?)", (filename, ts))
    conn.execute(
        """
        INSERT INTO detections(
            detection_id, image_filename, od_class_name, bbox_x, bbox_y, bbox_w, bbox_h, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
        """,
        (det_id, filename, species, *bbox),
    )


def test_fetch_bird_visits_groups_nearby_detections_as_single_visit():
    conn = _make_conn()
    try:
        _insert_det(
            conn,
            1,
            "a.webp",
            "20260101_120000",
            "parus_major",
            (0.10, 0.10, 0.20, 0.20),
        )
        _insert_det(
            conn,
            2,
            "b.webp",
            "20260101_120030",
            "parus_major",
            (0.12, 0.11, 0.21, 0.21),
        )
        conn.commit()

        data = fetch_bird_visits(conn)
        assert data["summary"]["total_visits"] == 1
        assert data["summary"]["total_detections"] == 2
        assert data["visits"][0]["photo_count"] == 2
    finally:
        conn.close()


def test_fetch_bird_visits_splits_visit_when_bbox_scale_changes_too_much():
    conn = _make_conn()
    try:
        _insert_det(
            conn,
            1,
            "a.webp",
            "20260101_120000",
            "parus_major",
            (0.10, 0.10, 0.30, 0.30),
        )
        _insert_det(
            conn,
            2,
            "b.webp",
            "20260101_120020",
            "parus_major",
            (0.12, 0.12, 0.05, 0.05),
        )
        conn.commit()

        data = fetch_bird_visits(conn)
        assert data["summary"]["total_visits"] == 2
        assert [v["photo_count"] for v in data["visits"]] == [1, 1]
    finally:
        conn.close()


def test_fetch_bird_visits_keeps_two_parallel_same_species_visits_separate():
    conn = _make_conn()
    try:
        # Bird A (left/top), Bird B (right/bottom), interleaved in time.
        _insert_det(conn, 1, "a1.webp", "20260101_120000", "parus_major", (0.10, 0.10, 0.16, 0.16))
        _insert_det(conn, 2, "b1.webp", "20260101_120005", "parus_major", (0.70, 0.70, 0.16, 0.16))
        _insert_det(conn, 3, "a2.webp", "20260101_120010", "parus_major", (0.11, 0.11, 0.16, 0.16))
        _insert_det(conn, 4, "b2.webp", "20260101_120015", "parus_major", (0.69, 0.69, 0.16, 0.16))
        conn.commit()

        data = fetch_bird_visits(conn)
        assert data["summary"]["total_visits"] == 2
        assert sorted(v["photo_count"] for v in data["visits"]) == [2, 2]
    finally:
        conn.close()
