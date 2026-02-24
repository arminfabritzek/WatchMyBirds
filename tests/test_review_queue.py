import sqlite3

from utils.db.review_queue import fetch_review_queue_count, fetch_review_queue_images


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_confidence REAL,
            score REAL,
            od_model_id TEXT
        );
        """
    )
    return conn


def test_review_queue_uses_score_threshold_no_dead_zone():
    conn = _make_conn()
    try:
        # Orphan (no detections)
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("orphan.jpg", "20260210_000000", "untagged"),
        )

        # Low score but high OD confidence -> must appear in review queue (dead-zone fix)
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("low.jpg", "20260210_000001", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low.jpg", 0.80, 0.50, "yolo"),
        )

        # High score -> not in review queue
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("high.jpg", "20260210_000002", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("high.jpg", 0.80, 0.90, "yolo"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        filenames = [row["filename"] for row in rows]
        assert "orphan.jpg" in filenames
        assert "low.jpg" in filenames
        assert "high.jpg" not in filenames

        count = fetch_review_queue_count(conn, gallery_threshold=0.7)
        assert count == 2
    finally:
        conn.close()


def test_review_queue_can_exclude_deep_scanned_items():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("ds.jpg", "20260210_000003", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("ds.jpg", 0.20, 0.20, "deep_scan_tiled"),
        )
        conn.commit()

        rows = fetch_review_queue_images(
            conn, gallery_threshold=0.7, exclude_deep_scanned=True
        )
        assert [row["filename"] for row in rows] == []
    finally:
        conn.close()


def test_review_queue_exposes_top_bbox_for_low_score_items():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("low_bbox.jpg", "20260210_000010", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low_bbox.jpg", 0.70, 0.40, "yolo"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low_bbox.jpg", 0.65, 0.55, "yolo"),
        )
        conn.execute(
            """
            UPDATE detections
            SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?
            WHERE image_filename = ? AND score = ?
            """,
            (0.10, 0.20, 0.30, 0.40, "low_bbox.jpg", 0.55),
        )
        conn.execute(
            """
            UPDATE detections
            SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?
            WHERE image_filename = ? AND score = ?
            """,
            (0.50, 0.60, 0.10, 0.10, "low_bbox.jpg", 0.40),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        target = next(row for row in rows if row["filename"] == "low_bbox.jpg")
        assert target["review_reason"] == "low_score"
        assert target["bbox_x"] == 0.10
        assert target["bbox_y"] == 0.20
        assert target["bbox_w"] == 0.30
        assert target["bbox_h"] == 0.40
    finally:
        conn.close()
