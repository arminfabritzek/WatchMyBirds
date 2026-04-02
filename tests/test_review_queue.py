import sqlite3

from utils.db.detections import set_manual_bbox_review
from utils.db.review_queue import (
    fetch_recent_review_species,
    fetch_review_queue_count,
    fetch_review_queue_images,
)


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
            od_model_id TEXT,
            status TEXT DEFAULT 'active',
            thumbnail_path TEXT,
            decision_state TEXT,
            bbox_quality REAL,
            unknown_score REAL,
            decision_reasons TEXT,
            manual_species_override TEXT,
            manual_bbox_review TEXT,
            bbox_reviewed_at TEXT
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


def test_review_queue_exposes_detection_context_and_bbox_review_state():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("ctx.jpg", "20260210_000020", "untagged"),
        )
        cur = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, bbox_quality, unknown_score
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ctx.jpg", 0.75, 0.42, "yolo", 0.25, 0.68),
        )
        detection_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (detection_id, "Parus_major", 0.84, 1, "active"),
        )
        set_manual_bbox_review(conn, detection_id, "wrong")
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        target = next(row for row in rows if row["filename"] == "ctx.jpg")

        assert target["best_detection_id"] == detection_id
        assert target["species_key"] == "Parus_major"
        assert target["manual_bbox_review"] == "wrong"
        assert target["cls_confidence"] == 0.84
        assert target["od_confidence"] == 0.75
    finally:
        conn.close()


def test_review_queue_returns_detection_items_per_unresolved_sibling():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("siblings.jpg", "20260210_000030", "untagged"),
        )
        first = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("siblings.jpg", 0.91, 0.91, "yolo", "uncertain"),
        ).lastrowid
        second = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("siblings.jpg", 0.66, 0.41, "yolo", "unknown"),
        ).lastrowid
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        sibling_rows = [row for row in rows if row["filename"] == "siblings.jpg"]

        assert len(sibling_rows) == 2
        assert {row["item_kind"] for row in sibling_rows} == {"detection"}
        assert {row["active_detection_id"] for row in sibling_rows} == {first, second}
        assert all(row["sibling_detection_count"] == 2 for row in sibling_rows)
    finally:
        conn.close()


def test_review_queue_excludes_images_marked_no_bird():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("trashed_import.jpg", "20260210_000040", "no_bird"),
        )
        conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("trashed_import.jpg", 0.62, 0.31, "yolo", "unknown", "active"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        filenames = [row["filename"] for row in rows]
        assert "trashed_import.jpg" not in filenames

        count = fetch_review_queue_count(conn, gallery_threshold=0.7)
        assert count == 0
    finally:
        conn.close()


def test_recent_review_species_returns_recent_frequent_species():
    conn = _make_conn()
    try:
        conn.executemany(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            [
                ("a.jpg", "20990110_000001", "untagged"),
                ("b.jpg", "20990110_000002", "untagged"),
                ("c.jpg", "20990109_230000", "untagged"),
            ],
        )
        detections = []
        for filename, species in [
            ("a.jpg", "Parus_major"),
            ("b.jpg", "Parus_major"),
            ("c.jpg", "Erithacus_rubecula"),
        ]:
            cur = conn.execute(
                """
                INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
                VALUES (?, ?, ?, ?)
                """,
                (filename, 0.8, 0.9, "yolo"),
            )
            detections.append((cur.lastrowid, species))

        conn.executemany(
            """
            INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(det_id, species, 0.9, 1, "active") for det_id, species in detections],
        )
        conn.commit()

        rows = fetch_recent_review_species(conn, limit=8, lookback_days=3650)
        species = [row["species_key"] for row in rows]
        assert species[:2] == ["Parus_major", "Erithacus_rubecula"]
    finally:
        conn.close()
