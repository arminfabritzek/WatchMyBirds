import sqlite3

from utils.db.detections import (
    fetch_active_detection_ids_in_date_range,
    fetch_active_detection_selection_by_source_type,
    fetch_bbox_centers,
    fetch_count_last_24h,
    fetch_daily_covers,
    fetch_day_count,
    fetch_detection_species_summary,
    fetch_detections_for_gallery,
    fetch_detections_last_24h,
    fetch_hourly_counts,
    fetch_random_favorites,
    fetch_sibling_detections,
)


def _build_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL
        );

        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged',
            downloaded_timestamp TEXT
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
            rating REAL,
            rating_source TEXT,
            is_favorite INTEGER DEFAULT 0,
            decision_state TEXT,
            bbox_quality REAL,
            unknown_score REAL,
            decision_reasons TEXT
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


def _seed_visibility_fixture(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO sources(source_id, name, type) VALUES (?, ?, ?)",
        [
            (1, "Garden Cam", "ipcam"),
            (2, "Inbox Import", "folder_upload"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO images(filename, timestamp, source_id, review_status)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("20260327_120000_visible.jpg", "20260327_120000", 1, "untagged"),
            ("20260327_130000_mixed.jpg", "20260327_130000", 2, "untagged"),
            ("20260328_080000_reviewed.jpg", "20260328_080000", 2, "confirmed_bird"),
            ("20260328_090000_hidden.jpg", "20260328_090000", 2, "untagged"),
            ("20260328_100000_trash.jpg", "20260328_100000", 1, "no_bird"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO detections(
            detection_id,
            image_filename,
            status,
            created_at,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            od_class_name,
            od_confidence,
            score,
            is_favorite,
            decision_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "20260327_120000_visible.jpg",
                "active",
                "20260327_120000",
                0.10,
                0.10,
                0.20,
                0.20,
                "Visible_species",
                0.91,
                0.60,
                1,
                "confirmed",
            ),
            (
                2,
                "20260327_130000_mixed.jpg",
                "active",
                "20260327_130000",
                0.15,
                0.15,
                0.18,
                0.18,
                "Mixed_species",
                0.88,
                0.65,
                1,
                "confirmed",
            ),
            (
                3,
                "20260327_130000_mixed.jpg",
                "active",
                "20260327_130500",
                0.55,
                0.55,
                0.20,
                0.20,
                "Hidden_species",
                0.72,
                0.95,
                1,
                "uncertain",
            ),
            (
                4,
                "20260328_080000_reviewed.jpg",
                "active",
                "20260328_080000",
                0.20,
                0.20,
                0.25,
                0.25,
                "Reviewed_species",
                0.79,
                0.80,
                1,
                "uncertain",
            ),
            (
                5,
                "20260328_090000_hidden.jpg",
                "active",
                "20260328_090000",
                0.25,
                0.25,
                0.22,
                0.22,
                "Unknown_species",
                0.61,
                0.99,
                1,
                "unknown",
            ),
            (
                6,
                "20260328_100000_trash.jpg",
                "active",
                "20260328_100000",
                0.30,
                0.30,
                0.22,
                0.22,
                "Trash_species",
                0.67,
                0.88,
                1,
                "confirmed",
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (1, "Visible_species", 0.91, 1, "active"),
            (2, "Mixed_species", 0.88, 1, "active"),
            (3, "Hidden_species", 0.72, 1, "active"),
            (4, "Reviewed_species", 0.79, 1, "active"),
            (5, "Unknown_species", 0.61, 1, "active"),
            (6, "Trash_species", 0.67, 1, "active"),
        ],
    )
    conn.commit()


def test_visibility_policy_filters_gallery_like_surfaces():
    conn = _build_conn()
    _seed_visibility_fixture(conn)

    gallery_ids = [
        row["detection_id"] for row in fetch_detections_for_gallery(conn, order_by="score")
    ]
    assert gallery_ids == [2, 1]

    mixed_row = next(
        row for row in fetch_detections_for_gallery(conn, order_by="score")
        if row["detection_id"] == 2
    )
    assert mixed_row["sibling_count"] == 1

    sibling_ids = [
        row["detection_id"]
        for row in fetch_sibling_detections(conn, "20260327_130000_mixed.jpg")
    ]
    assert sibling_ids == [2]

    favorite_ids = {
        row["detection_id"] for row in fetch_random_favorites(conn, limit=10)
    }
    assert favorite_ids == {1, 2}

    covers = {
        row["date_key"]: {
            "relative_path": row["relative_path"],
            "image_count": row["image_count"],
        }
        for row in fetch_daily_covers(conn, min_score=0.0)
    }
    assert covers["2026-03-27"]["relative_path"].endswith(
        "20260327_130000_mixed.webp"
    )
    assert covers["2026-03-27"]["image_count"] == 2

    assert "2026-03-28" not in covers

    species_summary = [
        (row["species"], row["count"])
        for row in fetch_detection_species_summary(conn, "2026-03-28")
    ]
    assert species_summary == []

    assert fetch_day_count(conn, "2026-03-27") == 2
    assert fetch_day_count(conn, "2026-03-28") == 0

    hourly_counts = [
        (row["hour"], row["count"]) for row in fetch_hourly_counts(conn, "2026-03-28")
    ]
    assert hourly_counts == []

    center_species = {row["species"] for row in fetch_bbox_centers(conn, limit=10)}
    assert center_species == {"Visible_species", "Mixed_species"}


def test_visibility_policy_propagates_to_rolling_and_bulk_selection_queries():
    conn = _build_conn()
    _seed_visibility_fixture(conn)

    last_24h_ids = [
        row["detection_id"]
        for row in fetch_detections_last_24h(
            conn,
            "20260328_000000",
            order_by="score",
        )
    ]
    assert last_24h_ids == []
    assert fetch_count_last_24h(conn, "20260328_000000") == 0

    ids_in_range = fetch_active_detection_ids_in_date_range(
        conn, "2026-03-27", "2026-03-28"
    )
    assert ids_in_range == [1, 2]

    selection = fetch_active_detection_selection_by_source_type(conn, "folder_upload")
    assert selection == {
        "detection_ids": [2],
        "image_filenames": ["20260327_130000_mixed.jpg"],
        "image_count": 1,
    }


def test_uncertain_detection_never_appears_in_gallery_even_if_image_is_confirmed():
    conn = _build_conn()
    conn.execute(
        """
        INSERT INTO images(filename, timestamp, source_id, review_status)
        VALUES (?, ?, ?, ?)
        """,
        ("20260329_070225_reviewed.jpg", "20260329_070225", 1, "confirmed_bird"),
    )
    conn.execute(
        """
        INSERT INTO detections(
            detection_id,
            image_filename,
            status,
            created_at,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            od_class_name,
            od_confidence,
            score,
            is_favorite,
            decision_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            99,
            "20260329_070225_reviewed.jpg",
            "active",
            "20260329_070225",
            0.10,
            0.10,
            0.20,
            0.20,
            "Cyanistes_caeruleus",
            0.76,
            0.52,
            0,
            "unknown",
        ),
    )
    conn.execute(
        """
        INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (99, "Cyanistes_caeruleus", 0.29, 1, "active"),
    )
    conn.commit()

    gallery_ids = [
        row["detection_id"] for row in fetch_detections_for_gallery(conn, order_by="score")
    ]
    assert gallery_ids == []
