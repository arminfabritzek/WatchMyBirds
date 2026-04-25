import sqlite3
from datetime import datetime, timedelta

from utils.db.analytics import fetch_event_intelligence_summary


def _make_conn() -> sqlite3.Connection:
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
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            score REAL,
            status TEXT NOT NULL DEFAULT 'active',
            decision_state TEXT,
            manual_species_override TEXT,
            species_source TEXT
        );

        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )
    return conn


def _timestamp(base: str, minutes: int) -> str:
    dt = datetime.strptime(base, "%Y%m%d_%H%M%S") + timedelta(minutes=minutes)
    return dt.strftime("%Y%m%d_%H%M%S")


def _insert_detection(
    conn: sqlite3.Connection,
    det_id: int,
    timestamp: str,
    species: str,
    *,
    review_status: str = "confirmed_bird",
    decision_state: str | None = "confirmed",
) -> None:
    filename = f"{det_id:04d}.webp"
    conn.execute(
        """
        INSERT INTO images(filename, timestamp, source_id, review_status)
        VALUES (?, ?, 1, ?)
        """,
        (filename, timestamp, review_status),
    )
    conn.execute(
        """
        INSERT INTO detections(
            detection_id, image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, score, status, decision_state
        )
        VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 0.95, 'active', ?)
        """,
        (det_id, filename, decision_state),
    )
    conn.execute(
        """
        INSERT INTO classifications(
            detection_id, cls_class_name, cls_confidence, rank, status
        )
        VALUES (?, ?, 0.91, 1, 'active')
        """,
        (det_id, species),
    )


def test_event_intelligence_summary_estimates_representative_retention():
    conn = _make_conn()
    try:
        for index in range(15):
            _insert_detection(
                conn,
                index + 1,
                _timestamp("20260425_120000", index),
                "Passer_domesticus",
            )
        for offset, minute in enumerate([0, 9, 22], start=100):
            _insert_detection(
                conn,
                offset,
                _timestamp("20260425_080000", minute),
                "Cyanistes_caeruleus",
            )
        conn.commit()

        data = fetch_event_intelligence_summary(conn)

        assert data["summary"]["event_count"] == 3
        assert data["summary"]["detection_count"] == 18
        assert data["summary"]["representative_image_count"] == 10
        assert data["summary"]["reducible_image_count"] == 8
        assert data["summary"]["retention_savings_pct"] == 44.4
        assert data["largest_events"][0]["species"] == "Passer_domesticus"
        assert data["largest_events"][0]["photo_count"] == 15
        assert data["largest_events"][0]["representative_image_count"] == 7

        profile_counts = {
            row["profile"]: row["event_count"] for row in data["profile_distribution"]
        }
        assert profile_counts["flock_burst"] == 1
        assert profile_counts["short_station_visit"] == 2

        species_pressure = {row["species"]: row for row in data["species_pressure"]}
        assert species_pressure["Passer_domesticus"]["reducible_image_count"] == 8
        assert species_pressure["Cyanistes_caeruleus"]["event_count"] == 2
    finally:
        conn.close()


def test_event_intelligence_summary_excludes_trash_and_unknown_decisions():
    conn = _make_conn()
    try:
        _insert_detection(conn, 1, "20260425_120000", "Parus_major")
        _insert_detection(
            conn,
            2,
            "20260425_120100",
            "Parus_major",
            review_status="no_bird",
        )
        _insert_detection(
            conn,
            3,
            "20260425_120200",
            "Parus_major",
            decision_state="unknown",
        )
        conn.commit()

        data = fetch_event_intelligence_summary(conn)

        assert data["summary"]["event_count"] == 1
        assert data["summary"]["detection_count"] == 1
        assert data["largest_events"][0]["species"] == "Parus_major"
    finally:
        conn.close()
