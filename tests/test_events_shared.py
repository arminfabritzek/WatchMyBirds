"""Tests for the shared event + effort layer used by Analytics and Insights."""

import sqlite3
from datetime import datetime, timedelta

from utils.db.events import (
    calculate_effort,
    clear_events_cache,
    fetch_events_for_analysis,
    get_events_cached,
)


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


def _ts(base: str, minutes: int) -> str:
    dt = datetime.strptime(base, "%Y%m%d_%H%M%S") + timedelta(minutes=minutes)
    return dt.strftime("%Y%m%d_%H%M%S")


def _insert(
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
        "INSERT INTO images(filename, timestamp, source_id, review_status) "
        "VALUES (?, ?, 1, ?)",
        (filename, timestamp, review_status),
    )
    conn.execute(
        """
        INSERT INTO detections(
            detection_id, image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, score, status, decision_state
        ) VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 0.95, 'active', ?)
        """,
        (det_id, filename, decision_state),
    )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, "
        "cls_confidence, rank, status) VALUES (?, ?, 0.91, 1, 'active')",
        (det_id, species),
    )


# --- fetch_events_for_analysis -------------------------------------------------


def test_fetch_events_empty_db_returns_empty_list():
    clear_events_cache()
    conn = _make_conn()
    assert fetch_events_for_analysis(conn) == []


def test_fetch_events_30min_gap_rule():
    """Three detections at 0, 10, 45 minutes → two events (45 > 30 gap)."""
    clear_events_cache()
    conn = _make_conn()
    base = "20260425_120000"
    # Use Erithacus_rubecula - falls back to default_30m profile (gap=30)
    _insert(conn, 1, base, "Erithacus_rubecula")
    _insert(conn, 2, _ts(base, 10), "Erithacus_rubecula")
    _insert(conn, 3, _ts(base, 45), "Erithacus_rubecula")

    events = fetch_events_for_analysis(conn)
    assert len(events) == 2, "30-min gap rule must split detections at 45 min"
    assert sum(e.photo_count for e in events) == 3


def test_fetch_events_different_species_separate():
    clear_events_cache()
    conn = _make_conn()
    base = "20260425_120000"
    _insert(conn, 1, base, "Erithacus_rubecula")
    _insert(conn, 2, _ts(base, 5), "Turdus_merula")

    events = fetch_events_for_analysis(conn)
    assert len(events) == 2
    species = {e.species for e in events}
    assert species == {"Erithacus_rubecula", "Turdus_merula"}


def test_fetch_events_short_visit_profile_splits_meisen_burst():
    """Cyanistes_caeruleus uses short_station_visit (gap=12min)."""
    clear_events_cache()
    conn = _make_conn()
    base = "20260425_120000"
    _insert(conn, 1, base, "Cyanistes_caeruleus")
    _insert(conn, 2, _ts(base, 8), "Cyanistes_caeruleus")
    _insert(conn, 3, _ts(base, 25), "Cyanistes_caeruleus")  # 17min gap > 12

    events = fetch_events_for_analysis(conn)
    assert len(events) == 2, "short_station_visit must split at 17min gap"


def test_fetch_events_respects_visibility_predicate_no_bird():
    """Detections on no_bird-flagged images are hidden, not counted."""
    clear_events_cache()
    conn = _make_conn()
    base = "20260425_120000"
    _insert(conn, 1, base, "Erithacus_rubecula", review_status="no_bird")
    _insert(conn, 2, _ts(base, 5), "Erithacus_rubecula", review_status="confirmed_bird")

    events = fetch_events_for_analysis(conn)
    assert len(events) == 1
    assert events[0].photo_count == 1


def test_fetch_events_respects_visibility_predicate_uncertain():
    """decision_state=uncertain detections are hidden."""
    clear_events_cache()
    conn = _make_conn()
    base = "20260425_120000"
    _insert(conn, 1, base, "Erithacus_rubecula", decision_state="uncertain")
    _insert(conn, 2, _ts(base, 5), "Erithacus_rubecula", decision_state="confirmed")

    events = fetch_events_for_analysis(conn)
    assert len(events) == 1
    assert events[0].photo_count == 1


# --- calculate_effort ----------------------------------------------------------


def test_effort_empty_db():
    conn = _make_conn()
    effort = calculate_effort(conn)
    assert effort.first_ts is None
    assert effort.last_ts is None
    assert effort.total_days == 0
    assert effort.active_days == 0


def test_effort_single_image():
    conn = _make_conn()
    _insert(conn, 1, "20260425_120000", "Erithacus_rubecula")
    effort = calculate_effort(conn)
    assert effort.first_ts == "20260425_120000"
    assert effort.last_ts == "20260425_120000"
    assert effort.total_days == 1
    assert effort.active_days == 1


def test_effort_multi_day_with_gaps():
    """Day1 + Day3 + Day5 → total_days=5 (calendar span), active_days=3."""
    conn = _make_conn()
    _insert(conn, 1, "20260420_080000", "Erithacus_rubecula")
    _insert(conn, 2, "20260420_180000", "Erithacus_rubecula")  # same day
    _insert(conn, 3, "20260422_120000", "Erithacus_rubecula")
    _insert(conn, 4, "20260424_120000", "Erithacus_rubecula")

    effort = calculate_effort(conn)
    assert effort.first_ts == "20260420_080000"
    assert effort.last_ts == "20260424_120000"
    assert effort.total_days == 5  # 20→24 inclusive
    assert effort.active_days == 3  # 20, 22, 24


# --- get_events_cached ---------------------------------------------------------


def test_cached_events_returns_same_result():
    clear_events_cache()
    conn = _make_conn()
    _insert(conn, 1, "20260425_120000", "Erithacus_rubecula")

    first = get_events_cached(conn)
    second = get_events_cached(conn)
    assert first == second


def test_cached_events_invalidates_when_new_image_arrives():
    clear_events_cache()
    conn = _make_conn()
    _insert(conn, 1, "20260425_120000", "Erithacus_rubecula")
    first = get_events_cached(conn)
    assert len(first) == 1

    # Adding a much later detection (no gap-rule merge possible) bumps the
    # max(timestamp) cache key and forces a recompute.
    _insert(conn, 2, "20260425_180000", "Erithacus_rubecula")
    second = get_events_cached(conn)
    assert len(second) == 2
