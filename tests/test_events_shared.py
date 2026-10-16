"""Tests for the shared event + effort layer used by Analytics and Insights."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.events import BirdEvent
from utils.db.events import (
    calculate_effort,
    clear_events_cache,
    fetch_events_for_analysis,
    get_events_cached,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from utils.db.connection import _init_schema

    _init_schema(conn)
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
    decision_level: str | None = "species",
    quality_gallery_ok: int = 1,
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
            od_class_name, score, status, decision_state, decision_level,
            quality_gallery_ok
        ) VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 0.95, 'active', ?, ?, ?)
        """,
        (det_id, filename, decision_state, decision_level, quality_gallery_ok),
    )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, "
        "cls_confidence, rank, status) VALUES (?, ?, 0.91, 1, 'active')",
        (det_id, species),
    )


# --- fetch_events_for_analysis -------------------------------------------------


def test_analysis_cohort_keeps_valid_low_quality_photos_but_excludes_review_rows():
    conn = _make_conn()
    try:
        _insert(
            conn,
            1,
            "20260420_060000",
            "Parus_major",
            quality_gallery_ok=0,
        )
        _insert(
            conn,
            2,
            "20260420_070000",
            "Cyanistes_caeruleus",
            decision_level="species_review",
        )
        _insert(
            conn,
            3,
            "20260420_080000",
            "Turdus_merula",
            decision_state=None,
        )
        conn.commit()

        events = fetch_events_for_analysis(conn, min_score=0.0)

        assert [event.species for event in events] == ["Parus_major"]
        assert events[0].photo_count == 1
    finally:
        conn.close()


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


def test_cache_replaces_arrivals_across_ttl_and_bounds_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import read_cache
    from utils.db import events

    clear_events_cache()
    clock = [1.0]
    monkeypatch.setattr(read_cache, "monotonic", lambda: clock[0])
    conn = _make_conn()
    try:
        for i in range(1, 20):
            _insert(conn, i, f"20260425_{i:02d}0000", "Erithacus_rubecula")
            clock[0] += 120
            assert len(get_events_cached(conn)) == i
            assert events._cache_state.size() == 1
        for i in range(20):
            get_events_cached(conn, min_score=i / 100)
        assert events._cache_state.size() == 8
    finally:
        conn.close()
        clear_events_cache()


def test_cache_does_not_round_distinct_thresholds() -> None:
    clear_events_cache()
    conn = _make_conn()
    _insert(conn, 1, "20260425_120000", "Parus_major")
    conn.execute("UPDATE detections SET score = 0.95002")
    assert len(get_events_cached(conn, min_score=0.95001)) == 1
    assert get_events_cached(conn, min_score=0.95003) == []
    conn.close()
    clear_events_cache()


def test_cache_separates_databases_and_handles_same_timestamp_correction(
    tmp_path: Path,
) -> None:
    clear_events_cache()
    source = _make_conn()
    _insert(source, 1, "20260425_120000", "Parus_major")
    source.commit()
    target = sqlite3.connect(tmp_path / "events.db")
    source.backup(target)
    target.close()
    source.close()
    for journal in ("DELETE", "WAL"):
        writer = sqlite3.connect(tmp_path / "events.db")
        writer.row_factory = sqlite3.Row
        writer.execute(f"PRAGMA journal_mode = {journal}")
        writer.execute("UPDATE detections SET manual_species_override = 'Parus_major'")
        writer.commit()
        reader = sqlite3.connect(tmp_path / "events.db")
        reader.row_factory = sqlite3.Row
        assert get_events_cached(reader)[0].species == "Parus_major"
        writer.execute(
            "UPDATE detections SET manual_species_override = 'Turdus_merula'"
        )
        writer.commit()
        assert get_events_cached(reader)[0].species == "Turdus_merula"
        reader.close()
        reader = sqlite3.connect(tmp_path / "events.db")
        reader.row_factory = sqlite3.Row
        assert get_events_cached(reader)[0].species == "Turdus_merula"
        memory = _make_conn()
        assert get_events_cached(memory) == []
        memory.close()
        writer.close()
        reader.close()
        clear_events_cache()


def test_cache_reuses_result_across_closed_request_connections(tmp_path: Path) -> None:
    clear_events_cache()
    source = _make_conn()
    _insert(source, 1, "20260425_120000", "Parus_major")
    source.commit()
    path = tmp_path / "requests.db"
    writer = sqlite3.connect(path)
    source.backup(writer)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.close()
    source.close()
    reader = sqlite3.connect(path)
    reader.row_factory = sqlite3.Row
    first = get_events_cached(reader)
    reader.close()
    reader = sqlite3.connect(path)
    reader.row_factory = sqlite3.Row
    assert get_events_cached(reader) is first
    reader.close()
    writer = sqlite3.connect(path)
    writer.execute("UPDATE detections SET manual_species_override = 'Turdus_merula'")
    writer.commit()
    writer.close()
    reader = sqlite3.connect(path)
    reader.row_factory = sqlite3.Row
    assert get_events_cached(reader)[0].species == "Turdus_merula"
    reader.close()
    clear_events_cache()


def test_uncommitted_file_changes_do_not_leak_through_cache(tmp_path: Path) -> None:
    clear_events_cache()
    source = _make_conn()
    _insert(source, 1, "20260425_120000", "Parus_major")
    source.commit()
    writer = sqlite3.connect(tmp_path / "transaction.db")
    writer.row_factory = sqlite3.Row
    source.backup(writer)
    source.close()
    first = get_events_cached(writer)
    writer.execute("UPDATE detections SET manual_species_override = 'Turdus_merula'")
    assert get_events_cached(writer)[0].species == "Turdus_merula"
    writer.rollback()
    assert get_events_cached(writer) is first
    writer.close()
    clear_events_cache()


def test_concurrent_event_requests_share_one_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from utils.db import events

    clear_events_cache()
    source = _make_conn()
    _insert(source, 1, "20260425_120000", "Parus_major")
    source.commit()
    path = tmp_path / "parallel.db"
    writer = sqlite3.connect(path)
    source.backup(writer)
    writer.close()
    source.close()
    barrier = threading.Barrier(4)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    original = events.fetch_events_for_analysis

    def build(conn: sqlite3.Connection, *, min_score: float) -> list[BirdEvent]:
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return original(conn, min_score=min_score)

    monkeypatch.setattr(events, "fetch_events_for_analysis", build)

    def read() -> list[BirdEvent]:
        reader = sqlite3.connect(path)
        reader.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=3)
            return get_events_cached(reader)
        finally:
            reader.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(read) for _ in range(4)]
        assert entered.wait(3)
        release.set()
        results = [job.result(timeout=3) for job in jobs]
    assert len(calls) == 1
    assert all(result is results[0] for result in results)
    clear_events_cache()


def test_events_use_current_answers_and_historical_admission() -> None:
    from core.human_label_core import HumanAnswer, record_human_answer
    from tests.test_station_observations import answer, provenance, seed

    conn = _make_conn()
    try:
        seed(conn, identifier=1, species="Parus_major", level=None)
        filename = seed(conn, identifier=2, species="Turdus_merula", state="uncertain")
        answer(conn, 2, filename, species="Pica_pica")
        absent = seed(conn, identifier=3, species="Cyanistes_caeruleus")
        record_human_answer(
            conn,
            HumanAnswer(absent, detection_id=3, object_bird_presence="absent"),
            provenance(),
        )
        assert {event.species for event in fetch_events_for_analysis(conn)} == {
            "Parus_major",
            "Pica_pica",
        }
    finally:
        conn.close()
