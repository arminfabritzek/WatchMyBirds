"""Retention Executor — file-delete-first, then DB status/log.

execute_plan(conn, output_dir, settings, now=...) deletes deletable
originals and records original_present=0 / original_deleted_at. Contracts:
files-first, missing-file-never-blocks-DB, survivors immutable,
OUTPUT_DIR-contained, idempotent.

Uses the shared production-shaped seed helper so the derivative layout
matches what PersistenceService writes (optimized <stem>.webp + per-crop
thumbs <stem>_crop_N.webp).
"""

import datetime as dt
import hashlib
import threading
import time

import pytest

from core import retention_core
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection, get_connection

NOW = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=dt.UTC)

SETTINGS = {
    "RETENTION_ENABLED": True,
    "RETENTION_DAYS": 90,
    "RETENTION_PROTECT_FAVORITES": True,
    "RETENTION_PROTECT_UNREVIEWED": True,
}


@pytest.fixture(autouse=True)
def wipe_schema_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_deletion_removes_original_keeps_derivatives_and_marks_db(output_dir):
    fn = "20260101_120000_a.jpg"
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir, orig_bytes=2048)
        opt_digest = _digest(paths["optimized"])
        thumb_digest = _digest(paths["thumbs"][0])

        result = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
        row = conn.execute(
            "SELECT original_present, original_deleted_at, timestamp, review_status "
            "FROM images WHERE filename=?",
            (fn,),
        ).fetchone()
        detection_count = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE image_filename=?", (fn,)
        ).fetchone()[0]

    assert result["deleted"] == 1
    assert result["freed_bytes"] == 2048
    assert not paths["original"].exists()  # original gone
    assert paths["optimized"].exists()  # derivatives preserved
    assert paths["thumbs"][0].exists()
    # Derivatives byte-identical (no recompression / rewrite).
    assert _digest(paths["optimized"]) == opt_digest
    assert _digest(paths["thumbs"][0]) == thumb_digest
    # DB marked.
    assert row["original_present"] == 0
    assert row["original_deleted_at"] is not None
    assert row["timestamp"] == fn[:15]
    assert row["review_status"] == "confirmed_bird"
    assert detection_count == 1


def test_missing_original_still_marks_db_and_does_not_raise(output_dir):
    fn = "20260101_120001_b.jpg"
    # Derivatives exist (so it's deletable) but the original is already gone,
    # while the DB still says original_present=1.
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, write_original=False)
        result = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename=?", (fn,)
        ).fetchone()

    # File was missing -> counted as missing, but DB is still updated.
    assert result["missing"] == 1
    assert row["original_present"] == 0


def test_protected_originals_are_untouched(output_dir):
    fn = "20260530_120000_c.jpg"  # too recent -> protected
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir)
        orig_digest = _digest(paths["original"])
        result = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename=?", (fn,)
        ).fetchone()

    assert result["deleted"] == 0
    assert paths["original"].exists()
    assert _digest(paths["original"]) == orig_digest  # immutable
    assert row["original_present"] == 1


def test_idempotent_second_run_deletes_nothing_new(output_dir):
    fn = "20260101_120002_d.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir)
        first = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
        second = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert first["deleted"] == 1
    assert second["deleted"] == 0
    assert second["errors"] == 0


def test_disabled_setting_deletes_nothing(output_dir):
    fn = "20260101_120003_e.jpg"
    settings = {**SETTINGS, "RETENTION_ENABLED": False}
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir)
        result = retention_core.execute_plan(conn, str(output_dir), settings, now=NOW)

    assert result["deleted"] == 0
    assert paths["original"].exists()


def test_repeated_runs_pick_up_newly_aged_and_new_images(output_dir):
    first_old = "20260101_120010_first.jpg"
    newly_old = "20260520_120010_newly.jpg"
    added_later = "20260102_120010_added.jpg"
    with closing_connection() as conn:
        first_paths = seed_image(conn, first_old, output_dir)
        newly_paths = seed_image(conn, newly_old, output_dir)
        first = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
        added_paths = seed_image(conn, added_later, output_dir)
        second = retention_core.execute_plan(
            conn,
            str(output_dir),
            SETTINGS,
            now=dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.UTC),
        )

    assert first["deleted"] == 1
    assert not first_paths["original"].exists()
    assert newly_paths["original"].exists() is False
    assert added_paths["original"].exists() is False
    assert second["deleted"] == 2


def test_protection_is_rechecked_immediately_before_delete(output_dir, monkeypatch):
    fn = "20260101_120011_changed.jpg"
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir)
        original_current_decision = retention_core._current_decision
        changed = False

        def add_favorite_then_decide(*args, **kwargs):
            nonlocal changed
            if not changed:
                conn.execute(
                    "UPDATE detections SET is_favorite=1, rating_source='manual' "
                    "WHERE image_filename=?",
                    (fn,),
                )
                conn.commit()
                changed = True
            return original_current_decision(*args, **kwargs)

        monkeypatch.setattr(
            retention_core, "_current_decision", add_favorite_then_decide
        )
        result = retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert result["deleted"] == 0
    assert result["protected"]["export_relevant"] == 1
    assert paths["original"].exists()


def test_concurrent_favorite_commit_precedes_delete_decision(output_dir):
    fn = "20260101_120012_concurrent.jpg"
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir)

    favorite_conn = get_connection()
    favorite_conn.execute("BEGIN IMMEDIATE")
    favorite_conn.execute(
        "UPDATE detections SET is_favorite=1, rating_source='manual' "
        "WHERE image_filename=?",
        (fn,),
    )

    started = threading.Event()
    finished = threading.Event()
    outcome = {}

    def run_retention():
        started.set()
        with closing_connection() as conn:
            outcome.update(
                retention_core.execute_plan(conn, str(output_dir), SETTINGS, now=NOW)
            )
        finished.set()

    worker = threading.Thread(target=run_retention)
    worker.start()
    assert started.wait(timeout=1)
    time.sleep(0.1)
    assert paths["original"].exists()

    favorite_conn.commit()
    favorite_conn.close()
    assert finished.wait(timeout=2)
    worker.join(timeout=1)

    assert outcome["deleted"] == 0
    assert outcome["protected"]["export_relevant"] == 1
    assert paths["original"].exists()


def test_stop_before_protected_candidate_batch_is_reported(output_dir):
    fn = "20260101_120013_protected.jpg"
    with closing_connection() as conn:
        paths = seed_image(
            conn, fn, output_dir, write_optimized=False, write_thumbs=False
        )
        result = retention_core.execute_plan(
            conn,
            str(output_dir),
            SETTINGS,
            now=NOW,
            stop_requested=lambda: True,
        )

    assert result["stopped"] is True
    assert result["deleted"] == 0
    assert paths["original"].exists()
