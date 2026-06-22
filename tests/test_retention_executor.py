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

import pytest

from core import retention_core
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection

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
            "SELECT original_present, original_deleted_at FROM images WHERE filename=?",
            (fn,),
        ).fetchone()

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
