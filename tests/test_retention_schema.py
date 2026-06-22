"""Retention schema — the additive original-presence columns on `images`."""

import pytest

from utils.db.connection import closing_connection


@pytest.fixture(autouse=True)
def wipe_schema_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_retention_columns_created_on_images():
    with closing_connection() as conn:
        cur = conn.execute("PRAGMA table_info(images);")
        cols = {row["name"]: row for row in cur.fetchall()}
    assert "original_present" in cols
    assert "original_deleted_at" in cols


def test_original_present_defaults_to_one():
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
            ("20260101_120000_x.jpg", "20260101_120000"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT original_present, original_deleted_at FROM images "
            "WHERE filename = ?",
            ("20260101_120000_x.jpg",),
        ).fetchone()
    assert row["original_present"] == 1
    assert row["original_deleted_at"] is None
