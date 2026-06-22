"""Retention Planner — the read-only dry-run engine.

build_plan(conn, output_dir, settings, now=...) partitions candidate
originals into deletable vs protected (with reasons), totals the
reclaimable bytes, and performs ZERO writes/deletes.

Uses the shared production-shaped seed helper (tests.retention_helpers) so
the layout matches what PersistenceService actually writes.
"""

import datetime as dt

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


def test_old_reviewed_image_with_derivatives_is_deletable(output_dir):
    fn = "20260101_120000_a.jpg"  # ~151 days before NOW
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, orig_bytes=4242)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert [d["filename"] for d in plan.deletable] == [fn]
    assert plan.estimated_bytes == 4242
    assert plan.deletable[0]["bytes"] == 4242


def test_recent_image_is_not_deletable(output_dir):
    # Recent images are pre-filtered out by the day-prefix cutoff; the
    # Policy P2 rule is unit-tested separately in test_retention_policy.
    fn = "20260530_120000_b.jpg"  # 2 days before NOW
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []


def test_missing_derivative_protects_original(output_dir):
    fn = "20260101_120001_c.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, write_optimized=False, write_thumbs=False)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("missing_derivative") == 1


def test_favorite_is_protected(output_dir):
    fn = "20260101_120002_d.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, favorite=True)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    # A favourite is also export-relevant; P5 fires before P4, so the
    # reason is export_relevant — the point is it is NOT deletable.
    assert plan.deletable == []
    assert sum(plan.protected_counts.values()) == 1


def test_unreviewed_is_protected(output_dir):
    fn = "20260101_120003_e.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, review_status="untagged")
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("unreviewed") == 1


def test_already_deleted_original_is_not_redeletable(output_dir):
    # original_present=0 rows are pre-filtered out of the candidate set (the
    # planner never re-processes them). P1 still guards this at the Policy
    # level — see test_retention_policy.test_p1_protected_when_original_already_deleted.
    fn = "20260101_120004_f.jpg"
    with closing_connection() as conn:
        seed_image(
            conn, fn, output_dir, original_present=0, write_original=False,
            original_deleted_at="2026-05-01T00:00:00+00:00",
        )
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("already_deleted", 0) == 0


def test_dry_run_writes_nothing(output_dir):
    fn = "20260101_120000_a.jpg"
    with closing_connection() as conn:
        paths = seed_image(conn, fn, output_dir)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)
        # DB unchanged: original_present still 1.
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename=?", (fn,)
        ).fetchone()

    assert len(plan.deletable) == 1
    assert row["original_present"] == 1
    assert paths["original"].exists()  # file untouched by a dry run
