"""Retention against production-shaped artifacts.

These tests build the exact derivative layout PersistenceService writes
(optimized ``<stem>.webp`` + per-detection thumbs ``<stem>_crop_N.webp``
recorded in ``detections.thumbnail_path``, preview ``<stem>_preview.webp``
for orphans) via the shared helper, so P3 derivative-presence is exercised
against real names — not fabricated ``<stem>.webp`` thumbs.
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
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def test_production_artifacts_make_original_deletable(output_dir):
    # optimized <stem>.webp + thumb <stem>_crop_1.webp (the REAL layout).
    fn = "20260101_120000_a.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, orig_bytes=4242)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert [d["filename"] for d in plan.deletable] == [fn]
    assert plan.estimated_bytes == 4242


def test_missing_optimized_protects_original(output_dir):
    fn = "20260101_120001_b.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, write_optimized=False)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("missing_derivative") == 1


def test_missing_thumb_protects_original(output_dir):
    # optimized exists but the crop thumb was never written.
    fn = "20260101_120002_c.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, write_thumbs=False)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("missing_derivative") == 1


def test_multi_detection_image_one_thumb_enough(output_dir):
    # 3 detections -> 3 crop thumbs; the original is the same single file.
    fn = "20260101_120003_d.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, detections=3)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert [d["filename"] for d in plan.deletable] == [fn]


def test_orphan_image_preview_thumb_counts(output_dir):
    # No active detections, but a preview thumb exists (the display
    # derivative for orphan images). Optimized exists too.
    fn = "20260101_120004_e.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, detections=0, write_preview=True)
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert [d["filename"] for d in plan.deletable] == [fn]


def test_orphan_image_without_any_thumb_is_protected(output_dir):
    fn = "20260101_120005_f.jpg"
    with closing_connection() as conn:
        seed_image(
            conn, fn, output_dir, detections=0, write_preview=False
        )
        plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)

    assert plan.deletable == []
    assert plan.protected_counts.get("missing_derivative") == 1
