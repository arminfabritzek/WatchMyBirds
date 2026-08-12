"""Review-cleanup Executor — reversible detection move-to-Trash.

execute_plan moves active detections into Trash without turning untagged images
into No Bird training data. It must delete ZERO files.
"""

import pytest

from core import review_cleanup_core
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection
from utils.db.detections import restore_detections
from utils.db.review_queue import (
    fetch_review_queue_images,
)

THRESHOLD = 0.7


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


def _seed_mixed_queue(conn, output_dir):
    seed_image(conn, "20260101_120000_q.jpg", output_dir, review_status="untagged")
    seed_image(
        conn,
        "20260101_120001_o.jpg",
        output_dir,
        review_status="untagged",
        detections=0,
    )


def test_execute_moves_detections_but_keeps_orphan_images(output_dir):
    with closing_connection() as conn:
        _seed_mixed_queue(conn, output_dir)
        result = review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)
        conn.commit()
        remaining = fetch_review_queue_images(conn, gallery_threshold=THRESHOLD)

    assert result == {"images_moved": 0, "detections_moved": 1}
    assert {(row["item_kind"], row["item_id"]) for row in remaining} == {
        ("image", "20260101_120001_o.jpg")
    }


def test_execute_uses_reversible_states_not_deletion(output_dir):
    with closing_connection() as conn:
        _seed_mixed_queue(conn, output_dir)
        review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)
        conn.commit()

        img = conn.execute(
            "SELECT review_status FROM images WHERE filename = ?",
            ("20260101_120001_o.jpg",),
        ).fetchone()
        det = conn.execute(
            "SELECT status FROM detections WHERE image_filename = ?",
            ("20260101_120000_q.jpg",),
        ).fetchone()

    assert img["review_status"] == "untagged"
    assert det["status"] == "rejected"


def test_execute_deletes_no_files(output_dir, monkeypatch):
    import utils.file_gc as file_gc

    calls = []
    real = file_gc._safe_delete
    monkeypatch.setattr(
        file_gc, "_safe_delete", lambda *a, **k: calls.append(a) or real(*a, **k)
    )

    with closing_connection() as conn:
        paths = seed_image(
            conn, "20260101_120000_q.jpg", output_dir, review_status="untagged"
        )
        review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)
        conn.commit()

    assert calls == []
    # Original + derivatives still on disk.
    assert paths["original"].exists()
    assert paths["optimized"].exists()
    assert all(t.exists() for t in paths["thumbs"])


def test_moved_detection_restores_while_orphan_stays_in_review(output_dir):
    with closing_connection() as conn:
        _seed_mixed_queue(conn, output_dir)
        review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)
        conn.commit()

        det_id = conn.execute(
            "SELECT detection_id FROM detections WHERE image_filename = ?",
            ("20260101_120000_q.jpg",),
        ).fetchone()["detection_id"]
        restore_detections(conn, [det_id])
        conn.commit()

        restored = fetch_review_queue_images(conn, gallery_threshold=THRESHOLD)

    kinds = {(r["item_kind"], r["item_id"]) for r in restored}
    assert ("detection", str(det_id)) in kinds
    assert ("image", "20260101_120001_o.jpg") in kinds


def test_favorite_and_export_relevant_are_moved_not_protected(output_dir):
    with closing_connection() as conn:
        seed_image(
            conn,
            "20260101_120000_f.jpg",
            output_dir,
            review_status="untagged",
            favorite=True,
        )
        plan = review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)
        review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)
        conn.commit()
        remaining = fetch_review_queue_images(conn, gallery_threshold=THRESHOLD)

    assert plan.favorite_count == 1
    assert plan.export_relevant_count == 1
    assert remaining == []


def test_execute_on_empty_queue_is_noop(output_dir):
    with closing_connection() as conn:
        seed_image(
            conn, "20260101_120002_c.jpg", output_dir, review_status="confirmed_bird"
        )
        result = review_cleanup_core.execute_plan(conn, gallery_threshold=THRESHOLD)

    assert result == {"images_moved": 0, "detections_moved": 0}
