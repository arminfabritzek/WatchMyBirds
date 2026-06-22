"""Review-cleanup Planner — read-only dry-run for "Move Review Queue to Trash".

build_plan(conn, gallery_threshold) partitions the LIVE review queue into
the two reversible-action buckets:
  - orphan/untagged images   -> image_filenames (-> review_status='no_bird')
  - active unresolved dets    -> detection_ids   (-> status='rejected')

The membership MUST equal fetch_review_queue_images() exactly (parity), and
the plan must surface favorite / export-relevant sub-counts. ZERO writes.
"""

import pytest

from core import review_cleanup_core
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection
from utils.db.review_queue import fetch_review_queue_images

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


def test_plan_membership_matches_review_queue_predicate(output_dir):
    # A mixed DB: a queued low-score detection, an orphan untagged image,
    # and a confirmed_bird image that must NOT be selected.
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_q.jpg", output_dir, review_status="untagged")
        seed_image(
            conn,
            "20260101_120001_o.jpg",
            output_dir,
            review_status="untagged",
            detections=0,
        )
        seed_image(
            conn, "20260101_120002_c.jpg", output_dir, review_status="confirmed_bird"
        )

        rows = fetch_review_queue_images(conn, gallery_threshold=THRESHOLD)
        plan = review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)

    expected_images = {r["item_id"] for r in rows if r["item_kind"] == "image"}
    expected_dets = {int(r["item_id"]) for r in rows if r["item_kind"] == "detection"}

    assert set(plan.image_filenames) == expected_images
    assert set(plan.detection_ids) == expected_dets
    # Confirmed image leaks into neither bucket.
    assert "20260101_120002_c.jpg" not in plan.image_filenames


def test_plan_counts_images_detections_events(output_dir):
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_q.jpg", output_dir, review_status="untagged")
        seed_image(
            conn,
            "20260101_120001_o.jpg",
            output_dir,
            review_status="untagged",
            detections=0,
        )
        plan = review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)

    assert len(plan.detection_ids) == 1
    assert len(plan.image_filenames) == 1
    assert plan.event_count >= 1


def test_plan_surfaces_favorite_and_export_relevant_counts(output_dir):
    with closing_connection() as conn:
        # A favorite is also export-relevant (favorites bucket is in the
        # export union), so this single image lifts both counts.
        seed_image(
            conn,
            "20260101_120000_f.jpg",
            output_dir,
            review_status="untagged",
            favorite=True,
        )
        plan = review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)

    assert plan.favorite_count == 1
    assert plan.export_relevant_count == 1
    # Still included in the action — counted, not protected.
    assert len(plan.detection_ids) == 1


def test_build_plan_performs_no_writes(output_dir):
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_q.jpg", output_dir, review_status="untagged")
        review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)
        # The queued image is untouched after planning.
        row = conn.execute(
            "SELECT review_status FROM images WHERE filename = ?",
            ("20260101_120000_q.jpg",),
        ).fetchone()
        det = conn.execute(
            "SELECT status FROM detections WHERE image_filename = ?",
            ("20260101_120000_q.jpg",),
        ).fetchone()

    assert row["review_status"] == "untagged"
    assert det["status"] == "active"


def test_empty_queue_yields_empty_plan(output_dir):
    with closing_connection() as conn:
        seed_image(
            conn, "20260101_120002_c.jpg", output_dir, review_status="confirmed_bird"
        )
        plan = review_cleanup_core.build_plan(conn, gallery_threshold=THRESHOLD)

    assert plan.image_filenames == []
    assert plan.detection_ids == []
    assert plan.event_count == 0
    assert plan.favorite_count == 0
    assert plan.export_relevant_count == 0
