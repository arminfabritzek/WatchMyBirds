"""Verify the gallery cover ranker prefers PTZ-preset frames.

Two layers:
1. SQL: fetch_daily_covers picks the preset row over an overview row
   when all other quality fields tie.
2. Python: _story_board_candidate_quality returns a tuple that ranks
   preset rows above overview/NULL rows but BELOW HUMAN favorites.
"""

import pytest

from core.gallery_core import _story_board_candidate_quality
from utils.db.connection import closing_connection
from utils.db.detections import fetch_daily_covers


@pytest.fixture(autouse=True)
def isolate_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def _seed_image(conn, filename: str, ptz_origin: str | None) -> None:
    conn.execute(
        "INSERT INTO images (filename, timestamp, source_id, ptz_origin, "
        "review_status) VALUES (?, ?, ?, ?, 'confirmed_bird');",
        (filename, filename, 1, ptz_origin),
    )


def _seed_detection(
    conn,
    image_filename: str,
    *,
    score: float = 0.9,
    aesthetic_score: float | None = None,
    rating: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, score, aesthetic_score, rating,
            status, decision_state, created_at
        ) VALUES (?, 0.1, 0.1, 0.3, 0.3, 'bird', ?, ?, ?, 'active', 'confirmed', ?);
        """,
        (image_filename, score, aesthetic_score, rating, image_filename),
    )
    conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# SQL ranker
# ---------------------------------------------------------------------------


def test_fetch_daily_covers_prefers_preset_row_over_overview_row():
    with closing_connection() as conn:
        # Two frames on the same day; same detector score; one captured
        # at preset, one at overview. The preset one should win the
        # daily cover slot.
        _seed_image(conn, "20260515_120000_000000.jpg", "overview")
        _seed_image(conn, "20260515_120100_000000.jpg", "preset")
        det_overview = _seed_detection(
            conn, "20260515_120000_000000.jpg", score=0.9
        )
        det_preset = _seed_detection(
            conn, "20260515_120100_000000.jpg", score=0.9
        )

        covers = fetch_daily_covers(conn)

        # One cover per day, and the day's cover must be the preset one.
        assert len(covers) == 1
        cover = covers[0]
        assert cover["detection_id"] == det_preset, (
            f"Expected preset detection {det_preset} to win cover, "
            f"got {cover['detection_id']} (overview detection was {det_overview})."
        )


def test_fetch_daily_covers_rating_still_wins_over_preset():
    """HUMAN rating outranks PTZ preset bias — the auto-picker never
    overrides a HUMAN choice."""
    with closing_connection() as conn:
        # Overview row has manual 5-star rating; preset row has none.
        _seed_image(conn, "20260516_120000_000000.jpg", "overview")
        _seed_image(conn, "20260516_120100_000000.jpg", "preset")
        det_rated = _seed_detection(
            conn, "20260516_120000_000000.jpg", score=0.9, rating=5
        )
        _seed_detection(conn, "20260516_120100_000000.jpg", score=0.9)

        covers = fetch_daily_covers(conn)

        assert len(covers) == 1
        assert covers[0]["detection_id"] == det_rated


def test_fetch_daily_covers_handles_null_ptz_origin_as_overview_equivalent():
    """Legacy rows with NULL ptz_origin must not throw and rank below preset."""
    with closing_connection() as conn:
        _seed_image(conn, "20260517_120000_000000.jpg", None)  # legacy
        _seed_image(conn, "20260517_120100_000000.jpg", "preset")
        _seed_detection(conn, "20260517_120000_000000.jpg", score=0.9)
        det_preset = _seed_detection(
            conn, "20260517_120100_000000.jpg", score=0.9
        )

        covers = fetch_daily_covers(conn)

        assert len(covers) == 1
        assert covers[0]["detection_id"] == det_preset


# ---------------------------------------------------------------------------
# Python ranker (story-board candidate quality tuple)
# ---------------------------------------------------------------------------


def test_story_board_quality_preset_beats_overview_at_equal_aesthetic():
    preset_det = {
        "is_favorite": 0,
        "is_gallery_eligible": 0,
        "bbox_x": 0.3,
        "bbox_y": 0.3,
        "bbox_w": 0.4,
        "bbox_h": 0.4,
        "ptz_origin": "preset",
        "aesthetic_score": 0.5,
        "score": 0.8,
        "bbox_quality": 0.7,
        "image_timestamp": "20260515_120000",
        "detection_id": 100,
    }
    overview_det = {**preset_det, "ptz_origin": "overview", "detection_id": 101}

    # Higher tuple = better. preset row must rank higher.
    assert _story_board_candidate_quality(preset_det) > _story_board_candidate_quality(
        overview_det
    )


def test_story_board_quality_favorite_beats_preset():
    """is_favorite is the first tuple element — HUMAN choice wins."""
    favorite_overview_det = {
        "is_favorite": 1,
        "is_gallery_eligible": 0,
        "bbox_x": 0.3,
        "bbox_y": 0.3,
        "bbox_w": 0.4,
        "bbox_h": 0.4,
        "ptz_origin": "overview",
        "aesthetic_score": 0.3,  # worse aesthetic, still wins on priority
        "score": 0.8,
        "bbox_quality": 0.7,
        "image_timestamp": "20260515_120000",
        "detection_id": 200,
    }
    plain_preset_det = {**favorite_overview_det, "is_favorite": 0, "ptz_origin": "preset", "detection_id": 201}

    assert _story_board_candidate_quality(
        favorite_overview_det
    ) > _story_board_candidate_quality(plain_preset_det)


def test_story_board_quality_manual_drive_treated_as_preset():
    """A future manual_drive marker (from the user-manual-control plan)
    must rank the same as auto preset — both mean 'operator-targeted'."""
    auto_preset = {
        "is_favorite": 0,
        "is_gallery_eligible": 0,
        "bbox_x": 0.3,
        "bbox_y": 0.3,
        "bbox_w": 0.4,
        "bbox_h": 0.4,
        "ptz_origin": "preset",
        "aesthetic_score": 0.5,
        "score": 0.8,
        "bbox_quality": 0.7,
        "image_timestamp": "20260515_120000",
        "detection_id": 300,
    }
    manual_drive = {**auto_preset, "ptz_origin": "manual_drive", "detection_id": 301}

    # Slice 1 of the gallery ranker treats both equally.
    auto_key = _story_board_candidate_quality(auto_preset)
    manual_key = _story_board_candidate_quality(manual_drive)
    # The PTZ-preset slot (index 1, between is_favorite and is_gallery_eligible)
    # must be identical for both PTZ origins.
    assert auto_key[1] == manual_key[1] == 1


def test_story_board_quality_null_ptz_does_not_throw():
    null_det = {
        "is_favorite": 0,
        "is_gallery_eligible": 0,
        "bbox_x": 0.3,
        "bbox_y": 0.3,
        "bbox_w": 0.4,
        "bbox_h": 0.4,
        # ptz_origin missing entirely — legacy/ingest row
        "aesthetic_score": 0.5,
        "score": 0.8,
        "bbox_quality": 0.7,
        "image_timestamp": "20260515_120000",
        "detection_id": 400,
    }

    key = _story_board_candidate_quality(null_det)
    # PTZ slot (index 1) should default to 0 (legacy treated as non-preset).
    assert key[1] == 0
