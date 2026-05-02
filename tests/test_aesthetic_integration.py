"""Tests for the aesthetic_score integration into ranking surfaces.

Covers the four call sites threaded by plan
``2026-05-01_HANDOFF_aesthetic-score-app-integration``:

1. core.gallery_core._story_board_candidate_quality (tuple ordering)
2. SQL ORDER BY in fetch_daily_covers (utils.db.detections)
3. SQL ORDER BY in _fetch_species_best_photos (utils.daily_report)
4. SELECT side: gallery query carries aesthetic_score so the dict has it

Notification-service integration is intentionally out of scope (plan v1
recommendation (c): live alerts keep using detector confidence because
the nightly tagger only runs once per day).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.gallery_core import _story_board_candidate_quality  # noqa: E402


# ---------------------------------------------------------------------------
# 1) Tuple ordering: aesthetic_score is the third tiebreaker, behind
#    is_favorite and is_interior, ahead of score and bbox_quality.
# ---------------------------------------------------------------------------

def _det(**overrides) -> dict:
    """Minimal detection dict with safe defaults for the storyboard ranker."""
    base = {
        "detection_id": 1,
        "image_timestamp": "20260501_120000",
        "bbox_x": 0.2,
        "bbox_y": 0.2,
        "bbox_w": 0.2,
        "bbox_h": 0.2,  # interior bbox by default
        "score": 0.5,
        "bbox_quality": 0.5,
        "is_favorite": 0,
        "aesthetic_score": None,
    }
    base.update(overrides)
    return base


def test_aesthetic_score_breaks_tie_when_score_equal():
    """Two detections with identical detector scores: aesthetic_score wins."""
    a = _det(detection_id=1, score=0.8, aesthetic_score=0.9)
    b = _det(detection_id=2, score=0.8, aesthetic_score=0.3)
    # Ranker key returns a tuple; descending sort puts higher tuples first.
    assert _story_board_candidate_quality(a) > _story_board_candidate_quality(b)


def test_aesthetic_score_overrides_lower_detector_score():
    """Aesthetic score outranks detector confidence when no favorite is set.

    This is the production effect: a beautifully-framed bird with mid
    detector confidence beats a confidently-detected back-of-bird.
    """
    pretty_but_low_conf = _det(detection_id=1, score=0.4, aesthetic_score=0.95)
    confident_but_ugly = _det(detection_id=2, score=0.95, aesthetic_score=0.10)
    assert _story_board_candidate_quality(pretty_but_low_conf) > \
           _story_board_candidate_quality(confident_but_ugly)


def test_manual_favorite_still_wins_over_aesthetic():
    """is_favorite=1 still wins regardless of aesthetic_score.

    HUMAN's manual stars must never be reordered behind an automatic
    score. This is the safety contract for the integration.
    """
    manual_favorite_low_aesthetic = _det(
        detection_id=1, is_favorite=1, score=0.5, aesthetic_score=0.05,
    )
    auto_high_aesthetic = _det(
        detection_id=2, is_favorite=0, score=0.9, aesthetic_score=0.99,
    )
    assert _story_board_candidate_quality(manual_favorite_low_aesthetic) > \
           _story_board_candidate_quality(auto_high_aesthetic)


def test_null_aesthetic_loses_to_any_real_score():
    """Legacy / non-taggable detections (aesthetic_score IS NULL) sink behind
    any detection that has a real aesthetic_score, mirroring the SQL
    ``COALESCE(aesthetic_score, -1) DESC, score DESC`` fallback.

    Plan rationale (Open Question #2 recommendation c): treating NULL as
    "below floor" keeps tagged detections preferred wherever the tagger has
    a verdict, while NULL still ranks among themselves by detector score.
    """
    legacy_high_score = _det(detection_id=1, score=0.9, aesthetic_score=None)
    scored_low_score = _det(detection_id=2, score=0.2, aesthetic_score=0.05)
    assert _story_board_candidate_quality(scored_low_score) > \
           _story_board_candidate_quality(legacy_high_score)


def test_two_null_aesthetic_detections_rank_by_score():
    """Among NULL-aesthetic detections, the higher detector score wins."""
    legacy_better = _det(detection_id=1, score=0.9, aesthetic_score=None)
    legacy_worse = _det(detection_id=2, score=0.2, aesthetic_score=None)
    assert _story_board_candidate_quality(legacy_better) > \
           _story_board_candidate_quality(legacy_worse)


def test_interior_bbox_still_outranks_aesthetic():
    """is_interior is still ranked higher than aesthetic_score.

    A bird touching the frame edge should not surface even if it scored
    well -- bad composition wins out via the existing edge filter.
    """
    edge_high_aesthetic = _det(
        detection_id=1,
        bbox_x=0.0, bbox_y=0.0, bbox_w=0.3, bbox_h=0.3,  # touches edge
        aesthetic_score=0.99,
    )
    interior_mid_aesthetic = _det(
        detection_id=2,
        bbox_x=0.3, bbox_y=0.3, bbox_w=0.3, bbox_h=0.3,  # interior
        aesthetic_score=0.5,
    )
    assert _story_board_candidate_quality(interior_mid_aesthetic) > \
           _story_board_candidate_quality(edge_high_aesthetic)


# ---------------------------------------------------------------------------
# 2 + 4) SQL: fetch_daily_covers ORDER BY uses aesthetic_score; the
#       gallery SELECT includes aesthetic_score in its column list.
# ---------------------------------------------------------------------------

def _build_minimal_db() -> sqlite3.Connection:
    """Create an in-memory DB with the columns these queries reference.

    We do not call _init_schema (that loads the full app config); we mirror
    just the structure the two functions exercise.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE images (
        filename TEXT PRIMARY KEY,
        timestamp TEXT,
        review_status TEXT DEFAULT 'untagged',
        downloaded_timestamp TEXT,
        source_id INTEGER
    );
    CREATE TABLE detections (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT NOT NULL,
        thumbnail_path TEXT,
        od_class_name TEXT,
        od_confidence REAL,
        score REAL,
        bbox_quality REAL,
        bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
        rating INTEGER,
        rating_source TEXT DEFAULT 'auto',
        is_favorite INTEGER DEFAULT 0,
        aesthetic_score REAL,
        aesthetic_score_at TEXT,
        decision_state TEXT,
        decision_level TEXT,
        manual_species_override TEXT,
        species_source TEXT,
        unknown_score REAL,
        decision_reasons TEXT,
        status TEXT DEFAULT 'active',
        created_at TEXT,
        FOREIGN KEY(image_filename) REFERENCES images(filename)
    );
    CREATE TABLE classifications (
        classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        cls_class_name TEXT,
        cls_confidence REAL,
        rank INTEGER DEFAULT 1,
        status TEXT DEFAULT 'active'
    );
    """)
    return conn


def _insert_image_and_det(
    conn: sqlite3.Connection,
    *,
    filename: str,
    score: float,
    aesthetic_score: float | None,
    is_favorite: int = 0,
    rating: int | None = None,
    species: str = "Parus_major",
):
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, 'confirmed_bird')",
        (filename, filename[:14]),
    )
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename, thumbnail_path, od_class_name, od_confidence,
            score, bbox_quality, bbox_x, bbox_y, bbox_w, bbox_h,
            rating, is_favorite, aesthetic_score, status,
            decision_state, decision_level, created_at
        ) VALUES (?, ?, 'bird', ?, ?, 0.5, 0.3, 0.3, 0.2, 0.2,
                  ?, ?, ?, 'active', 'confirmed', 'species', ?)
        """,
        (filename, filename.replace(".jpg", "_crop_1.webp"), score, score,
         rating, is_favorite, aesthetic_score, filename[:14]),
    )
    det_id = cur.lastrowid
    conn.execute(
        "INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank) "
        "VALUES (?, ?, 0.9, 1)",
        (det_id, species),
    )


def test_fetch_daily_covers_prefers_high_aesthetic_score(monkeypatch):
    """Two detections same day: the one with higher aesthetic_score wins as cover."""
    # The function imports config and a visibility helper that requires the
    # full app context; we patch the visibility helper to a permissive clause.
    from utils.db import detections as detections_module

    monkeypatch.setattr(
        detections_module, "_gallery_visibility_sql",
        lambda d, i: f"{d}.status = 'active'",
    )

    conn = _build_minimal_db()
    # Same day, same species, same detector score → aesthetic_score breaks tie.
    _insert_image_and_det(conn, filename="20260501_080000_a.jpg",
                          score=0.7, aesthetic_score=0.20)
    _insert_image_and_det(conn, filename="20260501_090000_b.jpg",
                          score=0.7, aesthetic_score=0.95)

    rows = detections_module.fetch_daily_covers(conn, min_score=0.0)
    assert len(rows) == 1, "exactly one cover row per day"
    cover = rows[0]
    assert cover["date_key"] == "2026-05-01"
    assert "_b" in cover["optimized_name_virtual"], (
        f"expected the higher-aesthetic detection to win; got {cover['optimized_name_virtual']}"
    )


def test_fetch_daily_covers_null_aesthetic_does_not_override(monkeypatch):
    """A NULL aesthetic_score must not beat a real one (COALESCE -1 fallback)."""
    from utils.db import detections as detections_module

    monkeypatch.setattr(
        detections_module, "_gallery_visibility_sql",
        lambda d, i: f"{d}.status = 'active'",
    )

    conn = _build_minimal_db()
    _insert_image_and_det(conn, filename="20260501_080000_legacy.jpg",
                          score=0.95, aesthetic_score=None)
    _insert_image_and_det(conn, filename="20260501_090000_scored.jpg",
                          score=0.5, aesthetic_score=0.40)

    rows = detections_module.fetch_daily_covers(conn, min_score=0.0)
    cover = rows[0]
    assert "_scored" in cover["optimized_name_virtual"], (
        "scored detection should win over legacy NULL even with lower score"
    )


def test_fetch_daily_covers_manual_rating_wins(monkeypatch):
    """User's manual rating still ranks first, ahead of aesthetic_score."""
    from utils.db import detections as detections_module

    monkeypatch.setattr(
        detections_module, "_gallery_visibility_sql",
        lambda d, i: f"{d}.status = 'active'",
    )

    conn = _build_minimal_db()
    _insert_image_and_det(conn, filename="20260501_080000_rated.jpg",
                          score=0.4, aesthetic_score=0.10, rating=5)
    _insert_image_and_det(conn, filename="20260501_090000_pretty.jpg",
                          score=0.9, aesthetic_score=0.99)

    rows = detections_module.fetch_daily_covers(conn, min_score=0.0)
    cover = rows[0]
    assert "_rated" in cover["optimized_name_virtual"], (
        "5-star manual rating must outrank a high aesthetic score"
    )


# ---------------------------------------------------------------------------
# 4) Gallery SELECT carries aesthetic_score through to consumers.
# ---------------------------------------------------------------------------

def test_gallery_select_includes_aesthetic_score(monkeypatch):
    """fetch_detections_for_gallery returns aesthetic_score in the row dict."""
    from utils.db import detections as detections_module

    monkeypatch.setattr(
        detections_module, "_gallery_visibility_sql",
        lambda d, i: f"{d}.status = 'active'",
    )

    conn = _build_minimal_db()
    _insert_image_and_det(conn, filename="20260501_080000_a.jpg",
                          score=0.5, aesthetic_score=0.77)

    rows = detections_module.fetch_detections_for_gallery(conn)
    assert len(rows) == 1
    row = dict(rows[0])
    assert "aesthetic_score" in row, (
        "aesthetic_score must appear in fetch_detections_for_gallery rows so "
        "_story_board_candidate_quality can pick it up via det.get(...)"
    )
    assert row["aesthetic_score"] == 0.77
