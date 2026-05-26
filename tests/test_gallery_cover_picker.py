"""Verify the shared cover picker honours the PTZ bias.

Until 2026-05-18 the Today's-Visitors-Summary tile row, the Species tab
and the subgallery's Species-of-Day section ran through closures inside
``web.web_interface.start_web_app`` that had no PTZ awareness. Those
closures are now thin wrappers around
``core.gallery_core.pick_cover_for_group`` and
``core.gallery_core.cover_quality_tuple``, so the four gallery surfaces
share one ranking DNA with the SQL pickers
(``fetch_daily_covers``, ``fetch_species_story_board_candidates``,
``_fetch_species_best_photos``). These tests pin that DNA.
"""

from __future__ import annotations

import random

from core.gallery_core import (
    cover_quality_tuple,
    is_favorite,
    is_gallery_eligible,
    is_ptz_preset,
    pick_cover_for_group,
)


def _det(**overrides) -> dict:
    """Build a baseline detection dict; override any field per test."""
    base = {
        "detection_id": 1,
        "is_favorite": 0,
        "is_gallery_eligible": 0,
        "ptz_origin": None,
        "aesthetic_score": None,
        "score": 0.8,
        "bbox_quality": 0.5,
        "bbox_x": 0.3,
        "bbox_y": 0.3,
        "bbox_w": 0.3,
        "bbox_h": 0.3,
        "image_timestamp": "20260518_120000",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def test_is_ptz_preset_matches_preset_and_manual_drive():
    assert is_ptz_preset(_det(ptz_origin="preset"))
    assert is_ptz_preset(_det(ptz_origin="manual_drive"))


def test_is_ptz_preset_rejects_overview_and_null():
    assert not is_ptz_preset(_det(ptz_origin="overview"))
    assert not is_ptz_preset(_det(ptz_origin=None))
    assert not is_ptz_preset(_det())  # field missing entirely


# ---------------------------------------------------------------------------
# cover_quality_tuple — sort key for the Today's-Visitors-Summary row.
# The tuple is compared element-wise; higher tuple → higher rank.
# ---------------------------------------------------------------------------


def test_cover_quality_favorite_beats_everything():
    fav = _det(detection_id=1, is_favorite=1, ptz_origin="overview", score=0.5)
    ptz_ki = _det(
        detection_id=2,
        is_favorite=0,
        is_gallery_eligible=1,
        ptz_origin="preset",
        aesthetic_score=0.99,
        score=0.99,
    )
    assert cover_quality_tuple(fav) > cover_quality_tuple(ptz_ki)


def test_cover_quality_ptz_beats_ki_only():
    """PTZ-preset (slot #2) outranks gallery-eligible (slot #3)."""
    ptz_only = _det(detection_id=1, ptz_origin="preset", is_gallery_eligible=0)
    ki_only = _det(detection_id=2, ptz_origin="overview", is_gallery_eligible=1)
    assert cover_quality_tuple(ptz_only) > cover_quality_tuple(ki_only)


def test_cover_quality_manual_drive_treated_as_preset():
    auto = _det(detection_id=1, ptz_origin="preset")
    manual = _det(detection_id=2, ptz_origin="manual_drive")
    # The PTZ slot (index 1) must be identical for both PTZ origins.
    assert cover_quality_tuple(auto)[1] == cover_quality_tuple(manual)[1] == 1


def test_cover_quality_null_ptz_origin_safe():
    """Legacy rows / non-PTZ cams have NULL ptz_origin — must not throw."""
    null_det = _det(ptz_origin=None)
    key = cover_quality_tuple(null_det)
    assert key[1] == 0  # PTZ slot defaults to 0


def test_cover_quality_tuple_length_matches_story_board_dna():
    """The Today's-Visitors-Summary key must have the same shape as the
    Best-of-Species story-board key so the two ranking paths can be
    reasoned about together. See ``_story_board_candidate_quality``."""
    from core.gallery_core import _story_board_candidate_quality

    sample = _det(ptz_origin="preset", is_gallery_eligible=1)
    assert len(cover_quality_tuple(sample)) == len(
        _story_board_candidate_quality(sample)
    )


# ---------------------------------------------------------------------------
# pick_cover_for_group — five-tier selection used by Species tab,
# Today's-Visitors-Summary, and subgallery Species-of-Day.
# ---------------------------------------------------------------------------


def test_pick_cover_empty_returns_none():
    assert pick_cover_for_group([]) is None


def test_pick_cover_human_favorite_wins_over_ptz_preset():
    """A manual favorite must never be hijacked by automation — including a
    physically-closer PTZ preset frame."""
    fav = _det(detection_id=1, is_favorite=1, ptz_origin="overview")
    preset = _det(
        detection_id=2, is_favorite=0, ptz_origin="preset", is_gallery_eligible=1
    )
    rng = random.Random(0)  # deterministic for tests
    picked = pick_cover_for_group([preset, fav], rng=rng)
    assert picked is fav


def test_pick_cover_ptz_ki_intersection_wins_over_plain_ptz():
    """When PTZ-preset frames exist AND one of them is also AI-approved,
    the intersection should win — strongest non-human signal."""
    ptz_plain = _det(
        detection_id=1, ptz_origin="preset", is_gallery_eligible=0
    )
    ptz_ki = _det(
        detection_id=2, ptz_origin="manual_drive", is_gallery_eligible=1
    )
    rng = random.Random(0)
    picked = pick_cover_for_group([ptz_plain, ptz_ki], rng=rng)
    assert picked is ptz_ki


def test_pick_cover_plain_ptz_wins_over_ki_only():
    """No PTZ-∩-AI frame available, but a plain PTZ-preset frame should
    still outrank a pure aesthetic-tagger pick — close-up beats generic."""
    ptz_only = _det(
        detection_id=1, ptz_origin="preset", is_gallery_eligible=0
    )
    ki_only = _det(
        detection_id=2, ptz_origin="overview", is_gallery_eligible=1
    )
    rng = random.Random(0)
    picked = pick_cover_for_group([ki_only, ptz_only], rng=rng)
    assert picked is ptz_only


def test_pick_cover_ki_wins_over_interior_fallback():
    """No favorite, no PTZ — an AI pick beats an interior-only fallback."""
    ki = _det(detection_id=1, is_gallery_eligible=1)
    plain_interior = _det(detection_id=2)  # interior, not AI-approved
    rng = random.Random(0)
    picked = pick_cover_for_group([plain_interior, ki], rng=rng)
    assert picked is ki


def test_pick_cover_score_fallback_for_legacy_rows():
    """All else equal (no favorite, no PTZ, no AI flag), the highest-score
    interior row wins. Mirrors the legacy fallback behaviour."""
    weak = _det(detection_id=1, score=0.5)
    strong = _det(detection_id=2, score=0.95)
    rng = random.Random(0)
    picked = pick_cover_for_group([weak, strong], rng=rng)
    assert picked is strong


def test_pick_cover_random_choice_within_tier_is_actually_random():
    """Two equally-qualified PTZ frames should both be selectable across
    multiple calls — the picker isn't accidentally deterministic in tier."""
    a = _det(detection_id=1, ptz_origin="preset", is_gallery_eligible=1)
    b = _det(detection_id=2, ptz_origin="preset", is_gallery_eligible=1)
    seen_ids: set[int] = set()
    for seed in range(20):
        rng = random.Random(seed)
        picked = pick_cover_for_group([a, b], rng=rng)
        assert picked is not None
        seen_ids.add(picked["detection_id"])
    # With 20 seeds, both detection IDs should appear at least once.
    assert seen_ids == {1, 2}


# ---------------------------------------------------------------------------
# Convenience predicates (sanity)
# ---------------------------------------------------------------------------


def test_is_favorite_handles_truthy_and_zero():
    assert is_favorite(_det(is_favorite=1))
    assert not is_favorite(_det(is_favorite=0))
    assert not is_favorite(_det(is_favorite=None))


def test_is_gallery_eligible_handles_truthy_and_zero():
    assert is_gallery_eligible(_det(is_gallery_eligible=1))
    assert not is_gallery_eligible(_det(is_gallery_eligible=0))
    assert not is_gallery_eligible(_det(is_gallery_eligible=None))
