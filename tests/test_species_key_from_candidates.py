"""Tests for utils.species_names.species_key_from_candidates.

This helper is the Python mirror of
utils.db.detections.effective_species_sql() and replaces 4+ hand-rolled
"bird"/"unknown"/"unclassified" blocklists across the codebase.
"""

from __future__ import annotations

import pytest

from utils.species_names import (
    UNKNOWN_SPECIES_KEY,
    is_non_species_od_token,
    species_key_from_candidates,
)


# ---------------------------------------------------------------------------
# is_non_species_od_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["bird", "BIRD", "Bird", "unknown", "Unclassified", "", "   ", None],
)
def test_is_non_species_od_token_rejects_placeholder_tokens(token):
    assert is_non_species_od_token(token) is True


@pytest.mark.parametrize(
    "token",
    ["squirrel", "cat", "marten_mustelid", "hedgehog", "Turdus_merula"],
)
def test_is_non_species_od_token_accepts_real_species(token):
    assert is_non_species_od_token(token) is False


# ---------------------------------------------------------------------------
# species_key_from_candidates — priority chain
# ---------------------------------------------------------------------------


def test_manual_override_wins_over_everything():
    assert (
        species_key_from_candidates(
            manual_override="Parus_major",
            cls_class_name="Cyanistes_caeruleus",
            od_class_name="bird",
            species_key="Something_else",
        )
        == "Parus_major"
    )


def test_species_key_wins_when_no_manual_override():
    assert (
        species_key_from_candidates(
            manual_override=None,
            species_key="Turdus_merula",
            cls_class_name="Parus_major",
            od_class_name="bird",
        )
        == "Turdus_merula"
    )


def test_cls_wins_when_no_species_key():
    assert (
        species_key_from_candidates(
            cls_class_name="Parus_major",
            od_class_name="bird",
        )
        == "Parus_major"
    )


# ---------------------------------------------------------------------------
# The critical bird/non-bird split
# ---------------------------------------------------------------------------


def test_bird_od_class_does_not_leak_as_species():
    """'bird' as od_class_name must not become species truth."""
    assert (
        species_key_from_candidates(
            cls_class_name=None,
            od_class_name="bird",
        )
        == UNKNOWN_SPECIES_KEY
    )


def test_bird_od_class_case_insensitive():
    assert (
        species_key_from_candidates(
            cls_class_name=None,
            od_class_name="BIRD",
        )
        == UNKNOWN_SPECIES_KEY
    )


@pytest.mark.parametrize(
    "garden_animal",
    ["squirrel", "cat", "marten_mustelid", "hedgehog"],
)
def test_non_bird_od_class_passes_through_as_species(garden_animal):
    """OD class names for garden animals ARE the species identity."""
    assert (
        species_key_from_candidates(
            cls_class_name=None,
            od_class_name=garden_animal,
        )
        == garden_animal
    )


def test_unknown_and_unclassified_are_also_rejected():
    assert (
        species_key_from_candidates(
            cls_class_name=None,
            od_class_name="unknown",
        )
        == UNKNOWN_SPECIES_KEY
    )
    assert (
        species_key_from_candidates(
            cls_class_name=None,
            od_class_name="unclassified",
        )
        == UNKNOWN_SPECIES_KEY
    )


def test_empty_candidates_return_unknown():
    assert species_key_from_candidates() == UNKNOWN_SPECIES_KEY


def test_whitespace_only_candidates_skipped():
    assert (
        species_key_from_candidates(
            manual_override="   ",
            cls_class_name="  ",
            species_key="",
            od_class_name="squirrel",
        )
        == "squirrel"
    )


def test_non_bird_od_with_failed_cls_still_uses_od():
    """Non-bird detection where CLS was skipped/failed: OD name IS species."""
    assert (
        species_key_from_candidates(
            manual_override=None,
            species_key=None,
            cls_class_name=None,  # CLS was skipped (non-bird track)
            od_class_name="squirrel",
        )
        == "squirrel"
    )
