"""Tests for gallery_core helpers.

Concurrent-visit grouping for the Subgallery keeps same-species
BirdEvents separate in the data layer and groups them only visually via
``group_concurrent_observations``. These tests pin the pure-function
semantics of that grouping so future pagination and sort changes can
rely on a stable contract.
"""

from __future__ import annotations

from core.gallery_core import group_concurrent_observations


def _obs(obs_id: int, start: str, end: str, species: str = "Pica_pica") -> dict:
    """Build a minimal observation dict for grouping tests."""
    return {
        "observation_id": obs_id,
        "species": species,
        "start_time": start,
        "end_time": end,
        "detection_ids": [obs_id],
        "photo_count": 1,
        "duration_sec": 0.0,
        "best_score": 0.9,
        "cover_detection_id": obs_id,
    }


def test_empty_input_returns_empty_list():
    assert group_concurrent_observations([]) == []


def test_single_observation_yields_one_visit_window_of_size_one():
    obs = _obs(1, "20260408_162000", "20260408_162030")
    result = group_concurrent_observations([obs])
    assert len(result) == 1
    assert result[0] == [obs]


def test_two_overlapping_observations_group_into_one_visit_window():
    a = _obs(1, "20260408_162000", "20260408_162200", species="Pica_pica")
    b = _obs(2, "20260408_162100", "20260408_162230", species="Parus_major")
    result = group_concurrent_observations([a, b])
    assert len(result) == 1
    assert len(result[0]) == 2
    # Same-species rule: both species stay as separate observations
    # inside the shared window.
    species_in_window = {obs["species"] for obs in result[0]}
    assert species_in_window == {"Pica_pica", "Parus_major"}


def test_far_apart_observations_become_separate_visit_windows():
    a = _obs(1, "20260408_120000", "20260408_120030")
    b = _obs(2, "20260408_163000", "20260408_163030")  # ~4h later
    result = group_concurrent_observations([a, b])
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1


def test_three_observations_two_concurrent_plus_one_solo():
    a = _obs(1, "20260408_162000", "20260408_162100", species="Pica_pica")
    b = _obs(2, "20260408_162030", "20260408_162130", species="Parus_major")
    c = _obs(3, "20260408_180000", "20260408_180100", species="Turdus_merula")
    result = group_concurrent_observations([a, b, c])
    assert len(result) == 2
    sizes = sorted(len(w) for w in result)
    assert sizes == [1, 2]


def test_tolerance_window_default_five_minutes():
    # Gap of 3 minutes → within 5-min tolerance → one window.
    a = _obs(1, "20260408_162000", "20260408_162030")
    b = _obs(2, "20260408_162330", "20260408_162400")
    result = group_concurrent_observations([a, b])
    assert len(result) == 1
    assert len(result[0]) == 2

    # Gap of 10 minutes → outside default tolerance → two windows.
    c = _obs(1, "20260408_162000", "20260408_162030")
    d = _obs(2, "20260408_163100", "20260408_163130")
    result = group_concurrent_observations([c, d])
    assert len(result) == 2


def test_tolerance_window_is_configurable():
    a = _obs(1, "20260408_162000", "20260408_162030")
    b = _obs(2, "20260408_162900", "20260408_162930")  # 8.5 min gap
    # Default tolerance (5 min) → two windows.
    assert len(group_concurrent_observations([a, b])) == 2
    # Widened tolerance (10 min) → one window.
    result = group_concurrent_observations([a, b], window_minutes=10)
    assert len(result) == 1
    assert len(result[0]) == 2


def test_grouping_is_deterministic_regardless_of_input_order():
    a = _obs(1, "20260408_162000", "20260408_162100", species="Pica_pica")
    b = _obs(2, "20260408_162030", "20260408_162130", species="Parus_major")
    c = _obs(3, "20260408_180000", "20260408_180100", species="Turdus_merula")

    forward = group_concurrent_observations([a, b, c])
    reversed_ = group_concurrent_observations([c, b, a])

    # Same bucketing regardless of input order — the helper sorts
    # internally before walking.
    def _snapshot(windows):
        return [[obs["observation_id"] for obs in w] for w in windows]

    assert _snapshot(forward) == _snapshot(reversed_)


def test_does_not_mutate_input_list():
    a = _obs(1, "20260408_162000", "20260408_162100")
    b = _obs(2, "20260408_180000", "20260408_180100")
    original = [a, b]
    snapshot = list(original)
    group_concurrent_observations(original)
    assert original == snapshot
