"""Tests for the pure-Python biodiversity metric module."""

import math
from datetime import datetime, timedelta

import pytest

from core.biodiversity import (
    chao1_richness,
    diel_activity_curve,
    hill_numbers,
    observed_richness,
    pielou_evenness,
    relative_activity_index,
    sample_coverage,
    shannon_entropy,
    simpson_concentration,
    simpson_index,
    species_accumulation,
    species_event_counts,
    species_niche_pca,
)
from core.events import BirdEvent


def _ev(species, start="20260425_120000", count=1) -> BirdEvent:
    return BirdEvent(
        event_key=f"{species}_{start}",
        species=species,
        species_source="classifier",
        detection_ids=list(range(count)),
        photo_count=count,
        duration_sec=0.0,
        start_time=start,
        end_time=start,
        cover_detection_id=0,
        eligibility="event_eligible",
        fallback_reason=None,
        touched_filenames=[f"{i}.webp" for i in range(count)],
    )


# --- counts + richness ---------------------------------------------------------


def test_species_event_counts_drops_unknown():
    events = [_ev("A"), _ev("A"), _ev("B"), _ev(None)]
    assert species_event_counts(events) == {"A": 2, "B": 1}


def test_observed_richness_distinct_species():
    assert observed_richness([_ev("A"), _ev("A"), _ev("B")]) == 2
    assert observed_richness([]) == 0


# --- diversity indices ---------------------------------------------------------


def test_shannon_zero_on_empty():
    assert shannon_entropy({}) == 0.0


def test_shannon_zero_on_single_species():
    assert shannon_entropy({"A": 10}) == 0.0


def test_shannon_known_value():
    """Two species, equal counts → ln(2) ≈ 0.693."""
    assert shannon_entropy({"A": 5, "B": 5}) == pytest.approx(math.log(2), rel=1e-6)


def test_simpson_concentration_and_diversity():
    # Single species → concentration 1.0, diversity 0.0
    assert simpson_concentration({"A": 10}) == pytest.approx(1.0)
    assert simpson_index({"A": 10}) == pytest.approx(0.0)
    # Equal split → concentration 0.5, diversity 0.5
    assert simpson_concentration({"A": 5, "B": 5}) == pytest.approx(0.5)
    assert simpson_index({"A": 5, "B": 5}) == pytest.approx(0.5)


def test_pielou_evenness_perfect_split():
    assert pielou_evenness({"A": 5, "B": 5}) == pytest.approx(1.0)


def test_pielou_evenness_dominant_species():
    e = pielou_evenness({"A": 99, "B": 1})
    assert 0.0 < e < 0.2


# --- Hill numbers --------------------------------------------------------------


def test_hill_numbers_canonical_fixture():
    """[10, 5, 2, 1] (n=18) → q0=4, q1≈2.97, q2=1/Σpᵢ² ≈ 2.49."""
    counts = {"a": 10, "b": 5, "c": 2, "d": 1}
    h = hill_numbers(counts)
    # q=0 = richness
    assert h[0.0] == pytest.approx(4.0)
    # q=1 = exp(Shannon). Σ-p ln p with p=[10/18, 5/18, 2/18, 1/18]:
    #   H = 1.090 → exp(H) = 2.973
    assert h[1.0] == pytest.approx(2.973, abs=0.01)
    # q=2 = 1/Σpᵢ². Σpᵢ² = (100+25+4+1)/324 = 130/324 = 0.4012
    #   1/0.4012 = 2.492
    assert h[2.0] == pytest.approx(2.492, abs=0.01)


def test_hill_numbers_empty():
    h = hill_numbers({})
    assert h == {0.0: 0.0, 1.0: 0.0, 2.0: 0.0}


def test_hill_q0_equals_richness():
    counts = {"a": 1, "b": 1, "c": 1}
    h = hill_numbers(counts, q_values=(0.0,))
    assert h[0.0] == 3.0


# --- Chao1 ---------------------------------------------------------------------


def test_chao1_no_singletons_returns_observed():
    chao, se = chao1_richness({"a": 5, "b": 4, "c": 3})
    assert chao == 3.0
    assert se == 0.0


def test_chao1_with_singletons_lifts_estimate():
    """4 species observed (S_obs=4), 2 singletons (f1=2), 1 doubleton (f2=1)
    → chao1 = 4 + (2²)/(2·1) = 4 + 2 = 6.0."""
    chao, se = chao1_richness({"a": 5, "b": 1, "c": 1, "d": 2})
    assert chao == pytest.approx(6.0)
    assert se > 0.0


def test_chao1_empty():
    chao, se = chao1_richness({})
    assert chao == 0.0 and se == 0.0


# --- sample coverage -----------------------------------------------------------


def test_coverage_no_singletons_full():
    assert sample_coverage({"a": 5, "b": 5}) == pytest.approx(1.0)


def test_coverage_all_singletons_zero():
    assert sample_coverage({"a": 1, "b": 1, "c": 1}) == pytest.approx(0.0)


def test_coverage_mixed():
    # 1 singleton, 5 total individuals → 1 - 1/5 = 0.8
    assert sample_coverage({"a": 4, "b": 1}) == pytest.approx(0.8)


def test_coverage_empty():
    assert sample_coverage({}) == 0.0


# --- accumulation curve --------------------------------------------------------


def test_accumulation_monotonic_non_decreasing():
    events = [
        _ev("A", "20260420_120000"),
        _ev("B", "20260421_120000"),
        _ev("A", "20260422_120000"),  # already seen → no change
        _ev("C", "20260423_120000"),
    ]
    curve = species_accumulation(events)
    days = [d for d, _ in curve]
    cum = [c for _, c in curve]
    assert days == sorted(days)
    assert cum == sorted(cum), "accumulation curve must be non-decreasing"
    assert cum[-1] == 3
    assert cum[0] == 1


def test_accumulation_empty():
    assert species_accumulation([]) == []


# --- RAI -----------------------------------------------------------------------


def test_rai_zero_effort_raises():
    with pytest.raises(ValueError):
        relative_activity_index([_ev("A")], effort_days=0)


def test_rai_normal_effort():
    """5 events of A, 1 of B, over 10 effort-days → A=50, B=10 per 100 days."""
    events = [_ev("A") for _ in range(5)] + [_ev("B")]
    rai = relative_activity_index(events, effort_days=10.0)
    assert rai["A"] == pytest.approx(50.0)
    assert rai["B"] == pytest.approx(10.0)


# --- diel curve ----------------------------------------------------------------


def test_diel_curve_integrates_to_one():
    base = datetime(2026, 4, 25, 6, 0, 0)
    events = [
        _ev("A", (base + timedelta(hours=h)).strftime("%Y%m%d_%H%M%S"))
        for h in range(0, 12)
    ]
    curve = diel_activity_curve(events, samples=240)
    step = 24.0 / 240
    integral = sum(d for _, d in curve) * step
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_diel_curve_peak_near_synthetic_peak():
    """All events at 15:00 → peak near hour 15 in the density curve."""
    events = [_ev("A", "20260425_150000") for _ in range(50)]
    curve = diel_activity_curve(events, bandwidth_hours=0.5, samples=240)
    peak_hour = max(curve, key=lambda hd: hd[1])[0]
    assert 14.5 <= peak_hour <= 15.5


def test_diel_curve_empty():
    assert diel_activity_curve([]) == []


def test_diel_curve_species_filter():
    events = [_ev("A", "20260425_060000"), _ev("B", "20260425_180000")]
    curve_a = diel_activity_curve(events, species="A", samples=120)
    peak_a = max(curve_a, key=lambda hd: hd[1])[0]
    assert 5.0 <= peak_a <= 7.0


# --- Species niche PCA --------------------------------------------------------


def test_pca_empty_returns_not_ok():
    result = species_niche_pca([])
    assert result["ok"] is False
    assert result["species"] == []


def test_pca_single_species_returns_not_ok():
    """PCA needs ≥2 points to define an axis; single-species input degenerates."""
    events = [_ev("A", "20260425_060000")] * 5
    result = species_niche_pca(events)
    assert result["ok"] is False
    assert result["species"] == ["A"]
    assert result["coords"] == [[0.0, 0.0]]


def test_pca_two_species_separates_them():
    """Two species with disjoint diel windows must land at distinct coords."""
    morning = [_ev("Robin", f"20260425_{h:02d}0000") for h in (5, 6, 7, 8)]
    evening = [_ev("Owl", f"20260425_{h:02d}0000") for h in (19, 20, 21, 22)]
    result = species_niche_pca(morning + evening)
    assert result["ok"] is True
    assert sorted(result["species"]) == ["Owl", "Robin"]
    coords = result["coords"]
    # The two species must end up at different points on PC1.
    assert abs(coords[0][0] - coords[1][0]) > 0.01


def test_pca_peak_hours_match_distribution():
    morning = [_ev("Robin", "20260425_060000") for _ in range(10)]
    evening = [_ev("Owl", "20260425_200000") for _ in range(10)]
    result = species_niche_pca(morning + evening)
    sp_to_peak = dict(zip(result["species"], result["peak_hours"], strict=True))
    assert 5.0 <= sp_to_peak["Robin"] <= 7.0
    assert 19.0 <= sp_to_peak["Owl"] <= 21.0


def test_pca_event_counts_align_with_species_order():
    events = [_ev("A", "20260425_060000")] * 3 + [_ev("B", "20260425_180000")] * 7
    result = species_niche_pca(events)
    sp_to_count = dict(zip(result["species"], result["event_counts"], strict=True))
    assert sp_to_count["A"] == 3
    assert sp_to_count["B"] == 7


def test_pca_min_events_filter_drops_rare_species():
    rare = [_ev("Rare", "20260425_060000")]
    common_a = [_ev("CommonA", "20260425_060000")] * 5
    common_b = [_ev("CommonB", "20260425_200000")] * 5
    result = species_niche_pca(rare + common_a + common_b, min_events_per_species=3)
    assert "Rare" not in result["species"]
    assert sorted(result["species"]) == ["CommonA", "CommonB"]


def test_pca_variance_pct_sums_at_most_100():
    evs = []
    for sp, hour in [("A", 5), ("B", 10), ("C", 15), ("D", 20)]:
        evs.extend([_ev(sp, f"20260425_{hour:02d}0000") for _ in range(4)])
    result = species_niche_pca(evs)
    assert result["ok"] is True
    pc1, pc2 = result["variance_pct"]
    assert 0.0 <= pc1 <= 100.0
    assert 0.0 <= pc2 <= 100.0
    assert pc1 + pc2 <= 100.001  # tolerance for float rounding


def test_pca_handles_microseconds_suffix_in_start_time():
    """Production timestamps carry a `_microseconds` suffix.

    Regression: an earlier strptime("%Y%m%d_%H%M%S") call rejected the
    longer string, so every diel profile collapsed to zeros and PCA
    silently returned ok=False. Verify both formats parse and produce
    a working PCA.
    """
    morning_short = [_ev("Robin", f"20260425_06{m:02d}00") for m in (0, 5, 10)]
    morning_long = [_ev("Robin", f"20260425_06{m:02d}00_123456") for m in (15, 20, 25)]
    evening_long = [
        _ev("Owl", f"20260425_20{m:02d}00_654321") for m in (0, 5, 10, 15, 20, 25)
    ]
    result = species_niche_pca(morning_short + morning_long + evening_long)
    assert result["ok"] is True
    sp_to_peak = dict(zip(result["species"], result["peak_hours"], strict=True))
    assert 5.0 <= sp_to_peak["Robin"] <= 7.0
    assert 19.0 <= sp_to_peak["Owl"] <= 21.0


def test_diel_curve_handles_microseconds_suffix():
    """Diel KDE must also tolerate the production timestamp format."""
    events = [_ev("A", f"20260425_15{m:02d}00_999999") for m in (0, 10, 20, 30)]
    curve = diel_activity_curve(events, bandwidth_hours=0.5, samples=240)
    assert curve, "diel curve must not be empty when timestamps are valid"
    peak_hour = max(curve, key=lambda hd: hd[1])[0]
    assert 14.5 <= peak_hour <= 15.5


def test_accumulation_handles_microseconds_suffix():
    """species_accumulation must parse the long timestamp shape."""
    events = [
        _ev("A", "20260420_120000_111111"),
        _ev("B", "20260421_120000_222222"),
    ]
    curve = species_accumulation(events)
    assert len(curve) == 2
    assert curve[-1][1] == 2  # cumulative richness reaches 2
