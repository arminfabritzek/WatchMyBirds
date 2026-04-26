"""Pure-Python biodiversity metrics for the Insights dashboard.

Every function in this module takes events (list of `BirdEvent`) or species
counts (dict[str, int]) and returns primitives or dataclasses. **No DB
access. No I/O.** This keeps the metrics trivially unit-testable and lets
the Insights blueprint compose them without hitting SQLite per request.

Numerical conventions:
    - Shannon entropy uses the natural logarithm (matches the existing
      `_compute_biodiversity_indices` helper in `utils/db/analytics.py`).
    - Simpson is reported as 1 - Σpᵢ² (Gini-Simpson) for intuitive
      "diversity" reading; the raw concentration index is exposed under
      `simpson_concentration`.
    - Hill numbers: q=0 = richness, q=1 = exp(Shannon), q=2 = 1/Σpᵢ².

References:
    Hill 1973 — diversity as a unified family of indices.
    Jost 2006 — entropy vs. effective number of species.
    Chao 1984, 1987 — Chao1 richness estimator.
    Good 1953 — sample coverage (Good–Turing).
    Sollmann et al. — RAI is activity, not abundance.
    Ridout & Linkie 2009 — diel activity from camera-trap timestamps.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime

from core.events import BirdEvent


def _parse_event_start(ts: str | None) -> datetime | None:
    """Parse a BirdEvent.start_time leniently.

    Production timestamps come in two shapes:
        ``YYYYMMDD_HHMMSS``           (15 chars, legacy)
        ``YYYYMMDD_HHMMSS_microseconds`` (22+ chars, current pipeline)

    Anything beyond the second underscore is discarded, so both shapes
    parse cleanly.
    """
    if not ts:
        return None
    try:
        return datetime.strptime(ts[:15], "%Y%m%d_%H%M%S")
    except (ValueError, TypeError):
        return None


# --- core counting -------------------------------------------------------------


def species_event_counts(events: Iterable[BirdEvent]) -> dict[str, int]:
    """Number of events per resolved species. Unknown species are dropped."""
    counts: Counter[str] = Counter()
    for ev in events:
        if ev.species:
            counts[ev.species] += 1
    return dict(counts)


def observed_richness(events: Iterable[BirdEvent]) -> int:
    """Hill q=0 — the count of distinct species observed (naive richness)."""
    return len({ev.species for ev in events if ev.species})


# --- diversity indices ---------------------------------------------------------


def shannon_entropy(counts: Mapping[str, int]) -> float:
    """Shannon entropy H' (natural log)."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        h -= p * math.log(p)
    return h


def simpson_concentration(counts: Mapping[str, int]) -> float:
    """Σpᵢ² — concentration. Closer to 1 = more dominated."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum((n / total) ** 2 for n in counts.values() if n > 0)


def simpson_index(counts: Mapping[str, int]) -> float:
    """1 - Σpᵢ² (Gini-Simpson). Closer to 1 = more diverse."""
    return 1.0 - simpson_concentration(counts)


def pielou_evenness(counts: Mapping[str, int]) -> float:
    """Pielou's evenness J' = H' / ln(S). 1 = perfectly even."""
    s = sum(1 for n in counts.values() if n > 0)
    if s <= 1:
        return 1.0
    return shannon_entropy(counts) / math.log(s)


def hill_numbers(
    counts: Mapping[str, int],
    q_values: Iterable[float] = (0.0, 1.0, 2.0),
) -> dict[float, float]:
    """Hill numbers (Jost 2006).

    q=0 = richness, q=1 = exp(Shannon), q=2 = 1 / Σpᵢ². Returns one entry
    per q value in `q_values`.
    """
    total = sum(counts.values())
    out: dict[float, float] = {}
    if total <= 0:
        for q in q_values:
            out[float(q)] = 0.0
        return out

    proportions = [n / total for n in counts.values() if n > 0]
    for q in q_values:
        qf = float(q)
        if qf == 0.0:
            out[qf] = float(len(proportions))
        elif qf == 1.0:
            h = -sum(p * math.log(p) for p in proportions)
            out[qf] = math.exp(h)
        else:
            s = sum(p**qf for p in proportions)
            out[qf] = s ** (1.0 / (1.0 - qf)) if s > 0 else 0.0
    return out


def chao1_richness(counts: Mapping[str, int]) -> tuple[float, float]:
    """Chao1 lower-bound richness estimator + standard error.

    Chao 1984 closed-form for the bias-corrected estimator. Returns
    `(estimate, standard_error)`. With no singletons the estimate equals
    observed richness.
    """
    s_obs = sum(1 for n in counts.values() if n > 0)
    f1 = sum(1 for n in counts.values() if n == 1)
    f2 = sum(1 for n in counts.values() if n == 2)

    if s_obs == 0:
        return 0.0, 0.0
    if f1 == 0:
        return float(s_obs), 0.0

    if f2 > 0:
        chao = s_obs + (f1 * f1) / (2.0 * f2)
    else:
        chao = s_obs + (f1 * (f1 - 1)) / 2.0

    # Variance per Chao 1987 / Colwell 2013
    if f2 > 0:
        ratio = f1 / f2
        var = f2 * (0.5 * ratio**2 + ratio**3 + 0.25 * ratio**4)
    else:
        var = (
            0.5 * f1 * (f1 - 1)
            + 0.25 * f1 * (2 * f1 - 1) ** 2
            - 0.25 * f1**4 / max(s_obs, 1)
        )
    se = math.sqrt(max(var, 0.0))
    return chao, se


def sample_coverage(counts: Mapping[str, int]) -> float:
    """Good-Turing sample coverage estimator: 1 - f1 / n.

    Returns the estimated proportion of the local community already
    detected. Range [0, 1]. With no singletons → 1.0; with all
    singletons → 0.0.
    """
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    f1 = sum(1 for c in counts.values() if c == 1)
    return max(0.0, min(1.0, 1.0 - f1 / n))


# --- accumulation curve --------------------------------------------------------


def species_accumulation(events: Iterable[BirdEvent]) -> list[tuple[date, int]]:
    """Cumulative new species per calendar day, sorted ascending.

    Returns `[(day, cumulative_richness), ...]`. Days with no new species
    are still included so the curve has one point per active day.
    """
    by_day: dict[date, set[str]] = defaultdict(set)
    for ev in events:
        if not ev.species:
            continue
        dt = _parse_event_start(ev.start_time)
        if dt is None:
            continue
        by_day[dt.date()].add(ev.species)

    if not by_day:
        return []

    seen: set[str] = set()
    out: list[tuple[date, int]] = []
    for day in sorted(by_day.keys()):
        seen.update(by_day[day])
        out.append((day, len(seen)))
    return out


# --- activity rate -------------------------------------------------------------


def relative_activity_index(
    events: Iterable[BirdEvent],
    effort_days: float,
) -> dict[str, float]:
    """Events per species per 100 trap-days (RAI).

    **Caveat (Sollmann et al.):** RAI is a relative activity / encounter
    rate. It is NOT an abundance estimate. Detection probability varies
    by species, weather, and time-of-day, so equal RAI does not imply
    equal abundance. Always communicate as "activity rate" in the UI.

    Raises ValueError on non-positive effort to avoid div-by-zero.
    """
    if effort_days <= 0:
        raise ValueError("effort_days must be > 0 to compute RAI")
    counts = species_event_counts(events)
    return {species: round(100.0 * n / effort_days, 2) for species, n in counts.items()}


# --- diel activity -------------------------------------------------------------


def diel_activity_curve(
    events: Iterable[BirdEvent],
    *,
    species: str | None = None,
    bandwidth_hours: float = 1.0,
    samples: int = 144,
) -> list[tuple[float, float]]:
    """24h activity density via circular Gaussian KDE.

    Returns `[(hour_in_[0,24), density), ...]`. Density integrates to 1.0
    across the full 24h domain (within sampling tolerance). Wrapping is
    achieved by replicating events at ±24h before convolution.

    `species` filters to one species; `None` includes all events.
    `bandwidth_hours` controls the Gaussian sigma in hours.
    """
    if samples <= 0 or bandwidth_hours <= 0:
        return []

    hours: list[float] = []
    for ev in events:
        if species and ev.species != species:
            continue
        dt = _parse_event_start(ev.start_time)
        if dt is None:
            continue
        h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        hours.append(h)

    if not hours:
        return []

    xs = [24.0 * i / samples for i in range(samples)]
    bw = float(bandwidth_hours)
    densities: list[float] = []
    n = len(hours)
    inv_norm = 1.0 / (n * bw * math.sqrt(2.0 * math.pi))
    for x in xs:
        s = 0.0
        for h in hours:
            # Circular distance (smallest of |x-h|, 24-|x-h|).
            d = abs(x - h)
            d = min(d, 24.0 - d)
            s += math.exp(-0.5 * (d / bw) ** 2)
        densities.append(s * inv_norm)

    # Renormalise so the discrete trapezoidal integral over [0, 24) equals 1.
    step = 24.0 / samples
    integral = sum(densities) * step
    if integral > 0:
        densities = [d / integral for d in densities]

    return list(zip(xs, densities, strict=True))


# --- species niche PCA ---------------------------------------------------------


def _species_diel_profile(
    events: Iterable[BirdEvent],
    species: str,
    bins: int = 24,
) -> list[float]:
    """Hour-of-day histogram for one species, length=bins.

    Returns a probability vector that sums to 1.0 (or the zero vector if the
    species has no events). Used as the per-species feature vector for PCA.
    """
    hist = [0.0] * bins
    n = 0
    for ev in events:
        if ev.species != species:
            continue
        dt = _parse_event_start(ev.start_time)
        if dt is None:
            continue
        idx = int(dt.hour * bins / 24)
        if 0 <= idx < bins:
            hist[idx] += 1.0
            n += 1
    if n == 0:
        return hist
    return [x / n for x in hist]


def _peak_hour(profile: list[float]) -> float:
    """Argmax of a 24-bin diel profile, returned in hours [0, 24)."""
    if not profile:
        return 0.0
    bins = len(profile)
    idx = max(range(bins), key=lambda i: profile[i])
    return idx * 24.0 / bins


def species_niche_pca(
    events: Iterable[BirdEvent],
    *,
    bins: int = 24,
    min_events_per_species: int = 1,
) -> dict[str, object]:
    """PCA over species × diel-profile features.

    Each species becomes a point in 2D niche space. Position carries the
    24h activity profile; species with similar daily routines cluster.

    Returns a dict with:
        species:        list of species names (kept after min_events filter)
        coords:         list of [pc1, pc2] per species
        event_counts:   list of event counts per species (for marker size)
        peak_hours:     list of float hour-of-peak per species (for colour)
        variance_pct:   [pct1, pct2] explained variance for the two PCs
        ok:             True if PCA was computed; False on degenerate input
                        (fewer than 2 species, or all-zero feature matrix)

    Algorithm:
        1. Build species × bins matrix from per-species diel histograms.
        2. Drop species with fewer than `min_events_per_species` events.
        3. Mean-centre columns.
        4. SVD: U Σ Vᵀ. Project rows onto first two right singular vectors.
        5. Explained-variance ratio from squared singular values.

    No NumPy dependency in input/output, but uses NumPy internally.
    """
    import numpy as np

    events_list = list(events)
    counts = species_event_counts(events_list)
    species = sorted(
        s for s, n in counts.items() if n >= max(1, int(min_events_per_species))
    )

    if len(species) < 2:
        return {
            "species": species,
            "coords": [[0.0, 0.0] for _ in species],
            "event_counts": [counts.get(s, 0) for s in species],
            "peak_hours": [0.0 for _ in species],
            "variance_pct": [0.0, 0.0],
            "ok": False,
        }

    profiles = [_species_diel_profile(events_list, s, bins=bins) for s in species]
    matrix = np.array(profiles, dtype=float)  # shape: (n_species, bins)

    # If every row is all zeros (no timestamps anywhere), bail out gracefully.
    if not np.any(matrix):
        return {
            "species": species,
            "coords": [[0.0, 0.0] for _ in species],
            "event_counts": [counts.get(s, 0) for s in species],
            "peak_hours": [0.0 for _ in species],
            "variance_pct": [0.0, 0.0],
            "ok": False,
        }

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # full_matrices=False returns the economy SVD; faster and enough for PCA.
    _, sing, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T  # shape: (n_species, 2)

    total_var = float((sing**2).sum())
    if total_var > 0:
        variance_pct = [
            round(100.0 * (sing[0] ** 2) / total_var, 1),
            round(100.0 * (sing[1] ** 2) / total_var, 1),
        ]
    else:
        variance_pct = [0.0, 0.0]

    return {
        "species": species,
        "coords": [[float(c[0]), float(c[1])] for c in coords],
        "event_counts": [counts.get(s, 0) for s in species],
        "peak_hours": [_peak_hour(p) for p in profiles],
        "variance_pct": variance_pct,
        "ok": True,
    }


__all__ = [
    "chao1_richness",
    "diel_activity_curve",
    "hill_numbers",
    "observed_richness",
    "pielou_evenness",
    "relative_activity_index",
    "sample_coverage",
    "shannon_entropy",
    "simpson_concentration",
    "simpson_index",
    "species_accumulation",
    "species_event_counts",
    "species_niche_pca",
]
