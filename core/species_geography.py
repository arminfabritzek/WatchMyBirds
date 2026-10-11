"""Rank image candidates using independent geographic occurrence scores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeographicCandidate:
    species: str
    image_score: float
    occurrence_score: float | None
    weighted_score: float


@dataclass(frozen=True)
class GeographicAssessment:
    raw_species: str
    suggested_species: str
    ranking: tuple[GeographicCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_geographically(
    candidates: Sequence[tuple[str, float]],
    occurrence: Mapping[str, float] | None = None,
) -> GeographicAssessment:
    """Compare original candidates; weighted scores are not probabilities."""
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Expected distinct, nonempty image candidates")
    if (
        any(
            not name or not math.isfinite(score) or not 0 <= score <= 1
            for name, score in candidates
        )
        or not 0 < sum(score for _, score in candidates) <= 1.02
    ):
        raise ValueError("Invalid image scores")
    ranked = []
    for name, score in candidates:
        prior = occurrence.get(name) if occurrence is not None else None
        if prior is not None and (not math.isfinite(prior) or not 0 <= prior <= 1):
            raise ValueError("Invalid occurrence score")
        # A finite floor preserves rare candidates; unmapped taxonomy is neutral.
        weight = math.sqrt(max(0.001, prior)) if prior is not None else 1.0
        ranked.append(GeographicCandidate(name, score, prior, score * weight))
    ranked.sort(key=lambda candidate: candidate.weighted_score, reverse=True)
    return GeographicAssessment(candidates[0][0], ranked[0].species, tuple(ranked))
