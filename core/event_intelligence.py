"""Local event-intelligence rules for biological BirdEvent grouping.

This module is intentionally small and deterministic. It gives the event
builder species-aware defaults while keeping the 30 minute camera-trap window
as the fallback for unknown or unprofiled species.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EventGroupingProfile:
    """Event-boundary and sampling defaults for one behaviour class."""

    name: str
    gap_minutes: float
    max_duration_minutes: float
    max_representative_images: int = 12


DEFAULT_EVENT_PROFILE = EventGroupingProfile(
    name="default_30m",
    gap_minutes=30.0,
    max_duration_minutes=60.0,
    max_representative_images=12,
)

SHORT_VISIT_PROFILE = EventGroupingProfile(
    name="short_station_visit",
    gap_minutes=12.0,
    max_duration_minutes=30.0,
    max_representative_images=10,
)

FLOCK_BURST_PROFILE = EventGroupingProfile(
    name="flock_burst",
    gap_minutes=10.0,
    max_duration_minutes=20.0,
    max_representative_images=12,
)

SLOW_VISITOR_PROFILE = EventGroupingProfile(
    name="slow_station_visit",
    gap_minutes=20.0,
    max_duration_minutes=45.0,
    max_representative_images=12,
)


# Small tits and similar garden regulars usually make short, repeated feeder
# visits. A 30 minute gap hides those returns as one event on fixed cameras.
SHORT_VISIT_SPECIES = frozenset(
    {
        "Aegithalos_caudatus",
        "Cyanistes_caeruleus",
        "Lophophanes_cristatus",
        "Parus_major",
        "Periparus_ater",
        "Poecile_montanus",
        "Poecile_palustris",
    }
)

# Species that often generate dense burst sequences at feeders. The shorter
# cap keeps Review from receiving one giant event when many near-identical
# frames arrive over hours.
FLOCK_BURST_SPECIES = frozenset(
    {
        "Carduelis_carduelis",
        "Chloris_chloris",
        "Emberiza_citrinella",
        "Fringilla_coelebs",
        "Fringilla_montifringilla",
        "Passer_domesticus",
        "Passer_montanus",
        "Spinus_spinus",
        "Sturnus_vulgaris",
    }
)

# Larger garden visitors tend to linger or move slowly, so they get a longer
# per-visit window than flock bursts without falling back to unlimited events.
SLOW_VISITOR_SPECIES = frozenset(
    {
        "Columba_palumbus",
        "Corvus_corone",
        "Corvus_monedula",
        "Garrulus_glandarius",
        "Pica_pica",
        "Streptopelia_decaocto",
    }
)


def normalize_species_key(species_key: str | None) -> str:
    """Return the canonical-ish key used by WMB species profiles."""

    return str(species_key or "").strip()


def resolve_event_profile(
    species_key: str | None,
    *,
    overrides: Mapping[str, EventGroupingProfile] | None = None,
) -> EventGroupingProfile:
    """Resolve the event grouping profile for a species key."""

    key = normalize_species_key(species_key)
    if overrides and key in overrides:
        return overrides[key]
    if key in SHORT_VISIT_SPECIES:
        return SHORT_VISIT_PROFILE
    if key in FLOCK_BURST_SPECIES:
        return FLOCK_BURST_PROFILE
    if key in SLOW_VISITOR_SPECIES:
        return SLOW_VISITOR_PROFILE
    return DEFAULT_EVENT_PROFILE


def representative_image_budget(
    photo_count: int,
    *,
    profile: EventGroupingProfile | None = None,
    rare_bonus: int = 0,
    uncertainty_bonus: int = 0,
) -> int:
    """Compute how many full images are worth keeping for one event.

    The formula keeps a few anchors plus logarithmic growth, so a huge burst
    gets more evidence than a three-frame event without scaling linearly.
    """

    count = max(0, int(photo_count or 0))
    if count <= 0:
        return 0

    active_profile = profile or DEFAULT_EVENT_PROFILE
    base = 3 + math.ceil(math.log2(max(count, 1)))
    budget = base + max(0, int(rare_bonus)) + max(0, int(uncertainty_bonus))
    return min(
        count, max(1, min(int(active_profile.max_representative_images), budget))
    )
