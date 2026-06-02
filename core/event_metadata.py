"""Canonical metadata envelope for image-export burn-in.

A single source of truth for the species/location/provenance facts that
get serialised into XMP (and, partially, EXIF) when a human downloads an
image. This is a *derived view* of current DB state at download time —
nothing here is ever read back into the DB.

The envelope is pure ``core/``: no Flask, no detector imports (H-02).
It serialises to the XMP packet via :func:`utils.image_ops.build_xmp_packet`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bumped whenever the field set or serialisation changes in a way a
# downstream reader would need to know about. Carried into the wmb:
# namespace so a re-imported WMB copy can tell which envelope produced it.
# v2 (2026-06-02): added image-wide rating/favorite/review fields and the
#   xmp:Rating / xmp:Label / wmb:* mapping for them.
METADATA_SCHEMA_VERSION = 2

# dwc constants are gated on bird detections only; a squirrel/marten must
# never receive Aves/Animalia. Non-bird species carry no dwc class block.
_BIRD_DWC_CLASS = "Aves"
_BIRD_DWC_KINGDOM = "Animalia"


@dataclass(frozen=True)
class SpeciesEntry:
    """One detected species on a frame.

    ``scientific`` is the bare binomial with an underscore replaced by a
    space and no authorship token — the form iNaturalist matches on
    (plan Finding 1). ``common`` is the localized display name.
    """

    scientific: str
    common: str
    is_bird: bool
    detection_id: int | None = None
    confidence: float | None = None
    is_favorite: bool = False
    rating: int | None = None

    @property
    def genus(self) -> str:
        """First token of the binomial (``""`` if not resolvable)."""
        return self.scientific.split(" ", 1)[0] if self.scientific else ""

    @property
    def dwc_class(self) -> str | None:
        return _BIRD_DWC_CLASS if self.is_bird else None

    @property
    def dwc_kingdom(self) -> str | None:
        return _BIRD_DWC_KINGDOM if self.is_bird else None

    def caption(self) -> str:
        """Human caption ``"Common (Scientific)"`` for ``dc:description``.

        Falls back to whichever half is present; empty string if neither.
        """
        common = (self.common or "").strip()
        scientific = (self.scientific or "").strip()
        if common and scientific:
            return f"{common} ({scientific})"
        return common or scientific


@dataclass(frozen=True)
class EventMetadata:
    """All metadata burned into one exported image copy.

    A frame may carry multiple detections, so ``species`` is a list —
    ``dc:subject`` becomes an ``rdf:Bag`` with one entry per species name
    (common + scientific), and the per-detection ``wmb:`` block is a list.
    There is no one-bird-per-image assumption.
    """

    species: list[SpeciesEntry] = field(default_factory=list)
    detector_model: str | None = None
    classifier_model: str | None = None
    schema_version: int = METADATA_SCHEMA_VERSION
    # Image-wide (not per-detection) facts. A JPEG carries one rating/label,
    # so these aggregate across the frame's detections.
    creator_tool: str | None = None
    creator: str | None = None
    review_status: str | None = None

    @property
    def has_content(self) -> bool:
        """True when there is at least one resolvable species to burn in."""
        return any(s.scientific or s.common for s in self.species)

    @property
    def is_favorite(self) -> bool:
        """True when any detection on this frame is a manual favorite."""
        return any(s.is_favorite for s in self.species)

    def xmp_rating(self) -> int | None:
        """Image-wide ``xmp:Rating`` (0-5) for foto tools.

        A favorite anywhere on the frame pins the image to 5 stars; otherwise
        the highest manual/auto star rating across detections wins (one JPEG,
        one rating). ``None`` when nothing rates the image, so no tag is
        written rather than a misleading 0.
        """
        if self.is_favorite:
            return 5
        ratings = [s.rating for s in self.species if s.rating]
        return max(ratings) if ratings else None

    def xmp_label(self) -> str | None:
        """Image-wide ``xmp:Label`` — ``"Favorite"`` when starred, else None."""
        return "Favorite" if self.is_favorite else None

    def subject_keywords(self) -> list[str]:
        """Flat, de-duplicated keyword list for ``dc:subject`` (rdf:Bag).

        Each species contributes both its scientific binomial (the iNat
        match key) and its common name. Order is preserved; duplicates and
        empties are dropped.
        """
        keywords: list[str] = []
        seen: set[str] = set()
        for entry in self.species:
            for term in (entry.scientific, entry.common):
                term = (term or "").strip()
                if term and term not in seen:
                    seen.add(term)
                    keywords.append(term)
        return keywords

    def primary_caption(self) -> str:
        """Caption for ``dc:description`` — the first species with content."""
        for entry in self.species:
            caption = entry.caption()
            if caption:
                return caption
        return ""
