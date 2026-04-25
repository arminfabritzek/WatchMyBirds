"""BirdEvent aggregation layer.

This module is the single biological aggregation unit used across the
Review surface, the Gallery sub-gallery, and the planned ``/insights``
dashboard. An *event* is a run of detections of the same species whose
timestamps fit the active event profile (30 minutes for unknown/unprofiled
species, shorter windows for station-regular or burst-prone species).

Why profiles, with a 30 minute fallback
---------------------------------------
Camera-trap research commonly uses independence windows on the order of
30 minutes to avoid inflating metrics with burst-mode serial
photographs. Treating many near-identical frames as independent
observations would bias downstream event, activity, and occupancy-style
metrics. Fixed feeder cameras also produce very species-specific local
patterns, so known short-visit species and dense feeder bursts use tighter
profiles while the 30 minute rule remains the conservative fallback.

Migration history
-----------------
This module was introduced as a reusable event library and later wired
into the Review blueprint and templates. The legacy
``core/review_core.py`` cluster builder was retired, and
``core/gallery_core.group_detections_into_observations`` now delegates
here while preserving its legacy dict shape for gallery, stream, and
analytics callers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.event_intelligence import (
    DEFAULT_EVENT_PROFILE,
    EventGroupingProfile,
    representative_image_budget,
    resolve_event_profile,
)
from core.gallery_core import _bbox_dist, _ts_to_epoch

EVENT_GAP_MINUTES_DEFAULT = 30
"""Default independence window in minutes. Documented in the module docstring."""

EVENT_MAX_DURATION_MINUTES_DEFAULT = DEFAULT_EVENT_PROFILE.max_duration_minutes
"""Default safety cap for a single Event when no species profile is available."""

# Maximum normalised start-to-end bbox displacement inside a single
# event. Anything above this is flagged ``bbox_jump``; the event is
# still kept as one group so the reviewer sees the full run.
_EVENT_MAX_SPAN_DIST = 0.42

_UNKNOWN_SPECIES_TOKENS = frozenset(
    {
        "",
        "bird",
        "unknown",
        "unknown_species",
        "unclassified",
    }
)
# Kept as a superset of utils.species_names._NON_SPECIES_OD_TOKENS so that
# "Unknown_species" (the literal UNKNOWN_SPECIES_KEY) is also rejected as
# event species. Sync any changes with utils.species_names.is_non_species_od_token.


@dataclass(frozen=True)
class BirdEvent:
    """A single biological event: one species, gap <= ``gap_minutes``.

    Event instances are frozen so they can be cached, shared between
    surfaces, and passed into pure metric functions in
    ``core/biodiversity.py`` without the risk of downstream mutation.

    Fields
    ------
    event_key
        Stable string identifier derived from species, start time, and
        cover detection id. Safe to use as a DOM id or cache key.
    species
        Resolved species, or ``None`` if the event is fully unknown.
    species_source
        ``"manual"`` when at least one member carries a manual
        override, ``"classifier"`` when the species comes from the
        classifier chain, ``"unknown"`` otherwise.
    detection_ids
        Detection ids in chronological order. Always non-empty.
    photo_count
        Number of detections in the event.
    duration_sec
        Seconds between the first and last detection. ``0.0`` for
        single-detection events.
    start_time / end_time
        ``YYYYMMDD_HHMMSS`` strings of the first and last member.
    cover_detection_id
        Detection id chosen to represent the event (the newest member,
        matching the existing review ordering convention.
    eligibility
        ``"event_eligible"`` when the event is a clean single-species
        run without ambiguity, ``"event_ineligible"`` otherwise.
    fallback_reason
        ``None`` when eligible. Otherwise one of ``"unknown_species"``,
        ``"partial_unknown_species"``, ``"multi_bird_ambiguity"``,
        ``"bbox_jump"``.
    touched_filenames
        Unique filenames of source images covered by this event, in
        chronological order, used by the Review image-visibility
        recompute and by the review-template filmstrip.
    bbox_trail
        Per-detection bbox snapshots with percentage coordinates and a
        ``trail_role`` label (``start`` / ``mid`` / ``step`` / ``end``)
        consumed by the motion-trail UI. Each entry also
        carries a ``context_only`` flag so the Review template can mute
        the matching cell when the bbox belongs to a Gallery anchor.
    context_only_count
        Number of members that came from
        ``fetch_review_cluster_context`` (i.e. confirmed-bird Gallery
        anchors that the Review surface treats as read-only). ``0`` for
        events that are pure-untagged. The Review blueprint drops events
        where ``context_only_count == photo_count`` because there is
        nothing left to review.
    context_anchored
        ``True`` when the event mixes at least one untagged member with
        at least one Gallery context member. Informational signal for
        the Review surface to render the "Connected to N confirmed
        frames already in the Gallery" banner. Does **not** change
        ``eligibility`` — context anchoring is a continuation hint, not
        a disqualifying reason.
    grouping_profile / event_gap_minutes / max_duration_minutes
        The behaviour profile that shaped the event boundary. Unknown or
        unprofiled species keep the 30 minute fallback; known short-visit
        or flocking species may use tighter local rules.
    representative_image_count
        Suggested number of full-size event images worth keeping once a
        retention layer exists. Metadata and thumbnails can still remain.
    """

    event_key: str
    species: str | None
    species_source: str
    detection_ids: list[int]
    photo_count: int
    duration_sec: float
    start_time: str
    end_time: str
    cover_detection_id: int
    eligibility: str
    fallback_reason: str | None
    touched_filenames: list[str]
    bbox_trail: list[dict[str, Any]] = field(default_factory=list)
    context_only_count: int = 0
    context_anchored: bool = False
    source_id: str | None = None
    grouping_profile: str = DEFAULT_EVENT_PROFILE.name
    event_gap_minutes: float = EVENT_GAP_MINUTES_DEFAULT
    max_duration_minutes: float = EVENT_MAX_DURATION_MINUTES_DEFAULT
    representative_image_count: int = 0

    @property
    def is_eligible(self) -> bool:
        return self.eligibility == "event_eligible"


def _normalize_species(value: Any) -> str | None:
    species = str(value or "").strip()
    if not species or species.lower() in _UNKNOWN_SPECIES_TOKENS:
        return None
    return species


def _resolve_detection_species(det: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve a detection's species via the manual → curated key → classifier chain.

    The order matches the existing gallery / stream / analytics
    resolution: a manual override always wins, then the curated
    ``species_key`` (post-processed canonical species), then the raw
    classifier output, then the object-detector class as a last resort.
    """
    manual_species = _normalize_species(det.get("manual_species_override"))
    if manual_species:
        return manual_species, "manual"

    species_source = str(det.get("species_source") or "").strip().lower()
    species_key = _normalize_species(det.get("species_key"))
    if species_key and species_source != "manual":
        return species_key, "classifier"

    cls_species = _normalize_species(det.get("cls_class_name"))
    if cls_species:
        return cls_species, "classifier"

    # Non-bird OD classes (squirrel, cat, etc.) pass through
    # _normalize_species because they are NOT in _UNKNOWN_SPECIES_TOKENS.
    # Their species identity IS the OD class name; mark the source as
    # "detector" (not "classifier") so downstream UI can distinguish
    # OD-assigned non-bird species from CLS-assigned bird species.
    od_species = _normalize_species(det.get("od_class_name"))
    if od_species:
        return od_species, "detector"

    return None, "unknown"


def _build_bbox_trail(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trail: list[dict[str, Any]] = []
    total = len(members)
    preview_indexes = {0, max(total // 2, 0), max(total - 1, 0)}
    for index, member in enumerate(members):
        bbox_x = float(member.get("bbox_x") or 0.0)
        bbox_y = float(member.get("bbox_y") or 0.0)
        bbox_w = float(member.get("bbox_w") or 0.0)
        bbox_h = float(member.get("bbox_h") or 0.0)
        if index == 0:
            role = "start"
        elif index == total - 1:
            role = "end"
        elif index in preview_indexes:
            role = "mid"
        else:
            role = "step"
        trail.append(
            {
                "detection_id": int(member.get("detection_id") or 0),
                "timestamp": member.get("timestamp") or "",
                "filename": member.get("filename") or "",
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "bbox_x_pct": round(bbox_x * 100, 2),
                "bbox_y_pct": round(bbox_y * 100, 2),
                "bbox_w_pct": round(bbox_w * 100, 2),
                "bbox_h_pct": round(bbox_h * 100, 2),
                "center_x_pct": round((bbox_x + (bbox_w / 2.0)) * 100, 2),
                "center_y_pct": round((bbox_y + (bbox_h / 2.0)) * 100, 2),
                "trail_role": role,
                "context_only": bool(member.get("context_only") or False),
            }
        )
    return trail


def _resolve_event_species(
    members: list[dict[str, Any]],
) -> tuple[str | None, str, str | None]:
    """Return ``(species, species_source, fallback_reason_or_None)``.

    Reads the already-resolved ``species_resolved`` / ``species_source``
    projections that ``_collect_items`` cached on each member, so we
    do not pay for ``_resolve_detection_species`` twice per detection.

    Strict rule: an event must carry exactly one non-unknown species.
    Everything else becomes an ineligibility reason.
    """
    resolved_species: set[str] = set()
    manual_count = 0
    classifier_count = 0
    unknown_count = 0

    for member in members:
        species = member.get("species_resolved")
        source = member.get("species_source") or "unknown"
        if source == "manual":
            manual_count += 1
        elif source == "classifier":
            classifier_count += 1
        else:
            unknown_count += 1
        if species:
            resolved_species.add(species)

    total = len(members)
    if unknown_count == total:
        return None, "unknown", "unknown_species"
    if unknown_count > 0:
        # Keep the known species label for display, but mark the event
        # ineligible so approval stays blocked.
        single_species = (
            next(iter(resolved_species)) if len(resolved_species) == 1 else None
        )
        source = "manual" if manual_count else "classifier"
        return single_species, source, "partial_unknown_species"
    if len(resolved_species) != 1:
        # Strict same-species split means this branch should never be
        # reached in the normal grouping path, but we keep the guard so
        # a future caller that feeds pre-grouped detections gets a
        # clean failure mode.
        return None, "classifier", "multi_bird_ambiguity"

    species = next(iter(resolved_species))
    source = "manual" if manual_count else "classifier"
    return species, source, None


def _detect_bbox_jump(members: list[dict[str, Any]]) -> bool:
    if len(members) < 2:
        return False
    first = members[0]
    last = members[-1]
    span = _bbox_dist(
        first["bbox_x"],
        first["bbox_y"],
        first["bbox_w"],
        first["bbox_h"],
        last["bbox_x"],
        last["bbox_y"],
        last["bbox_w"],
        last["bbox_h"],
    )
    return span > _EVENT_MAX_SPAN_DIST


def _detect_multi_bird(members: list[dict[str, Any]]) -> bool:
    return any(
        int(member.get("sibling_detection_count") or 0) > 1 for member in members
    )


def _build_event_key(
    species: str | None, start_time: str, cover_detection_id: int
) -> str:
    species_token = species or "unknown"
    digest = hashlib.blake2b(
        f"{species_token}|{start_time}|{cover_detection_id}".encode(),
        digest_size=6,
    ).hexdigest()
    return f"bird-event-{digest}"


def _normalize_source_id(value: Any) -> str | None:
    source = str(value or "").strip()
    return source or None


def _group_key_for_item(item: dict[str, Any]) -> tuple[str | None, str | None]:
    return (_normalize_source_id(item.get("source_id")), item.get("species_resolved"))


def _collect_items(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the minimum state the grouping pass needs.

    Only fields actually consumed downstream are kept. Raw detection
    rows are intentionally not preserved - the public BirdEvent
    contract carries detection ids plus the resolved projections, and
    callers who need the raw row should look it up via the DB layer
    instead of holding it in memory for every frame. If this code ever
    needs to support very large queries, the per-detection dicts can be
    replaced with a slotted dataclass as a later optimization.
    """
    items: list[dict[str, Any]] = []
    for det in detections:
        detection_id = int(
            det.get("detection_id")
            or det.get("active_detection_id")
            or det.get("best_detection_id")
            or 0
        )
        if detection_id <= 0:
            continue
        timestamp = str(det.get("timestamp") or det.get("image_timestamp") or "")
        species, source = _resolve_detection_species(det)
        items.append(
            {
                "detection_id": detection_id,
                "filename": str(det.get("filename") or det.get("image_filename") or ""),
                "timestamp": timestamp,
                "epoch": _ts_to_epoch(timestamp),
                "bbox_x": float(det.get("bbox_x") or 0.0),
                "bbox_y": float(det.get("bbox_y") or 0.0),
                "bbox_w": float(det.get("bbox_w") or 0.0),
                "bbox_h": float(det.get("bbox_h") or 0.0),
                "species_resolved": species,
                "species_source": source,
                "source_id": _normalize_source_id(det.get("source_id")),
                "sibling_detection_count": int(det.get("sibling_detection_count") or 0),
                "context_only": bool(det.get("context_only") or False),
            }
        )
    return items


def build_bird_events(
    detections: list[dict[str, Any]],
    *,
    gap_minutes: float = EVENT_GAP_MINUTES_DEFAULT,
    max_duration_minutes: float | None = None,
    profile_overrides: Mapping[str, EventGroupingProfile] | None = None,
) -> list[BirdEvent]:
    """Group detections into biological events.

    Contract
    --------
    - Same-species split. Detections with a resolved species only group
      with detections of that same species. Unknown-species detections
      attach to the nearest open known-species group inside the window
      (preserving the ``partial_unknown_species`` fallback semantics);
      they only form their own group when no such open group exists, in
      which case the resulting event is flagged ``unknown_species`` and
      always ineligible.
    - ``gap_minutes`` controls the independence window. The default
      is species-aware: unprofiled species use the 30 minute fallback,
      while known short-visit or flocking species use tighter windows.
      Passing a non-default ``gap_minutes`` overrides every profile.
    - ``max_duration_minutes`` caps a single event even when detections
      keep arriving continuously. This prevents one six-hour burst from
      becoming one unreviewable event. Passing ``None`` uses the species
      profile's cap.
    - Within a species + window group, spatial proximity is still used
      to flag bbox jumps via ``fallback_reason == "bbox_jump"``, but the
      detections are not re-split. This keeps the event intact while
      still signaling that the reviewer should take a closer look.
    - Output is sorted newest-first (by ``end_time`` descending, ties
      broken by highest ``cover_detection_id``) to match the existing
      Bulk Review browser order.
    """
    if not detections:
        return []

    items = _collect_items(detections)
    if not items:
        return []

    use_profile_gap = float(gap_minutes) == float(EVENT_GAP_MINUTES_DEFAULT)

    def _profile_for_species(species_key: str | None) -> EventGroupingProfile:
        return resolve_event_profile(species_key, overrides=profile_overrides)

    def _effective_gap_minutes(profile: EventGroupingProfile) -> float:
        return profile.gap_minutes if use_profile_gap else float(gap_minutes)

    def _effective_max_minutes(profile: EventGroupingProfile) -> float:
        if max_duration_minutes is not None:
            return float(max_duration_minutes)
        return profile.max_duration_minutes

    items.sort(key=lambda item: (item["epoch"], item["detection_id"]))

    # Groups are keyed by (source_id, species_resolved or None) so the
    # same-species rule is enforced at split time and different fixed
    # camera sources can never accidentally merge when callers provide
    # ``source_id``. Unknown detections only form their own group when
    # there is no open known-species group from the same source within
    # that known group's event profile.
    group_key_t = tuple[str | None, str | None]
    open_groups: dict[group_key_t, list[dict[str, Any]]] = {}
    # Track the last-attached epoch per group separately from the
    # member list. This is what the close-pass tests against, so the
    # close-pass stays correct regardless of whether the most recent
    # attach was an unknown-species redirect or a normal append.
    open_last_epoch: dict[group_key_t, float] = {}
    open_start_epoch: dict[group_key_t, float] = {}
    open_profiles: dict[group_key_t, EventGroupingProfile] = {}
    closed_groups: list[list[dict[str, Any]]] = []

    def _finalize(open_key: group_key_t) -> None:
        group = open_groups.pop(open_key, None)
        open_last_epoch.pop(open_key, None)
        open_start_epoch.pop(open_key, None)
        open_profiles.pop(open_key, None)
        if group:
            closed_groups.append(group)

    def _can_attach(open_key: group_key_t, epoch: float) -> bool:
        profile = open_profiles[open_key]
        max_gap_sec = max(_effective_gap_minutes(profile) * 60.0, 0.0)
        max_duration_sec = max(_effective_max_minutes(profile) * 60.0, 0.0)
        if epoch - open_last_epoch[open_key] > max_gap_sec:
            return False
        return epoch - open_start_epoch[open_key] <= max_duration_sec

    for item in items:
        species_key = item["species_resolved"]
        epoch = item["epoch"]
        source_id = item.get("source_id")

        # Close any open group whose last attached member is beyond
        # the window relative to *this* item, or whose total event span
        # would exceed its profile's max duration if this item joined.
        for open_key in list(open_groups.keys()):
            if not _can_attach(open_key, epoch):
                _finalize(open_key)

        attach_key: group_key_t = _group_key_for_item(item)
        if species_key is None:
            # Unknown detection: prefer adopting an open known-species
            # group from the same source whose last attached member is
            # still inside that known group's profile. Pick the group
            # with the most recent activity to mirror how a human
            # reviewer would interpret the burst.
            best_known_key: group_key_t | None = None
            best_known_epoch: float = -1.0
            for open_key in open_groups:
                open_source_id, open_species = open_key
                if open_source_id != source_id or open_species is None:
                    continue
                last_epoch = open_last_epoch[open_key]
                if _can_attach(open_key, epoch) and last_epoch > best_known_epoch:
                    best_known_epoch = last_epoch
                    best_known_key = open_key
            if best_known_key is not None:
                attach_key = best_known_key

        group = open_groups.get(attach_key)
        if group is None:
            open_groups[attach_key] = [item]
            open_start_epoch[attach_key] = epoch
            open_profiles[attach_key] = _profile_for_species(attach_key[1])
        else:
            group.append(item)
        open_last_epoch[attach_key] = epoch

    for open_key in list(open_groups.keys()):
        _finalize(open_key)

    events: list[BirdEvent] = []
    for members in closed_groups:
        members.sort(key=lambda item: (item["epoch"], item["detection_id"]))
        trail = _build_bbox_trail(members)
        species, source, species_reason = _resolve_event_species(members)

        fallback_reason: str | None = species_reason
        if fallback_reason is None and _detect_multi_bird(members):
            fallback_reason = "multi_bird_ambiguity"
        if fallback_reason is None and _detect_bbox_jump(members):
            fallback_reason = "bbox_jump"

        eligibility = (
            "event_eligible" if fallback_reason is None else "event_ineligible"
        )

        start_time = members[0]["timestamp"]
        end_time = members[-1]["timestamp"]
        # Prefer the newest actionable (non-context) member as cover so the
        # Review surface always gets a frame the operator can act on. Falls
        # back to the newest member overall when every member is context.
        actionable_members = [
            member for member in members if not member.get("context_only")
        ]
        cover_source = actionable_members[-1] if actionable_members else members[-1]
        cover_detection_id = cover_source["detection_id"]
        duration_sec = max(
            round(members[-1]["epoch"] - members[0]["epoch"], 1),
            0.0,
        )

        event_key = _build_event_key(species, start_time, cover_detection_id)

        detection_ids = [member["detection_id"] for member in members]
        touched_filenames = list(
            dict.fromkeys(
                member["filename"] for member in members if member["filename"]
            )
        )

        context_only_count = sum(1 for member in members if member.get("context_only"))
        actionable_count = len(members) - context_only_count
        context_anchored = context_only_count > 0 and actionable_count > 0
        source_id = _normalize_source_id(members[0].get("source_id"))
        profile = _profile_for_species(species)
        event_gap_minutes = _effective_gap_minutes(profile)
        event_max_minutes = _effective_max_minutes(profile)
        representative_count = representative_image_budget(
            len(members),
            profile=profile,
            uncertainty_bonus=4 if fallback_reason is not None else 0,
        )

        events.append(
            BirdEvent(
                event_key=event_key,
                species=species,
                species_source=source,
                detection_ids=detection_ids,
                photo_count=len(members),
                duration_sec=duration_sec,
                start_time=start_time,
                end_time=end_time,
                cover_detection_id=cover_detection_id,
                eligibility=eligibility,
                fallback_reason=fallback_reason,
                touched_filenames=touched_filenames,
                bbox_trail=trail,
                context_only_count=context_only_count,
                context_anchored=context_anchored,
                source_id=source_id,
                grouping_profile=profile.name,
                event_gap_minutes=event_gap_minutes,
                max_duration_minutes=event_max_minutes,
                representative_image_count=representative_count,
            )
        )

    events.sort(
        key=lambda event: (event.end_time, event.cover_detection_id),
        reverse=True,
    )
    return events
