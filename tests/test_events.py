"""Tests for the BirdEvent layer in core.events.

These tests describe the contract that ``build_bird_events`` must satisfy.
The Event layer sits above the existing 60 s frame-cluster layer and
matches the biological independence rule used by camera-trap research:
same species, profile-aware gap, and a 30 minute fallback.

The tests intentionally do not cover Review-blueprint or template
behaviour. That surface is covered separately by review-specific tests.
"""

from __future__ import annotations

from core.event_intelligence import EventGroupingProfile, representative_image_budget
from core.events import (
    EVENT_GAP_MINUTES_DEFAULT,
    EVENT_MAX_DURATION_MINUTES_DEFAULT,
    BirdEvent,
    build_bird_events,
)


def _det(
    detection_id: int,
    timestamp: str,
    *,
    bbox: tuple[float, float, float, float] = (0.20, 0.20, 0.18, 0.18),
    cls_species: str | None = "Cyanistes_caeruleus",
    manual_species: str | None = None,
    sibling_detection_count: int = 1,
) -> dict:
    """Build a synthetic detection row matching the shape Review uses."""
    return {
        "detection_id": detection_id,
        "active_detection_id": detection_id,
        "filename": f"frame-{detection_id}.jpg",
        "timestamp": timestamp,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "cls_class_name": cls_species,
        "manual_species_override": manual_species,
        "species_key": manual_species or cls_species,
        "species_source": "manual" if manual_species else "classifier",
        "sibling_detection_count": sibling_detection_count,
    }


def test_default_gap_is_thirty_minutes():
    """The independence rule is exactly 30 minutes."""
    assert EVENT_GAP_MINUTES_DEFAULT == 30


def test_empty_input_returns_empty_list():
    assert build_bird_events([]) == []


def test_burst_collapses_into_single_event():
    """5 serial frames within 12 s of the same bird sitting still are one event."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000"),
            _det(2, "20260407_080003"),
            _det(3, "20260407_080006"),
            _det(4, "20260407_080009"),
            _det(5, "20260407_080012"),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.photo_count == 5
    assert event.detection_ids == [1, 2, 3, 4, 5]
    assert event.species == "Cyanistes_caeruleus"
    assert event.eligibility == "event_eligible"
    assert event.fallback_reason is None
    assert event.duration_sec == 12.0
    assert event.start_time == "20260407_080000"
    assert event.end_time == "20260407_080012"
    assert event.cover_detection_id == 5


def test_thirty_minute_gap_keeps_events_together():
    """A 25-minute gap is still inside the fallback event window."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Erithacus_rubecula"),
            _det(2, "20260407_082500", cls_species="Erithacus_rubecula"),
        ]
    )

    assert len(events) == 1
    assert events[0].photo_count == 2


def test_gap_above_thirty_minutes_starts_new_event():
    """A 35-minute gap splits unprofiled species into two events."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Erithacus_rubecula"),
            _det(2, "20260407_083500", cls_species="Erithacus_rubecula"),
        ]
    )

    assert len(events) == 2
    assert {event.photo_count for event in events} == {1}


def test_short_visit_species_use_tighter_gap_than_default():
    """Local feeder regulars such as blue tits should split earlier than 30 min."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Cyanistes_caeruleus"),
            _det(2, "20260407_081300", cls_species="Cyanistes_caeruleus"),
        ]
    )

    assert len(events) == 2
    assert {event.grouping_profile for event in events} == {"short_station_visit"}
    assert {event.event_gap_minutes for event in events} == {12.0}


def test_short_visit_species_keep_nearby_frames_together():
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Cyanistes_caeruleus"),
            _det(2, "20260407_080900", cls_species="Cyanistes_caeruleus"),
        ]
    )

    assert len(events) == 1
    assert events[0].photo_count == 2
    assert events[0].grouping_profile == "short_station_visit"


def test_max_event_duration_splits_continuous_bursts():
    """Even constant same-species detections must not become one giant event."""
    detections = [
        _det(
            idx + 1,
            f"20260407_{8 + idx // 6:02d}{(idx % 6) * 10:02d}00",
            cls_species="Passer_domesticus",
        )
        for idx in range(19)
    ]

    events = build_bird_events(detections)

    assert len(events) > 1
    assert all(event.grouping_profile == "flock_burst" for event in events)
    assert all(event.max_duration_minutes == 20.0 for event in events)
    assert max(event.duration_sec for event in events) <= 20 * 60


def test_profile_overrides_can_tune_one_station_species():
    profile = EventGroupingProfile(
        name="user_blue_tit",
        gap_minutes=5.0,
        max_duration_minutes=15.0,
        max_representative_images=6,
    )

    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Cyanistes_caeruleus"),
            _det(2, "20260407_080600", cls_species="Cyanistes_caeruleus"),
        ],
        profile_overrides={"Cyanistes_caeruleus": profile},
    )

    assert len(events) == 2
    assert {event.grouping_profile for event in events} == {"user_blue_tit"}


def test_custom_gap_minutes_overrides_default():
    """Caller can tighten the window when needed."""
    events_default = build_bird_events(
        [
            _det(1, "20260407_080000"),
            _det(2, "20260407_080900"),
        ]
    )
    events_strict = build_bird_events(
        [
            _det(1, "20260407_080000"),
            _det(2, "20260407_080900"),
        ],
        gap_minutes=5,
    )

    assert len(events_default) == 1
    assert len(events_strict) == 2


def test_event_carries_representative_image_budget():
    events = build_bird_events(
        [
            _det(idx + 1, f"20260407_0800{idx:02d}", cls_species="Erithacus_rubecula")
            for idx in range(8)
        ]
    )

    assert len(events) == 1
    assert events[0].max_duration_minutes == EVENT_MAX_DURATION_MINUTES_DEFAULT
    assert events[0].representative_image_count == representative_image_budget(8)


def test_two_species_at_overlapping_times_split_into_two_events():
    """Same time, different species, must NOT merge or be flagged ineligible."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Cyanistes_caeruleus"),
            _det(2, "20260407_080004", cls_species="Parus_major"),
            _det(3, "20260407_080008", cls_species="Cyanistes_caeruleus"),
            _det(4, "20260407_080012", cls_species="Parus_major"),
        ]
    )

    assert len(events) == 2
    species_to_count = {event.species: event.photo_count for event in events}
    assert species_to_count == {
        "Cyanistes_caeruleus": 2,
        "Parus_major": 2,
    }
    for event in events:
        assert event.eligibility == "event_eligible"
        assert event.fallback_reason is None


def test_all_unknown_species_yields_single_ineligible_event():
    """Detections with no species at all are still grouped, but flagged."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species=None),
            _det(2, "20260407_080004", cls_species=None),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.species is None
    assert event.eligibility == "event_ineligible"
    assert event.fallback_reason == "unknown_species"
    assert event.photo_count == 2


def test_partial_unknown_species_marks_event_ineligible():
    """Mixed known + unknown classifier output is partial-unknown, not split."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Cyanistes_caeruleus"),
            _det(2, "20260407_080004", cls_species=None),
            _det(3, "20260407_080008", cls_species="Cyanistes_caeruleus"),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.eligibility == "event_ineligible"
    assert event.fallback_reason == "partial_unknown_species"
    assert event.species == "Cyanistes_caeruleus"


def test_unknown_redirect_updates_close_pass_pivot():
    """An unknown detection that attaches to group A must keep A alive.

    The close-pass is relative to the last *attached* epoch per group,
    not to the iteration item. A later iteration item for group B must
    not close group A just because A's most recent *own-species*
    member is older than the window.
    """
    events = build_bird_events(
        [
            _det(1, "20260407_080000", cls_species="Erithacus_rubecula"),
            _det(2, "20260407_082000", cls_species=None),
            _det(3, "20260407_082500", cls_species="Erithacus_rubecula"),
            _det(4, "20260407_083000", cls_species="Sitta_europaea"),
        ]
    )

    assert len(events) == 2
    species_to_count = {event.species: event.photo_count for event in events}
    assert species_to_count == {
        "Erithacus_rubecula": 3,
        "Sitta_europaea": 1,
    }
    robin = next(e for e in events if e.species == "Erithacus_rubecula")
    assert robin.eligibility == "event_ineligible"
    assert robin.fallback_reason == "partial_unknown_species"
    assert robin.detection_ids == [1, 2, 3]


def test_multi_bird_ambiguity_marks_event_ineligible():
    """Any member with sibling_detection_count > 1 makes the event ineligible."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", sibling_detection_count=1),
            _det(2, "20260407_080004", sibling_detection_count=2),
            _det(3, "20260407_080008", sibling_detection_count=1),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.eligibility == "event_ineligible"
    assert event.fallback_reason == "multi_bird_ambiguity"


def test_bbox_jump_within_event_window_flags_ineligible():
    """Same species, same 30 min, but bbox jumps far -> still one event but ineligible."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", bbox=(0.10, 0.18, 0.18, 0.18)),
            _det(2, "20260407_080010", bbox=(0.74, 0.18, 0.18, 0.18)),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.eligibility == "event_ineligible"
    assert event.fallback_reason == "bbox_jump"


def test_manual_species_override_takes_priority_over_classifier():
    """Manual override is the source of truth for species resolution."""
    events = build_bird_events(
        [
            _det(
                1,
                "20260407_080000",
                cls_species="Parus_major",
                manual_species="Cyanistes_caeruleus",
            ),
            _det(2, "20260407_080005", cls_species="Cyanistes_caeruleus"),
        ]
    )

    assert len(events) == 1
    assert events[0].species == "Cyanistes_caeruleus"
    assert events[0].eligibility == "event_eligible"


def test_events_returned_sorted_by_end_time_descending():
    """Newest event first, matching today's Bulk Review browser order."""
    events = build_bird_events(
        [
            _det(1, "20260407_073000"),
            _det(2, "20260407_073004"),
            _det(3, "20260407_090000", cls_species="Parus_major"),
            _det(4, "20260407_090003", cls_species="Parus_major"),
            _det(5, "20260407_080000", cls_species="Erithacus_rubecula"),
            _det(6, "20260407_080004", cls_species="Erithacus_rubecula"),
        ]
    )

    assert [event.species for event in events] == [
        "Parus_major",
        "Erithacus_rubecula",
        "Cyanistes_caeruleus",
    ]


def test_event_carries_touched_filenames_and_bbox_trail():
    """Review templates need filenames and a trail to render the filmstrip."""
    events = build_bird_events(
        [
            _det(1, "20260407_080000", bbox=(0.12, 0.18, 0.18, 0.18)),
            _det(2, "20260407_080004", bbox=(0.18, 0.20, 0.18, 0.18)),
            _det(3, "20260407_080008", bbox=(0.24, 0.20, 0.18, 0.18)),
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.touched_filenames == [
        "frame-1.jpg",
        "frame-2.jpg",
        "frame-3.jpg",
    ]
    assert len(event.bbox_trail) == 3
    roles = [point["trail_role"] for point in event.bbox_trail]
    assert roles[0] == "start"
    assert roles[-1] == "end"


def test_event_key_is_stable_for_same_input():
    """Event keys must be deterministic so the Review surface can dedupe."""
    detections = [
        _det(1, "20260407_080000"),
        _det(2, "20260407_080004"),
    ]
    first = build_bird_events(detections)
    second = build_bird_events(detections)

    assert [event.event_key for event in first] == [event.event_key for event in second]


def _context_det(detection_id: int, timestamp: str, **kwargs) -> dict:
    """Synthetic Gallery context row (read-only confirmed_bird neighbour)."""
    base = _det(detection_id, timestamp, **kwargs)
    base["context_only"] = True
    return base


def test_pure_untagged_event_has_no_context_state():
    events = build_bird_events([_det(1, "20260407_080000")])
    assert len(events) == 1
    event = events[0]
    assert event.context_only_count == 0
    assert event.context_anchored is False


def test_context_only_event_has_count_but_is_not_anchored():
    events = build_bird_events(
        [
            _context_det(101, "20260407_080000"),
            _context_det(102, "20260407_080010"),
        ]
    )
    assert len(events) == 1
    event = events[0]
    assert event.photo_count == 2
    assert event.context_only_count == 2
    # Pure context: nothing actionable, so context_anchored stays False.
    assert event.context_anchored is False


def test_mixed_context_and_actionable_event_is_anchored():
    events = build_bird_events(
        [
            _context_det(101, "20260407_080000"),
            _context_det(102, "20260407_080020"),
            _det(201, "20260407_080500"),
            _det(202, "20260407_080510"),
        ]
    )
    assert len(events) == 1
    event = events[0]
    assert event.photo_count == 4
    assert event.context_only_count == 2
    assert event.context_anchored is True
    # Cover must come from the newest actionable member, never from a
    # read-only context detection — the operator has to be able to act
    # on the cover.
    assert event.cover_detection_id == 202
    # Trail entries carry the context_only flag through to the template.
    trail_flags = [point["context_only"] for point in event.bbox_trail]
    assert trail_flags == [True, True, False, False]


def test_birdevent_dataclass_is_frozen():
    """Events must be immutable so they can be cached and shared."""
    events = build_bird_events([_det(1, "20260407_080000")])
    assert isinstance(events[0], BirdEvent)
    try:
        events[0].photo_count = 99  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("BirdEvent must be frozen")
