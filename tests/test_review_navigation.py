from dataclasses import replace

from core.events import BirdEvent, build_bird_events
from core.review_navigation import select_review_day
from core.station_report import StationEventReview


def _event(detection_id: int, timestamp: str) -> BirdEvent:
    return BirdEvent(
        event_key=f"event-{detection_id}",
        species="Parus_major",
        species_source="classifier",
        detection_ids=[detection_id],
        photo_count=1,
        duration_sec=0,
        start_time=timestamp,
        end_time=timestamp,
        cover_detection_id=detection_id,
        eligibility="event_eligible",
        fallback_reason=None,
        touched_filenames=[f"frame-{detection_id}.jpg"],
    )


def test_latest_day_counts_only_actionable_unreviewed_events() -> None:
    older = _event(1, "20261010_090000")
    newest = _event(2, "20261012_090000")
    context = replace(_event(3, "20261013_090000"), context_only_count=1)
    completed = _event(4, "20261014_090000")
    review = StationEventReview(
        review_id=1,
        event_key_snapshot=completed.event_key,
        detection_ids=(4,),
        outcome="unresolved_bird",
        candidate_species_key=None,
        species_key=None,
        evidence_quality="limited",
        event_start=completed.start_time,
        event_end=completed.end_time,
        created_at="2026-10-14T10:00:00Z",
    )

    result = select_review_day(
        [older, newest, context, completed], {frozenset({4}): review}, "latest"
    )

    assert result.selected_day == "2026-10-12"
    assert result.events == (newest,)
    assert result.days == (("2026-10-12", 1), ("2026-10-10", 1))
    assert result.total_events == 2
    assert result.older_day == "2026-10-10"
    assert result.newer_day is None


def test_empty_selected_day_keeps_date_and_links_to_nearest_open_days() -> None:
    result = select_review_day(
        [_event(1, "20261010_090000"), _event(2, "20261012_090000")],
        {},
        "2026-10-11",
    )

    assert result.selected_day == "2026-10-11"
    assert result.events == ()
    assert result.older_day == "2026-10-10"
    assert result.newer_day == "2026-10-12"


def test_all_days_and_empty_queue() -> None:
    event = _event(1, "20261010_090000")
    result = select_review_day([event], {}, "all")
    assert result.events == (event,)
    assert result.older_day is None
    assert result.newer_day is None
    empty = select_review_day([], {}, "latest")
    assert empty.selected_day == "all"
    assert empty.days == ()
    assert empty.total_events == 0


def test_midnight_event_keeps_both_days_and_gallery_context() -> None:
    events = build_bird_events(
        [
            {
                "detection_id": detection_id,
                "filename": f"frame-{detection_id}.jpg",
                "timestamp": timestamp,
                "species_key": "Parus_major",
                "cls_class_name": "Parus_major",
                "bbox_x": 0.2,
                "bbox_y": 0.2,
                "bbox_w": 0.2,
                "bbox_h": 0.2,
                "context_only": context,
            }
            for detection_id, timestamp, context in [
                (1, "20261010_235900", True),
                (2, "20261011_000100", False),
            ]
        ]
    )
    assert len(events) == 1
    result = select_review_day(events, {}, "2026-10-10")
    assert result.events[0].detection_ids == [1, 2]
    assert result.events[0].context_only_count == 1
    assert result.events[0].context_anchored
    assert select_review_day(events, {}, "2026-10-11").events == ()
