"""Day selection for complete Review events, independent of their grouping."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.events import BirdEvent
from core.station_report import StationEventReview


@dataclass(frozen=True)
class ReviewDaySelection:
    events: tuple[BirdEvent, ...]
    days: tuple[tuple[str, int], ...]
    selected_day: str
    total_events: int
    older_day: str | None
    newer_day: str | None


def select_review_day(
    events: Sequence[BirdEvent],
    completed_reviews: Mapping[frozenset[int], StationEventReview],
    day: str,
) -> ReviewDaySelection:
    actionable = []
    for event in events:
        review = completed_reviews.get(frozenset(event.detection_ids))
        if review is not None and review.outcome in {
            "verified_species",
            "unresolved_bird",
            "no_bird",
        }:
            continue
        if event.context_only_count == event.photo_count:
            continue
        actionable.append(event)

    def event_day(event: BirdEvent) -> str:
        timestamp = event.start_time
        return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

    counts = Counter(event_day(event) for event in actionable)
    days = tuple(sorted(counts.items(), reverse=True))
    selected_day = (days[0][0] if days else "all") if day == "latest" else day
    selected = tuple(
        event
        for event in actionable
        if selected_day == "all" or event_day(event) == selected_day
    )
    older = [value for value, _ in days if value < selected_day]
    newer = [value for value, _ in days if value > selected_day]
    return ReviewDaySelection(
        events=selected,
        days=days,
        selected_day=selected_day,
        total_events=len(actionable),
        older_day=max(older) if older and selected_day != "all" else None,
        newer_day=min(newer) if newer and selected_day != "all" else None,
    )
