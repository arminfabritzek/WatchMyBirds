"""Write-path membership checks must not depend on display materialization."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from core.events import BirdEvent
from web.blueprints import review


def test_action_lookup_keeps_context_members() -> None:
    event = BirdEvent(
        event_key="bird-event-012345abcdef",
        species="Parus_major",
        species_source="cls",
        detection_ids=[1, 2],
        photo_count=2,
        duration_sec=20,
        start_time="20261011_235950",
        end_time="20261012_000010",
        cover_detection_id=2,
        eligibility="event_eligible",
        fallback_reason=None,
        touched_filenames=["anchor.jpg", "review.jpg"],
        context_only_count=1,
        context_anchored=True,
    )
    row_map = {
        1: {"best_detection_id": 1, "context_only": True},
        2: {"best_detection_id": 2, "active_detection_id": 2},
    }
    kwargs = dict(
        event_key=event.event_key,
        gallery_threshold=0.7,
        output_dir="/tmp",
        species_locale="DE",
        common_names={},
    )
    with (
        patch.object(
            review,
            "_fetch_review_event_payloads",
            return_value=([event], row_map, [], False),
        ) as fetch,
        patch.object(
            review, "_build_review_item", side_effect=AssertionError("display queue")
        ),
    ):
        result = review._load_single_review_event(MagicMock(), **kwargs)
        assert result["detection_ids"] == [1, 2]
        assert result["members"] == [
            {"best_detection_id": 1, "context_only": True},
            {"best_detection_id": 2, "context_only": False},
        ]
        assert result["candidate_species"] == "Parus_major"
        assert result["eligibility"] == "event_eligible"
        fetch.return_value = (
            [replace(event, context_only_count=2)],
            row_map,
            [],
            False,
        )
        assert review._load_single_review_event(MagicMock(), **kwargs) is None
        fetch.return_value = ([event], {1: row_map[1]}, [], False)
        assert review._load_single_review_event(MagicMock(), **kwargs) is None
