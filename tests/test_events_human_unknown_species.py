"""Events must not rebuild an event around a species a person withdrew.

``core/events.py`` carries its own copy of the species fallback chain, kept
local on purpose so ``core`` stays free of ``utils`` imports. That copy did
not know about explicit human "species unknown" answers.

The failure is subtle: ``_normalize_species`` correctly maps the guarded
``Unknown_species`` key to ``None``, which makes the guarded value vanish and
hands control to the raw ``cls_class_name`` one line below - so the safety
token itself opened the hole. The Review Desk then headlined the withdrawn
species as ``MODEL PROPOSAL`` next to an Approve button.
"""

from __future__ import annotations

from core.events import build_bird_events

AI_SPECIES = "Sitta_europaea"


def _row(detection_id: int, timestamp: str, **overrides) -> dict:
    """A review-queue shaped row, as ``build_bird_events`` receives it."""
    row = {
        "detection_id": detection_id,
        "filename": f"{timestamp}_x.jpg",
        "timestamp": timestamp,
        "species_key": "Unknown_species",
        "manual_species_override": None,
        "species_source": "manual_unknown",
        "cls_class_name": AI_SPECIES,
        "cls_confidence": 0.92,
        "od_class_name": "bird",
        "od_confidence": 0.93,
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_w": 0.2,
        "bbox_h": 0.2,
        "context_only": False,
    }
    row.update(overrides)
    return row


def test_explicit_unknown_does_not_become_the_event_species() -> None:
    events = build_bird_events([_row(1, "20260921_094825")])
    assert len(events) == 1
    assert events[0].species != AI_SPECIES, (
        "the event adopted the species the human explicitly withdrew"
    )


def test_explicit_unknown_event_is_not_approvable() -> None:
    """An unresolved species must not be one Approve click from a record."""
    events = build_bird_events([_row(1, "20260921_094825")])
    assert events[0].eligibility != "event_eligible", (
        "an explicitly unknown bird was offered as an approvable event"
    )
    assert events[0].fallback_reason == "unknown_species"


def test_species_wrong_is_treated_the_same_way() -> None:
    events = build_bird_events(
        [_row(1, "20260921_094825", species_source="manual_wrong")]
    )
    assert events[0].species != AI_SPECIES
    assert events[0].eligibility != "event_eligible"


def test_a_withdrawn_frame_does_not_contaminate_an_untouched_one() -> None:
    """The guard is per detection, and deliberately no wider.

    Events group by species first, so a withdrawn frame and an ordinary
    unanswered frame form two separate events. The withdrawn one must be
    unresolved and unapprovable; the untouched one must stay a normal,
    approvable proposal rather than being blocked by its neighbour.
    """
    events = build_bird_events(
        [
            _row(1, "20260921_094825"),
            _row(
                2,
                "20260921_094830",
                species_key=AI_SPECIES,
                species_source=None,
            ),
        ]
    )
    by_detection = {tuple(e.detection_ids): e for e in events}
    withdrawn = by_detection[(1,)]
    untouched = by_detection[(2,)]

    assert withdrawn.species is None
    assert withdrawn.eligibility == "event_ineligible"
    assert withdrawn.fallback_reason == "unknown_species"

    assert untouched.species == AI_SPECIES
    assert untouched.eligibility == "event_eligible", (
        "an unrelated detection was blocked by a neighbour's withdrawn species"
    )


# --- the guard must stay narrow -------------------------------------------


def test_an_untouched_proposal_still_forms_an_eligible_event() -> None:
    events = build_bird_events(
        [
            _row(
                1,
                "20260921_094825",
                species_key=AI_SPECIES,
                species_source=None,
            )
        ]
    )
    assert events[0].species == AI_SPECIES
    assert events[0].eligibility == "event_eligible"


def test_a_human_confirmed_species_still_forms_an_eligible_event() -> None:
    events = build_bird_events(
        [
            _row(
                1,
                "20260921_094825",
                species_key="Parus_major",
                manual_species_override="Parus_major",
                species_source="manual",
                cls_class_name=AI_SPECIES,
            )
        ]
    )
    assert events[0].species == "Parus_major"
    assert events[0].species_source == "manual"
    assert events[0].eligibility == "event_eligible"


# --- the Review Desk card must not arm Approve with the withdrawn species ---


def test_review_event_payload_does_not_preselect_the_withdrawn_species() -> None:
    """``data-species`` on the card is what an Approve would submit.

    The event species is resolved to None correctly, but the payload then
    falls back to the classifier's default suggestion - the same
    "None means unset, so use a default" confusion that the per-detection
    resolver already had to fix.
    """
    from web.blueprints.review import _resolve_review_event_selected_species

    # Event species unresolved (explicit human unknown), AI default available.
    assert (
        _resolve_review_event_selected_species(
            candidate_species=None,
            default_species=AI_SPECIES,
            fallback_reason="unknown_species",
        )
        is None
    ), "Approve would be armed with the species the human withdrew"


def test_review_event_payload_still_defaults_for_ordinary_events() -> None:
    from web.blueprints.review import _resolve_review_event_selected_species

    assert (
        _resolve_review_event_selected_species(
            candidate_species=None,
            default_species=AI_SPECIES,
            fallback_reason=None,
        )
        == AI_SPECIES
    )
    assert (
        _resolve_review_event_selected_species(
            candidate_species="Parus_major",
            default_species=AI_SPECIES,
            fallback_reason=None,
        )
        == "Parus_major"
    )


def test_review_member_candidate_species_respects_an_explicit_unknown() -> None:
    """The tile's relabel control must not carry the withdrawn species.

    ``data-current-species`` seeds the picker's "current" value, and
    ``review_grid.js`` refuses to write when the chosen species equals it.
    Carrying the withdrawn species there would silently block the operator
    from later deciding the bird *is* that species after all.
    """
    from web.blueprints.review import _resolve_review_member_candidate_species

    assert (
        _resolve_review_member_candidate_species(
            {
                "manual_species_override": None,
                "cls_class_name": AI_SPECIES,
                "species_source": "manual_unknown",
            }
        )
        is None
    )


def test_review_member_candidate_species_unchanged_for_normal_rows() -> None:
    from web.blueprints.review import _resolve_review_member_candidate_species

    assert (
        _resolve_review_member_candidate_species(
            {
                "manual_species_override": None,
                "cls_class_name": AI_SPECIES,
                "species_source": None,
            }
        )
        == AI_SPECIES
    )
    assert (
        _resolve_review_member_candidate_species(
            {
                "manual_species_override": "Parus_major",
                "cls_class_name": AI_SPECIES,
                "species_source": "manual",
            }
        )
        == "Parus_major"
    )
