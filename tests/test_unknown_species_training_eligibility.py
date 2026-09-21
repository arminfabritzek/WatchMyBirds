"""What an explicit "species unknown" does and does not exclude from training.

"The human said unknown, so the row is useless" is too crude a rule and the
readiness contract does not implement it. OD and CLS readiness are derived on
independent axes:

* CLS must not receive a positive label for a species the human withdrew.
* OD suitability never consults the species at all - it rests on the explicit
  bird-presence and box-quality answers.
* An unknown *bird* is still a bird. It must never become a bird-free negative.

These pin the exact exclusion reasons per fact set, so a later change cannot
quietly widen or narrow them. They exercise the pure readiness function, so
the fact sets are written out here rather than recorded through
``record_human_answer`` — which now derives a ``bird_presence`` fact from an
answered species axis (see ``test_unknown_species_implies_bird_presence``).
"""

from __future__ import annotations

from core.human_label_core import object_training_readiness


def _fact(fact_type: str, answer_value: str, species_key: str | None = None) -> dict:
    return {
        "scope": "object",
        "fact_type": fact_type,
        "answer_value": answer_value,
        "species_key": species_key,
    }


# A bare species answer with no other axis filled — the shape a caller
# produces when it writes facts directly rather than through record_human_answer.
PRODUCTION_FACTS = [_fact("species_identity", "unknown")]


def test_cls_is_excluded_and_names_the_species_reason() -> None:
    readiness = object_training_readiness(PRODUCTION_FACTS)
    assert readiness["cls"]["ready"] is False
    assert "species_unknown" in readiness["cls"]["reasons"]


def test_cls_carries_no_species_key_for_an_unknown_answer() -> None:
    """The withdrawn species must not ride along as a usable CLS label."""
    readiness = object_training_readiness(PRODUCTION_FACTS)
    assert "Sitta_europaea" not in str(readiness)


def test_od_exclusion_is_about_the_box_axes_not_the_species() -> None:
    """The real reason OD skips this row is the missing box/presence answers."""
    readiness = object_training_readiness(PRODUCTION_FACTS)
    assert readiness["od"]["ready"] is False
    assert set(readiness["od"]["reasons"]) == {
        "object_bird_presence_unknown",
        "bbox_quality_unknown",
    }
    assert "species_unknown" not in readiness["od"]["reasons"], (
        "OD readiness must not depend on the species axis"
    )


def test_unknown_species_with_box_facts_becomes_od_usable() -> None:
    """The decisive proof that species and OD suitability are independent.

    Same unknown species, but the human also confirmed a bird is present and
    the box is good: OD may use it, CLS still may not.
    """
    facts = [
        _fact("species_identity", "unknown"),
        _fact("bird_presence", "present"),
        _fact("bbox_quality", "suitable"),
    ]
    readiness = object_training_readiness(facts)
    assert readiness["od"]["ready"] is True, (
        "an explicitly unknown bird with a confirmed, suitable box is valid "
        "OD training data"
    )
    assert readiness["cls"]["ready"] is False
    assert readiness["cls"]["reasons"] == ["species_unknown"]


def test_an_unknown_bird_is_not_a_bird_free_negative() -> None:
    """'Species unknown' and 'no bird here' must never collapse together."""
    unknown = object_training_readiness(
        [_fact("species_identity", "unknown"), _fact("bird_presence", "present")]
    )
    absent = object_training_readiness([_fact("bird_presence", "absent")])

    assert "object_bird_absent" not in unknown["od"]["reasons"], (
        "an unknown bird was treated as a bird-free negative"
    )
    assert "object_bird_absent" in absent["od"]["reasons"]
    assert "object_bird_absent" in absent["cls"]["reasons"]


def test_a_confirmed_species_stays_fully_usable() -> None:
    facts = [
        _fact("species_identity", "confirmed", "Parus_major"),
        _fact("bird_presence", "present"),
        _fact("bbox_quality", "suitable"),
    ]
    readiness = object_training_readiness(facts)
    assert readiness["od"]["ready"] is True
    assert readiness["cls"]["ready"] is True


def test_species_wrong_is_distinct_from_species_unknown() -> None:
    """Different answers keep different reason codes even though both block CLS."""
    wrong = object_training_readiness([_fact("species_identity", "wrong")])
    unknown = object_training_readiness([_fact("species_identity", "unknown")])
    assert "species_wrong" in wrong["cls"]["reasons"]
    assert "species_unknown" in unknown["cls"]["reasons"]
    assert wrong["cls"]["reasons"] != unknown["cls"]["reasons"]


def test_never_answered_is_distinct_from_explicitly_unknown() -> None:
    """The distinction the display bug erased must survive in the reason codes."""
    never = object_training_readiness([_fact("bird_presence", "present")])
    explicit = object_training_readiness(
        [_fact("bird_presence", "present"), _fact("species_identity", "unknown")]
    )
    assert "species_identity_unknown" in never["cls"]["reasons"]
    assert "species_unknown" in explicit["cls"]["reasons"]
    assert never["cls"]["reasons"] != explicit["cls"]["reasons"]
