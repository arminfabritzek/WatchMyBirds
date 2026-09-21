"""The detail-modal head must show the human answer, not the withdrawn AI guess.

Reproduces the production state of a real record: a human opened a detection
whose classifier had proposed ``Sitta_europaea`` at 92% and answered
"bird, but I cannot tell the species". ``core.human_label_core`` records that
by *clearing* ``manual_species_override`` and stamping
``species_source='manual_unknown'``.

The head used to key "has a human touched this?" off
``species_source == 'manual'`` alone. ``manual_unknown`` failed that test, so
the modal fell back through ``cls_class_name`` and presented the withdrawn
species as the current identity, complete with the old CLS confidence and a
Wikipedia link to the wrong bird.

These render the real macro rather than grepping its source: the earlier
string-only template tests passed while this defect was live.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from flask import Flask, render_template_string

ROOT = Path(__file__).resolve().parents[1]

AI_SPECIES = "Sitta_europaea"
AI_COMMON = "Kleiber"
UNKNOWN_COMMON = "Bird · species unknown"


def _app() -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    app.secret_key = "test"
    app.jinja_env.globals["wikipedia_species_url"] = lambda common, key: (
        "https://en.wikipedia.org/wiki/" + str(key).replace("_", " ") if key else None
    )
    return app


def _detection(**overrides) -> dict:
    """The real production row for detection 74759, verified read-only."""
    det = {
        "detection_id": 74759,
        "species_key": "Unknown_species",
        "common_name": UNKNOWN_COMMON,
        "manual_species_override": None,
        "species_source": "manual_unknown",
        "human_review_state": "reviewed_unknown",
        "human_species_key": None,
        "human_common_name": "",
        "od_class_name": "bird",
        "od_confidence": 0.931461155414581,
        "cls_class_name": AI_SPECIES,
        "cls_confidence": 0.92,
        "score": 0.921875,
        "decision_state": "unknown",
        "review_status": "untagged",
        "formatted_date": "20.09.2026",
        "formatted_time": "09:48:25",
        "gallery_date": "2026-09-20",
        "siblings": [],
        "sibling_count": 1,
        "bbox_x": 0.125,
        "bbox_y": 0.370138888888889,
        "bbox_w": 0.093359375,
        "bbox_h": 0.274305555555556,
        "is_favorite": False,
        "image_filename": "20260920_094825_673647.jpg",
        "original_path": "/uploads/originals/2026-09-20/20260920_094825_673647.jpg",
        "full_path": "/uploads/derivatives/optimized/2026-09-20/x.webp",
        "original_present": 1,
    }
    det.update(overrides)
    return det


def _render(det: dict, group: str = "species_overview") -> str:
    app = _app()
    with app.test_request_context("/gallery/2026-09-20"):
        return render_template_string(
            '{% from "components/detection_modal.html" import render_modal %}'
            "{{ render_modal(det, group) }}",
            det=det,
            group=group,
        )


def _head(rendered: str) -> str:
    """Isolate the modal header so body/JSON payloads cannot mask a defect."""
    start = rendered.index('class="modal-header')
    end = rendered.index('class="wm-modal__body"')
    return rendered[start:end]


# --- the head must not present the withdrawn species as current ------------


def test_head_title_is_not_the_withdrawn_ai_species() -> None:
    head = _head(_render(_detection()))
    title = re.search(r"data-editor-title-name[^>]*>(.*?)</em>", head, re.S)
    assert title is not None, "title element missing"
    assert AI_COMMON not in title.group(1), (
        "the head still presents the withdrawn AI common name as the identity"
    )


def test_head_subtitle_species_is_not_the_withdrawn_ai_species() -> None:
    head = _head(_render(_detection()))
    species = re.search(r"data-editor-title-species[^>]*>(.*?)</em>", head, re.S)
    assert species is not None, "subtitle species element missing"
    assert AI_SPECIES.replace("_", " ") not in species.group(1), (
        "the subtitle still shows the withdrawn AI species as the current one"
    )


def test_head_marks_the_species_as_explicitly_unknown() -> None:
    head = _head(_render(_detection()))
    assert "unknown" in head.lower(), (
        "nothing in the head tells the user the species is explicitly unknown"
    )


def test_head_never_shows_bare_cls_confidence_for_an_unknown_species() -> None:
    """The old 92% belonged to a species the human withdrew.

    Showing it unqualified next to the current identity would assert model
    support the model never gave for "unknown".
    """
    head = _head(_render(_detection()))
    conf = re.search(r"data-editor-title-confidence[^>]*>(.*?)</span>", head, re.S)
    assert conf is not None, "confidence element missing"
    text = conf.group(1)
    if "CLS" in text:
        assert AI_SPECIES.replace("_", " ") in text or "Original" in text, (
            "CLS confidence is shown without naming the original prediction "
            "it belonged to: " + text
        )


def test_head_has_no_wikipedia_link_to_the_withdrawn_species() -> None:
    head = _head(_render(_detection()))
    assert "wikipedia.org/wiki/Sitta" not in head.replace("%20", " ").replace(
        "_", " "
    ).replace("wikipedia.org/wiki/Sitta europaea", "wikipedia.org/wiki/Sitta"), (
        "the head links to the withdrawn species' Wikipedia article"
    )


def test_head_provenance_is_not_plain_ai_proposal() -> None:
    head = _head(_render(_detection()))
    prov = re.search(r"data-editor-title-provenance[^>]*>(.*?)</span>", head, re.S)
    assert prov is not None, "provenance element missing"
    assert prov.group(1).strip(), (
        "an explicitly answered bird shows no human provenance in the head"
    )


# --- the fix must stay narrow ----------------------------------------------


def test_untouched_ai_proposal_still_shows_its_species_and_confidence() -> None:
    """The guard must not blank out ordinary, unanswered AI proposals."""
    det = _detection(
        species_key=AI_SPECIES,
        common_name=AI_COMMON,
        species_source="model_top1",
        human_review_state="unreviewed",
        decision_state="confirmed",
    )
    head = _head(_render(det))
    assert AI_COMMON in head, "an ordinary AI proposal lost its species name"
    assert "CLS" in head, "an ordinary AI proposal lost its CLS confidence"


def test_human_confirmed_species_still_shows_that_species() -> None:
    det = _detection(
        species_key="Parus_major",
        common_name="Kohlmeise",
        manual_species_override="Parus_major",
        species_source="manual",
        human_review_state="confirmed",
        human_species_key="Parus_major",
        human_common_name="Kohlmeise",
    )
    head = _head(_render(det))
    assert "Kohlmeise" in head
    assert AI_COMMON not in head


# --- every gallery entrance behaves the same -------------------------------


@pytest.mark.parametrize(
    "group", ["species_overview", "species_summary", "stream-summary", "species"]
)
def test_all_detail_entrances_agree_on_the_unknown_head(group: str) -> None:
    """Only the Gallery route computes ``human_review_state``.

    Species / Species Overview / Stream render the same macro without it, so a
    fix that keys purely off that field would pass in Gallery and silently fail
    everywhere else. Drop it here to prove the head does not depend on it.
    """
    det = _detection()
    det.pop("human_review_state")
    det.pop("human_species_key")
    det.pop("human_common_name")
    head = _head(_render(det, group=group))
    assert AI_COMMON not in head, (
        f"{group}: withdrawn AI species resurfaces when the Gallery-only "
        "human_review_state field is absent"
    )


def test_modal_renders_when_the_row_omits_the_new_fields() -> None:
    """Stream and other surfaces build detection dicts without these keys.

    ``species_source`` arrives as Jinja ``Undefined`` there, and an Undefined
    inside the ``current_detection`` payload makes ``tojson`` raise, which
    took down the whole page rather than just the head.
    """
    det = _detection()
    for key in (
        "species_source",
        "human_review_state",
        "human_species_key",
        "human_common_name",
        "manual_species_override",
        "review_status",
    ):
        det.pop(key, None)

    rendered = _render(det, group="stream-summary")
    assert "data-current-detection" in rendered


# --- behaviour cover for rules previously pinned as source strings ---------


def test_ai_status_badge_follows_the_decision_state() -> None:
    """Replaces a source-string assertion on the badge literals."""
    confirmed = _head(
        _render(
            _detection(
                species_key=AI_SPECIES,
                common_name=AI_COMMON,
                species_source="model_top1",
                human_review_state="unreviewed",
                decision_state="confirmed",
            )
        )
    )
    assert "AI confirmed" in confirmed

    uncertain = _head(
        _render(
            _detection(
                species_key=AI_SPECIES,
                common_name=AI_COMMON,
                species_source="model_top1",
                human_review_state="unreviewed",
                decision_state="uncertain",
            )
        )
    )
    assert "AI uncertain" in uncertain


def test_manual_species_provenance_wins_over_the_ai_badge() -> None:
    """A human-identified species reads as human work, not an AI verdict.

    The provenance slot is exclusive: once a person has set the species it
    says "Manually identified" and no automatic assessment badge appears.
    """
    head = _head(
        _render(
            _detection(
                species_key="Parus_major",
                common_name="Kohlmeise",
                manual_species_override="Parus_major",
                species_source="manual",
                review_status="confirmed_bird",
                human_review_state="confirmed",
            )
        )
    )
    prov = re.search(r"data-editor-title-provenance[^>]*>(.*?)</span>", head, re.S)
    assert prov is not None
    assert "Manually identified" in prov.group(1)
    assert "AI confirmed" not in head, (
        "an automatic AI badge sits beside a human-identified species"
    )


def test_confidence_pair_is_rendered_for_an_untouched_proposal() -> None:
    """Replaces the `OD {{` / `/ CLS {{` source-string assertions."""
    head = _head(
        _render(
            _detection(
                species_key=AI_SPECIES,
                common_name=AI_COMMON,
                species_source="model_top1",
                human_review_state="unreviewed",
                decision_state="confirmed",
            )
        )
    )
    conf = re.search(r"data-editor-title-confidence[^>]*>(.*?)</span>", head, re.S)
    assert conf is not None
    assert re.search(r"OD\s+93%\s*/\s*CLS\s+92%", conf.group(1)), conf.group(1)


def test_manual_species_hides_the_old_cls_confidence() -> None:
    """A corrected species must not carry the previous model's CLS number."""
    head = _head(
        _render(
            _detection(
                species_key="Parus_major",
                common_name="Kohlmeise",
                manual_species_override="Parus_major",
                species_source="manual",
                human_review_state="corrected",
            )
        )
    )
    conf = re.search(r"data-editor-title-confidence[^>]*>(.*?)</span>", head, re.S)
    assert conf is not None
    assert "CLS" not in conf.group(1), (
        "old CLS confidence shown next to a human-corrected species: " + conf.group(1)
    )


def test_wikipedia_link_targets_the_displayed_species() -> None:
    head = _head(
        _render(
            _detection(
                species_key="Parus_major",
                common_name="Kohlmeise",
                manual_species_override="Parus_major",
                species_source="manual",
                human_review_state="confirmed",
            )
        )
    )
    assert "Parus" in head
    assert "Sitta" not in head, "link/title still references the withdrawn species"


def test_no_wikipedia_link_at_all_for_an_explicitly_unknown_species() -> None:
    """Uses the real URL builder, not a stub.

    ``build_species_wikipedia_url`` only declines an *empty* term, and the
    common name of an unknown bird is a non-empty phrase, so the head used to
    offer a confident-looking search link for the literal words the UI shows.
    """
    from utils.wikipedia import build_species_wikipedia_url

    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    app.secret_key = "test"
    app.jinja_env.globals["wikipedia_species_url"] = build_species_wikipedia_url

    with app.test_request_context("/gallery/2026-09-20"):
        rendered = render_template_string(
            '{% from "components/detection_modal.html" import render_modal %}'
            '{{ render_modal(det, "g") }}',
            det=_detection(),
        )
    head = _head(rendered)
    assert "wikipedia" not in head.lower(), (
        "the head offers a Wikipedia link for a species nobody identified"
    )


def test_a_real_species_still_gets_its_wikipedia_link() -> None:
    from utils.wikipedia import build_species_wikipedia_url

    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    app.secret_key = "test"
    app.jinja_env.globals["wikipedia_species_url"] = build_species_wikipedia_url

    with app.test_request_context("/gallery/2026-09-20"):
        rendered = render_template_string(
            '{% from "components/detection_modal.html" import render_modal %}'
            '{{ render_modal(det, "g") }}',
            det=_detection(
                species_key="Parus_major",
                common_name="Kohlmeise",
                manual_species_override="Parus_major",
                species_source="manual",
                human_review_state="confirmed",
            ),
        )
    head = _head(rendered)
    assert "wikipedia" in head.lower()
    assert "Parus" in head
