"""What the editor tells the server about the box on save.

Object-detection training needs an explicit verdict on the box
(`bbox_quality`); the classifier does not. The editor is the only place
in the everyday flow where that verdict can come from, and it must come
from what the person actually did:

* Redrawing a box *is* the verdict — the new box is the one they want.
* Correcting only the species says nothing about the geometry, so it
  must send nothing on this axis. Silence is not consent here.
* The explicit toggle overrides both.

These run the real module under Node rather than asserting on source
strings, so a body that is built but never sent cannot pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"

SAVED = {
    "objectKey": "detection:74759",
    "detectionId": 74759,
    "objectKind": "detection",
    "speciesKey": "Sitta_europaea",
    "commonName": "Nuthatch",
    "bbox": {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3},
}


def _build_body(draft: dict, initial: dict, filename: str = "a.jpg") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require(" + json.dumps(str(_MODULE)) + ");"
        "const draft = " + json.dumps(draft) + ";"
        "const initial = " + json.dumps(initial) + ";"
        "console.log(JSON.stringify("
        "editor.detectionAnswerBody(" + json.dumps(filename) + ", draft, initial)"
        "));"
    )
    out = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def _draft(**overrides) -> dict:
    draft = json.loads(json.dumps(SAVED))
    draft.update(overrides)
    return draft


def test_dragging_the_box_answers_the_quality_axis():
    """Redrawing a box is itself the verdict that the new box is right."""
    moved = _draft(bbox={"x": 0.4, "y": 0.2, "w": 0.3, "h": 0.3})

    body = _build_body(moved, SAVED)

    assert body["bbox_correction"] == moved["bbox"]
    assert body["bbox_quality"] == "suitable"


def test_correcting_only_the_species_says_nothing_about_the_box():
    """The decisive guard: silence on an axis is not an answer."""
    relabelled = _draft(speciesKey="Parus_major", commonName="Great Tit")

    body = _build_body(relabelled, SAVED)

    assert body["species_identity"] == "corrected"
    assert "bbox_quality" not in body
    assert "bbox_correction" not in body


def test_withdrawing_the_species_says_nothing_about_the_box():
    withdrawn = _draft(speciesKey=None, commonName=None)

    body = _build_body(withdrawn, SAVED)

    assert body["species_identity"] == "unknown"
    assert "bbox_quality" not in body


def test_the_toggle_answers_the_axis_without_touching_the_box():
    confirmed = _draft(bboxVerdict="suitable")

    body = _build_body(confirmed, SAVED)

    assert body["bbox_quality"] == "suitable"
    assert "bbox_correction" not in body


def test_the_toggle_can_report_an_unusable_box():
    rejected = _draft(bboxVerdict="unsuitable")

    body = _build_body(rejected, SAVED)

    assert body["bbox_quality"] == "unsuitable"


def test_an_explicit_verdict_outranks_the_drag_implication():
    """Someone who drags and then marks it wrong meant wrong."""
    moved_then_rejected = _draft(
        bbox={"x": 0.4, "y": 0.2, "w": 0.3, "h": 0.3}, bboxVerdict="unsuitable"
    )

    body = _build_body(moved_then_rejected, SAVED)

    assert body["bbox_correction"] == moved_then_rejected["bbox"]
    assert body["bbox_quality"] == "unsuitable"


def test_a_save_with_nothing_changed_carries_only_identity():
    body = _build_body(_draft(), SAVED)

    assert body == {"filename": "a.jpg", "detection_id": 74759}


def test_species_and_box_are_answered_together_when_both_changed():
    both = _draft(
        speciesKey="Parus_major",
        commonName="Great Tit",
        bbox={"x": 0.4, "y": 0.2, "w": 0.3, "h": 0.3},
    )

    body = _build_body(both, SAVED)

    assert body["species_identity"] == "corrected"
    assert body["species_key"] == "Parus_major"
    assert body["bbox_quality"] == "suitable"
    assert body["bbox_correction"] == both["bbox"]


def _next_verdict(current, pressed: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require(" + json.dumps(str(_MODULE)) + ");"
        "console.log(JSON.stringify({v: editor.nextBboxVerdict("
        + json.dumps(current)
        + ", "
        + json.dumps(pressed)
        + ")}));"
    )
    out = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)["v"]


def test_pressing_a_verdict_sets_it():
    assert _next_verdict(None, "suitable") == "suitable"
    assert _next_verdict(None, "unsuitable") == "unsuitable"


def test_pressing_the_active_verdict_clears_it():
    """A mis-click must be undoable back to 'unanswered'."""
    assert _next_verdict("suitable", "suitable") is None
    assert _next_verdict("unsuitable", "unsuitable") is None


def test_switching_between_verdicts():
    assert _next_verdict("suitable", "unsuitable") == "unsuitable"
    assert _next_verdict("unsuitable", "suitable") == "suitable"


def test_a_verdict_is_never_carried_into_a_later_save():
    """The verdict answers one save; a stale one would re-assert silently."""
    already_answered = _draft(bboxVerdict="suitable")
    initial_with_verdict = _draft(bboxVerdict="suitable")

    body = _build_body(already_answered, initial_with_verdict)

    assert body["bbox_quality"] == "suitable"
    assert "bbox_correction" not in body


def _verdict_visibility(item, in_edit_mode: bool) -> dict:
    """Ask the module whether the verdict controls belong on screen."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require(" + json.dumps(str(_MODULE)) + ");"
        "console.log(JSON.stringify(editor.bboxVerdictView("
        + json.dumps(item)
        + ", "
        + json.dumps(in_edit_mode)
        + ")));"
    )
    out = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def test_the_verdict_is_offered_while_simply_viewing_a_bird():
    """The decisive guard: judging a box must not require editing it first.

    Gating the controls on an active edit draft hid them during ordinary
    viewing, which is exactly where the question is supposed to be asked.
    """
    view = _verdict_visibility(SAVED, False)

    assert view["visible"] is True


def test_the_verdict_stays_offered_in_edit_mode():
    view = _verdict_visibility(SAVED, True)

    assert view["visible"] is True


def test_a_manually_added_bird_gets_no_verdict_control():
    """A box someone drew themselves needs no verdict on the model's box."""
    manual = _draft(objectKind="manual", detectionId=None)

    view = _verdict_visibility(manual, False)

    assert view["visible"] is False


def test_no_selected_bird_means_no_control():
    assert _verdict_visibility(None, False)["visible"] is False


def test_the_active_verdict_is_reported_for_the_pressed_state():
    view = _verdict_visibility(_draft(bboxVerdict="unsuitable"), False)

    assert view["visible"] is True
    assert view["active"] == "unsuitable"


def test_an_unanswered_box_reports_no_active_verdict():
    assert _verdict_visibility(SAVED, False)["active"] is None


def _click_mode(in_edit_mode: bool) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require(" + json.dumps(str(_MODULE)) + ");"
        "console.log(JSON.stringify({m: editor.bboxVerdictClickMode("
        + json.dumps(in_edit_mode)
        + ")}));"
    )
    out = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)["m"]


def test_a_verdict_given_while_viewing_is_written_immediately():
    """There is no Save button in view mode, so the answer must post itself."""
    assert _click_mode(False) == "post"


def test_a_verdict_given_while_editing_rides_along_with_save():
    """In edit mode the verdict joins the pending box change instead."""
    assert _click_mode(True) == "draft"
