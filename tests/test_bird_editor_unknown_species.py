"""The editor's own JS must not resurrect a withdrawn species either.

Python (``utils/species_names.py``) and SQL (``utils/db/detections.py``) both
learned to stop at an explicit human "species unknown". ``normalize()`` in
``assets/js/bird_editor.js`` carries a third, independent copy of the same
fallback chain and did not, so the editor header, the bird selector and the
box labels could still present the withdrawn species while the server-rendered
page said "unknown".

These run the real module under Node rather than asserting on its source.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"

AI_SPECIES = "Sitta_europaea"


def _normalize(item: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require("
        + json.dumps(str(MODULE))
        + ");"
        + "const item = "
        + json.dumps(item)
        + ";"
        + "if (typeof editor.normalize !== 'function') {"
        + "  console.log(JSON.stringify({__missing__: true})); "
        + "} else { console.log(JSON.stringify(editor.normalize(item))); }"
    )
    result = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    )
    payload = json.loads(result.stdout)
    if payload.get("__missing__"):
        pytest.fail(
            "bird_editor.js does not export normalize(); the JS species "
            "fallback cannot be verified except by reading source strings"
        )
    return payload


def _unknown_row(**overrides) -> dict:
    """Detection 74759 as the server now serialises it into the viewer."""
    row = {
        "detection_id": 74759,
        "object_key": "detection:74759",
        "manual_object_id": None,
        "species_key": None,
        "common_name": "Bird · species unknown",
        "species_source": "manual_unknown",
        "manual_species_override": None,
        "cls_class_name": AI_SPECIES,
        "cls_confidence": 0.92,
        "od_class_name": "bird",
        "od_confidence": 0.93,
        "provenance": "human_unknown",
        "bbox_x": 0.125,
        "bbox_y": 0.37,
        "bbox_w": 0.093,
        "bbox_h": 0.274,
    }
    row.update(overrides)
    return row


def test_explicit_unknown_keeps_no_species_key() -> None:
    got = _normalize(_unknown_row())
    assert got["speciesKey"] in (None, ""), (
        "the editor recovered the withdrawn species: " + repr(got["speciesKey"])
    )


def test_explicit_unknown_survives_a_missing_provenance_hint() -> None:
    """Surfaces that do not compute provenance must still be safe.

    ``species_source`` rides on every detection row; ``provenance`` is
    template-computed. The JS must not depend on the richer field alone.
    """
    got = _normalize(_unknown_row(provenance=None))
    assert got["speciesKey"] in (None, ""), (
        "without the template-computed provenance the JS fell back to CLS"
    )


def test_explicit_unknown_is_not_labelled_an_ai_proposal() -> None:
    got = _normalize(_unknown_row(provenance=None))
    assert got["provenance"] != "model_proposal", (
        "an explicitly answered bird is presented as an untouched AI proposal"
    )


def test_explicit_unknown_keeps_the_original_prediction_for_context() -> None:
    """The old prediction is still true history; only its *role* changed."""
    got = _normalize(_unknown_row())
    assert got["clsClassName"] == AI_SPECIES
    assert got["clsConfidence"] == pytest.approx(0.92)


def test_untouched_proposal_still_resolves_its_species() -> None:
    got = _normalize(
        _unknown_row(
            species_source="model_top1",
            provenance="model_proposal",
            common_name="Kleiber",
        )
    )
    assert got["speciesKey"] == AI_SPECIES
    assert got["provenance"] == "model_proposal"


def test_human_confirmed_species_is_unaffected() -> None:
    got = _normalize(
        _unknown_row(
            species_key="Parus_major",
            manual_species_override="Parus_major",
            species_source="manual",
            provenance="manually_identified",
            common_name="Kohlmeise",
        )
    )
    assert got["speciesKey"] == "Parus_major"


# --- the serialise -> normalise round trip must preserve the answer --------


def _roundtrip(item: dict) -> dict:
    """normalize -> the shape broadcastObjects() emits -> normalize again.

    Cross-modal updates travel as re-serialised plain objects, so an
    explicitly unknown bird has to survive the trip. If the serialised form
    drops the marker, the *second* modal of the same photo silently restores
    the withdrawn species.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const e = require("
        + json.dumps(str(MODULE))
        + ");"
        + "const first = e.normalize("
        + json.dumps(item)
        + ");"
        + """
const wire = {
  detection_id: first.detectionId, manual_object_id: first.manualObjectId,
  object_key: first.objectKey, species_key: first.speciesKey,
  common_name: first.commonName, provenance: first.provenance,
  human_review_state: first.humanReviewState,
  cls_class_name: first.clsClassName, cls_confidence: first.clsConfidence,
  od_class_name: first.odClassName, od_confidence: first.odConfidence,
  bbox_x: first.bbox.x, bbox_y: first.bbox.y,
  bbox_w: first.bbox.w, bbox_h: first.bbox.h
};
console.log(JSON.stringify(e.normalize(wire)));
"""
    )
    return json.loads(
        subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
    )


def test_unknown_survives_the_cross_modal_round_trip() -> None:
    got = _roundtrip(_unknown_row())
    assert got["speciesKey"] in (None, ""), (
        "a second modal of the same photo restored the withdrawn species"
    )
    assert got["provenance"] == "human_unknown"


def test_untouched_proposal_survives_the_round_trip_unchanged() -> None:
    got = _roundtrip(
        _unknown_row(
            species_key=AI_SPECIES,
            species_source="model_top1",
            provenance="model_proposal",
            common_name="Kleiber",
        )
    )
    assert got["speciesKey"] == AI_SPECIES
    assert got["provenance"] == "model_proposal"
