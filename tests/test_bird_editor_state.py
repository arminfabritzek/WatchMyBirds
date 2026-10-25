"""Executable contracts for species confirmation with pending geometry."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("move", [False, True])
@pytest.mark.parametrize("change_species", [False, True])
def test_confirmation_preserves_only_unsaved_geometry(
    move: bool, change_species: bool
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"
    script = (
        "const editor = require("
        + json.dumps(str(module))
        + ");"
        + """
const saved = {objectKey:'detection:1', speciesKey:'Sitta_europaea', commonName:'Kleiber',
 bbox:{x:0.2,y:0.2,w:0.3,h:0.3}, humanReviewState:'unreviewed'};
const draft = JSON.parse(JSON.stringify(saved));
"""
        + ("draft.bbox.x = 0.4;" if move else "")
        + (
            "draft.speciesKey='Parus_major';draft.commonName='Kohlmeise';"
            if change_species
            else ""
        )
        + """
const result = editor.confirmedSpeciesState(saved, draft,
    draft.speciesKey === saved.speciesKey ? 'confirmed' : 'corrected');
console.log(JSON.stringify({saved,draft,result}));
"""
    )
    payload = json.loads(
        subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
    )
    result = payload["result"]
    assert result["confirmed"]["bbox"] == payload["saved"]["bbox"]
    assert result["confirmed"]["speciesKey"] == (
        "Parus_major" if change_species else "Sitta_europaea"
    )
    expected_state = "corrected" if change_species else "confirmed"
    assert result["confirmed"]["humanReviewState"] == expected_state
    assert payload["saved"]["humanReviewState"] == "unreviewed"
    if move:
        assert result["draft"]["bbox"]["x"] == 0.4
        assert result["draft"]["speciesKey"] == result["confirmed"]["speciesKey"]
        assert result["draft"]["humanReviewState"] == expected_state
    else:
        assert result["draft"] is None


@pytest.mark.parametrize("verdict", [None, "suitable", "unsuitable"])
@pytest.mark.parametrize("object_status", ["active", "rejected"])
def test_box_state_survives_broadcast_roundtrip(
    verdict: str | None, object_status: str
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"
    script = (
        "const e=require("
        + json.dumps(str(module))
        + ");const v="
        + json.dumps(verdict)
        + ";const status="
        + json.dumps(object_status)
        + ";"
        + """
    const a = e.normalize({detection_id:1, bbox_x:.1,bbox_y:.2,bbox_w:.3,bbox_h:.4,
        species_key:'Parus_major', human_review_state:'corrected',human_bbox_verdict:v,object_status:status});
    const b = e.normalize(e.serializeObject(a));
    const draft = {...b,bboxVerdict:v === 'suitable' ? null : 'suitable',bboxVerdictTouched:true};
    const result=e.confirmedSpeciesState(b,draft,'corrected');
    console.log(JSON.stringify({a,b,result,draft,boxView:e.bboxVerdictView(b,false),speciesAction:e.speciesClickAction(b)}));
    """
    )
    data = json.loads(
        subprocess.run(
            [node, "-e", script], check=True, capture_output=True, text=True
        ).stdout
    )
    assert data["b"]["bboxVerdict"] == verdict
    assert data["b"]["objectStatus"] == object_status
    if object_status == "rejected":
        assert data["boxView"]["visible"] is False
        assert data["speciesAction"] == "none"
    assert data["b"]["humanReviewState"] == "corrected"
    assert data["result"]["confirmed"]["bboxVerdict"] == verdict
    assert data["result"]["draft"]["bboxVerdict"] == data["draft"]["bboxVerdict"]
    assert data["result"]["draft"]["bboxVerdictTouched"] is True


@pytest.mark.parametrize(
    "state,action",
    [
        ("unreviewed", "confirm"),
        ("confirmed", "retract"),
        ("corrected", "retract"),
        ("reviewed_unknown", "confirm"),
    ],
)
def test_species_click_uses_the_same_canonical_state_as_label_colour(
    state: str, action: str
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    module = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"
    script = (
        "const e=require("
        + json.dumps(str(module))
        + ");console.log(e.speciesClickAction({speciesKey:'Cyanistes_caeruleus',humanReviewState:"
        + json.dumps(state)
        + ",provenance:'manually_identified',objectStatus:'active'}));"
    )
    result = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == action
