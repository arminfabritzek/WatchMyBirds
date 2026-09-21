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
const result = editor.confirmedSpeciesState(saved, draft);
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
    assert result["confirmed"]["humanReviewState"] == "confirmed"
    assert payload["saved"]["humanReviewState"] == "unreviewed"
    if move:
        assert result["draft"]["bbox"]["x"] == 0.4
        assert result["draft"]["speciesKey"] == result["confirmed"]["speciesKey"]
        assert result["draft"]["humanReviewState"] == "confirmed"
    else:
        assert result["draft"] is None
