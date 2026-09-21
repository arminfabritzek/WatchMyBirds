"""Clicking a confirmed species again takes the confirmation back.

Unlike the box verdict — which is a draft until Save — confirming a
species writes immediately. The second click therefore has to be a second
write that undoes the first, not a local reset, so the two directions are
pinned here as one decision function.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "assets/js/bird_editor.js"


def _action(item: dict) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (
        "const editor = require(" + json.dumps(str(_MODULE)) + ");"
        "console.log(JSON.stringify({a: editor.speciesClickAction("
        + json.dumps(item)
        + ")}));"
    )
    out = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)["a"]


def test_an_unconfirmed_species_confirms():
    assert (
        _action({"provenance": "model_proposal", "speciesKey": "Parus_major"})
        == "confirm"
    )


def test_a_confirmed_species_retracts():
    assert (
        _action({"provenance": "human_confirmed", "speciesKey": "Parus_major"})
        == "retract"
    )


def test_a_corrected_species_also_retracts():
    """A relabel is a human answer too, so it is takeable back the same way."""
    assert (
        _action({"provenance": "manually_identified", "speciesKey": "Parus_major"})
        == "retract"
    )


def test_a_withdrawn_species_offers_nothing_to_confirm():
    """There is no species to confirm, and the unknown answer is its own axis."""
    assert _action({"provenance": "human_unknown", "speciesKey": None}) == "none"


def test_a_row_without_a_species_offers_nothing():
    assert _action({"provenance": "model_proposal", "speciesKey": None}) == "none"
