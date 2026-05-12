"""Regression: every species key that can flow out of the DB must have
a DE common name, otherwise the UI shows the raw scientific name twice
(once as the key, once as the fallback ``replace("_", " ")``).

The list below is the snapshot of species observed on the production Pi
(2026-05-11). New species discovered on the live system should be added
here AND to assets/common_names_DE.json in the same commit — the test
keeps the two in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "common_names_DE.json"


# Species observed on the live Pi 2026-05-11. Extend when new species
# appear (and add the translation at the same time).
#
# NOTE: ``Falco_rufigularis`` (a classifier hallucination — bat falcon,
# Central America) and ``Phoenicurus_sp.`` are observed in DB but
# DELIBERATELY not in the names map: the notification service uses
# "absent from common_names_DE.json" as a catalog-orphan gate that
# blocks live alerts for unverified classifier outputs. Adding them
# here would also surface them in live alerts. The cleaner long-term
# fix is to separate "display translation" from "catalog whitelist"
# in a follow-up plan; until then these two stay raw-Latin in the UI
# on purpose.
SPECIES_IN_USE = [
    "Columba_palumbus",
    "Cyanistes_caeruleus",
    "Dendrocopos_major",
    "Fringilla_sp.",
    "Garrulus_glandarius",
    "Parus_major",
    "Poecile_palustris",
    "Sylvia_sp.",
    "Turdus_sp.",
    "Unknown_species",
    "cat",
]


def _load_de_names() -> dict[str, str]:
    with _ASSETS.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("species_key", SPECIES_IN_USE)
def test_every_observed_species_has_de_translation(species_key):
    """If this test fails for a species, the operator sees Latin in the UI.
    Fix by adding the key to assets/common_names_DE.json."""
    de = _load_de_names()
    assert species_key in de, (
        f"Species {species_key!r} appeared on the live Pi but has no DE "
        f"common name; the UI will render the scientific name verbatim. "
        f"Add an entry to assets/common_names_DE.json."
    )
    assert de[species_key], f"DE name for {species_key} is empty"


def test_genus_sp_entries_follow_naming_convention():
    """Convention: all ``<Genus>_sp.`` keys read '<dt. Gattung> (Art unklar)'.

    Catches drift where a future contributor adds a new genus-fallback
    with a different format ("Spec. unklar", "Genus sp.", etc.).
    """
    de = _load_de_names()
    for key, value in de.items():
        if not key.endswith("_sp."):
            continue
        assert "(Art unklar)" in value, (
            f"{key!r} → {value!r} breaks the '<Gattung> (Art unklar)' "
            f"convention used by Fringilla_sp., Phylloscopus_sp., etc."
        )
