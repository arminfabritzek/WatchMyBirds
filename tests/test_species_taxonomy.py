from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.species_geography import rank_geographically
from core.species_geography_runtime import GeographicWeighting
from core.species_taxonomy import GeographicNameIndex, load_geographic_synonyms
from scripts.build_geographic_synonyms import SOURCE_QUERY, SOURCE_URL, build_reference


def test_normalized_exact_names_preserve_model_identity() -> None:
    index = GeographicNameIndex(["Parus_major", "Sitta europaea"], [])
    assert index.resolve(" parus   major ") == "Parus_major"
    assert index.resolve("SITTA_EUROPAEA") == "Sitta europaea"


def test_only_documented_synonyms_resolve_without_epithet_guessing() -> None:
    index = GeographicNameIndex(["Newgenus alba"], [("Oldgenus alba", "Newgenus alba")])
    assert index.resolve("Oldgenus_alba") == "Newgenus alba"
    assert index.resolve("Othergenus_alba") is None
    assert index.resolve("Oldgenus_alba_subspecies") is None
    assert index.resolve("Oldgenus_alba_x_Othergenus_alba") is None


def test_conflicts_and_splits_do_not_merge_geographic_outputs() -> None:
    groups = [("Oldgenus alba", "Newgenus alba", "Newgenus nigra")]
    index = GeographicNameIndex(["Newgenus alba", "Newgenus nigra"], groups)
    assert index.resolve("Oldgenus_alba") is None
    assert index.resolve("Newgenus_alba") == "Newgenus alba"
    conflicting = GeographicNameIndex(
        ["Newgenus alba", "Newgenus nigra"],
        [("Oldgenus alba", "Newgenus alba"), ("Oldgenus alba", "Newgenus nigra")],
    )
    assert conflicting.resolve("Oldgenus_alba") is None
    assert (
        GeographicNameIndex(["Parus major", "Parus_major"], []).resolve("Parus_major")
        is None
    )


@pytest.mark.parametrize(
    "original,target",
    [
        ("Regulus_calendula", "Corthylio_calendula"),
        ("Picoides_pubescens", "Dryobates_pubescens"),
        ("Chen_rossii", "Anser_rossii"),
        ("Anas_clypeata", "Spatula_clypeata"),
        ("Sylvia_communis", "Curruca_communis"),
    ],
)
def test_shipped_reference_covers_independent_genus_changes(
    original: str, target: str
) -> None:
    groups, checksum = load_geographic_synonyms()
    index = GeographicNameIndex([target], groups)
    assert index.resolve(original) == target
    assert len(checksum) == 64


def test_every_shipped_synonym_group_resolves_against_its_unique_target() -> None:
    groups, _checksum = load_geographic_synonyms()
    for group in groups:
        index = GeographicNameIndex([group[0]], [group])
        assert all(index.resolve(name) == group[0] for name in group)


def test_reference_build_excludes_partial_synonyms_and_infraspecific_links() -> None:
    from copy import deepcopy

    valid = {
        "canonicalName": "Oldgenus alba",
        "species": "Newgenus alba",
        "rank": "SPECIES",
        "taxonomicStatus": "HOMOTYPIC_SYNONYM",
        "acceptedKey": 123,
        "speciesKey": 123,
        "nameType": "SCIENTIFIC",
    }
    rows = [valid]
    for field, value in (
        ("rank", "SUBSPECIES"),
        ("taxonomicStatus", "PROPARTE_SYNONYM"),
        ("acceptedKey", 456),
        ("acceptedKey", None),
        ("nameType", "HYBRID"),
        ("canonicalName", "Othergenus alba subspecies"),
    ):
        row = deepcopy(valid)
        row["canonicalName"] = "Othergenus alba"
        row[field] = value
        rows.append(row)
    source = {"url": SOURCE_URL, "params": SOURCE_QUERY, "records": rows}
    reference = build_reference(json.dumps(source).encode(), "2026-10-11")
    assert reference["groups"] == [["Newgenus alba", "Oldgenus alba"]]


def test_partial_taxonomy_cannot_promote_an_unmapped_candidate() -> None:
    candidates = [("Known_species", 0.7), ("Unmapped_species", 0.2)]
    result = rank_geographically(candidates, {"Known_species": 0.001})
    assert result.suggested_species == "Known_species"
    assert not result.weighting_applied
    assert result.unmapped_species == ("Unmapped_species",)
    assert [(r.species, r.weighted_score) for r in result.ranking] == candidates


def test_absence_candidate_does_not_disable_complete_bird_taxonomy() -> None:
    result = rank_geographically(
        [("Known_species", 0.7), ("Other_species", 0.2), ("background", 0.01)],
        {"Known_species": 0.001, "Other_species": 0.9},
    )
    assert result.weighting_applied
    assert not result.unmapped_species
    assert result.suggested_species == "Other_species"


def test_runtime_uses_synonyms_and_records_lookup_without_renaming_candidates() -> None:
    from unittest.mock import Mock

    weighting = GeographicWeighting({})
    weighting._model = Mock()
    weighting._model.predict.return_value = (
        {"Corthylio_calendula": 0.0, "Parus_major": 0.9},
        {"latitude": 10, "longitude": 20},
    )
    result = weighting.assess(
        [("Regulus_calendula", 0.8), ("Parus_major", 0.15)],
        datetime(2026, 10, 1),
        location={"latitude": 10, "longitude": 20},
    )
    assert result is not None
    assessment, evidence = result
    assert assessment.suggested_species == "Parus_major"
    assert assessment.weighting_applied
    assert assessment.ranking[1].species == "Regulus_calendula"
    assert assessment.ranking[1].occurrence_score == 0.0
    assert evidence["name_matches"]["Regulus_calendula"] == "Corthylio_calendula"
    assert len(evidence["taxonomy_sha256"]) == 64
    assert "latitude" not in evidence and "longitude" not in evidence
    json.dumps(assessment.to_dict(), allow_nan=False)
