"""Tests for extended species locale handling."""

import json

from utils import species_names


def test_load_extended_species_uses_multilang_fields_and_fallbacks(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps(
            {
                "Unknown_species": "Unknown species",
                "Parus_major": "Kohlmeise",
            }
        ),
        encoding="utf-8",
    )
    (assets_dir / "common_names_NO.json").write_text(
        json.dumps({"Parus_major": "Kjøttmeis"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Picus_canus",
                    "common_de": "Grauspecht",
                    "common_en": "Grey-headed Woodpecker",
                    "common_no": "Gråspett",
                },
                {
                    "scientific": "Corvus_corax",
                    "common_de": "Kolkrabe",
                    "common_en": "Common Raven",
                    "common_no": "",
                },
                {
                    "scientific": "Aquila_test",
                    "common_de": "",
                    "common_en": "Test Eagle",
                    "common_no": "",
                },
                {
                    "scientific": "Species_only",
                    "common_de": "",
                    "common_en": "",
                    "common_no": "",
                },
                {
                    "scientific": "Parus_major",
                    "common_de": "Kohlmeise",
                    "common_en": "Great Tit",
                    "common_no": "Kjøttmeis",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)

    entries = species_names.load_extended_species("NO")
    common_by_scientific = {row["scientific"]: row["common"] for row in entries}

    assert "Parus_major" not in common_by_scientific
    assert common_by_scientific["Picus_canus"] == "Gråspett"
    assert common_by_scientific["Corvus_corax"] == "Kolkrabe"
    assert common_by_scientific["Aquila_test"] == "Test Eagle"
    assert common_by_scientific["Species_only"] == "Species only"
