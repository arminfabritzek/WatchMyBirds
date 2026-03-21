"""Tests for the extended species catalog generator."""

from scripts import generate_extended_species as catalog


def test_build_entries_emits_common_nb(monkeypatch):
    locale_rows = {
        "common_de": {
            "Picus_canus": {
                "scientific": "Picus_canus",
                "common_de": "Grauspecht",
                "common_en": "Grey-headed Woodpecker",
            }
        },
        "common_nb": {
            "Picus_canus": {
                "scientific": "Picus_canus",
                "common_nb": "Gråspett",
                "common_en": "Grey-headed Woodpecker",
            }
        },
    }

    def fake_fetch(locale: str, field_name: str):
        return locale_rows[field_name]

    monkeypatch.setattr(catalog, "fetch_locale_entries", fake_fetch)

    entries = catalog.build_entries()

    assert entries == [
        {
            "scientific": "Picus_canus",
            "common_de": "Grauspecht",
            "common_en": "Grey-headed Woodpecker",
            "common_nb": "Gråspett",
        }
    ]
    assert "common_no" not in entries[0]
