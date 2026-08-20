"""Tests for extended species locale handling."""

import json

import pytest

from utils import species_names


@pytest.fixture(autouse=True)
def _clear_species_caches():
    """Clear ``species_names`` caches around each test.

    Each test monkeypatches ``_ASSETS_DIR`` to a tmp tree; the per-locale
    ``@lru_cache`` on ``load_common_names`` / ``load_extended_species``
    would otherwise leak fake-asset data into later tests in the run.
    """
    species_names.load_common_names.cache_clear()
    species_names.load_extended_species.cache_clear()
    species_names._extended_species_keys.cache_clear()
    yield
    species_names.load_common_names.cache_clear()
    species_names.load_extended_species.cache_clear()
    species_names._extended_species_keys.cache_clear()


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
                    "common_nb": "Gråspett",
                },
                {
                    "scientific": "Corvus_corax",
                    "common_de": "Kolkrabe",
                    "common_en": "Common Raven",
                    "common_nb": "",
                },
                {
                    "scientific": "Aquila_test",
                    "common_de": "",
                    "common_en": "Test Eagle",
                    "common_nb": "",
                },
                {
                    "scientific": "Species_only",
                    "common_de": "",
                    "common_en": "",
                    "common_nb": "",
                },
                {
                    "scientific": "Parus_major",
                    "common_de": "Kohlmeise",
                    "common_en": "Great Tit",
                    "common_nb": "Kjøttmeis",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    species_names.load_common_names.cache_clear()
    species_names.load_extended_species.cache_clear()
    species_names._extended_species_keys.cache_clear()

    entries = species_names.load_extended_species("NO")
    common_by_scientific = {row["scientific"]: row["common"] for row in entries}

    assert "Parus_major" not in common_by_scientific
    assert common_by_scientific["Picus_canus"] == "Gråspett"
    assert common_by_scientific["Corvus_corax"] == "Common Raven"
    assert common_by_scientific["Aquila_test"] == "Test Eagle"
    assert common_by_scientific["Species_only"] == "Species only"


def test_load_extended_species_uses_de_fallback_order(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Unknown_species": "Unknown species"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Species_de",
                    "common_de": "Deutsch",
                    "common_en": "English",
                    "common_nb": "Norsk",
                },
                {
                    "scientific": "Species_en",
                    "common_de": "",
                    "common_en": "English only",
                    "common_nb": "",
                },
                {
                    "scientific": "Species_only",
                    "common_de": "",
                    "common_en": "",
                    "common_nb": "",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    species_names.load_common_names.cache_clear()
    species_names.load_extended_species.cache_clear()
    species_names._extended_species_keys.cache_clear()

    entries = species_names.load_extended_species("DE")
    common_by_scientific = {row["scientific"]: row["common"] for row in entries}

    assert common_by_scientific["Species_de"] == "Deutsch"
    assert common_by_scientific["Species_en"] == "English only"
    assert common_by_scientific["Species_only"] == "Species only"


def test_load_common_names_uses_english_overlay(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps(
            {
                "Unknown_species": "Vogel (Art unklar)",
                "Parus_major": "Kohlmeise",
            }
        ),
        encoding="utf-8",
    )
    (assets_dir / "common_names_EN.json").write_text(
        json.dumps(
            {
                "Unknown_species": "Bird (species uncertain)",
                "Parus_major": "Great Tit",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)

    assert species_names.load_common_names("EN") == {
        "Unknown_species": "Bird (species uncertain)",
        "Parus_major": "Great Tit",
    }


def test_active_inat_labels_extend_known_species_and_localized_names(
    tmp_path, monkeypatch
):
    import config as config_module

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Unknown_species": "Vogel (Art unklar)"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Cyanocitta_cristata",
                    "common_de": "Blauhäher",
                    "common_en": "Blue Jay",
                    "common_nb": "Blåskrike",
                }
            ]
        ),
        encoding="utf-8",
    )
    labels_dir = tmp_path / "models" / "classifier" / "inat_bird_mobilenet_v2"
    labels_dir.mkdir(parents=True)
    (labels_dir / "inat_bird_labels.txt").write_text(
        "Cyanocitta cristata (Blue Jay)\n"
        "Picoides pubescens (Downy Woodpecker)\n"
        "background\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda: {
            "CLASSIFIER_BACKEND": "inat_tflite",
            "MODEL_BASE_PATH": str(tmp_path / "models"),
        },
    )
    species_names.clear_species_name_caches()

    names = species_names.load_common_names("DE")

    assert names["Cyanocitta_cristata"] == "Blauhäher"
    assert names["Picoides_pubescens"] == "Downy Woodpecker"
    assert species_names.is_known_species("Picoides_pubescens", "DE") is True
    assert "background" not in names


def test_load_extended_species_uses_en_fallback_order(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Unknown_species": "Vogel (Art unklar)"}),
        encoding="utf-8",
    )
    (assets_dir / "common_names_EN.json").write_text(
        json.dumps({"Unknown_species": "Bird (species uncertain)"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Species_en",
                    "common_de": "Deutsch",
                    "common_en": "English",
                    "common_nb": "Norsk",
                },
                {
                    "scientific": "Species_de",
                    "common_de": "Deutsch only",
                    "common_en": "",
                    "common_nb": "",
                },
                {
                    "scientific": "Species_only",
                    "common_de": "",
                    "common_en": "",
                    "common_nb": "",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)

    entries = species_names.load_extended_species("EN")
    common_by_scientific = {row["scientific"]: row["common"] for row in entries}

    assert common_by_scientific["Species_en"] == "English"
    assert common_by_scientific["Species_de"] == "Deutsch only"
    assert common_by_scientific["Species_only"] == "Species only"


def test_load_common_names_result_survives_caller_inplace_refresh(
    tmp_path, monkeypatch
):
    """Regression for issue #55.

    ``web_interface`` keeps a long-lived ``COMMON_NAMES`` dict and refreshes
    it in place (``clear()`` + ``update()``) whenever the locale setting is
    re-applied — which happens on *every* settings save, because the form
    POSTs the locale field each time. ``load_common_names`` is ``@lru_cache``'d
    and returns the SAME dict object per locale, so binding ``COMMON_NAMES``
    straight to that object made ``clear()`` wipe the cache (and the
    ``update`` source, which was the same object) — leaving every species
    showing its bare scientific name until a restart.

    The fix is that the caller owns a private ``dict(...)`` copy. This test
    pins that the documented app idiom keeps the names AND leaves the cache
    intact for other readers.
    """
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Parus_major": "Kohlmeise"}),
        encoding="utf-8",
    )
    (assets_dir / "common_names_NO.json").write_text(
        json.dumps({"Parus_major": "Kjøttmeis"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    species_names.load_common_names.cache_clear()

    # Mount: caller owns a private copy (the issue-#55 fix).
    common_names = dict(species_names.load_common_names("NO"))
    assert common_names["Parus_major"] == "Kjøttmeis"

    # Settings save re-applies the (unchanged) locale -> in-place refresh.
    new_names = species_names.load_common_names("NO")
    common_names.clear()
    common_names.update(new_names)

    # Names must survive the refresh...
    assert common_names["Parus_major"] == "Kjøttmeis"
    # ...and the shared cache must NOT have been poisoned for other readers.
    assert species_names.load_common_names("NO")["Parus_major"] == "Kjøttmeis"
    assert species_names.load_common_names("DE")["Parus_major"] == "Kohlmeise"


def test_load_extended_species_ignores_legacy_common_no_field(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Unknown_species": "Unknown species"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Legacy_species",
                    "common_de": "",
                    "common_en": "English fallback",
                    "common_no": "Legacy Norwegian",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    species_names.load_common_names.cache_clear()
    species_names.load_extended_species.cache_clear()
    species_names._extended_species_keys.cache_clear()

    entries = species_names.load_extended_species("NO")
    assert entries == [{"scientific": "Legacy_species", "common": "English fallback"}]
