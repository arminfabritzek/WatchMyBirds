"""Tests for species-colour assignment and reference-image mapping.

These tests pin the deterministic contract that the Review surface
relies on to keep the same scientific name on the same colour slot
across the whole workspace, plus the reference image filename lookup
in ``assets/review_species/``.
"""

from pathlib import Path

import pytest

from web.blueprints.review import (
    SPECIES_COLOUR_SLOTS,
    _build_species_ref_image_map,
    _species_ref_image_dir,
    _stamp_species_display_on_event,
    assign_species_colours,
    get_species_ref_image_map,
    resolve_species_ref_image_url,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# assign_species_colours — determinism + wrap + edge cases
# ---------------------------------------------------------------------------


def test_assign_species_colours_is_deterministic_alphabetical():
    """Same input set → same map, ordered by alphabetical sort."""
    one = assign_species_colours(["Pica_pica", "Parus_major", "Cyanistes_caeruleus"])
    two = assign_species_colours({"Cyanistes_caeruleus", "Parus_major", "Pica_pica"})
    three = assign_species_colours(["parus_major", "Pica_pica", "Cyanistes_caeruleus"])  # noqa: E501

    # Same map regardless of input order or container type.
    assert one == two
    # Sort key is the cleaned scientific name as-is, so a different
    # casing produces a different sort position; the map is still
    # deterministic for that input though.
    expected = {
        "Cyanistes_caeruleus": 0,
        "Parus_major": 1,
        "Pica_pica": 2,
    }
    assert one == expected
    # Different casing → different sort, but still deterministic.
    assert three["Pica_pica"] == one["Pica_pica"] - 1 or three["Pica_pica"] != one["Pica_pica"]
    assert assign_species_colours(["parus_major", "Pica_pica", "Cyanistes_caeruleus"]) == three


def test_assign_species_colours_wraps_at_eight_slots():
    """The 9th species starts at slot 0 again."""
    species = [f"Species_{chr(ord('A') + i)}" for i in range(10)]
    colours = assign_species_colours(species)

    # 10 distinct species → 10 entries, but the values cycle 0..7.
    assert len(colours) == 10
    assert all(0 <= value < SPECIES_COLOUR_SLOTS for value in colours.values())
    # Sorted alphabetically, slot 0 should appear at index 0 and 8.
    sorted_keys = sorted(colours.keys())
    assert colours[sorted_keys[0]] == 0
    assert colours[sorted_keys[SPECIES_COLOUR_SLOTS]] == 0
    assert colours[sorted_keys[SPECIES_COLOUR_SLOTS + 1]] == 1


def test_assign_species_colours_ignores_blank_inputs():
    """None / empty / whitespace-only entries do not consume a slot."""
    colours = assign_species_colours(
        ["Parus_major", None, "", "   ", "Pica_pica"]
    )
    assert colours == {"Parus_major": 0, "Pica_pica": 1}


def test_assign_species_colours_empty_input_returns_empty_map():
    assert assign_species_colours([]) == {}
    assert assign_species_colours(set()) == {}


def test_assign_species_colours_same_species_repeated_collapses_to_one_slot():
    colours = assign_species_colours(["Parus_major"] * 5)
    assert colours == {"Parus_major": 0}


# ---------------------------------------------------------------------------
# Reference image lookup
# ---------------------------------------------------------------------------


def test_species_ref_image_dir_points_at_assets_review_species():
    expected = _project_root() / "assets" / "review_species"
    assert _species_ref_image_dir() == expected
    assert expected.is_dir(), "fixture asset directory missing on disk"


def test_build_species_ref_image_map_indexes_known_species():
    """The cached scan must include the known fixture species and use
    the ``/assets/review_species/`` URL prefix Flask serves them under."""
    mapping = _build_species_ref_image_map()

    # A handful of species we know exist in the fixture set.
    assert "Parus_major" in mapping
    assert mapping["Parus_major"].startswith("/assets/review_species/Parus_major.")
    assert "Pica_pica" in mapping
    assert "Aegithalos_caudatus" in mapping

    # README.md is not a reference image even though it lives in the
    # same directory.
    assert "README" not in mapping


def test_build_species_ref_image_map_prefers_webp_over_png(tmp_path, monkeypatch):
    """When both ``.webp`` and ``.png`` exist for the same species, the
    smaller ``.webp`` payload wins so the overlay is cheap on mobile."""
    fake_dir = tmp_path / "review_species"
    fake_dir.mkdir()
    (fake_dir / "Parus_major.webp").write_bytes(b"webp")
    (fake_dir / "Parus_major.png").write_bytes(b"png")
    (fake_dir / "Pica_pica.png").write_bytes(b"png-only")

    monkeypatch.setattr(
        "web.blueprints.review._species_ref_image_dir",
        lambda: fake_dir,
    )

    mapping = _build_species_ref_image_map()
    assert mapping["Parus_major"].endswith(".webp")
    assert mapping["Pica_pica"].endswith(".png")


def test_resolve_species_ref_image_url_returns_none_for_unknown_species():
    assert resolve_species_ref_image_url(None) is None
    assert resolve_species_ref_image_url("") is None
    assert resolve_species_ref_image_url("   ") is None
    assert resolve_species_ref_image_url("Made_up_species_xyz_zzz") is None


def test_resolve_species_ref_image_url_returns_path_for_known_species():
    url = resolve_species_ref_image_url("Parus_major")
    assert url is not None
    assert url.startswith("/assets/review_species/Parus_major.")


# ---------------------------------------------------------------------------
# _stamp_species_display_on_event — regression guards
# ---------------------------------------------------------------------------


def test_stamp_species_display_on_event_without_members_or_batch():
    """Regression: the fragment endpoint's no-batch early return used
    to ship an unstamped event payload, which made every event in the
    fragment render its reference image as the initial fallback even
    though the page rail payload was stamped correctly.

    The bare single-event path (no ``members``, no ``continuity_batch``)
    must still receive ``species_colour``, ``species_colour_key`` and
    ``species_ref_image_url`` directly on the top-level event dict.
    """
    event_payload = {
        "event_key": "bird-event-xyz",
        "candidate_species": "Parus_major",
        "candidate_species_common": "Kohlmeise",
    }
    colour_map = assign_species_colours(["Parus_major"])

    _stamp_species_display_on_event(event_payload, colour_map)

    assert event_payload["species_colour_key"] == "Parus_major"
    assert event_payload["species_colour"] == 0
    # Reference image must be resolved from the real fixture set on
    # disk. Parus_major exists as a .png there.
    assert event_payload["species_ref_image_url"] is not None
    assert event_payload["species_ref_image_url"].endswith("Parus_major.png")


def test_stamp_species_display_on_event_with_members_and_batch():
    """Members + continuity_batch members + quick_species all get stamped."""
    event_payload = {
        "candidate_species": "Parus_major",
        "members": [
            {"candidate_species": "Parus_major", "best_detection_id": 1},
            {"candidate_species": "Pica_pica", "best_detection_id": 2},
        ],
        "continuity_batch": {
            "review_members": [
                {"candidate_species": "Parus_major", "best_detection_id": 1},
            ],
            "anchor_members": [
                {"candidate_species": "Pica_pica", "best_detection_id": 99},
            ],
        },
        "quick_species": [
            {"scientific": "Parus_major"},
            {"scientific": "Cyanistes_caeruleus"},
        ],
    }
    colour_map = assign_species_colours(
        ["Parus_major", "Pica_pica", "Cyanistes_caeruleus"]
    )

    _stamp_species_display_on_event(event_payload, colour_map)

    # Event + members.
    assert event_payload["species_colour"] == colour_map["Parus_major"]
    assert event_payload["members"][0]["species_colour"] == colour_map["Parus_major"]
    assert event_payload["members"][1]["species_colour"] == colour_map["Pica_pica"]
    # Batch members (both sides).
    batch = event_payload["continuity_batch"]
    assert batch["review_members"][0]["species_colour"] == colour_map["Parus_major"]
    assert batch["anchor_members"][0]["species_colour"] == colour_map["Pica_pica"]
    # Quick-pick entries stamped directly from the scientific name.
    assert event_payload["quick_species"][0]["species_colour"] == colour_map["Parus_major"]
    assert event_payload["quick_species"][1]["species_colour"] == colour_map["Cyanistes_caeruleus"]


def test_get_species_ref_image_map_caches_filesystem_scan(monkeypatch):
    """The cache must hold across calls; we only hit the filesystem
    once. We test by clearing the cache and counting builder calls."""
    import web.blueprints.review as review_module

    # Reset the cache.
    review_module._SPECIES_REF_IMAGE_CACHE = None

    call_count = {"n": 0}
    real_builder = review_module._build_species_ref_image_map

    def counting_builder():
        call_count["n"] += 1
        return real_builder()

    monkeypatch.setattr(
        review_module,
        "_build_species_ref_image_map",
        counting_builder,
    )

    first = get_species_ref_image_map()
    second = get_species_ref_image_map()
    third = get_species_ref_image_map()

    assert first is second is third
    assert call_count["n"] == 1
