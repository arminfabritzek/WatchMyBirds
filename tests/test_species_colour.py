"""Tests for species-colour assignment and reference-image mapping.

These tests pin the deterministic contract that the Review surface
relies on to keep the same scientific name on the same colour slot
across the whole workspace, plus the reference image filename lookup
in ``assets/review_species/``.
"""

from core.species_colours import SPECIES_COLOUR_SLOTS, assign_species_colours

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
    assert (
        three["Pica_pica"] == one["Pica_pica"] - 1
        or three["Pica_pica"] != one["Pica_pica"]
    )
    assert (
        assign_species_colours(["parus_major", "Pica_pica", "Cyanistes_caeruleus"])
        == three
    )


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
    colours = assign_species_colours(["Parus_major", None, "", "   ", "Pica_pica"])
    assert colours == {"Parus_major": 0, "Pica_pica": 1}


def test_assign_species_colours_empty_input_returns_empty_map():
    assert assign_species_colours([]) == {}
    assert assign_species_colours(set()) == {}


def test_assign_species_colours_same_species_repeated_collapses_to_one_slot():
    colours = assign_species_colours(["Parus_major"] * 5)
    assert colours == {"Parus_major": 0}
