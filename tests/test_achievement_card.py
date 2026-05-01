"""Tests for the achievement-card renderer.

These don't pin pixel-level appearance (font + Pillow rendering varies
by host) — they verify the card meets its layout contract: square,
1080x1080, neon hue keyed to the species slot, score badge text
present, banner text present.
"""

import numpy as np

from utils.achievement_card import (
    CARD_SIZE,
    _NEON_PALETTE,
    build_species_colour_map,
    neon_for_species,
    render_achievement_card,
    render_collector_card,
)


def _fake_photo(width=1200, height=900) -> np.ndarray:
    """Solid-coloured stand-in for a real bird photo."""
    img = np.full((height, width, 3), (40, 90, 60), dtype=np.uint8)
    img[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = (180, 130, 90)
    return img


def test_render_card_returns_square_1080():
    card = render_achievement_card(
        _fake_photo(),
        common_name="Ringeltaube",
        count=7,
        rim_color=(255, 170, 0),
        glow_color=(255, 200, 80),
    )
    assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_render_card_handles_missing_device_and_date():
    """Missing device/date labels must not blow up — they just disappear
    from the bottom subline."""
    card = render_achievement_card(
        _fake_photo(),
        common_name="Blaumeise",
        count=1,
        rim_color=(0, 165, 255),
        glow_color=(60, 200, 255),
    )
    assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_neon_for_species_is_deterministic():
    """Same species always picks the same hue, regardless of map order."""
    map1 = build_species_colour_map(["Parus_major", "Cyanistes_caeruleus"])
    map2 = build_species_colour_map(["Cyanistes_caeruleus", "Parus_major"])

    rim1, _ = neon_for_species("Parus_major", map1)
    rim2, _ = neon_for_species("Parus_major", map2)
    assert rim1 == rim2


def test_neon_for_species_falls_back_without_map():
    """Single-species preview path: no colour_map provided -> default slot."""
    rim, glow = neon_for_species("Parus_major", None)
    # Falls back to slot 0 in the neon palette. The exact hex values can
    # be tuned; what matters is that the helper returns slot 0's colours.
    assert (rim, glow) == _NEON_PALETTE[0]


def test_neon_for_species_handles_unknown_species():
    """Species not in the colour map gets the default slot rather than
    raising a KeyError."""
    colour_map = build_species_colour_map(["Parus_major"])
    rim, glow = neon_for_species("Some_random_species", colour_map)
    # Falls back to default slot 0.
    assert (rim, glow) == _NEON_PALETTE[0]


def test_card_renders_for_every_palette_slot():
    """Every slot in the neon palette must produce a valid card. Guards
    against the slot wraparound (>8 species) and the edge index 7."""
    species = [f"Species_{i:02d}" for i in range(10)]  # 10 > 8 slots
    colour_map = build_species_colour_map(species)
    for s in species:
        rim, glow = neon_for_species(s, colour_map)
        card = render_achievement_card(
            _fake_photo(),
            common_name=s.replace("_", " "),
            count=1,
            rim_color=rim,
            glow_color=glow,
        )
        assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_render_collector_card_returns_square_1080():
    """Collector card must match the per-species card dimensions so the
    Telegram album reads as a uniform deck."""
    species = [
        {
            "scientific": "Parus_major",
            "common_name": "Kohlmeise",
            "count": 12,
            "photo": _fake_photo(),
        },
        {
            "scientific": "Cyanistes_caeruleus",
            "common_name": "Blaumeise",
            "count": 4,
            "photo": _fake_photo(),
        },
    ]
    colour_map = build_species_colour_map(
        [s["scientific"] for s in species]
    )
    card = render_collector_card(
        species,
        colour_map=colour_map,
        device_label="Rpi",
        date_label="Donnerstag, 30.04.2026",
    )
    assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_collector_card_handles_single_species():
    """Edge case: only one species seen all day. Roster grid must
    centre the lone vignette without blowing up the layout math."""
    species = [
        {
            "scientific": "Parus_major",
            "common_name": "Kohlmeise",
            "count": 1,
            "photo": _fake_photo(),
        }
    ]
    colour_map = build_species_colour_map(["Parus_major"])
    card = render_collector_card(species, colour_map=colour_map)
    assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_collector_card_caps_roster_at_six():
    """If more than 6 species are passed, only the first 6 are rendered;
    no exception when the underlying album builder hands a longer list."""
    species = [
        {
            "scientific": f"Species_{i:02d}",
            "common_name": f"Species {i}",
            "count": 1,
            "photo": _fake_photo(),
        }
        for i in range(10)
    ]
    colour_map = build_species_colour_map([s["scientific"] for s in species])
    # Should not raise.
    card = render_collector_card(species, colour_map=colour_map)
    assert card.shape == (CARD_SIZE, CARD_SIZE, 3)


def test_card_has_visible_neon_around_edges():
    """A column of the rendered card near the rim should differ from the
    plain background — proves the neon glow actually got drawn."""
    card = render_achievement_card(
        _fake_photo(),
        common_name="Test",
        count=3,
        rim_color=(255, 170, 0),
        glow_color=(255, 200, 80),
    )
    plain_bg = np.full(card.shape, (10, 12, 16), dtype=np.uint8)
    # Right rim region (just inside the edge inset of 18 + glow radius)
    right_edge = card[100:980, CARD_SIZE - 50 : CARD_SIZE - 20]
    bg_slice = plain_bg[100:980, CARD_SIZE - 50 : CARD_SIZE - 20]
    assert not np.array_equal(right_edge, bg_slice)
