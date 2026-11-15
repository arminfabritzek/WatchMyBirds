"""The Favorites page in a real browser, by mouse and by touch."""

from __future__ import annotations

import json
import os

import pytest
from flask import Flask

from tests.test_favorites_only_filter import local_db_app as local_db_app
from tests.test_favorites_page import photo
from tests.test_nav_feedback_browser import station


@pytest.fixture
def album(local_db_app: Flask) -> dict[str, int]:
    return {
        "tit_old": photo("Parus_major", "20260101"),
        "blackbird": photo("Turdus_merula", "20260102"),
        "magpie": photo("Pica_pica", "20260103"),
        "tit_new": photo("Parus_major", "20260104"),
        "plain": photo("Erithacus_rubecula", "20260105", star=False),
    }


@pytest.mark.parametrize(
    ("width", "touch", "theme"), [(1440, False, "light"), (390, True, "dark")]
)
def test_a_visitor_browses_filters_and_opens_a_photo(
    local_db_app: Flask, album: dict[str, int], width: int, touch: bool, theme: str
) -> None:
    with station(local_db_app, width=width, touch=touch, theme=theme) as (
        page,
        base,
        _requests,
        _delay,
    ):
        page.goto(base + "/gallery")
        if width < 900:
            page.locator(".app-bar__toggle").click()
        menu = page.locator('a.app-bar__link[href="/favorites"]')
        (menu.tap if touch else menu.click)()
        page.wait_for_url("**/favorites")

        tiles = page.locator("#favorites-grid .wm-tile")
        assert tiles.count() == 4
        assert page.locator(".inline-edit-toggle").count() == 0
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )

        page.select_option('select[name="species"]', "Parus_major")
        page.select_option('select[name="sort"]', "species")
        apply = page.locator('.station-filters button[type="submit"]')
        (apply.tap if touch else apply.click)()
        page.wait_for_url("**/favorites?*species=Parus_major*")
        assert [
            int(tile.get_attribute("data-detection-id")) for tile in tiles.all()
        ] == [album["tit_new"], album["tit_old"]]
        assert not page.evaluate(
            "document.documentElement.classList.contains('nav-pending')"
        )

        opener = tiles.first.locator(".wm-tile__button")
        (opener.tap if touch else opener.click)()
        page.locator(f"#modal-favorites-{album['tit_new']}").wait_for(state="visible")


@pytest.mark.parametrize(
    ("address", "picked"),
    [
        ("/favorites", ("magpie", "tit_old")),
        ("/species/overview?species_key=Parus_major", ("tit_new", "tit_old")),
    ],
)
def test_the_signed_in_owner_collects_photos_for_one_download(
    local_db_app: Flask, album: dict[str, int], address: str, picked: tuple[str, str]
) -> None:
    with station(local_db_app, width=1440, touch=False, theme="light") as (
        page,
        base,
        _requests,
        _delay,
    ):
        page.goto(base + "/login")
        page.fill("#password", os.environ["EDIT_PASSWORD"])
        page.click('form[action="/login"] button[type="submit"]')
        page.wait_for_url("**/gallery")

        page.goto(base + address)
        page.click(".inline-edit-toggle")
        for name in picked:
            page.locator(f"#detection-{album[name]} .wm-tile__media").click()
            # Selecting must not also open the viewer on top of the selection.
            page.wait_for_timeout(400)
            assert page.locator(".modal.show").count() == 0
        assert page.locator("#inlineEditCounter").inner_text().startswith("2 ")

        with page.expect_request("**/api/moderation/bulk/download") as sent:
            page.click("#inlineEditFooter .btn--success")
        assert sorted(json.loads(sent.value.post_data)["detection_ids"]) == sorted(
            album[name] for name in picked
        )


MENU_EDGE = """() => Math.max(...[...document.getElementById('appBarNav').children]
    .filter(entry => entry.getClientRects().length)
    .map(entry => entry.getBoundingClientRect().right))"""


@pytest.mark.parametrize("width", [1101, 1200, 1320, 1321, 1440])
def test_the_signed_in_menu_stays_inside_the_window(
    local_db_app: Flask, width: int
) -> None:
    with station(local_db_app, width=width, touch=False, theme="light") as (
        page,
        base,
        _requests,
        _delay,
    ):
        page.goto(base + "/login")
        page.fill("#password", os.environ["EDIT_PASSWORD"])
        page.click('form[action="/login"] button[type="submit"]')
        page.wait_for_url("**/gallery")

        page.goto(base + "/favorites")

        assert page.locator(".app-bar__toggle").is_hidden()
        assert page.locator('a.app-bar__link[href="/logout"]').is_visible()
        assert page.evaluate(MENU_EDGE) <= width
