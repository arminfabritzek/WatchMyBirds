"""The Favorites page in a real browser, by mouse and by touch."""

from __future__ import annotations

import json
import os
from typing import Any

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
        sign_in(page, base)

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


def sign_in(page: Any, base: str) -> None:
    page.goto(base + "/login")
    page.fill("#password", os.environ["EDIT_PASSWORD"])
    page.click('form[action="/login"] button[type="submit"]')
    page.wait_for_url("**/gallery")


STARRED = ("tit_old", "blackbird", "magpie", "tit_new")
LAST_PHOTO_CLEAR_OF_THE_BAR = """() => {
    const photos = document.querySelectorAll('[data-batch-target="detection"]');
    const bar = document.getElementById('inlineEditFooter').getBoundingClientRect();
    return photos[photos.length - 1].getBoundingClientRect().bottom <= bar.top;
}"""


@pytest.mark.parametrize(
    ("width", "touch", "theme"), [(1440, False, "light"), (390, True, "dark")]
)
def test_the_owner_removes_the_star_from_every_photo_on_the_page(
    local_db_app: Flask, album: dict[str, int], width: int, touch: bool, theme: str
) -> None:
    with station(local_db_app, width=width, touch=touch, theme=theme) as (
        page,
        base,
        _requests,
        _delay,
    ):

        def press(selector: str) -> None:
            target = page.locator(selector)
            (target.tap if touch else target.click)()

        sign_in(page, base)
        page.goto(base + "/favorites")
        press(".inline-edit-toggle")
        press("#inlineEditSelectAll")

        assert page.locator("#favorites-grid .wm-tile.selected").count() == 4
        assert page.locator("#inlineEditCounter").inner_text() == "4 selected"
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        assert page.evaluate(LAST_PHOTO_CLEAR_OF_THE_BAR)
        if touch:
            controls = page.locator(
                "#inlineEditFooter label:visible, #inlineEditFooter .btn:visible"
            )
            assert controls.count() == 6
            for control in controls.all():
                assert control.bounding_box()["height"] >= 44, control.inner_text()
            # The bar may take a third of a 720px-high phone screen at most.
            assert page.locator("#inlineEditFooter").bounding_box()["height"] <= 240

        asked: list[str] = []

        def agree(dialog: Any) -> None:
            asked.append(dialog.message)
            dialog.accept()

        page.once("dialog", agree)
        with page.expect_request("**/api/moderation/bulk/unfavorite") as sent:
            press("#inlineEditUnfavorite")

        assert sorted(json.loads(sent.value.post_data)["detection_ids"]) == sorted(
            album[name] for name in STARRED
        )
        assert "4 item(s)" in asked[0]
        assert "automatic deletion" in asked[0]
        page.locator(".empty-state").wait_for()
        assert page.locator("#favorites-grid").count() == 0


def test_declining_the_question_keeps_every_star(
    local_db_app: Flask, album: dict[str, int]
) -> None:
    with station(local_db_app, width=1440, touch=False, theme="light") as (
        page,
        base,
        _requests,
        _delay,
    ):
        sign_in(page, base)
        page.goto(base + "/favorites")
        page.click(".inline-edit-toggle")
        page.click("#inlineEditSelectAll")
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url))
        page.once("dialog", lambda dialog: dialog.dismiss())

        page.click("#inlineEditUnfavorite")
        page.wait_for_timeout(400)

        assert not [url for url in sent if "bulk/unfavorite" in url]
        page.reload()
        assert page.locator("#favorites-grid .wm-tile").count() == 4


def test_selecting_the_whole_page_follows_single_changes(
    local_db_app: Flask, album: dict[str, int]
) -> None:
    with station(local_db_app, width=1440, touch=False, theme="light") as (
        page,
        base,
        _requests,
        _delay,
    ):
        sign_in(page, base)
        page.goto(base + "/favorites")
        whole_page = page.locator("#inlineEditSelectAll")
        counter = page.locator("#inlineEditCounter")

        def state() -> tuple[str, bool, bool]:
            return (
                counter.inner_text(),
                whole_page.is_checked(),
                whole_page.evaluate("box => box.indeterminate"),
            )

        page.click(".inline-edit-toggle")
        whole_page.click()
        assert state() == ("4 selected", True, False)

        page.locator(f"#detection-{album['magpie']} .wm-tile__media").click()
        assert state() == ("3 selected", False, True)

        whole_page.click()
        assert state() == ("4 selected", True, False)

        whole_page.click()
        assert state() == ("0 selected", False, False)
        assert page.locator(".wm-tile.selected").count() == 0

        whole_page.click()
        page.click(".inline-edit-toggle")
        page.click(".inline-edit-toggle")
        assert state() == ("0 selected", False, False)


@pytest.mark.parametrize(
    "address", ["/species/overview?species_key=Parus_major", "/gallery/2026-01-04"]
)
def test_the_whole_page_can_be_selected_wherever_batch_edit_exists(
    local_db_app: Flask, album: dict[str, int], address: str
) -> None:
    with station(local_db_app, width=1440, touch=False, theme="light") as (
        page,
        base,
        _requests,
        _delay,
    ):
        sign_in(page, base)
        page.goto(base + address)
        page.click(".inline-edit-toggle")
        page.click("#inlineEditSelectAll")

        targets = page.locator('[data-batch-target="detection"]')
        assert targets.count() >= 1
        assert page.locator('[data-batch-target="detection"].selected').count() == (
            targets.count()
        )
        assert page.locator("#inlineEditCounter").inner_text() == (
            f"{targets.count()} selected"
        )
        assert page.locator("#inlineEditUnfavorite").count() == 0


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
        sign_in(page, base)

        page.goto(base + "/favorites")

        assert page.locator(".app-bar__toggle").is_hidden()
        assert page.locator('a.app-bar__link[href="/logout"]').is_visible()
        assert page.evaluate(MENU_EDGE) <= width
