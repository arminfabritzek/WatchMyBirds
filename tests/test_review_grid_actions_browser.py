from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from tests.test_station_observations import seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.path_manager import get_path_manager
from web.services import station_service


@pytest.fixture(params=["chromium", "firefox"])
def review_page(local_db_app: Flask, request: pytest.FixtureRequest) -> Iterator[Any]:
    api = pytest.importorskip("playwright.sync_api")
    paths = get_path_manager()
    with closing_connection() as conn:
        for identifier in (101, 102, 103, 201, 202, 203):
            filename = seed(
                conn,
                identifier=identifier,
                species="Pica_pica" if identifier < 200 else "Parus_major",
                state="uncertain",
            )
            original = paths.get_original_path(filename)
            original.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640, 420), (100, 130, 110)).save(original)
    with api.sync_playwright() as playwright:
        try:
            browser = getattr(playwright, request.param).launch()
        except api.Error as error:
            pytest.skip(
                f"{request.param} is unavailable: {error.message.splitlines()[0]}"
            )
        server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(f"http://127.0.0.1:{server.server_port}/login")
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()
            page.goto(f"http://127.0.0.1:{server.server_port}/admin/review?day=all")
            yield page
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)


def _card(page: Any, identifier: int) -> Any:
    return page.locator("[data-review-grid-card]").filter(
        has=page.locator(f'.review-grid__tile[data-detection-id="{identifier}"]')
    )


def _assert_no_selection(page: Any) -> None:
    from playwright.sync_api import expect

    expect(page.locator(".review-grid__tile-checkbox:checked")).to_have_count(0)
    expect(page.locator(".review-grid__tile.is-selected")).to_have_count(0)
    expect(page.locator("[data-review-grid-card].has-selection")).to_have_count(0)
    expect(page.locator("[data-review-grid-card-selected]:visible")).to_have_count(0)


@pytest.mark.parametrize(
    "action", ["approve_event", "relabel_event", "unresolved_event", "no_bird_event"]
)
def test_review_answers_clear_selection_without_changing_scope(
    review_page: Any, action: str
) -> None:
    from playwright.sync_api import expect

    page = review_page
    card = _card(page, 201)
    card.locator('input[value="201"]').check()
    with (
        page.expect_navigation(wait_until="load"),
        page.expect_response(
            lambda response: (
                "/api/labels/answer" in response.url
                and response.request.method == "POST"
            )
        ) as result,
    ):
        card.locator(f'[data-review-grid-action="{action}"]').click()
        if action == "relabel_event":
            page.get_by_placeholder("Search species...").fill("Pica pica")
            page.locator(".wm-species-picker-item").filter(
                has_text="Pica pica"
            ).first.click()
    assert result.value.status == 200
    payload = result.value.request.post_data_json
    assert payload["detection_ids"] == [201]
    assert payload["review_only"] is True
    assert "image_bird_presence" not in payload
    if action == "no_bird_event":
        assert payload["object_bird_presence"] == "absent"
    expected_facts = {
        "approve_event": {
            (201, "bird_presence", "present", None),
            (201, "species_identity", "confirmed", "Parus_major"),
        },
        "relabel_event": {
            (201, "bird_presence", "present", None),
            (201, "species_identity", "corrected", "Pica_pica"),
        },
        "unresolved_event": {
            (201, "bird_presence", "present", None),
            (201, "species_identity", "unknown", None),
        },
        "no_bird_event": {(201, "bird_presence", "absent", None)},
    }[action]
    with closing_connection() as conn:
        assert {
            tuple(row)
            for row in conn.execute(
                "SELECT detection_id, fact_type, answer_value, species_key "
                "FROM current_human_label_facts"
            )
        } == expected_facts
        assert (
            conn.execute(
                "SELECT cls_class_name FROM classifications WHERE detection_id=201"
            ).fetchone()[0]
            == "Parus_major"
        )
    _assert_no_selection(page)
    expect(page.locator('.review-grid__tile[data-detection-id="201"]')).to_have_count(0)
    card = _card(page, 202)
    card.locator('input[value="202"]').check()
    card.locator('input[value="203"]').check()
    with page.expect_navigation(wait_until="load"):
        card.locator('[data-review-grid-action="unresolved_event"]').click()
    _assert_no_selection(page)
    expect(page.locator("[data-review-grid-card]")).to_have_count(1)
    expect(_card(page, 101).locator(".review-grid__tile-checkbox")).to_have_count(3)
    for identifier in (202, 203):
        expected_facts.update(
            {
                (identifier, "bird_presence", "present", None),
                (identifier, "species_identity", "unknown", None),
            }
        )
    with closing_connection() as conn:
        assert {
            tuple(row)
            for row in conn.execute(
                "SELECT detection_id, fact_type, answer_value, species_key "
                "FROM current_human_label_facts"
            )
        } == expected_facts
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM current_human_label_facts WHERE scope='image'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id IN (101,102,103)"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_review_header_trash_preserves_labels_and_originals(
    review_page: Any, tmp_path: Path, width: int, theme: str
) -> None:
    from playwright.sync_api import expect

    page = review_page
    page.set_viewport_size({"width": width, "height": 900})
    page.emulate_media(color_scheme=theme)
    page.evaluate("theme => localStorage.setItem('wmb-theme', theme)", theme)
    page.reload()
    expect(page.locator("html")).to_have_attribute("data-theme", theme)
    card = _card(page, 201)
    expect(
        card.get_by_role("button", name="No bird(s) in crop(s)", exact=True)
    ).to_be_visible()
    expect(card.get_by_role("button", name="Move to Trash", exact=True)).to_be_visible()
    assert page.evaluate(
        "() => document.documentElement.scrollWidth <= window.innerWidth"
    )
    card.locator(".review-grid__card-header").screenshot(
        path=str(tmp_path / f"review-header-{width}-{theme}.png")
    )
    card.locator('input[value="201"]').check()
    prompts: list[str] = []

    def accept(dialog: Any) -> None:
        prompts.append(dialog.message)
        dialog.accept()

    page.once("dialog", accept)
    with (
        page.expect_navigation(wait_until="load"),
        page.expect_response(
            lambda response: "/api/trash/move" in response.url
        ) as result,
    ):
        card.get_by_role("button", name="Move to Trash (1)", exact=True).click()
    assert result.value.status == 200
    assert result.value.request.post_data_json == {"detection_ids": [201]}
    assert prompts == ["Move 1 selected object(s) to Trash?"]
    _assert_no_selection(page)
    page.once("dialog", accept)
    with (
        page.expect_navigation(wait_until="load"),
        page.expect_response(
            lambda response: "/api/trash/move" in response.url
        ) as result,
    ):
        _card(page, 202).get_by_role("button", name="Move to Trash", exact=True).click()
    assert result.value.status == 200
    assert set(result.value.request.post_data_json["detection_ids"]) == {202, 203}
    assert prompts[-1] == "Move all 2 open object(s) in this event to Trash?"
    _assert_no_selection(page)
    with closing_connection() as conn:
        rows = conn.execute(
            "SELECT detection_id, status, trashed_at, image_filename FROM detections"
        ).fetchall()
        assert {row["detection_id"] for row in rows if row["status"] == "rejected"} == {
            201,
            202,
            203,
        }
        assert all(row["trashed_at"] for row in rows if row["status"] == "rejected")
        assert all(
            get_path_manager().get_original_path(row["image_filename"]).is_file()
            for row in rows
        )
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 0


def test_review_cancel_and_failure_keep_selection_but_restoration_clears_it(
    review_page: Any,
) -> None:
    from playwright.sync_api import expect

    page = review_page
    card = _card(page, 201)
    checkbox = card.locator('input[value="201"]')
    checkbox.check()
    card.locator('[data-review-grid-action="relabel_event"]').click()
    page.get_by_placeholder("Search species...").wait_for(state="visible")
    page.keyboard.press("Escape")
    expect(checkbox).to_be_checked()
    page.once("dialog", lambda dialog: dialog.dismiss())
    card.locator('[data-review-grid-action="trash_event"]').click()
    expect(checkbox).to_be_checked()

    def fail(route: Any) -> None:
        route.fulfill(
            status=409,
            content_type="application/json",
            body='{"status":"error","message":"The selection has changed"}',
        )

    for endpoint, action in (
        ("/api/labels/answer", "no_bird_event"),
        ("/api/trash/move", "trash_event"),
    ):
        page.route("**" + endpoint, fail)
        if action == "trash_event":
            page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_response(
            lambda response, endpoint=endpoint: endpoint in response.url
        ) as result:
            card.locator(f'[data-review-grid-action="{action}"]').click()
        assert result.value.status == 409
        expect(checkbox).to_be_checked()
        expect(card.locator(f'[data-review-grid-action="{action}"]')).to_be_enabled()
        page.unroute("**" + endpoint, fail)
    page.evaluate("() => window.dispatchEvent(new PageTransitionEvent('pagehide'))")
    _assert_no_selection(page)
    checkbox.evaluate("checkbox => { checkbox.checked = true; }")
    page.evaluate(
        "() => window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true}))"
    )
    _assert_no_selection(page)
    checkbox.check()
    page.goto(page.url.split("/admin/review")[0] + "/privacy")
    page.go_back()
    _assert_no_selection(page)
    with closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM detections WHERE status != 'active'"
            ).fetchone()[0]
            == 0
        )


def test_review_shift_click_stays_within_its_event(review_page: Any) -> None:
    from playwright.sync_api import expect

    page = review_page
    first = _card(page, 201)
    second = _card(page, 101)
    first.locator("[data-tile-image]").nth(2).click()
    expect(first.locator('input[value="203"]')).to_be_checked()
    second.locator("[data-tile-image]").first.click(modifiers=["Shift"])
    expect(second.locator('input[value="101"]')).to_be_checked()
    expect(second.locator(".review-grid__tile-checkbox:checked")).to_have_count(1)
    first.locator("[data-tile-image]").first.click(modifiers=["Shift"])
    expect(first.locator(".review-grid__tile-checkbox:checked")).to_have_count(3)
    expect(second.locator(".review-grid__tile-checkbox:checked")).to_have_count(1)


@pytest.fixture
def paginated_review_page(review_page: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    original = station_service.grouped_review

    def grouped_review(**kwargs: Any) -> dict[str, Any]:
        return {**original(**kwargs), "page_count": 30, "page": kwargs["page"]}

    monkeypatch.setattr(station_service, "grouped_review", grouped_review)
    review_page.goto(
        review_page.url.split("?")[0] + "?day=all&start=2026-10-01&end=2026-10-20"
    )
    return review_page


@pytest.mark.parametrize(
    ("control", "key"), [("link", "Enter"), ("jump", "Enter"), ("jump", "Space")]
)
def test_review_pagination_keyboard_never_answers_objects(
    paginated_review_page: Any, control: str, key: str
) -> None:
    from playwright.sync_api import expect

    page = paginated_review_page
    writes: list[str] = []
    page.on(
        "request",
        lambda request: (
            writes.append(request.url) if request.method == "POST" else None
        ),
    )
    if control == "link":
        target = page.get_by_role("link", name="Page 2", exact=True)
    else:
        page.get_by_label("Go to page").fill("2")
        page.get_by_label("Go to page").press("Tab")
        target = page.get_by_role("button", name="Go", exact=True)
        expect(target).to_be_focused()
    target.focus()
    try:
        with page.expect_navigation(wait_until="load", timeout=5000):
            target.press(key)
    finally:
        assert writes == []
    assert parse_qs(urlparse(page.url).query) == {
        "answered": ["0"],
        "day": ["all"],
        "start": ["2026-10-01"],
        "end": ["2026-10-20"],
        "page": ["2"],
    }
    expect(page.locator(".wm-modal.show")).to_have_count(0)
    with closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 0
