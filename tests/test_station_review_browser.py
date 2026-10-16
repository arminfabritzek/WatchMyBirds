from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
from flask import Flask
from werkzeug.serving import make_server

from core.human_label_core import (
    HumanAnswer,
    record_human_answer,
)
from tests.test_station_observations import provenance, seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.db.detections import fetch_observation_page, fetch_species_covers


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_review_answers_complete_and_can_be_corrected(
    local_db_app: Flask, tmp_path: Path, width: int, theme: str
) -> None:
    api = pytest.importorskip("playwright.sync_api")
    with closing_connection() as conn:
        seed(conn, identifier=42, species="Pica_pica", state="uncertain")
        seed(conn, identifier=43, species="bird", state="uncertain")
    with api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception:
            pytest.skip("Chromium for Playwright is unavailable")
        server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        page = browser.new_page(
            viewport={"width": width, "height": 900}, color_scheme=theme
        )
        page.add_init_script(f"localStorage.setItem('wmb-theme', '{theme}')")
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(base + "/login")
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()

            def open_object(identifier: int, answered: bool = False) -> object:
                response = page.goto(base + f"/admin/review?answered={int(answered)}")
                assert response.status == 200
                if answered:
                    page.locator(
                        f'article[data-detection-id="{identifier}"] .wm-tile__button'
                    ).click()
                else:
                    page.locator(
                        f'.review-grid__tile[data-detection-id="{identifier}"] [data-tile-image]'
                    ).dblclick()
                modal = page.locator(f"#modal-station_review-{identifier}")
                modal.wait_for(state="visible")
                return modal

            for identifier in (42, 43):
                modal = open_object(identifier)
                with page.expect_response(
                    lambda r: (
                        "/api/labels/answer" in r.url and r.request.method == "POST"
                    )
                ) as result:
                    modal.get_by_role(
                        "button", name="Bird · species unknown", exact=True
                    ).click()
                assert result.value.status == 200
                with closing_connection() as conn:
                    assert identifier not in [
                        r["detection_id"]
                        for r in fetch_observation_page(conn, answered=False)[0]
                    ]
                    assert identifier in [
                        r["detection_id"]
                        for r in fetch_observation_page(conn, answered=True)[0]
                    ]
            modal = open_object(42, answered=True)
            modal.get_by_role("button", name="Change species", exact=True).click()
            page.get_by_placeholder("Search species...").fill("Elster")
            page.locator(".wm-species-picker-item").filter(
                has_text="Pica pica"
            ).first.click()
            with page.expect_response(
                lambda r: "/api/labels/answer" in r.url and r.request.method == "POST"
            ) as result:
                modal.get_by_role("button", name="Save bird", exact=True).click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_species_covers(conn)[0]["species_key"] == "Pica_pica"
            modal = open_object(42, answered=True)
            with page.expect_response(
                lambda r: "/api/labels/answer" in r.url and r.request.method == "POST"
            ) as result:
                modal.get_by_role("button", name="No bird", exact=True).click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert all(
                    row["species_key"] == "Unknown_species"
                    for row in fetch_species_covers(conn)
                )
                assert fetch_observation_page(conn, answered=False)[1] == 0
            with closing_connection() as conn:
                filename = seed(conn, identifier=44, state="uncertain")
                record_human_answer(
                    conn,
                    HumanAnswer(filename, image_bird_presence="absent"),
                    provenance(),
                )
            modal = open_object(44, answered=True)
            notice = modal.locator("[data-image-absence]")
            notice.wait_for(state="visible")
            assert page.locator("html").get_attribute("data-theme") == theme
            page.wait_for_function(
                "id => getComputedStyle(document.getElementById(id)).opacity === '1'",
                arg="modal-station_review-44",
            )
            page.screenshot(
                path=str(tmp_path / f"review-{width}-{theme}.png"),
                animations="disabled",
            )
            with (
                page.expect_navigation(wait_until="load"),
                page.expect_response(
                    lambda r: (
                        "/api/labels/answer" in r.url and r.request.method == "POST"
                    )
                ) as result,
            ):
                notice.get_by_role(
                    "button", name="Withdraw full-image answer", exact=True
                ).click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM current_human_label_facts WHERE image_filename=?",
                        (filename,),
                    ).fetchone()[0]
                    == 0
                )
                assert 44 in [
                    r["detection_id"]
                    for r in fetch_observation_page(conn, answered=False)[0]
                ]
                record_human_answer(
                    conn,
                    HumanAnswer(filename, detection_id=44, species_identity="unknown"),
                    provenance(),
                )
            with closing_connection() as conn:
                seed(conn, identifier=71, species="Pica_pica", state="uncertain")
                seed(conn, identifier=72, species="Pica_pica", state="uncertain")
            page.goto(base + "/admin/review")
            card = page.locator("[data-review-grid-card]").filter(
                has=page.locator('[data-detection-id="71"]')
            )
            card.locator('input[value="71"]').check()
            with (
                page.expect_navigation(wait_until="load"),
                page.expect_response(lambda r: "/api/labels/answer" in r.url) as result,
            ):
                card.locator('[data-review-grid-action="unresolved_event"]').click()
            assert result.value.status == 200
            page.goto(base + "/admin/review")
            assert (
                page.locator('.review-grid__tile[data-detection-id="71"]').count() == 0
            )
            assert (
                page.locator('.review-grid__tile[data-detection-id="72"]').count() == 1
            )
            with (
                page.expect_navigation(wait_until="load"),
                page.expect_response(lambda r: "/api/labels/answer" in r.url) as result,
            ):
                page.locator('[data-review-grid-action="approve_event"]').click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_observation_page(conn, answered=False)[1] == 0
                seed(conn, identifier=73, species="Pica_pica", state="uncertain")
            page.goto(base + "/admin/review")
            with (
                page.expect_navigation(wait_until="load"),
                page.expect_response(lambda r: "/api/labels/answer" in r.url) as result,
            ):
                page.locator('[data-review-grid-action="no_bird_event"]').click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_observation_page(conn, answered=False)[1] == 0
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
