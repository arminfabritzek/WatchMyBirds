from __future__ import annotations

from threading import Thread

import pytest
from flask import Flask
from werkzeug.serving import make_server

from core.human_label_core import (
    HumanAnswer,
    append_fact,
    ensure_object_subject,
    record_human_answer,
)
from tests.test_station_observations import provenance, seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.db.detections import fetch_observation_page, fetch_species_covers


def test_review_answers_complete_and_can_be_corrected(local_db_app: Flask) -> None:
    api = pytest.importorskip("playwright.sync_api")
    with closing_connection() as conn:
        seed(conn, identifier=42, species="Pica_pica")
        seed(conn, identifier=43, species="bird")
    with api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception:
            pytest.skip("Chromium for Playwright is unavailable")
        server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(base + "/login")
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()

            def open_object(identifier: int, answered: bool = False) -> object:
                page.goto(base + f"/admin/review?answered={int(answered)}")
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
                filename = seed(conn, identifier=55478)
                record_human_answer(
                    conn,
                    HumanAnswer(filename, image_bird_presence="absent"),
                    provenance(),
                )
                append_fact(
                    conn,
                    subject_id=ensure_object_subject(conn, 55478),
                    fact_type="species_identity",
                    answer_value="corrected",
                    species_key="Passer_domesticus",
                    provenance=provenance(),
                )
            modal = open_object(55478)
            assert modal.locator("[data-review-conflict]").is_visible()
            with page.expect_response(
                lambda r: "/api/labels/answer" in r.url and r.request.method == "POST"
            ) as result:
                modal.get_by_role(
                    "button", name="This image contains a bird", exact=True
                ).click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_observation_page(conn, answered=False)[1] == 0
                assert "Passer_domesticus" in [
                    row["species_key"] for row in fetch_species_covers(conn)
                ]
            with closing_connection() as conn:
                seed(conn, identifier=71, species="Pica_pica")
                seed(conn, identifier=72, species="Pica_pica")
            page.goto(base + "/admin/review")
            card = page.locator("[data-review-grid-card]").filter(
                has=page.locator('[data-detection-id="71"]')
            )
            card.locator('input[value="71"]').check()
            with page.expect_response(
                lambda r: "/api/review/answers" in r.url
            ) as result:
                card.locator('[data-review-grid-action="unresolved_event"]').click()
            assert result.value.status == 200
            page.wait_for_load_state("load")
            page.goto(base + "/admin/review")
            assert (
                page.locator('.review-grid__tile[data-detection-id="71"]').count() == 0
            )
            assert (
                page.locator('.review-grid__tile[data-detection-id="72"]').count() == 1
            )
            with page.expect_response(
                lambda r: "/api/review/answers" in r.url
            ) as result:
                page.locator('[data-review-grid-action="approve_event"]').click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_observation_page(conn, answered=False)[1] == 0
                seed(conn, identifier=73, species="Pica_pica")
            page.goto(base + "/admin/review")
            with page.expect_response(
                lambda r: "/api/review/answers" in r.url
            ) as result:
                page.locator('[data-review-grid-action="no_bird_event"]').click()
            assert result.value.status == 200
            with closing_connection() as conn:
                assert fetch_observation_page(conn, answered=False)[1] == 0
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
