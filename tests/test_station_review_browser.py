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


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_done_tabs_include_manual_birds_and_image_statements_without_photos(
    local_db_app: Flask, tmp_path: Path, width: int, theme: str
) -> None:
    from PIL import Image

    from core.human_label_core import BBox, append_fact, ensure_object_subject
    from core.manual_object_core import ManualObjectDraft, create_manual_object
    from tests.test_station_observations import answer

    api = pytest.importorskip("playwright.sync_api")
    with closing_connection() as conn:
        absent = seed(conn, identifier=1, state="uncertain")
        answer(conn, 1, absent, species="Pica_pica")
        append_fact(
            conn,
            subject_id=ensure_object_subject(conn, 1),
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance(),
        )
        wrong = seed(conn, identifier=2, state="uncertain")
        answer(conn, 2, wrong, state="wrong", species=None)
        manual = seed(conn, identifier=3, state="uncertain")
        original = tmp_path / "temporary.jpg"
        Image.new("RGB", (600, 400), (115, 130, 110)).save(original)
        create_manual_object(
            conn,
            ManualObjectDraft(
                manual, BBox(0.5, 0.5, 0.2, 0.2), "Parus_major", "manual-review-browser"
            ),
            provenance(),
            original_path=original,
            locale="DE",
        )
        for filename, stamp, options in [
            ("empty.jpg", "20261012_100000", {"image_bird_presence": "absent"}),
            ("missed.jpg", "20261013_120000", {"detector_miss": True}),
        ]:
            conn.execute(
                "INSERT INTO images(filename,timestamp,original_present) VALUES (?,?,0)",
                (filename, stamp),
            )
            record_human_answer(conn, HumanAnswer(filename, **options), provenance())
        conn.execute("UPDATE images SET original_present=0")
    with api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
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
            page.goto(base + "/admin/review?day=all")
            tabs = page.get_by_role("navigation", name="Review tabs")
            assert (
                tabs.get_by_role("link", name="Open", exact=True).get_attribute(
                    "aria-current"
                )
                == "page"
            )
            page.screenshot(
                path=str(tmp_path / f"open-{width}-{theme}.png"), full_page=True
            )
            tabs.get_by_role("link", name="Done", exact=True).click()
            assert (
                page.get_by_role("link", name="Done", exact=True).get_attribute(
                    "aria-current"
                )
                == "page"
            )
            tile = page.locator('article[data-detection-id="1"]')
            assert tile.locator(".wm-tile__name").inner_text() == "No bird"
            assert tile.locator(".wm-tile__latin").count() == 0
            assert tile.locator(".station-media-placeholder").is_visible()
            assert (
                page.locator("article .wm-tile__name")
                .filter(has_text="Wrong species")
                .count()
                == 1
            )
            assert (
                page.get_by_text("Manually added bird · Answered", exact=True).count()
                == 1
            )
            assert (
                page.locator("article .wm-tile__name")
                .filter(has_text="No birds in full image")
                .count()
                == 1
            )
            assert (
                page.locator("article .wm-tile__name")
                .filter(has_text="Bird missed by the detector")
                .count()
                == 1
            )
            assert page.locator("html").get_attribute("data-theme") == theme
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(
                path=str(tmp_path / f"done-{width}-{theme}.png"), full_page=True
            )
            # An image-only answer can use the shared viewer and its withdrawal gate.
            page.locator(
                'article[data-image-filename="empty.jpg"] .wm-tile__button'
            ).click()
            modal = page.locator(".gallery-modal.show")
            notice = modal.locator("[data-image-absence]")
            notice.wait_for(state="visible")
            with page.expect_navigation(wait_until="load"):
                notice.get_by_role(
                    "button", name="Withdraw full-image answer", exact=True
                ).click()
            assert (
                page.locator("article .wm-tile__name")
                .filter(has_text="No birds in full image")
                .count()
                == 0
            )
            page.locator("#review-day").select_option("2026-10-13")
            page.locator(".review-day-nav__form").get_by_role(
                "button", name="Apply", exact=True
            ).click()
            assert "answered=1" in page.url
            assert (
                page.locator("article .wm-tile__name")
                .filter(has_text="Bird missed by the detector")
                .count()
                == 1
            )
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
