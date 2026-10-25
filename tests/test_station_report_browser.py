from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from config import get_config
from tests.test_station_observations import seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.path_manager import PathManager


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_report_stays_still_during_capture_and_refreshes_explicitly(
    local_db_app: Flask, tmp_path: Path, width: int, theme: str
) -> None:
    api = pytest.importorskip("playwright.sync_api")
    paths = PathManager(get_config()["OUTPUT_DIR"])
    with closing_connection() as conn:
        for identifier, species in [(42, "Pica_pica"), (43, "Parus_major")]:
            filename = seed(conn, identifier=identifier, species=species, quality=1)
            for path in (
                paths.get_original_path(filename),
                paths.get_derivative_path(filename, "optimized"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (600, 400), (115, 130, 110)).save(path)
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
        requests: list[str] = []
        navigations: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                requests.append(request.url)
                if "/api/station-report" in request.url
                else None
            ),
        )
        page.on("framenavigated", lambda frame: navigations.append(frame.url))
        try:
            page.goto(base + "/login")
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()
            page.goto(base + "/analytics")
            page.wait_for_selector('[data-report-state="ready"]')
            page.locator("[data-report-time]").wait_for()
            assert page.locator("html").get_attribute("data-theme") == theme
            assert (
                "2 recorded species"
                in page.locator(".station-report__overview").inner_text()
            )
            assert page.locator("[data-report-time]").inner_text().startswith("as of ")
            page.screenshot(
                path=str(tmp_path / f"report-{width}-{theme}.png"), full_page=True
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.locator('input[name="start"]').fill("2026-10-01")
            page.locator('details[data-report-details="models"]').evaluate(
                "node => node.open = true"
            )
            page.locator(".station-report__table-scroll").first.evaluate(
                "node => node.scrollLeft = 40"
            )
            page.evaluate("window.scrollTo(0, 600)")
            before = page.locator("[data-current-report]").inner_html()
            position = page.evaluate("window.scrollY")
            report_requests = len(requests)
            page.clock.install()
            for minute in range(1, 11):
                with closing_connection() as conn:
                    seed(
                        conn,
                        identifier=100 + minute,
                        species="Picus_viridis",
                        quality=1,
                    )
                folder = paths.originals_dir / f"capture-{minute:02}"
                folder.mkdir()
                Image.new("RGB", (32, 32), (minute, 100, 100)).save(
                    folder / "capture.jpg"
                )
                page.clock.fast_forward(60_000)
                assert page.locator("[data-current-report]").inner_html() == before
                assert page.evaluate("window.scrollY") == position
                assert page.locator("[data-current-report]").is_visible()
                assert not page.locator("#report-progress").is_visible()
                assert len(requests) == report_requests
            revision = page.locator("[data-report-state]").get_attribute(
                "data-report-revision"
            )
            page.get_by_role("button", name="Refresh", exact=True).click()
            page.clock.resume()
            page.wait_for_function(
                '(revision) => document.querySelector("[data-report-state]").dataset.reportRevision !== revision',
                arg=revision,
            )
            assert (
                "3 recorded species"
                in page.locator(".station-report__overview").inner_text()
            )
            assert "Picus viridis" in page.locator("[data-current-report]").inner_text()
            assert page.locator('input[name="start"]').input_value() == "2026-10-01"
            assert page.locator('details[data-report-details="models"]').evaluate(
                "node => node.open"
            )
            # Correct the existing card through the real shared editor and answer endpoint.
            overview = page.locator(".station-report__overview").inner_text()
            tile = page.locator('article[data-detection-id="42"]')
            tile.locator(".wm-tile__button").click()
            modal = page.locator("#modal-station_report-42")
            modal.wait_for(state="visible")
            modal.locator('[data-editor-species="detection:42"]').click()
            page.get_by_placeholder("Search species...").fill("Eichelhäher")
            page.locator(".wm-species-picker-item").filter(
                has_text="Garrulus glandarius"
            ).first.click()
            with page.expect_response(
                lambda r: "/api/labels/answer" in r.url and r.request.method == "POST"
            ) as result:
                modal.get_by_role("button", name="Save bird", exact=True).click()
            assert result.value.status == 200
            assert tile.locator(".wm-tile__latin").inner_text() == "Garrulus glandarius"
            notices = page.locator("[data-report-outdated]")
            assert notices.count() == 2
            for notice in notices.all():
                assert notice.is_visible()
                assert notice.get_attribute("role") == "status"
            assert page.locator(".station-report__overview").inner_text() == overview
            modal.locator("button.modal-action-bar__close").click()
            modal.wait_for(state="hidden")
            assert page.locator("[data-current-report]").is_visible()
            notices = page.locator("[data-report-outdated]")
            assert notices.count() == 2
            for notice in notices.all():
                assert notice.is_visible()
                assert notice.get_attribute("role") == "status"
            assert page.locator(".station-report__overview").inner_text() == overview
            page.locator(".wm-toast").wait_for(state="hidden")
            species_section = page.locator('section[aria-labelledby="report-species"]')
            assert "Garrulus glandarius" not in species_section.inner_text()
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.screenshot(
                path=str(tmp_path / f"outdated-controls-{width}-{theme}.png")
            )
            species_section.locator(".station-report__table-scroll").evaluate(
                "node => node.scrollLeft = 0"
            )
            page.evaluate("""() => {
                const section = document.querySelector('section[aria-labelledby="report-species"]');
                window.scrollTo({top: window.scrollY + section.getBoundingClientRect().top - 100, behavior: 'instant'});
            }""")
            page.wait_for_function(
                """() => Math.abs(document.querySelector('section[aria-labelledby="report-species"]').getBoundingClientRect().top - 100) < 2"""
            )
            page.screenshot(
                path=str(tmp_path / f"outdated-species-{width}-{theme}.png")
            )
            species_section.get_by_role("button", name="Refresh", exact=True).click()
            page.wait_for_function(
                '() => [...document.querySelectorAll("[data-report-outdated]")].every(node => node.hidden)'
            )
            assert page.locator("[data-report-outdated]").count() == 2
            for notice in page.locator("[data-report-outdated]").all():
                assert not notice.is_visible()
            assert (
                "3 recorded species"
                in page.locator(".station-report__overview").inner_text()
            )
            assert "Garrulus glandarius" in species_section.inner_text()
            assert "Pica pica" not in species_section.inner_text()
            assert not page.locator("#report-progress").is_visible()
            tile = page.locator('article[data-detection-id="43"]')
            more_actions = tile.get_by_role("button", name="More actions", exact=True)
            more_actions.focus()
            more_actions.press("Enter")
            tile.locator('[data-action="change-species"]').click()
            page.get_by_placeholder("Search species...").fill("Grünspecht")
            with page.expect_response(
                lambda r: "/api/labels/answer" in r.url and r.request.method == "POST"
            ) as result:
                page.locator(".wm-species-picker-item").filter(
                    has_text="Picus viridis"
                ).first.click()
            assert result.value.status == 200
            assert tile.locator(".wm-tile__latin").inner_text() == "Picus viridis"
            notices = page.locator("[data-report-outdated]")
            assert notices.count() == 2
            for notice in notices.all():
                assert notice.is_visible()
                assert notice.get_attribute("role") == "status"
            assert (
                "3 recorded species"
                in page.locator(".station-report__overview").inner_text()
            )
            species_section.get_by_role("button", name="Refresh", exact=True).click()
            page.wait_for_function(
                '() => [...document.querySelectorAll("[data-report-outdated]")].every(node => node.hidden)'
            )
            assert (
                "2 recorded species"
                in page.locator(".station-report__overview").inner_text()
            )
            for notice in page.locator("[data-report-outdated]").all():
                assert not notice.is_visible()
            assert len([url for url in navigations if "/analytics" in url]) == 1
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
            cache = local_db_app.extensions.get("station_report")
            if cache:
                cache.close()
