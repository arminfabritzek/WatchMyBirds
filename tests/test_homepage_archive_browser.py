"""Deferred archive cards retain the shared viewer and do not block Live."""

from pathlib import Path
from threading import Event, Thread

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from config import get_config
from tests.test_index_blueprint import _seed_detection
from tests.test_index_blueprint import local_db_app as local_db_app
from utils.db.connection import closing_connection
from utils.path_manager import PathManager


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_homepage_archive_retry_loading_and_shared_dialog(
    local_db_app: Flask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    theme: str,
) -> None:
    api = pytest.importorskip("playwright.sync_api")
    from web import view_helpers, web_interface

    local_db_app.context_processor(lambda: {"local_development": True})
    paths = PathManager(get_config()["OUTPUT_DIR"])
    with closing_connection() as conn:
        for i in range(25):
            name = f"20200101_120000_{i:03}.jpg"
            _seed_detection(
                conn,
                filename=name,
                timestamp="20200101_120000",
                species="Parus_major",
                decision_state="confirmed",
            )
            for path in (
                paths.get_original_path(name),
                paths.get_derivative_path(name, "optimized"),
                paths.get_derivative_path(name.replace(".jpg", "_crop_1.webp")),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (200, 150), (90, 130, 100)).save(path)
    web_interface.invalidate_best_species_cache()
    entered, release = Event(), Event()
    fetch = view_helpers.fetch_best_species_pools
    attempts = 0

    def delayed(**kwargs: object) -> tuple[list[dict], list[dict]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary test failure")
        entered.set()
        assert release.wait(10)
        return fetch(**kwargs)

    monkeypatch.setattr(view_helpers, "fetch_best_species_pools", delayed)
    server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(
                viewport={"width": width, "height": 900}, color_scheme=theme
            )
            page.add_init_script(f"localStorage.setItem('wmb-theme', '{theme}')")
            page.route(
                "**/api/v1/ptz/auto/status",
                lambda route: route.fulfill(json={"enabled": False}),
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            base = f"http://127.0.0.1:{server.server_port}"
            with local_db_app.test_client() as client:
                with client.session_transaction() as session:
                    session["authenticated"] = True
                cookie = client.get_cookie(local_db_app.config["SESSION_COOKIE_NAME"])
            page.context.add_cookies(
                [{"name": cookie.key, "value": cookie.value, "url": base}]
            )
            page.goto(base + "/", wait_until="domcontentloaded")
            page.locator("[data-archive-retry]").wait_for(state="visible")
            assert page.locator(".stream-embed__unavailable").is_visible()
            page.get_by_role("button", name="Retry", exact=True).click()
            assert entered.wait(5)
            assert (
                page.locator("[data-archive-status]").inner_text()
                == "Loading your archive…"
            )
            assert page.locator("#last24hSummarySection").is_visible()
            release.set()
            page.locator("[data-homepage-archive-result]").wait_for()
            assert page.locator('[data-modal-group="stream-best"]').count() == 4
            assert page.evaluate("window.__ledTickerStats.total") == 1
            assert page.locator("html").get_attribute("data-theme") == theme
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.wait_for_function(
                """() => [...document.querySelectorAll(
                    '#homepageArchive .best-species-board__featured .best-species-card'
                )].every(el => getComputedStyle(el).opacity === '1')"""
            )
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.screenshot(
                path=str(tmp_path / f"homepage-{width}-{theme}.png"), full_page=True
            )
            page.locator(
                "#homepageArchive .best-species-card__media-link"
            ).first.click()
            modal = page.locator('[data-modal-group="stream-best"].show')
            modal.wait_for(state="visible")
            assert modal.locator(".wm-bird-editor").is_visible()
            with page.expect_response(
                lambda response: response.url == base + "/" and response.status == 200
            ):
                page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
            assert modal.is_visible()
            assert page.locator('[data-modal-group="stream-best"]').count() == 4
            page.wait_for_function(
                """() => {
                    const modal = document.querySelector(
                        '[data-modal-group="stream-best"].show'
                    );
                    return getComputedStyle(modal).opacity === '1' &&
                        getComputedStyle(modal.querySelector('.modal-dialog')).transform === 'none';
                }"""
            )
            page.screenshot(
                path=str(tmp_path / f"homepage-dialog-{width}-{theme}.png"),
            )
            assert not errors
            assert attempts == 2
            browser.close()
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
