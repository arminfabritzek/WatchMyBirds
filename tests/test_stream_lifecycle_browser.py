"""Visibility recovery must retain the embedded player's fullscreen document."""

from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from tests.test_index_blueprint import local_db_app as local_db_app


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_live_player_survives_visibility_in_fullscreen(
    local_db_app: Flask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    theme: str,
) -> None:
    api = pytest.importorskip("playwright.sync_api")
    from web.blueprints import index

    monkeypatch.setitem(index.config, "STREAM_SOURCE_MODE", "auto")
    local_db_app.context_processor(lambda: {"local_development": False})
    snapshot = BytesIO()
    Image.new("RGB", (640, 360), (70, 100, 70)).save(snapshot, format="JPEG")
    player = """<!doctype html><html><body style="margin:0;background:#345;color:white">
        <main id="player" style="height:100vh;display:grid;place-items:center">
            <button onclick="document.getElementById('player').requestFullscreen()">
                Fullscreen
            </button>
        </main></body></html>"""
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
            page.add_init_script("localStorage.setItem('wmb-status-rail', 'visible')")
            player_loads = []
            status_requests = []
            preset_requests = []
            auto_enabled = False
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))

            def serve_player(route: Any) -> None:
                if urlsplit(route.request.url).path == "/stream.html":
                    player_loads.append(route.request.url)
                    route.fulfill(content_type="text/html", body=player)
                else:
                    route.fulfill(body="")

            page.route("http://127.0.0.1:1984/**", serve_player)

            def serve_status(route: Any) -> None:
                status_requests.append(route.request.url)
                route.fulfill(
                    json={
                        "status": "success",
                        "overlay_enabled": True,
                        "auto_ptz": {
                            "configured_enabled": auto_enabled,
                            "camera_id": 0,
                            "state": "idle",
                        },
                    }
                )

            def serve_camera(route: Any) -> None:
                nonlocal auto_enabled
                path = urlsplit(route.request.url).path
                if path.endswith("/auto/enabled"):
                    auto_enabled = route.request.post_data_json["enabled"]
                elif path.endswith("/ptz/goto"):
                    preset_requests.append(route.request.post_data_json)
                route.fulfill(
                    json={
                        "status": "success",
                        "config": {"overview_preset": "Preset022"},
                        "capabilities": {},
                    }
                )

            page.route("**/api/v1/ptz/auto/status", serve_status)
            page.route("**/api/v1/cameras/0/**", serve_camera)
            page.route("**/api/status", lambda route: route.fulfill(json={}))
            page.route(
                "**/api/v1/public/go2rtc/health",
                lambda route: route.fulfill(json={"healthy": True}),
            )
            page.route(
                "**/api/snapshot?*",
                lambda route: route.fulfill(
                    content_type="image/jpeg", body=snapshot.getvalue()
                ),
            )
            base = f"http://127.0.0.1:{server.server_port}"
            with local_db_app.test_client() as client:
                with client.session_transaction() as session:
                    session["authenticated"] = True
                cookie = client.get_cookie(local_db_app.config["SESSION_COOKIE_NAME"])
            page.context.add_cookies(
                [{"name": cookie.key, "value": cookie.value, "url": base}]
            )
            page.goto(
                base + "/",
                wait_until="domcontentloaded",
            )
            frame = page.frame_locator("#go2rtc-frame")
            frame.get_by_role("button", name="Fullscreen").wait_for()
            api.expect(page.locator("#stream-loading-static")).to_have_css(
                "opacity", "0"
            )
            api.expect(page.locator("#ptzDpadHome")).to_be_enabled()
            page.locator(".stream-embed").hover()
            page.locator("#ptzDpadHome").click()
            assert preset_requests == [{"preset_token": "Preset022"}]
            page.wait_for_timeout(650)
            assert len(status_requests) == 1, (
                "Auto Cam off must not run the overlay poll"
            )
            assert page.locator("html").get_attribute("data-theme") == theme
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(path=str(tmp_path / f"stream-{width}-{theme}.png"))
            frame.get_by_role("button", name="Fullscreen").click()
            page.wait_for_function("() => !!document.fullscreenElement")
            page.evaluate(
                """async () => {
                    document.dispatchEvent(new Event('visibilitychange'));
                    Object.defineProperty(document, 'hidden', {configurable: true, value: true});
                    document.dispatchEvent(new Event('visibilitychange'));
                    delete document.hidden;
                    document.dispatchEvent(new Event('visibilitychange'));
                    await new Promise(resolve => setTimeout(resolve, 100));
                }"""
            )
            assert page.evaluate("!!document.fullscreenElement")
            assert len(player_loads) == 1
            page.evaluate("document.exitFullscreen()")
            page.wait_for_function("() => !document.fullscreenElement")
            toggle = page.locator("#appbarAutoCamToggle")
            before_tracking = len(status_requests)
            toggle.click()
            api.expect(page.locator("#trackingOverlay")).to_be_visible()
            page.wait_for_timeout(650)
            assert len(status_requests) > before_tracking + 1
            toggle.click()
            api.expect(page.locator("#trackingOverlay")).to_be_hidden()
            after_disabled = len(status_requests)
            page.wait_for_timeout(650)
            assert len(status_requests) == after_disabled
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
