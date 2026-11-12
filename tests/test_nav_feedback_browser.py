"""Page-change feedback and tap cues in a real browser, by mouse and by touch."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Thread
from typing import Any

import pytest
from flask import Flask
from werkzeug.serving import make_server

from tests.test_gallery_blueprint import _seed_detection
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection

PENDING = "document.documentElement.classList.contains('nav-pending')"

# Playwright holds its calls until a pending navigation has finished, so the
# page being left writes down what it shows; the next page hands it over.
WATCH = """
repeats => {
    const link = document.querySelector('a.app-bar__link[href="/species"]');
    sessionStorage.removeItem('navSeen');
    const ticker = window.wmbLedTicker;
    const setNotice = ticker.setNotice;
    let tickerNotice;
    ticker.setNotice = (top, bottom) => {
        tickerNotice = [top, bottom];
        setNotice(top, bottom);
    };
    setInterval(() => {
        const seen = JSON.parse(sessionStorage.getItem('navSeen') || '{}');
        if (document.documentElement.classList.contains('nav-pending')) {
            seen.pending = true;
            seen.busy = link.getAttribute('aria-busy');
            seen.opening = link.classList.contains('is-opening');
            seen.ticker = tickerNotice;
            const line = document.querySelector('.nav-progress');
            seen.line = Math.max(seen.line || 0, line.getBoundingClientRect().width);
            for (; repeats > 0; repeats -= 1) link.click();
        }
        const notice = document.getElementById('navNotice');
        if (!notice.hidden) {
            seen.notice = notice.innerText.trim();
            seen.noticePointerEvents = getComputedStyle(notice).pointerEvents;
            seen.noticeDisplay = getComputedStyle(notice).display;
            seen.announcement = document.getElementById('navAnnouncement').textContent;
        }
        sessionStorage.setItem('navSeen', JSON.stringify(seen));
    }, 50);
}
"""


@contextmanager
def station(
    app: Flask, *, width: int, touch: bool, theme: str
) -> Iterator[tuple[Any, str, Counter[str], dict[str, float]]]:
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    with closing_connection() as conn:
        _seed_detection(
            conn,
            filename="20260101_120000_tit.jpg",
            timestamp="20260101_120000",
            species="Parus_major",
            decision_state="confirmed",
        )
    requests: Counter[str] = Counter()
    delay = {"seconds": 0.0}

    def slow(environ: dict[str, Any], start_response: Callable[..., Any]) -> Any:
        if environ["PATH_INFO"] == "/species":
            requests["species"] += 1
            time.sleep(delay["seconds"])
        return app(environ, start_response)

    server = make_server("127.0.0.1", 0, slow, threaded=True)
    Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as api:
        browser = api.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": 900},
            color_scheme=theme,
            has_touch=touch,
            is_mobile=touch,
        )
        page = context.new_page()
        page.add_init_script(
            f"localStorage.setItem('wmb-theme', '{theme}');"
            "localStorage.setItem('wmb-status-rail', 'visible');"
        )
        page.route(
            "**/api/v1/ptz/auto/status",
            lambda route: route.fulfill(json={"enabled": False}),
        )
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            yield page, f"http://127.0.0.1:{server.server_port}", requests, delay
            assert errors == [], errors
        finally:
            browser.close()
            server.shutdown()


def open_species(page: Any, *, width: int, touch: bool, repeats: int) -> dict[str, Any]:
    """Leave for the slow species page and return what the old page showed."""
    if width < 900:
        page.locator(".app-bar__toggle").click()
    link = page.locator('a.app-bar__link[href="/species"]')
    page.evaluate(WATCH, repeats)
    (link.tap if touch else link.click)(no_wait_after=True)
    page.wait_for_url("**/species")
    return json.loads(page.evaluate("sessionStorage.getItem('navSeen')") or "{}")


@pytest.mark.parametrize(("width", "touch"), [(1440, False), (390, True)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_slow_page_is_acknowledged_once_and_leaves_no_trace(
    local_db_app: Flask, width: int, touch: bool, theme: str
) -> None:
    with station(local_db_app, width=width, touch=touch, theme=theme) as (
        page,
        base,
        requests,
        delay,
    ):
        page.goto(base + "/gallery")
        assert not page.evaluate(PENDING)
        assert page.locator("html").get_attribute("data-theme") == theme
        assert page.locator("#ledTickerCanvas").is_visible()
        assert page.evaluate("typeof window.wmbLedTicker.setNotice") == "function"

        delay["seconds"] = 2.0
        seen = open_species(page, width=width, touch=touch, repeats=3)
        assert seen["pending"] is True
        assert seen["busy"] == "true" and seen["opening"] is True
        assert seen["line"] > 0
        assert seen["ticker"] == ["OPENING", "Species"]
        assert "notice" not in seen
        assert requests["species"] == 1
        assert not page.evaluate(PENDING)

        name = page.locator(".wm-tile__body--link .wm-tile__name").first
        cue = name.evaluate("node => getComputedStyle(node, '::after').content")
        assert "›" in cue
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )

        delay["seconds"] = 0.0
        page.go_back()
        page.wait_for_url("**/gallery")
        assert not page.evaluate(PENDING)
        assert page.locator("#navNotice").is_hidden()
        assert page.locator(".is-opening").count() == 0


def test_long_wait_shows_a_notice_that_never_blocks_the_page(
    local_db_app: Flask,
) -> None:
    with station(local_db_app, width=1440, touch=False, theme="light") as (
        page,
        base,
        _requests,
        delay,
    ):
        page.goto(base + "/gallery")
        delay["seconds"] = 5.5
        seen = open_species(page, width=1440, touch=False, repeats=0)
        assert re.fullmatch(r"Still loading Species · \d+ s", seen["notice"])
        assert seen["noticePointerEvents"] == "none"
        assert seen["noticeDisplay"] != "none"
        assert seen["announcement"] == "Still loading Species"
        assert page.locator("#navNotice").is_hidden()
