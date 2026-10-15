from __future__ import annotations

import json
from pathlib import Path

import pytest


def report_html(revision: str, *, ready: bool = True) -> str:
    report = (
        """<div data-current-report style="height:2200px">
      <div class="station-report__table-scroll" style="width:100px;overflow:auto">
        <div style="width:1000px;height:30px">Calendar</div>
      </div>
      <details data-report-details="models"><summary>Models</summary>History</details>
    </div>"""
        if ready
        else ""
    )
    return f"""<!doctype html><html><body>
      <div data-report-state="{"ready" if ready else "building"}" data-report-revision="{revision}">
        <input id="filter-draft" value="">
        <p id="report-progress" {"hidden" if ready else ""}>Building</p>{report}
      </div>
      <div data-report-modals><div class="gallery-modal"><input value="saved"></div></div>
      <script src="/station_report.js"></script>
    </body></html>"""


def test_updates_preserve_page_state_and_defer_until_editor_closes() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    script = (
        Path(__file__).resolve().parents[1] / "assets/js/station_report.js"
    ).read_text()
    status = {"state": "building", "revision": "station:0"}
    navigations = []
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception:
            pytest.skip("Chromium for Playwright is unavailable")
        try:
            page = browser.new_page(viewport={"width": 390, "height": 700})
            page.add_init_script("""const originalTimeout = window.setTimeout;
                window.setTimeout = (callback, delay) => originalTimeout(callback, Math.min(delay, 30));""")
            page.on("framenavigated", lambda frame: navigations.append(frame.url))
            page.route(
                "**/station_report.js",
                lambda route: route.fulfill(
                    content_type="text/javascript", body=script
                ),
            )
            page.route(
                "**/api/station-report",
                lambda route: route.fulfill(
                    content_type="application/json", body=json.dumps(status)
                ),
            )
            page.route(
                "**/analytics",
                lambda route: route.fulfill(
                    content_type="text/html",
                    body=report_html(
                        status["revision"], ready=status["state"] == "ready"
                    ),
                ),
            )
            page.goto("http://station.test/analytics")
            page.locator("#filter-draft").fill("Unsaved period")
            status.update(state="ready", revision="station:1")
            page.wait_for_selector('[data-report-revision="station:1"]')
            page.evaluate("""() => {
                document.querySelector('details').open = true;
                document.querySelector('.station-report__table-scroll').scrollLeft = 200;
                window.scrollTo(0, 800);
            }""")
            for revision in range(2, 5):
                status["revision"] = f"station:{revision}"
                page.wait_for_selector(f'[data-report-revision="station:{revision}"]')
                assert page.locator("#filter-draft").input_value() == "Unsaved period"
                assert page.evaluate("window.scrollY") == 800
                assert page.locator("details").evaluate("node => node.open")
                assert (
                    page.locator(".station-report__table-scroll").evaluate(
                        "node => node.scrollLeft"
                    )
                    == 200
                )

            status["state"] = "error"
            page.wait_for_function(
                "document.querySelector('[data-current-report]').inert"
            )
            assert (
                page.locator("[data-current-report]").evaluate(
                    "node => node.getBoundingClientRect().height"
                )
                == 2200
            )
            status["state"] = "ready"
            page.wait_for_function(
                "!document.querySelector('[data-current-report]').inert"
            )

            page.evaluate("""() => {
                const modal = document.querySelector('.gallery-modal');
                window.disposals = 0;
                modal._wmBirdEditorDispose = () => window.disposals++;
                modal.classList.add('show');
                modal.dispatchEvent(new Event('show.bs.modal', {bubbles:true}));
                modal.querySelector('input').value = 'Unsaved bird answer';
                document.dispatchEvent(new CustomEvent('wmb:bird-objects-updated'));
            }""")
            status["revision"] = "station:5"
            page.wait_for_function(
                "document.querySelector('[data-current-report]').inert"
            )
            page.wait_for_timeout(150)
            assert (
                page.locator("[data-report-state]").get_attribute(
                    "data-report-revision"
                )
                == "station:4"
            )
            assert (
                page.locator(".gallery-modal input").input_value()
                == "Unsaved bird answer"
            )
            assert page.evaluate("window.disposals") == 0
            page.evaluate("""() => {
                const modal = document.querySelector('.gallery-modal');
                modal.classList.remove('show');
                modal.dispatchEvent(new Event('hidden.bs.modal', {bubbles:true}));
            }""")
            page.wait_for_selector('[data-report-revision="station:5"]')
            assert page.evaluate("window.disposals") == 1
            assert page.locator(".gallery-modal input").input_value() == "saved"
            assert navigations == ["http://station.test/analytics"]
        finally:
            browser.close()
