"""Executable layout contracts for the detail-modal bird editor.

These render the real stylesheet in a headless browser and measure the
resulting boxes. A string check on the CSS source cannot catch the class of
defect these guard: the editor rail being clipped by the modal's
``overflow: hidden`` because the image consumed the whole height budget.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "assets" / "design-system.css"

# A viewport short enough that a naive "image = 100vh - fixed-rem" budget
# overflows once the editor rail is added below the image.
VIEWPORT = {"width": 1440, "height": 729}


def _render_and_measure() -> dict:
    """Lay the modal out in headless Chromium and return measured boxes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip("Playwright is unavailable")
    from playwright.sync_api import sync_playwright

    # Minimal DOM mirroring templates/components/detection_modal.html:
    # content > (header, body > (image-wrap > viewer > img, bird-editor)).
    html = textwrap.dedent(
        """
        <!doctype html><html><head><meta charset="utf-8">
        <link rel="stylesheet" href="design-system.css"></head>
        <body>
        <div class="modal gallery-modal wm-modal show" style="display:block">
          <div class="modal-dialog wm-modal__dialog">
            <div class="modal-content wm-modal__content">
              <div class="wm-modal__header">
                <div class="wm-modal__title">
                  <span class="wm-modal__title-text">Fixture</span>
                  <span class="wm-modal__title-sub">sub</span>
                </div>
              </div>
              <div class="wm-modal__body">
                <div class="wm-modal__image wm-toolbox-host">
                  <div class="modal-image-viewer wm-image-viewer">
                    <img class="wm-image-viewer__img bbox-base-image" alt="f"
                         src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjgwMCI+PHJlY3Qgd2lkdGg9IjEyMDAiIGhlaWdodD0iODAwIiBmaWxsPSIjOTk5Ii8+PC9zdmc+">
                    <div class="wm-bird-editor__layer" data-bird-editor-layer></div>
                  </div>
                </div>
                <div class="wm-bird-editor" data-bird-editor>
                  <div class="wm-bird-editor__toolbar">
                    <div class="wm-view-mode-toggle wm-view-mode-toggle--editor">
                      <button class="wm-view-mode-toggle__option">Full</button>
                    </div>
                    <button class="wm-bird-editor__species">1 &middot; Kohlmeise</button>
                    <div class="wm-bird-editor__actions">
                      <button class="btn">Adjust box</button>
                      <button class="btn">Save</button>
                    </div>
                  </div>
                  <div class="wm-bird-editor__status">Status</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        </body></html>
        """
    ).strip()

    measure = """() => {
      const R = (s) => { const e = document.querySelector(s);
        if (!e) return null; const r = e.getBoundingClientRect();
        return {top:+r.top.toFixed(1), bottom:+r.bottom.toFixed(1),
                height:+r.height.toFixed(1), width:+r.width.toFixed(1),
                clientH:e.clientHeight, scrollH:e.scrollHeight}; };
      return {content:R('.wm-modal__content'), body:R('.wm-modal__body'),
              img:R('.wm-image-viewer__img'),
              toolbar:R('.wm-bird-editor__toolbar'),
              status:R('.wm-bird-editor__status'),
              innerH: window.innerHeight};
    }"""

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "index.html").write_text(html, encoding="utf-8")
        (tmpdir / "design-system.css").write_text(
            CSS.read_text(encoding="utf-8"), encoding="utf-8"
        )
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:  # pragma: no cover - browser not installed
                pytest.skip("Chromium for Playwright is unavailable")
            try:
                page = browser.new_context(viewport=VIEWPORT).new_page()
                page.goto((tmpdir / "index.html").as_uri(), wait_until="load")
                page.wait_for_timeout(250)
                return page.evaluate(measure)
            finally:
                browser.close()


@pytest.fixture(scope="module")
def measured() -> dict:
    return _render_and_measure()


def test_editor_rail_is_not_clipped_by_the_modal(measured: dict) -> None:
    """The toolbar and status must stay inside the clipping content box.

    Regression: the image was capped against the raw viewport while the rail
    below it reserved no space, so the rail fell outside
    ``.wm-modal__content`` (``overflow: hidden``) and the user had to zoom the
    browser out to reach Save/Cancel.
    """
    content, toolbar, status = (
        measured["content"],
        measured["toolbar"],
        measured["status"],
    )
    assert toolbar["bottom"] <= content["bottom"] + 0.5, (
        f"toolbar clipped by {toolbar['bottom'] - content['bottom']:.1f}px"
    )
    assert status["bottom"] <= content["bottom"] + 0.5, (
        f"status clipped by {status['bottom'] - content['bottom']:.1f}px"
    )


def test_modal_body_does_not_overflow(measured: dict) -> None:
    """The body must fit its own box rather than relying on a scrollbar."""
    body = measured["body"]
    assert body["scrollH"] <= body["clientH"] + 1, (
        f"body overflows by {body['scrollH'] - body['clientH']}px"
    )


def test_image_yields_height_to_the_rail(measured: dict) -> None:
    """The image shrinks so the rail keeps its intrinsic height."""
    img, toolbar = measured["img"], measured["toolbar"]
    assert img["height"] > 100, "image collapsed instead of merely shrinking"
    assert toolbar["height"] >= 40, "rail was squashed instead of the image"
