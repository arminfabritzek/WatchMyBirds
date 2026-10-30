from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from config import get_config
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from tests.labeling_helpers import _seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.path_manager import PathManager


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_export_download_repeat_and_receipt(
    local_db_app: Flask,
    tmp_path: Path,
    width: int,
    theme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

    paths = PathManager(get_config()["OUTPUT_DIR"])
    filename = "20261001_120000_answer.jpg"
    with closing_connection() as conn:
        detection_id = _seed(conn, filename=filename, timestamp="20261001_120000")
        record_human_answer(
            conn,
            HumanAnswer(
                filename,
                detection_id,
                object_bird_presence="present",
                species_identity="confirmed",
                species_key="Parus_major",
            ),
            LabelProvenance(
                "0123456789abcdef0123456789abcdef",
                "test",
                "normal_correction",
                "watchmybirds_ui",
            ),
        )
    path = paths.get_original_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 400), (115, 130, 110)).save(path)
    from core import training_export_jobs

    started, release = Event(), Event()
    original_stream = training_export_jobs.stream_canonical_bundle

    def slow_archive(*args: Any, **kwargs: Any) -> Iterator[bytes]:
        started.set()
        assert release.wait(20)
        yield from original_stream(*args, **kwargs)

    monkeypatch.setattr(training_export_jobs, "stream_canonical_bundle", slow_archive)
    server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as api:
        browser = api.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": 900},
            color_scheme=theme,
            accept_downloads=True,
        )
        page.add_init_script(f"localStorage.setItem('wmb-theme', '{theme}')")
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            page.goto(base + "/login")
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()
            page.goto(base + "/admin/canonical-dataset")
            assert page.locator("html").get_attribute("data-theme") == theme
            assert (
                "1 photo is waiting for training"
                in page.get_by_role("status").inner_text()
            )
            with page.expect_download() as first_download:
                page.get_by_role(
                    "button", name="Export waiting photos", exact=True
                ).click()
                assert started.wait(5)
                page.locator("[data-job-progress]").first.wait_for()
                assert (
                    page.request.get(
                        base + "/admin/canonical-dataset", timeout=3000
                    ).status
                    == 200
                )
                page.screenshot(
                    path=str(tmp_path / f"preparing-{width}-{theme}.png"),
                    full_page=True,
                )
                release.set()
            first = tmp_path / "first.zip"
            first_download.value.save_as(first)
            assert first.stat().st_size > 0
            notice = page.get_by_role("status").filter(has_text="Not yet confirmed")
            assert notice.is_visible()
            assert (
                notice.evaluate("node => getComputedStyle(node).backgroundColor")
                != "rgba(0, 0, 0, 0)"
            )
            assert page.get_by_role(
                "button", name="Export waiting photos", exact=True
            ).is_disabled()
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.wait_for_function(
                "() => getComputedStyle(document.querySelector('.canonical-review')).opacity === '1'"
            )
            page.screenshot(
                path=str(tmp_path / f"export-{width}-{theme}.png"), full_page=True
            )
            with page.expect_download() as repeated_download:
                page.get_by_role("link", name="Download again", exact=True).click()
            second = tmp_path / "second.zip"
            repeated_download.value.save_as(second)
            assert first.read_bytes() == second.read_bytes()
            page.get_by_role("button", name="Export received", exact=True).click()
            page.wait_for_url(base + "/admin/canonical-dataset")
            assert (
                page.get_by_role("status").filter(has_text="Not yet confirmed").count()
                == 0
            )
            assert "Received" in page.locator(".canonical-dataset__export").inner_text()
            assert (
                page.get_by_role("button", name="Export received", exact=True).count()
                == 0
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.wait_for_function(
                "() => getComputedStyle(document.querySelector('.canonical-review')).opacity === '1'"
            )
            page.screenshot(
                path=str(tmp_path / f"received-{width}-{theme}.png"), full_page=True
            )
            assert errors == []
            page.get_by_role("button", name="Remove local ZIP", exact=True).click()
            assert page.get_by_role(
                "button", name="Prepare download again", exact=True
            ).is_visible()
            with page.expect_download() as rebuilt_download:
                page.get_by_role(
                    "button", name="Prepare download again", exact=True
                ).click()
            rebuilt = tmp_path / "rebuilt.zip"
            rebuilt_download.value.save_as(rebuilt)
            assert rebuilt.read_bytes() == first.read_bytes()
            assert (
                page.get_by_role("button", name="Export received", exact=True).count()
                == 0
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            assert errors == []
        finally:
            release.set()
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
            cache = local_db_app.extensions.get("station_report")
            if cache:
                cache.close()
            local_db_app.extensions["training_export_jobs"].close()
