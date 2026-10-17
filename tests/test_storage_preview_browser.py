from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from config import get_config, update_runtime_settings
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from core.training_export import create_export
from tests.retention_helpers import seed_image
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection
from utils.path_manager import PathManager


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_storage_preview_measurement_and_status_notices(
    local_db_app: Flask,
    tmp_path: Path,
    width: int,
    theme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from playwright.sync_api import sync_playwright

    paths = PathManager(get_config()["OUTPUT_DIR"])
    with closing_connection():
        pass
    update_runtime_settings({"AUTOMATIC_DELETION_ENABLED": False})
    monkeypatch.setattr(
        "config.ensure_go2rtc_stream_synced", lambda *_args, **_kwargs: None
    )
    provenance = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")
    with closing_connection() as conn:
        for suffix in ("open", "favorite", "waiting", "exported"):
            name = f"20260101_120000_{suffix}.jpg"
            seed_image(
                conn,
                name,
                paths.base_dir,
                review_status="untagged",
                favorite=suffix == "favorite",
            )
            Image.new("RGB", (200, 200)).save(paths.get_original_path(name))
            if suffix in {"waiting", "exported"}:
                identifier = conn.execute(
                    "SELECT detection_id FROM detections WHERE image_filename=?",
                    (name,),
                ).fetchone()[0]
                record_human_answer(
                    conn,
                    HumanAnswer(name, identifier, object_bird_presence="absent"),
                    provenance,
                )
            conn.commit()
        export_id, _ = create_export(
            conn,
            paths=paths,
            labeling_installation_id="0123456789abcdef0123456789abcdef",
            selection={f"detection:{identifier}"},
        )
    originals_before = {p: p.read_bytes() for p in paths.originals_dir.rglob("*.jpg")}
    server = make_server("127.0.0.1", 0, local_db_app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as api:
        browser = api.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": 900}, color_scheme=theme
        )
        page.add_init_script(f"localStorage.setItem('wmb-theme', '{theme}')")
        page.route(
            "**/api/v1/ptz/auto/status",
            lambda route: route.fulfill(json={"enabled": False}),
        )
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            page.goto(base + "/storage")
            assert "/login" in page.url
            page.get_by_label("Password").fill("test-password")
            page.get_by_role("button", name="Login", exact=True).click()
            page.goto(base + "/storage")
            assert page.locator("html").get_attribute("data-theme") == theme
            notice = page.locator("#deletion-notice")
            assert notice.is_visible()
            assert "Automatic deletion is off" in notice.inner_text()
            assert "fresh 14-day grace period" in notice.inner_text()
            assert "preview assumes activation today" in notice.inner_text()
            assert (
                notice.evaluate("node => getComputedStyle(node).backgroundColor")
                != "rgba(0, 0, 0, 0)"
            )
            page.get_by_role("button", name="Measure storage", exact=True).click()
            page.wait_for_function(
                "() => document.getElementById('storage-status').textContent.startsWith('Measurement finished')"
            )
            assert "Keep previews" in page.locator("#storage-policy").inner_text()
            assert (
                "Waiting for training: 1 photo"
                in page.locator("#storage-waiting").inner_text()
            )
            assert (
                "Waiting for export confirmation: 1 photo"
                in page.locator("#storage-waiting").inner_text()
            )
            assert export_id in page.locator("#storage-exports").inner_text()
            assert page.locator("#storage-classes tr").count() == 6
            assert "_open" in page.locator("#storage-upcoming").inner_text()
            assert "_favorite" not in page.locator("#storage-upcoming").inner_text()
            assert "_waiting" not in page.locator("#storage-upcoming").inner_text()
            assert "_exported" not in page.locator("#storage-upcoming").inner_text()
            assert (
                "job rechecks each photo"
                in page.locator("#storage-eligible").inner_text()
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(
                path=str(tmp_path / f"storage-{width}-{theme}.png"), full_page=True
            )
            page.reload()
            assert page.locator("#storage-upcoming tr").count() > 0
            page.goto(base + "/settings")
            page.locator("#panelRetention .settings-panel__trigger").click()
            page.wait_for_function(
                "() => document.getElementById('retentionResult').textContent.includes('Due within the next 14 days')"
            )
            assert (
                "Automatic deletion is off"
                in page.locator("#retentionResult").inner_text()
            )
            assert "1 photo," in page.locator("#retentionResult").inner_text()
            assert (
                "existing Review backlog"
                in page.locator("#retentionResult").inner_text()
            )
            assert page.locator("#retentionResult").get_attribute("role") == "status"
            page.on("dialog", lambda dialog: dialog.accept())
            page.get_by_role("button", name="Clean up now", exact=True).click()
            page.wait_for_function(
                "() => document.getElementById('retentionResult').textContent.includes('No files or rows were deleted')"
            )
            page.get_by_label("Automatic deletion", exact=True).select_option("true")
            page.get_by_label("Old bird photos", exact=True).select_option("delete_all")
            with page.expect_navigation(wait_until="domcontentloaded"):
                with page.expect_response(
                    lambda response: (
                        response.url.endswith("/api/v1/settings")
                        and response.request.method == "POST"
                    )
                ) as saved:
                    page.get_by_role(
                        "button", name="Apply Settings", exact=True
                    ).click()
            assert saved.value.ok, saved.value.text()
            page.wait_for_load_state("domcontentloaded")
            page.locator("#panelRetention .settings-panel__trigger").click()
            page.wait_for_function(
                "() => document.getElementById('retentionResult').textContent.includes('Automatic deletion is on')"
            )
            assert (
                page.get_by_label("Old bird photos", exact=True).input_value()
                == "delete_all"
            )
            assert get_config()["AUTOMATIC_DELETION_ENABLED_AT"]
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.locator("#panelRetention").screenshot(
                path=str(tmp_path / f"settings-{width}-{theme}.png")
            )
            page.goto(base + "/storage")
            page.get_by_role("button", name="Measure storage", exact=True).click()
            page.wait_for_function(
                "() => document.getElementById('storage-status').textContent.startsWith('Measurement finished')"
            )
            page.wait_for_function(
                "() => document.getElementById('storage-policy').textContent.startsWith('Old bird photos: Delete everything')"
            )
            assert (
                page.locator("#storage-policy")
                .inner_text()
                .startswith("Old bird photos: Delete everything")
            )
            assert (
                "on at measurement time"
                in page.locator("#deletion-notice").inner_text()
            )
            assert "14 days" in page.locator("#deletion-notice").inner_text()
            assert errors == []
            assert originals_before == {
                p: p.read_bytes() for p in paths.originals_dir.rglob("*.jpg")
            }
            with closing_connection() as conn:
                assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 4
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
            cache = local_db_app.extensions.get("station_report")
            if cache:
                cache.close()
