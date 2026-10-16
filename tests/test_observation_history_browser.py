from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
from flask import Flask
from PIL import Image
from werkzeug.serving import make_server

from tests.test_station_observations import seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_blurry_admitted_bird_in_history_surfaces(
    local_db_app: Flask, tmp_path: Path, width: int, theme: str
) -> None:
    api = pytest.importorskip("playwright.sync_api")
    from core.gallery_core import invalidate_cache
    from web.web_interface import invalidate_best_species_cache

    invalidate_cache()
    invalidate_best_species_cache()
    with closing_connection() as conn:
        filename = seed(conn, identifier=42, species="Pica_pica", quality=0)
        from config import get_config
        from utils.path_manager import PathManager

        paths = PathManager(get_config()["OUTPUT_DIR"])
        for path in (
            paths.get_original_path(filename),
            paths.get_derivative_path(filename, "optimized"),
            paths.get_derivative_path(filename.replace(".jpg", "_crop_1.jpg"), "thumb"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (600, 400), (115, 130, 110)).save(path)
        seed(conn, identifier=43, species="Picus_viridis", state="uncertain")
    with api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:
            pytest.skip(f"Chromium for Playwright is unavailable: {error}")
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
            for name, route in (
                ("home", "/"),
                ("gallery", "/gallery/2026-10-13"),
                ("species", "/species"),
                ("species-detail", "/species/overview?species_key=Pica_pica"),
                ("report", "/analytics"),
            ):
                response = page.goto(base + route)
                assert response.status == 200
                if name == "report":
                    page.locator("[data-current-report]").wait_for()
                assert page.get_by_text("Pica pica", exact=False).count() > 0
                assert page.get_by_text("Picus viridis", exact=False).count() == 0
                assert page.locator("html").get_attribute("data-theme") == theme
                page.screenshot(
                    path=str(tmp_path / f"{name}-{width}-{theme}.png"),
                    full_page=True,
                    animations="disabled",
                )
            assert errors == []
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)
            worker = local_db_app.extensions.get("station_report")
            if worker is not None:
                worker.close()
