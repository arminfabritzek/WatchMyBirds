"""HTTP contracts for canonical dataset preview and download."""

from __future__ import annotations

import zipfile
from contextlib import nullcontext
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from config import get_config
from tests.test_label_write_endpoints_contract import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from utils.path_manager import PathManager
from web.web_interface import create_web_interface


@pytest.fixture
def canonical_client(monkeypatch, tmp_path):
    _reset_test_config(monkeypatch, tmp_path)
    manager = MagicMock()
    manager.frame_lock = nullcontext()
    manager.latest_raw_timestamp = 0.0
    manager.last_good_frame_timestamp = 0.0
    manager._first_frame_received = False
    with (
        patch("web.services.auth_service.should_require_password_setup", return_value=False),
        patch("web.services.auth_service.is_default_password", return_value=False),
    ):
        app = create_web_interface(manager)
        app.config["TESTING"] = True
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["authenticated"] = True
                session["_csrf_token"] = "test-csrf-token"
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{today}_101500_canonical.jpg"
            with db_connection.closing_connection() as conn:
                detection_id = _seed(conn, filename=filename, timestamp=f"{today}_101500")
            path = PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"canonical-image")
            post(
                client,
                "/api/labels/answer",
                {
                    "filename": filename,
                    "detection_id": detection_id,
                    "object_bird_presence": "present",
                    "bbox_quality": "suitable",
                    "species_identity": "corrected",
                    "species_key": "Parus_major",
                },
            )
            yield client, filename


def test_preview_and_download_share_bundle_and_do_not_write(canonical_client) -> None:
    client, _ = canonical_client
    with db_connection.closing_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]

    preview = client.get("/api/canonical-dataset/preview")
    download = client.get("/api/canonical-dataset/download")

    assert preview.status_code == 200
    assert download.status_code == 200
    archive = zipfile.ZipFile(BytesIO(download.data))
    metadata = archive.read("bundle.json").decode()
    assert preview.get_json()["bundle_id"] in metadata
    assert "manifest.jsonl" in archive.namelist()
    assert "facts.jsonl" in archive.namelist()
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == before


def test_canonical_dataset_page_is_the_export_navigation_target(canonical_client) -> None:
    client, _ = canonical_client
    page = client.get("/admin/canonical-dataset")

    assert page.status_code == 200
    assert "Canonical Dataset" in page.get_data(as_text=True)
    appbar = open("templates/partials/appbar.html", encoding="utf-8").read()
    assert 'href="/admin/canonical-dataset"' in appbar
    assert 'href="/admin/groundtruth-export"' not in appbar
