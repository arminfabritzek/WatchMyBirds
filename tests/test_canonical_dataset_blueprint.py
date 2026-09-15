"""HTTP contracts for canonical dataset preview and download."""

from __future__ import annotations

import os
import zipfile
from contextlib import nullcontext
from datetime import datetime
from io import BytesIO
from pathlib import Path
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
        patch(
            "web.services.auth_service.should_require_password_setup",
            return_value=False,
        ),
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
                detection_id = _seed(
                    conn, filename=filename, timestamp=f"{today}_101500"
                )
            path = PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(
                filename
            )
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
        assert (
            conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
            == before
        )


def test_canonical_dataset_page_is_the_export_navigation_target(
    canonical_client,
) -> None:
    client, filename = canonical_client
    with db_connection.closing_connection() as conn:
        _seed(
            conn, filename=filename, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    page = client.get("/admin/canonical-dataset")

    assert page.status_code == 200
    content = page.get_data(as_text=True)
    assert "Canonical Dataset" in content
    assert "Ready for another bird?" in content
    assert 'href="/admin/review"' in content
    assert 'aria-label="Review the next bird"' in content
    appbar = Path("templates/partials/appbar.html").read_text(encoding="utf-8")
    assert 'href="/admin/canonical-dataset"' in appbar
    assert 'href="/admin/groundtruth-export"' not in appbar


def test_export_page_invites_sharing_without_interrupting_anyone(
    canonical_client,
) -> None:
    client, _ = canonical_client

    page = client.get("/admin/canonical-dataset")

    content = page.get_data(as_text=True)
    assert page.status_code == 200
    assert 'id="contribute-training-title"' in content
    assert "Entirely optional" in content
    assert "Nothing leaves this device unless" in content
    assert "no hurry and no obligation" in content
    assert "attach the archive to a public" in content
    assert "Training%20data%20contribution" in content


def test_sharing_invitation_never_interrupts_the_operator(canonical_client) -> None:
    """The ask lives on the Export page only — no modal, no favorite hijack."""
    client, _ = canonical_client

    content = client.get("/admin/canonical-dataset").get_data(as_text=True)

    assert "trainingDataInvite" not in content
    assert "wmb.trainingDataInvite" not in content
    assert "data-training-data-invite-trigger" not in content
    assert not os.path.exists("templates/partials/training_data_invite.html")


def test_canonical_dataset_page_links_to_gallery_when_queue_is_clear(
    canonical_client,
) -> None:
    client, _ = canonical_client

    page = client.get("/admin/canonical-dataset")

    content = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "All caught up" in content
    assert 'href="/gallery"' in content
    assert 'aria-label="Open bird gallery"' in content


@pytest.mark.parametrize("consume", [True, False])
def test_download_cleans_temporary_archive_on_close(
    canonical_client, monkeypatch, consume
):
    from web.services import canonical_dataset_service as service

    client, _ = canonical_client
    created = []
    render = service.render_canonical_bundle_to_tempdir

    def record(*args, **kwargs):
        result = render(*args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(service, "render_canonical_bundle_to_tempdir", record)
    response = client.get("/api/canonical-dataset/download", buffered=False)
    assert response.status_code == 200
    if consume:
        assert response.data
    response.close()
    assert created
    assert created[0][0].parent == Path(get_config()["OUTPUT_DIR"]) / "backup"
    assert not created[0][0].exists()


def test_backup_form_streams_selected_options_and_requires_csrf(
    canonical_client, monkeypatch
):
    from web.blueprints import backup

    client, _ = canonical_client
    called = []
    monkeypatch.setattr(backup.time, "sleep", lambda _: None)

    def stream(**options):
        called.append(options)
        yield b"archive-chunk"

    monkeypatch.setattr(backup.backup_restore_service, "stream_backup", stream)
    fields = {
        "include_db": "true",
        "include_originals": "false",
        "include_settings": "false",
        "include_derivatives": "true",
    }
    assert client.post("/api/backup/create", data=fields).status_code == 403
    response = client.post(
        "/api/backup/create", data={**fields, "_csrf_token": "test-csrf-token"}
    )
    assert response.status_code == 200
    assert response.data == b"archive-chunk"
    response.close()
    assert called == [
        {
            "include_db": True,
            "include_originals": False,
            "include_derivatives": True,
            "include_settings": False,
        }
    ]
