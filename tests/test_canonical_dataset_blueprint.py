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
    main_start = content.index('<main class="page">')
    main_content = content[main_start : content.index("</main>", main_start)]
    assert 'href="/admin/review"' not in main_content
    appbar = Path("templates/partials/appbar.html").read_text(encoding="utf-8")
    assert 'href="/admin/canonical-dataset"' in appbar
    assert 'href="/admin/groundtruth-export"' not in appbar


def test_canonical_dataset_page_shows_honest_readiness_figures(
    canonical_client,
) -> None:
    """The header states what the station can contribute, not a stale count.

    The fixture answers presence, box and species for one detection, so it
    is both CLS- and OD-ready and leaves nothing needing a box verdict.
    """
    client, _ = canonical_client

    page = client.get("/admin/canonical-dataset")

    assert page.status_code == 200
    content = page.get_data(as_text=True)
    assert "ready for species training" in content
    assert "ready for box training" in content
    assert "need a box verdict" in content


def test_dataset_explains_missing_original_without_losing_labels(
    canonical_client,
) -> None:
    client, filename = canonical_client
    path = PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(filename)
    before = client.get("/api/canonical-dataset/preview").get_json()
    assert "Missing originals block training export" not in client.get(
        "/admin/canonical-dataset"
    ).get_data(as_text=True)
    path.unlink()

    content = client.get("/admin/canonical-dataset").get_data(as_text=True)
    after = client.get("/api/canonical-dataset/preview").get_json()
    assert "1 labeled bird is excluded from training export" in content
    assert "Your corrections are saved." in content
    assert after["counts"]["facts"] == before["counts"]["facts"]
    assert after["counts"].get("cls_positive_included", 0) == 0


def test_canonical_dataset_page_offers_the_walkthrough_when_verdicts_are_missing(
    canonical_client,
) -> None:
    """A detection with species answered but no box verdict shows the CTA."""
    client, filename = canonical_client
    with db_connection.closing_connection() as conn:
        detection_id = _seed(
            conn,
            filename=filename,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )

    page = client.get("/admin/canonical-dataset")

    content = page.get_data(as_text=True)
    assert page.status_code == 200
    assert 'href="/admin/canonical-dataset/box-walkthrough"' in content
    assert "Go through them" in content


def test_box_walkthrough_shows_the_candidate_editor(canonical_client) -> None:
    """The walkthrough embeds the same bird editor the detail modal uses."""
    client, filename = canonical_client
    with db_connection.closing_connection() as conn:
        detection_id = _seed(
            conn,
            filename=filename,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )

    page = client.get("/admin/canonical-dataset/box-walkthrough")

    assert page.status_code == 200
    content = page.get_data(as_text=True)
    assert "data-current-detection=" in content
    assert "data-bird-editor" in content
    assert "Box fits" in content
    assert 'class="btn btn--primary btn--sm"' in content
    assert content.index('data-editor-action="bbox-verdict"') < content.index('data-editor-action="menu"')
    assert f"skip={detection_id}" in content
    assert "/assets/js/species_picker.js?v=" in content
    assert "/assets/js/gallery_utils.js?v=" in content
    assert "/assets/js/tile_actions.js?v=" in content
    assert "initSmartZoom(image)" in content


def test_box_walkthrough_is_empty_when_nothing_needs_a_verdict(
    canonical_client,
) -> None:
    """The fixture's one detection already carries a box verdict."""
    client, _ = canonical_client

    page = client.get("/admin/canonical-dataset/box-walkthrough")

    assert page.status_code == 200
    content = page.get_data(as_text=True)
    assert "Nothing is waiting on a box verdict" in content
    assert "data-current-detection=" not in content


def test_box_walkthrough_skip_moves_to_the_next_candidate(canonical_client) -> None:
    """A skipped detection ID is excluded, but nothing is written for it."""
    client, filename = canonical_client
    with db_connection.closing_connection() as conn:
        detection_id = _seed(
            conn,
            filename=filename,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )
    with db_connection.closing_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]

    page = client.get(f"/admin/canonical-dataset/box-walkthrough?skip={detection_id}")

    assert page.status_code == 200
    content = page.get_data(as_text=True)
    assert "Nothing is waiting on a box verdict" in content
    with db_connection.closing_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
    assert after == before


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


def test_canonical_dataset_page_has_no_action_when_no_box_verdict_is_needed(
    canonical_client,
) -> None:
    client, _ = canonical_client

    page = client.get("/admin/canonical-dataset")

    content = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "All caught up" in content
    assert 'href="/admin/canonical-dataset/box-walkthrough"' not in content


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
