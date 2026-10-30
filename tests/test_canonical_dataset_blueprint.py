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
from tests.export_helpers import download_export, prepare_export
from tests.labeling_helpers import _reset_test_config, _seed, post
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
        app.extensions["training_export_jobs"].close()


def test_preview_and_download_share_bundle_and_do_not_write(canonical_client) -> None:
    client, _ = canonical_client
    with db_connection.closing_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]

    preview = client.get("/api/canonical-dataset/preview")
    download = download_export(client)

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
    assert "Training export" in content
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
    assert content.index('data-editor-action="bbox-verdict"') < content.index(
        'data-editor-action="menu"'
    )
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
def test_prepared_download_does_not_rebuild_files(canonical_client, consume):
    client, _ = canonical_client
    root = Path(get_config()["OUTPUT_DIR"])
    identifier = prepare_export(client)
    before = {p.relative_to(root) for p in root.rglob("*") if p.is_file()}
    response = client.get(
        f"/api/canonical-dataset/exports/{identifier}/download", buffered=False
    )
    assert response.status_code == 200
    for chunk in response.response:
        assert chunk
        assert {p.relative_to(root) for p in root.rglob("*") if p.is_file()} == before
        if not consume:
            break
    response.close()
    assert {p.relative_to(root) for p in root.rglob("*") if p.is_file()} == before
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM labeling_exports WHERE received_at IS NULL"
            ).fetchone()[0]
            == 1
        )
    headers = {"X-CSRF-Token": "test-csrf-token"}
    assert (
        client.post(
            f"/api/canonical-dataset/exports/{identifier}/received", headers=headers
        ).status_code
        == 303
    )
    assert (
        client.post(
            f"/api/canonical-dataset/exports/{identifier}/remove-archive",
            headers=headers,
        ).status_code
        == 303
    )


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


def test_repeat_download_is_stable_after_answer_changes(canonical_client) -> None:
    client, filename = canonical_client
    original = download_export(client)
    with db_connection.closing_connection() as conn:
        export_id = conn.execute("SELECT export_id FROM labeling_exports").fetchone()[0]
        detection_id = conn.execute(
            "SELECT detection_id FROM detections WHERE image_filename=?", (filename,)
        ).fetchone()[0]
    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "corrected",
            "species_key": "Pica_pica",
        },
    )
    repeated = client.get(f"/api/canonical-dataset/exports/{export_id}/download")
    assert repeated.status_code == 200 and repeated.data == original.data
    page = client.get("/admin/canonical-dataset").get_data(as_text=True)
    assert "1 photo is waiting for training" in page
    assert "Not yet confirmed" in page
    assert (
        client.post(f"/api/canonical-dataset/exports/{export_id}/received").status_code
        == 403
    )
    response = client.post(
        f"/api/canonical-dataset/exports/{export_id}/received",
        data={"_csrf_token": "test-csrf-token"},
    )
    assert response.status_code == 303
    page = client.get(response.location).get_data(as_text=True)
    assert "Not yet confirmed" not in page
    assert "1 photo is waiting for training" in page


def test_export_actions_require_auth_csrf_and_valid_selection(canonical_client) -> None:
    client, _ = canonical_client
    assert client.get("/api/canonical-dataset/download").status_code == 405
    assert client.post("/api/canonical-dataset/download").status_code == 403
    for selection in (
        "detection:1",
        [False],
        ["detection:999999"],
        ["detection:999999999999999999999999999999999999"],
        [],
    ):
        response = post(
            client, "/api/canonical-dataset/download", {"selection": selection}
        )
        assert response.status_code in {400, 409}
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
    with client.session_transaction() as session:
        session.clear()
    assert (
        client.get("/api/canonical-dataset/exports/missing/download").status_code == 302
    )


def test_prepared_download_survives_missing_original_without_confirming(
    canonical_client,
) -> None:
    client, filename = canonical_client
    download_export(client)
    with db_connection.closing_connection() as conn:
        identifier = conn.execute("SELECT export_id FROM labeling_exports").fetchone()[
            0
        ]
    PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(filename).unlink()
    assert (
        client.get(f"/api/canonical-dataset/exports/{identifier}/download").status_code
        == 200
    )
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute("SELECT received_at FROM labeling_exports").fetchone()[0]
            is None
        )


def test_prepared_archive_supports_range_retry_without_receipt(canonical_client):
    client, _ = canonical_client
    identifier = prepare_export(client)
    endpoint = f"/api/canonical-dataset/exports/{identifier}/download"
    full = client.get(endpoint, buffered=True)
    first = client.get(endpoint, headers={"Range": "bytes=0-99"}, buffered=True)
    assert first.status_code == 206
    assert first.data == full.data[:100]
    assert first.headers["Content-Range"] == f"bytes 0-99/{len(full.data)}"
    rest = client.get(
        endpoint,
        headers={"Range": "bytes=100-", "If-Range": full.headers["ETag"]},
        buffered=True,
    )
    assert rest.status_code == 206
    assert first.data + rest.data == full.data
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute(
                "SELECT received_at FROM labeling_exports WHERE export_id=?",
                (identifier,),
            ).fetchone()[0]
            is None
        )


def test_job_actions_require_authentication_and_csrf(canonical_client):
    client, _ = canonical_client
    identifier = prepare_export(client)
    for endpoint in (
        f"/api/canonical-dataset/jobs/{identifier}/cancel",
        f"/api/canonical-dataset/jobs/{identifier}/resume",
        f"/api/canonical-dataset/exports/{identifier}/remove-archive",
        f"/api/canonical-dataset/exports/{identifier}/rebuild",
    ):
        assert client.post(endpoint).status_code == 403
    with client.session_transaction() as session:
        session.clear()
    assert client.get(f"/api/canonical-dataset/jobs/{identifier}").status_code == 302
