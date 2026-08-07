"""Tests for the selected-images ZIP download.

Both entry points share ``web.services.image_download_service``: the
date-scoped ``/api/edit/actions?action=download`` used by the Gallery
edit page, and ``/api/moderation/bulk/download``, which takes a bare id
list so the Species view can download favorites spanning many days.

Guarded behaviour:
- one entry per source image, even when several detections share a frame
- files missing on disk are skipped, not fatal
- ``downloaded_timestamp`` is stamped on the served filenames
- metadata burn-in replaces the raw original when enabled, and falls back
  to the raw original when burn-in raises
"""

from __future__ import annotations

import io
import zipfile
from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tests.conftest import reset_config_in_place
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image
from web.web_interface import create_web_interface

SHARED_COLOUR = (200, 30, 30)
SOLO_COLOUR = (30, 30, 200)


def _reset_test_config(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    ingest_dir = tmp_path / "ingest"
    output_dir.mkdir()
    ingest_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("INGEST_DIR", str(ingest_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    reset_config_in_place()
    return output_dir


def _seed_detection(
    conn,
    *,
    filename: str,
    timestamp: str,
    species: str,
    is_favorite: int = 0,
) -> int:
    """Insert one image + detection + classification, return the detection id."""
    existing = conn.execute(
        "SELECT 1 FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    if not existing:
        insert_image(
            conn,
            {
                "filename": filename,
                "timestamp": timestamp,
                "source_id": 1,
                "content_hash": f"hash-{filename}",
            },
        )
        conn.execute(
            "UPDATE images SET review_status = ? WHERE filename = ?",
            ("confirmed_bird", filename),
        )

    detection_id = insert_detection(
        conn,
        {
            "image_filename": filename,
            "bbox_x": 0.18,
            "bbox_y": 0.16,
            "bbox_w": 0.22,
            "bbox_h": 0.24,
            "od_class_name": "bird",
            "od_confidence": 0.93,
            "od_model_id": "yolo-test",
            "created_at": timestamp,
            "score": 0.95,
            "decision_state": "confirmed",
            "thumbnail_path": filename.replace(".jpg", "_crop_1.webp"),
        },
    )
    conn.execute(
        "UPDATE detections SET status = 'active', is_favorite = ? "
        "WHERE detection_id = ?",
        (is_favorite, detection_id),
    )
    insert_classification(
        conn,
        {
            "detection_id": detection_id,
            "cls_class_name": species,
            "cls_confidence": 0.97,
            "cls_model_id": "cls-test",
            "rank": 1,
            "created_at": timestamp,
        },
    )
    conn.commit()
    return detection_id


def _write_original(output_dir, date_folder: str, filename: str, marker: tuple):
    """Write a real (tiny) JPEG so the metadata burn-in path can open it.

    ``marker`` is an RGB tuple; it survives the burn-in re-encode and lets a
    test tell the two seeded frames apart without byte-exact comparison.
    """
    originals = output_dir / "originals" / date_folder
    originals.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), marker).save(originals / filename, "JPEG")


def _dominant_colour(raw: bytes) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(raw)) as img:
        r, g, b = img.convert("RGB").getpixel((4, 4))  # type: ignore[misc]
        return (r, g, b)


def _assert_colour(raw: bytes, expected: tuple[int, int, int]) -> None:
    """Compare with tolerance — JPEG is lossy, so exact equality is wrong."""
    actual = _dominant_colour(raw)
    assert all(abs(a - e) <= 8 for a, e in zip(actual, expected, strict=True)), (
        f"expected ~{expected}, got {actual}"
    )


@pytest.fixture
def download_app(monkeypatch, tmp_path):
    output_dir = _reset_test_config(monkeypatch, tmp_path)

    detection_manager = MagicMock()
    detection_manager.frame_lock = nullcontext()
    detection_manager.latest_raw_timestamp = 0.0
    detection_manager.last_good_frame_timestamp = 0.0
    detection_manager._first_frame_received = False

    with (
        patch(
            "web.services.auth_service.should_require_password_setup",
            return_value=False,
        ),
        patch("web.services.auth_service.is_default_password", return_value=False),
    ):
        app = create_web_interface(detection_manager)
        app.config["TESTING"] = True
        yield app, output_dir


@pytest.fixture
def seeded(download_app):
    """Two frames; the first carries two detections so dedup is observable."""
    app, output_dir = download_app
    today_iso = datetime.now().strftime("%Y-%m-%d")
    prefix = today_iso.replace("-", "")

    shared = f"{prefix}_120000_stream.jpg"
    solo = f"{prefix}_121000_anchor.jpg"

    with db_connection.closing_connection() as conn:
        det_a = _seed_detection(
            conn,
            filename=shared,
            timestamp=f"{prefix}_120000",
            species="Parus_major",
            is_favorite=1,
        )
        det_b = _seed_detection(
            conn,
            filename=shared,
            timestamp=f"{prefix}_120000",
            species="Pica_pica",
        )
        det_c = _seed_detection(
            conn,
            filename=solo,
            timestamp=f"{prefix}_121000",
            species="Pica_pica",
            is_favorite=1,
        )

    _write_original(output_dir, today_iso, shared, SHARED_COLOUR)
    _write_original(output_dir, today_iso, solo, SOLO_COLOUR)

    with app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
            session["_csrf_token"] = "test-csrf-token"
        yield {
            "client": client,
            "date_iso": today_iso,
            "output_dir": output_dir,
            "shared_name": shared,
            "solo_name": solo,
            "ids": {"shared_a": det_a, "shared_b": det_b, "solo": det_c},
        }


@pytest.fixture
def burn_in_off():
    """Pin burn-in OFF.

    ``EXPORT_BURN_IN_METADATA`` is a settings.yaml runtime key (not a boot
    env var), so it cannot be set via monkeypatch.setenv. Tests that care
    about the raw-original path patch the predicate directly; production
    default is ON, which the burn-in tests below cover.
    """
    with patch(
        "web.services.metadata_export_service.burn_in_enabled", return_value=False
    ):
        yield


def _post_download(client, date_iso, ids):
    return client.post(
        "/api/edit/actions",
        data={
            "_csrf_token": "test-csrf-token",
            "action": "download",
            "date_iso": date_iso,
            "ids": [str(i) for i in ids],
        },
    )


def _names_in_zip(response) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(response.get_data())) as zf:
        return sorted(zf.namelist())


def test_download_returns_zip_attachment(seeded, burn_in_off):
    response = _post_download(
        seeded["client"], seeded["date_iso"], [seeded["ids"]["solo"]]
    )

    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "attachment" in response.headers["Content-Disposition"]


def test_download_emits_one_entry_per_source_image(seeded, burn_in_off):
    """Two detections on one frame must not yield the original twice."""
    ids = [seeded["ids"]["shared_a"], seeded["ids"]["shared_b"]]

    response = _post_download(seeded["client"], seeded["date_iso"], ids)

    assert _names_in_zip(response) == [seeded["shared_name"]]


def test_download_zips_each_distinct_frame(seeded, burn_in_off):
    ids = [seeded["ids"]["shared_a"], seeded["ids"]["solo"]]

    response = _post_download(seeded["client"], seeded["date_iso"], ids)

    assert _names_in_zip(response) == sorted(
        [seeded["shared_name"], seeded["solo_name"]]
    )


def test_download_serves_the_matching_original(seeded, burn_in_off):
    response = _post_download(
        seeded["client"], seeded["date_iso"], [seeded["ids"]["solo"]]
    )

    with zipfile.ZipFile(io.BytesIO(response.get_data())) as zf:
        _assert_colour(zf.read(seeded["solo_name"]), SOLO_COLOUR)


def test_download_skips_files_missing_on_disk(seeded, burn_in_off):
    """A row whose original is gone is skipped; the rest still zip."""
    (
        seeded["output_dir"] / "originals" / seeded["date_iso"] / seeded["solo_name"]
    ).unlink()

    ids = [seeded["ids"]["shared_a"], seeded["ids"]["solo"]]
    response = _post_download(seeded["client"], seeded["date_iso"], ids)

    assert response.status_code == 200
    assert _names_in_zip(response) == [seeded["shared_name"]]


def test_download_stamps_downloaded_timestamp(seeded, burn_in_off):
    _post_download(seeded["client"], seeded["date_iso"], [seeded["ids"]["solo"]])

    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT downloaded_timestamp FROM images WHERE filename = ?",
            (seeded["solo_name"],),
        ).fetchone()

    assert row["downloaded_timestamp"]


def test_download_without_ids_redirects_instead_of_zipping(seeded):
    response = seeded["client"].post(
        "/api/edit/actions",
        data={
            "_csrf_token": "test-csrf-token",
            "action": "download",
            "date_iso": seeded["date_iso"],
        },
    )

    assert response.status_code in (301, 302)


def test_download_uses_burn_in_copy_when_enabled(seeded):
    """With burn-in on, the archive carries the export copy, not the raw file."""
    with (
        patch(
            "web.services.metadata_export_service.burn_in_enabled", return_value=True
        ),
        patch(
            "web.services.metadata_export_service.produce_copy_bytes",
            return_value=b"burned-in-bytes",
        ),
        patch(
            "web.services.metadata_export_service.export_filename",
            return_value="exported.jpg",
        ),
    ):
        response = _post_download(
            seeded["client"], seeded["date_iso"], [seeded["ids"]["solo"]]
        )

    with zipfile.ZipFile(io.BytesIO(response.get_data())) as zf:
        assert zf.namelist() == ["exported.jpg"]
        assert zf.read("exported.jpg") == b"burned-in-bytes"


def test_download_falls_back_to_raw_original_when_burn_in_fails(seeded):
    """Burn-in is best-effort: a failure must not lose the image."""
    with (
        patch(
            "web.services.metadata_export_service.burn_in_enabled", return_value=True
        ),
        patch(
            "web.services.metadata_export_service.produce_copy_bytes",
            side_effect=RuntimeError("exif writer exploded"),
        ),
    ):
        response = _post_download(
            seeded["client"], seeded["date_iso"], [seeded["ids"]["solo"]]
        )

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.get_data())) as zf:
        assert zf.namelist() == [seeded["solo_name"]]
        _assert_colour(zf.read(seeded["solo_name"]), SOLO_COLOUR)


def test_download_requires_authentication(download_app):
    app, _ = download_app
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["_csrf_token"] = "test-csrf-token"
        response = client.post(
            "/api/edit/actions",
            data={
                "_csrf_token": "test-csrf-token",
                "action": "download",
                "date_iso": datetime.now().strftime("%Y-%m-%d"),
                "ids": ["1"],
            },
        )

    assert response.status_code in (302, 401, 403)


# --- Date-independent bulk endpoint (Species view) -------------------------


def _post_bulk_download(client, ids):
    return client.post(
        "/api/moderation/bulk/download",
        json={"detection_ids": ids},
        headers={"X-CSRF-Token": "test-csrf-token"},
    )


def test_bulk_download_returns_zip(seeded, burn_in_off):
    response = _post_bulk_download(seeded["client"], [seeded["ids"]["solo"]])

    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert _names_in_zip(response) == [seeded["solo_name"]]


def test_bulk_download_spans_dates_without_a_date_argument(seeded, burn_in_off):
    """The Species view selects favorites across days; no date is passed."""
    ids = [seeded["ids"]["shared_a"], seeded["ids"]["solo"]]

    response = _post_bulk_download(seeded["client"], ids)

    assert _names_in_zip(response) == sorted(
        [seeded["shared_name"], seeded["solo_name"]]
    )


def test_bulk_download_dedups_detections_sharing_a_frame(seeded, burn_in_off):
    ids = [seeded["ids"]["shared_a"], seeded["ids"]["shared_b"]]

    response = _post_bulk_download(seeded["client"], ids)

    assert _names_in_zip(response) == [seeded["shared_name"]]


def test_bulk_download_rejects_empty_selection(seeded):
    response = _post_bulk_download(seeded["client"], [])

    assert response.status_code == 400


def test_bulk_download_rejects_non_numeric_ids(seeded):
    response = seeded["client"].post(
        "/api/moderation/bulk/download",
        json={"detection_ids": ["not-an-id"]},
        headers={"X-CSRF-Token": "test-csrf-token"},
    )

    assert response.status_code == 400


def test_bulk_download_404s_when_nothing_resolves(seeded):
    response = _post_bulk_download(seeded["client"], [999999])

    assert response.status_code == 404


def test_bulk_download_requires_csrf(seeded):
    response = seeded["client"].post(
        "/api/moderation/bulk/download",
        json={"detection_ids": [seeded["ids"]["solo"]]},
    )

    assert response.status_code == 403


def test_bulk_download_requires_authentication(download_app):
    app, _ = download_app
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["_csrf_token"] = "test-csrf-token"
        response = client.post(
            "/api/moderation/bulk/download",
            json={"detection_ids": [1]},
            headers={"X-CSRF-Token": "test-csrf-token"},
        )

    assert response.status_code in (302, 401, 403)


def test_species_overview_renders_download_button_for_moderators(seeded):
    """The favorites download must be reachable from the Species view."""
    response = seeded["client"].get("/species/overview?species_key=Parus_major")

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "inlineEditDownload()" in body
    assert "Download Selected" in body


def test_species_overview_hides_download_button_when_anonymous(download_app):
    app, _ = download_app
    with app.test_client() as client:
        response = client.get("/species/overview?species_key=Parus_major")

    body = response.get_data(as_text=True)
    assert "inlineEditDownload()" not in body
