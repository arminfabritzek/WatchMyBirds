from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask.testing import FlaskClient

import config
from utils import path_manager
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image
from web.web_interface import create_web_interface


def _reset_test_config(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    config._CONFIG = None
    db_connection._schema_initialized_paths.clear()
    path_manager._instance = None
    return output_dir


def _seed_detection(
    conn,
    *,
    filename: str,
    timestamp: str,
    species: str,
    review_status: str = "confirmed_bird",
    detection_status: str = "active",
    decision_state: str | None = None,
    score: float = 0.95,
    bbox: tuple[float, float, float, float] = (0.18, 0.16, 0.22, 0.24),
) -> int:
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
        (review_status, filename),
    )
    detection_id = insert_detection(
        conn,
        {
            "image_filename": filename,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "od_class_name": "bird",
            "od_confidence": 0.93,
            "od_model_id": "yolo-test",
            "created_at": timestamp,
            "score": score,
            "decision_state": decision_state,
            "thumbnail_path": filename.replace(".jpg", "_crop_1.webp"),
        },
    )
    conn.execute(
        "UPDATE detections SET status = ? WHERE detection_id = ?",
        (detection_status, detection_id),
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


@pytest.fixture
def local_db_app(monkeypatch, tmp_path):
    _reset_test_config(monkeypatch, tmp_path)

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

        # review.py keeps a module-level config dict; make sure it points at the
        # temp OUTPUT_DIR even if the module was imported earlier in the test run.
        import web.blueprints.review as review_blueprint

        review_blueprint.config = config.get_config()

        yield app


@pytest.fixture
def seeded_client(local_db_app):
    today_iso = datetime.now().strftime("%Y-%m-%d")
    today_prefix = today_iso.replace("-", "")

    with db_connection.closing_connection() as conn:
        # Gallery / Stream / Subgallery visible item.
        _seed_detection(
            conn,
            filename=f"{today_prefix}_120000_stream.jpg",
            timestamp=f"{today_prefix}_120000",
            species="Parus_major",
            review_status="confirmed_bird",
            decision_state="confirmed",
            score=0.98,
        )

        # Trash item for the colored trash surface.
        _seed_detection(
            conn,
            filename=f"{today_prefix}_090000_trash.jpg",
            timestamp=f"{today_prefix}_090000",
            species="Pica_pica",
            review_status="untagged",
            detection_status="rejected",
            score=0.81,
        )

        # Review continuity-batch fixture: one confirmed Gallery anchor plus two
        # actionable Review events with different predicted species.
        _seed_detection(
            conn,
            filename=f"{today_prefix}_120000_anchor.jpg",
            timestamp=f"{today_prefix}_120000",
            species="Pica_pica",
            review_status="confirmed_bird",
            decision_state="confirmed",
            score=0.96,
        )
        _seed_detection(
            conn,
            filename=f"{today_prefix}_121000_review_a.jpg",
            timestamp=f"{today_prefix}_121000",
            species="Passer_domesticus",
            review_status="untagged",
            decision_state="uncertain",
            score=0.42,
        )
        _seed_detection(
            conn,
            filename=f"{today_prefix}_122000_review_b.jpg",
            timestamp=f"{today_prefix}_122000",
            species="Sitta_europaea",
            review_status="untagged",
            decision_state="uncertain",
            score=0.39,
        )

    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        yield client, today_iso


def test_local_sqlite_routes_render_species_colour_attrs(seeded_client):
    client, today_iso = seeded_client

    stream_response = client.get("/")
    assert stream_response.status_code == 200
    stream_body = stream_response.get_data(as_text=True)
    assert 'class="quiet-preview__item' in stream_body
    assert 'data-species-colour="0"' in stream_body
    assert "--cell-species-colour: var(--species-colour-0)" in stream_body

    subgallery_response = client.get(f"/gallery/{today_iso}")
    assert subgallery_response.status_code == 200
    subgallery_body = subgallery_response.get_data(as_text=True)
    assert 'data-observation-card="true"' in subgallery_body
    assert 'data-species-colour="0"' in subgallery_body
    assert "--cell-species-colour: var(--species-colour-0)" in subgallery_body

    trash_response = client.get("/trash")
    assert trash_response.status_code == 200
    trash_body = trash_response.get_data(as_text=True)
    assert 'class="trash-item' in trash_body
    assert 'data-species-colour="0"' in trash_body
    assert "--cell-species-colour: var(--species-colour-0)" in trash_body


def test_review_keeps_current_grid_and_removes_diagnostic_surfaces(
    seeded_client: tuple[FlaskClient, str],
) -> None:
    client, _ = seeded_client
    response = client.get("/admin/review")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "data-review-grid-card" in body
    assert "review-grid__tile" in body
    assert 'data-review-grid-action="invert_selection"' in body
    assert "Diagnostic events" not in body
    assert "data-review-grid-evidence" not in body
    for path in (
        "/admin/review?diagnostic=1",
        "/admin/review?layout=legacy",
        "/admin/review?layout=grid",
        "/api/review/panel/detection/1",
        "/api/review/event-panel/bird-event-012345abcdef",
    ):
        assert client.get(path).status_code == 404


def test_authenticated_page_routes_render_without_build_errors(seeded_client):
    # Single-blueprint unit tests can't catch cross-blueprint url_for() refs
    # between settings and camera discovery; render the
    # heavy authenticated pages against the FULL app to surface BuildErrors.
    client, _today_iso = seeded_client

    for path in ("/settings", "/logs", "/privacy", "/analytics"):
        resp = client.get(path)
        assert resp.status_code == 200, (
            f"{path} rendered {resp.status_code} (likely a url_for BuildError "
            f"for a route that moved into a blueprint)"
        )


def test_static_file_routes_emit_browser_cache_headers(local_db_app):
    output_dir = Path(config.get_config()["OUTPUT_DIR"])
    thumb_dir = output_dir / "derivatives" / "thumbs" / "2026-03-27"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / "20260327_120000_visible_crop_1.webp"
    thumb_path.write_bytes(b"fake-webp")

    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True

        thumb_response = client.get(
            "/uploads/derivatives/thumbs/2026-03-27/20260327_120000_visible_crop_1.webp"
        )
        assert thumb_response.status_code == 200
        assert (
            thumb_response.headers["Cache-Control"]
            == "private, max-age=2592000, immutable"
        )

        asset_response = client.get("/assets/design-system.css")
        assert asset_response.status_code == 200
        assert asset_response.headers["Cache-Control"] == "public, max-age=604800"


@pytest.mark.parametrize("retired", [False, True])
def test_review_lazy_modal_preserves_siblings_and_retention_guards(
    seeded_client, retired: bool
) -> None:
    client, today = seeded_client
    from web.blueprints import review

    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT detection_id, image_filename FROM detections WHERE score = 0.42"
        ).fetchone()
        detection_id, filename = row
        crop_name = filename.replace(".jpg", "_crop_2.webp")
        conn.execute(
            "UPDATE detections SET thumbnail_path=? WHERE detection_id=?",
            (crop_name, detection_id),
        )
        sibling_id = insert_detection(
            conn,
            {
                "image_filename": filename,
                "bbox_x": 0.55,
                "bbox_y": 0.2,
                "bbox_w": 0.2,
                "bbox_h": 0.25,
                "od_class_name": "bird",
                "od_confidence": 0.85,
                "score": 0.3,
                "decision_state": "confirmed",
            },
        )
        conn.execute(
            "UPDATE images SET original_present = ? WHERE filename = ?",
            (0 if retired else 1, filename),
        )
        conn.commit()

    url = f"/api/review/modal/bird-event-012345abcdef/{detection_id}"
    with patch.object(
        review,
        "_fetch_review_event_payloads",
        side_effect=AssertionError("global queue"),
    ):
        response = client.get(url)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    modal = response.get_data(as_text=True)
    assert modal.count("gallery-modal wm-modal") == 1
    assert f"modal-review-{url.split('/')[-2]}-{detection_id}" in modal
    assert f'"detection_id": {sibling_id}' in modal
    assert "data-editor-object-select" in modal
    assert (f"/api/image/download/{detection_id}" not in modal) == retired

    with client.session_transaction() as session:
        session.clear()
    assert client.get(url).status_code == 302


@pytest.mark.parametrize(
    "suffix",
    ["invalid/1", "bird-event-012345abcdef/0", "bird-event-012345abcdef/999999"],
)
def test_review_lazy_modal_rejects_missing_or_invalid_detection(
    seeded_client, suffix: str
) -> None:
    client, _ = seeded_client
    assert client.get("/api/review/modal/" + suffix).status_code == 404


def test_event_action_lookup_preserves_fresh_members_without_display_work(
    seeded_client,
) -> None:
    from web.blueprints import review

    with db_connection.closing_connection() as conn:
        kwargs = dict(
            gallery_threshold=0.7,
            output_dir=config.get_config()["OUTPUT_DIR"],
            species_locale="DE",
            common_names={},
        )
        events, _, _, _ = review._fetch_review_event_payloads(conn, **kwargs)
        events = [
            event for event in events if event.context_only_count < event.photo_count
        ]
        assert events
        with (
            patch.object(
                review, "_build_review_item", side_effect=AssertionError("UI")
            ),
            patch.object(
                review,
                "_build_review_modal_detection",
                side_effect=AssertionError("modal"),
            ),
            patch.object(
                review,
                "_load_recent_review_species",
                side_effect=AssertionError("picker"),
            ),
        ):
            for expected in events:
                actual = review._load_single_review_event(
                    conn, event_key=expected.event_key, **kwargs
                )
                assert actual["event_key"] == expected.event_key
                assert actual["detection_ids"] == expected.detection_ids
                assert actual["eligibility"] == expected.eligibility
                assert actual["candidate_species"] == expected.species
            target = events[0]
            conn.executemany(
                "UPDATE detections SET status = 'rejected' WHERE detection_id = ?",
                [(value,) for value in target.detection_ids],
            )
            assert (
                review._load_single_review_event(
                    conn, event_key=target.event_key, **kwargs
                )
                is None
            )
            conn.rollback()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/inbox"),
        ("POST", "/api/inbox"),
        ("GET", "/api/inbox/status"),
        ("POST", "/api/inbox/process"),
        ("POST", "/api/ingest/start"),
    ],
)
def test_removed_image_import_routes_return_404(
    seeded_client, method: str, path: str
) -> None:
    client, _ = seeded_client
    with client.session_transaction() as session:
        session["_csrf_token"] = "test-token"
    response = client.open(path, method=method, headers={"X-CSRF-Token": "test-token"})
    assert response.status_code == 404


def test_station_pages_offer_date_filters_without_source_filter(seeded_client) -> None:
    client, _ = seeded_client
    for path in (
        "/species",
        "/species/overview?species_key=Parus_major",
        "/analytics",
        "/admin/review",
    ):
        response = client.get(path)
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert 'name="start"' in body and 'name="end"' in body
        assert 'name="source"' not in body
    settings = client.get("/settings").get_data(as_text=True)
    assert "Image Ingest" not in settings
    assert "/inbox" not in settings and "/api/ingest/start" not in settings


def test_removed_import_settings_are_ignored(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    legacy = tmp_path / "legacy-ingest"
    (output / "settings.yaml").write_text(
        "INGEST_DIR: /unused\nINBOX_REQUIRE_EXIF_DATETIME: true\nINBOX_REQUIRE_EXIF_GPS: true\n"
    )
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    monkeypatch.setenv("MODEL_BASE_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("INGEST_DIR", str(legacy))
    monkeypatch.setattr(config, "_CONFIG", None)
    loaded = config.get_config()
    config.ensure_app_directories(loaded)
    assert (
        not {"INGEST_DIR", "INBOX_REQUIRE_EXIF_DATETIME", "INBOX_REQUIRE_EXIF_GPS"}
        & loaded.keys()
    )
    assert not legacy.exists()
