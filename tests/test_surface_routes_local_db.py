from __future__ import annotations

import re
from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import config
from utils.db import insert_classification, insert_detection, insert_image
from utils.db import connection as db_connection
from web.web_interface import create_web_interface


def _reset_test_config(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    ingest_dir = tmp_path / "ingest"
    output_dir.mkdir()
    ingest_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("INGEST_DIR", str(ingest_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    config._CONFIG = None
    db_connection._schema_initialized_paths.clear()
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


def test_local_sqlite_review_routes_render_real_continuity_batch(seeded_client):
    client, _today_iso = seeded_client

    review_response = client.get("/admin/review")
    assert review_response.status_code == 200
    review_body = review_response.get_data(as_text=True)
    assert 'id="reviewEventBrowser"' in review_body
    event_keys = re.findall(r'data-event-key="([^"]+)"', review_body)
    assert len(event_keys) == 2

    panel_response = client.get(f"/api/review/event-panel/{event_keys[0]}")
    assert panel_response.status_code == 200
    panel_body = panel_response.get_data(as_text=True)
    assert "data-review-batch-panel" in panel_body
    assert "Already in Gallery" in panel_body
    assert "Review now" in panel_body
    assert 'data-zoom-pref-key="wmb_review_zoom_pref"' in panel_body
    assert "data-species-colour=" in panel_body
    assert 'data-review-panel-action="trash_event"' in panel_body
