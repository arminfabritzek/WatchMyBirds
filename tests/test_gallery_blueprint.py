"""Characterization tests for the gallery/edit/species page routes.

These pin the *current* behavior (status code + key response/render shape) of
the seven routes before they are extracted from ``web_interface.py`` into
``web/blueprints/gallery.py``. They build the real app via
``create_web_interface`` and seed a local SQLite DB, mirroring
``test_surface_routes_local_db.py``.

Routes covered:
- ``/api/daily_species_summary``  (daily_species_summary)
- ``/edit/<date_iso>``            (edit_page)
- ``/api/edit/actions``           (edit_actions)
- ``/species``                    (species)
- ``/species/overview``           (species_overview)
- ``/gallery``                    (gallery)
- ``/gallery/<date>``             (subgallery)
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import config
from utils import path_manager
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image
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
        yield app


@pytest.fixture
def seeded_client(local_db_app):
    today_iso = datetime.now().strftime("%Y-%m-%d")
    today_prefix = today_iso.replace("-", "")

    with db_connection.closing_connection() as conn:
        _seed_detection(
            conn,
            filename=f"{today_prefix}_120000_stream.jpg",
            timestamp=f"{today_prefix}_120000",
            species="Parus_major",
            review_status="confirmed_bird",
            decision_state="confirmed",
            score=0.98,
        )
        _seed_detection(
            conn,
            filename=f"{today_prefix}_121000_anchor.jpg",
            timestamp=f"{today_prefix}_121000",
            species="Pica_pica",
            review_status="confirmed_bird",
            decision_state="confirmed",
            score=0.96,
        )

    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        yield client, today_iso


def test_daily_species_summary_returns_json_for_seeded_date(seeded_client):
    client, today_iso = seeded_client

    response = client.get(f"/api/daily_species_summary?date={today_iso}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["date"] == today_iso
    assert "summary" in payload
    assert isinstance(payload["summary"], (list, dict))


def test_daily_species_summary_rejects_bad_date(seeded_client):
    client, _ = seeded_client

    response = client.get("/api/daily_species_summary?date=not-a-date")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_edit_page_renders_for_seeded_date(seeded_client):
    client, today_iso = seeded_client

    response = client.get(f"/edit/{today_iso}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # edit.html surfaces the species filter dropdown sourced from species_list
    assert "Parus_major" in body or "Pica_pica" in body


def test_edit_page_rejects_bad_date(seeded_client):
    client, _ = seeded_client

    response = client.get("/edit/2026-13-99")

    assert response.status_code == 400


def test_edit_actions_post_requires_csrf_token(seeded_client):
    client, today_iso = seeded_client

    # A bare POST (no CSRF token) is rejected by the security before_request
    # hook that guards all state-changing routes. This characterizes the
    # current behavior so the extraction must keep the route behind that hook.
    response = client.post(
        "/api/edit/actions",
        data={"action": "reject", "date_iso": today_iso},
    )

    assert response.status_code == 403


def test_species_route_renders(seeded_client):
    client, _ = seeded_client

    response = client.get("/species")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'current_path' not in body  # sanity: template rendered, not echoed
    assert "species" in body.lower()


def test_species_overview_renders_for_seeded_species(seeded_client):
    client, _ = seeded_client

    response = client.get("/species/overview?species_key=Parus_major")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Parus_major" in body or "Parus major" in body


def test_species_overview_blank_key_redirects(seeded_client):
    client, _ = seeded_client

    response = client.get("/species/overview")

    assert response.status_code in (301, 302)
    assert "/species" in response.headers["Location"]


def test_gallery_route_lists_seeded_day(seeded_client):
    client, today_iso = seeded_client

    response = client.get("/gallery")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # gallery.html links each day card to its subgallery
    assert f"/gallery/{today_iso}" in body


def test_subgallery_route_renders_observation_cards(seeded_client):
    client, today_iso = seeded_client

    response = client.get(f"/gallery/{today_iso}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'data-observation-card="true"' in body


def test_subgallery_route_rejects_bad_date(seeded_client):
    client, _ = seeded_client

    response = client.get("/gallery/not-a-date")

    assert response.status_code == 400
