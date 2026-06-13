"""Characterization tests for the index (dashboard ``/``) route.

These pin the *current* behavior of ``index_route`` before it is extracted
from the ``create_web_interface`` factory closure in ``web/web_interface.py``
into ``web/blueprints/index.py``. They build the real app via
``create_web_interface`` and seed a local SQLite DB, mirroring
``tests/test_surface_routes_local_db.py`` and ``tests/test_gallery_blueprint.py``.

The dashboard renders ``stream.html`` with many image-dependent template
context vars. The asset-stripped test repo cannot fully exercise those images,
so instead of asserting on rendered HTML these tests capture the *render
context* (by patching ``render_template``) and assert on the structure of the
context dict the route builds. That is the contract the extraction must
preserve byte-for-byte.
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
        from web.web_interface import create_web_interface

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


# The full set of template context keys the dashboard contract guarantees.
_EXPECTED_CONTEXT_KEYS = {
    "title",
    "current_path",
    "latest_detections",
    "visual_summary",
    "today_visitors_board",
    "today_visitors_modal_dets",
    "species_summary",
    "dashboard_stats",
    "empty_latest_message",
    "image_width",
    "today_iso",
    "today_detection_count",
    "is_quiet_today",
    "recent_archive_preview",
    "best_species_board",
    "best_species_preview",
    "best_species_modal_dets",
    "landing_status",
    "noise_hourly",
}


def _capture_index_context(client):
    """Hit ``/`` while capturing the kwargs passed to ``render_template``.

    Patches the ``render_template`` symbol on whichever module currently owns
    ``index_route`` so the test is agnostic to where the route lives (factory
    today, blueprint after extraction).
    """
    captured = {}

    def _fake_render(template_name, **context):
        captured["template"] = template_name
        captured["context"] = context
        return "OK"

    # Patch render_template at its source in flask so it is intercepted no
    # matter which module imported it.
    import flask

    targets = ["web.web_interface.render_template"]
    try:
        import web.blueprints.index as _index_mod  # noqa: F401

        targets.append("web.blueprints.index.render_template")
    except ImportError:
        pass

    from contextlib import ExitStack

    with ExitStack() as stack:
        patched_any = False
        for target in targets:
            try:
                stack.enter_context(patch(target, _fake_render))
                patched_any = True
            except (AttributeError, ModuleNotFoundError):
                continue
        if not patched_any:
            stack.enter_context(patch.object(flask, "render_template", _fake_render))
        resp = client.get("/")
    return resp, captured


def test_index_route_returns_200(seeded_client):
    client, _ = seeded_client
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_renders_stream_template_with_expected_context_keys(seeded_client):
    client, today_iso = seeded_client
    resp, captured = _capture_index_context(client)

    assert resp.status_code == 200
    assert captured["template"] == "stream.html"

    context = captured["context"]
    missing = _EXPECTED_CONTEXT_KEYS - set(context)
    assert not missing, f"index context dropped keys: {sorted(missing)}"

    assert context["current_path"] == "/"
    assert context["today_iso"] == today_iso
    assert context["empty_latest_message"] == "No detections in the last 24 hours."
    assert context["image_width"] == 150


def test_index_context_collection_shapes(seeded_client):
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    context = captured["context"]

    # The four board/list payloads are always lists.
    for key in (
        "latest_detections",
        "visual_summary",
        "recent_archive_preview",
        "best_species_preview",
        "best_species_modal_dets",
        "today_visitors_modal_dets",
    ):
        assert isinstance(context[key], list), f"{key} should be a list"

    # The two species boards are dicts with featured/grid sections.
    for key in ("today_visitors_board", "best_species_board"):
        board = context[key]
        assert isinstance(board, dict)
        assert "featured" in board and "grid" in board

    # dashboard_stats carries the live counters the header ticker reads.
    stats = context["dashboard_stats"]
    assert isinstance(stats, dict)
    for stat_key in ("total_species", "last_24h_count", "today_count"):
        assert stat_key in stats


def test_index_landing_status_reports_stream_offline_without_frames(seeded_client):
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    landing = captured["context"]["landing_status"]

    # detection_manager fixture has no frames -> Offline/bad with a last-detection line.
    assert landing["stream_state"] == "Offline"
    assert landing["stream_tone"] == "bad"
    assert "last_detection" in landing


def test_index_title_reflects_today_observation_count(seeded_client):
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    title = captured["context"]["title"]

    assert title.startswith("Live • ")
    assert title.endswith("Observations Today")
