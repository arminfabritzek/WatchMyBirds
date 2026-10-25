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
from flask import Flask
from flask.testing import FlaskClient

import config
from utils import path_manager
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image


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
            "decision_level": "species",
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
    "visual_summary",
    "today_visitors_board",
    "today_visitors_modal_dets",
    "dashboard_stats",
    "image_width",
    "today_iso",
    "best_species_board",
    "best_species_preview",
    "best_species_modal_dets",
}


def _capture_index_context(client, path="/"):
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
        resp = client.get(path)
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
    assert context["image_width"] == 150


def test_index_context_collection_shapes(seeded_client):
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    context = captured["context"]

    # The four board/list payloads are always lists.
    for key in (
        "visual_summary",
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
    for stat_key in ("total_species", "today_visits"):
        assert stat_key in stats


def test_homepage_does_not_fetch_retired_widget_history(
    seeded_client: tuple[FlaskClient, str],
) -> None:
    from web.services import db_service

    client, _ = seeded_client
    with patch.object(
        db_service,
        "fetch_detections_last_24h",
        side_effect=AssertionError("retired widget"),
    ) as retired:
        response = client.get("/")
    assert response.status_code == 200
    assert retired.call_count == 0


def test_index_title_reflects_today_observation_count(seeded_client):
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    title = captured["context"]["title"]

    assert title.startswith("Live • ")
    assert title.endswith("Observations Today")


def test_modal_rows_carry_the_image_filename(seeded_client):
    """Every modal row must name its source image.

    ``bird_editor_toolbar.html`` renders ``data-filename`` from
    ``image_filename``/``original_name``, and ``bird_editor.js`` posts that
    value to ``/api/labels/answer``. A row without it renders an empty
    attribute, so confirming a species from the dashboard fails with
    "filename required" while the same bird works from the gallery.
    """
    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    context = captured["context"]

    checked = 0
    for key in (
        "visual_summary",
        "today_visitors_modal_dets",
        "best_species_modal_dets",
    ):
        for row in context[key]:
            name = str(row.get("original_name") or row.get("image_filename") or "")
            assert name.strip(), (
                f"{key} row for detection {row.get('detection_id')} carries no "
                "image filename, so its editor cannot save"
            )
            checked += 1

    assert checked, "fixture produced no modal rows to check"


def test_homepage_refreshes_canonical_answers_for_cached_modals_and_siblings(
    seeded_client,
) -> None:
    from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer

    client, _ = seeded_client
    _, captured = _capture_index_context(client)
    row = captured["context"]["visual_summary"][0]
    detection_id = row["detection_id"]
    with db_connection.closing_connection() as conn:
        filename = conn.execute(
            "SELECT image_filename FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()[0]
    provenance = LabelProvenance(
        "test-station", "test", "normal_correction", "watchmybirds_ui"
    )

    _capture_index_context(client, "/homepage/archive")
    for species in (row["species_key"], "Cyanistes_caeruleus"):
        with db_connection.closing_connection() as conn:
            record_human_answer(
                conn,
                HumanAnswer(
                    filename,
                    detection_id=detection_id,
                    object_bird_presence="present",
                    species_identity="confirmed",
                    species_key=species,
                    bbox_quality="suitable",
                ),
                provenance,
            )
            conn.commit()
        checked = []
        for path in ("/", "/homepage/archive"):
            response, captured = _capture_index_context(client, path)
            assert response.status_code == 200
            for key in (
                "visual_summary",
                "today_visitors_modal_dets",
                "best_species_modal_dets",
            ):
                for item in captured["context"][key]:
                    for candidate in (item, *item.get("siblings", [])):
                        if candidate.get("detection_id") == detection_id:
                            checked.append(candidate)
                            assert candidate["human_review_state"] in {
                                "confirmed",
                                "corrected",
                            }
                            assert candidate["human_species_key"] == species
                            assert candidate["human_bbox_verdict"] == "suitable"
                            assert candidate["needs_review"] is False
        assert len(checked) >= 4
        from web import web_interface

        cached_modals = web_interface._best_species_cache["payload"]["modal_rows"]
        assert all("human_review_state" not in item for item in cached_modals)


@pytest.mark.parametrize("empty_batch", [False, True])
def test_archive_only_hydrates_visible_frame_companions(
    local_db_app: Flask, empty_batch: bool
) -> None:
    from web import view_helpers, web_interface
    from web.services import gallery_service

    with db_connection.closing_connection() as conn:
        for index in range(25):
            _seed_detection(
                conn,
                filename=f"20200101_120000_{index}.jpg",
                timestamp="20200101_120000",
                species="Parus_major",
                decision_state="confirmed",
            )
    web_interface.invalidate_best_species_cache()
    with (
        local_db_app.test_client() as client,
        patch.object(
            gallery_service,
            "get_sibling_detections_batch",
            **(
                {"return_value": {}}
                if empty_batch
                else {"wraps": gallery_service.get_sibling_detections_batch}
            ),
        ) as batch,
        patch.object(
            gallery_service,
            "get_sibling_detections",
            wraps=gallery_service.get_sibling_detections,
        ) as single,
        patch(
            "web.view_helpers.fetch_best_species_pools",
            wraps=view_helpers.fetch_best_species_pools,
        ) as pools,
    ):
        with client.session_transaction() as session:
            session["authenticated"] = True
        for _ in range(2):
            response, captured = _capture_index_context(client, "/homepage/archive")
            assert response.status_code == 200
            context = captured["context"]
            modals = context["best_species_modal_dets"]
            frames = context["best_species_board"]["featured"][0]["story_frames"]
            assert {row["detection_id"] for row in modals} == {
                row["detection_id"] for row in frames
            }
            assert len(modals) == 4
            assert all(
                len(row["siblings"]) == (0 if empty_batch else 1) for row in modals
            )
            assert len(batch.call_args.args[0]) == 4
        assert batch.call_count == 2  # Current companions on each response.
        assert pools.call_count == 1  # Archive ranking shared by both requests.
        assert single.call_count == 0


def test_homepage_does_not_wait_for_archive_counts_or_pools(
    seeded_client: tuple[FlaskClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from web import web_interface
    from web.services import db_service

    web_interface.invalidate_best_species_cache()
    monkeypatch.setattr(
        web_interface,
        "_ticker_stats_cache",
        {"key": None, "timestamp": 0.0, "payload": None},
    )
    client, _ = seeded_client
    with (
        patch.object(
            db_service,
            "fetch_gallery_total_species_count",
            side_effect=AssertionError("full scan"),
        ) as count,
        patch(
            "web.view_helpers.fetch_best_species_pools",
            side_effect=AssertionError("ranking"),
        ) as pools,
    ):
        response = client.get("/")
    assert response.status_code == 200
    assert b"Loading your archive" in response.data
    assert b"total: null" in response.data
    assert count.call_count == pools.call_count == 0
    assert web_interface._ticker_stats_cache["payload"] is None


def test_archive_total_comes_from_the_same_ranking(
    seeded_client: tuple[FlaskClient, str],
) -> None:
    from web import web_interface
    from web.services import db_service

    web_interface.invalidate_best_species_cache()
    client, _ = seeded_client
    with patch.object(
        db_service,
        "fetch_gallery_total_species_count",
        side_effect=AssertionError("duplicate scan"),
    ) as count:
        response = client.get("/homepage/archive")
    assert response.status_code == 200
    assert b'data-total-species="2"' in response.data
    assert count.call_count == 0
    assert response.headers["Cache-Control"] == "no-store"


def test_archive_failure_is_retryable_without_breaking_homepage(
    seeded_client: tuple[FlaskClient, str],
) -> None:
    from web import web_interface

    web_interface.invalidate_best_species_cache()
    client, _ = seeded_client
    with patch(
        "web.view_helpers.fetch_best_species_pools",
        side_effect=RuntimeError("temporary DB error"),
    ):
        assert client.get("/").status_code == 200
        assert client.get("/homepage/archive").status_code == 503
    assert client.get("/homepage/archive").status_code == 200


def test_concurrent_archive_requests_share_one_rebuild(local_db_app: Flask) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from web import view_helpers, web_interface

    web_interface.invalidate_best_species_cache()
    entered, release = Event(), Event()
    fetch = view_helpers.fetch_best_species_pools

    def slow_fetch(**kwargs: object) -> tuple[list[dict], list[dict]]:
        entered.set()
        assert release.wait(5)
        return fetch(**kwargs)

    def request_archive() -> int:
        with local_db_app.test_client() as client:
            return client.get("/homepage/archive").status_code

    with patch.object(
        view_helpers, "fetch_best_species_pools", side_effect=slow_fetch
    ) as build:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(request_archive)
            try:
                assert entered.wait(5)
                second = workers.submit(request_archive)
                with local_db_app.test_client() as client:
                    assert client.get("/").status_code == 200
                assert not first.done()
            finally:
                release.set()
            assert first.result(timeout=5) == second.result(timeout=5) == 200
        assert build.call_count == 1


def test_manual_archive_photo_has_a_shared_editable_dialog(local_db_app: Flask) -> None:
    from web import web_interface

    with db_connection.closing_connection() as conn:
        filename = "20200101_120000_manual.jpg"
        conn.execute(
            "INSERT INTO images(filename,timestamp) VALUES (?, '20200101_120000')",
            (filename,),
        )
        conn.execute(
            """INSERT INTO manual_objects(image_filename,bbox_x,bbox_y,bbox_w,bbox_h,
            species_key,species_state,created_at,updated_at)
            VALUES (?,.1,.1,.3,.3,'Picus_viridis','identified','now','now')""",
            (filename,),
        )
        conn.commit()
    web_interface.invalidate_best_species_cache()
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        response, captured = _capture_index_context(client, "/homepage/archive")
        assert response.status_code == 200
        row = captured["context"]["best_species_modal_dets"][0]
        assert row["manual_object_id"] == 1
        assert row["object_kind"] == "manual"
        assert row["detection_id"] == -1
        html = client.get("/homepage/archive").get_data(as_text=True)
        assert 'id="modal-stream-best--1"' in html
        assert 'data-bs-target="#modal-stream-best--1"' in html
