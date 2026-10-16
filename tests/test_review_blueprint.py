"""Tests for review blueprint decision actions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from web.blueprints.review import (
    _build_review_event_member,
)


def _sql_result(*, fetchone=None, fetchall=None):
    result = MagicMock()
    if fetchone is not None:
        result.fetchone.return_value = fetchone
    if fetchall is not None:
        result.fetchall.return_value = fetchall
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def app():
    project_root = _project_root()
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "assets"),
    )
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.review import review_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(review_bp)
    return app


@pytest.fixture
def client(app):
    with (
        patch(
            "web.blueprints.review.human_label_service.record_answer",
            return_value=[1],
        ) as label_recorder,
        app.test_client() as client,
    ):
        app.extensions["human_label_recorder"] = label_recorder
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def _label_recorder(client):
    return client.application.extensions["human_label_recorder"]


def test_review_decision_accepts_trash_alias(client):
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"detection_id": 17, "image_filename": "review-item.jpg"}
    ]

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.get_connection.return_value = mock_conn
        mock_db.update_review_status.return_value = 1

        response = client.post(
            "/api/review/decision",
            json={"filenames": ["review-item.jpg"], "action": "trash"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["action"] == "trash"
    assert data["review_status"] is None
    mock_db.reject_detections.assert_called_once_with(mock_conn, [17])
    _label_recorder(client).assert_not_called()
    mock_conn.close.assert_called_once()


def test_review_approve_requires_manual_species_and_bbox(client):
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = {
        "manual_bbox_review": None,
        "manual_species_override": None,
        "species_source": None,
    }

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={"filename": "review-item.jpg", "detection_id": 17},
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "species selection is required" in data["message"]


def test_build_review_event_member_labels_context_frame_as_in_gallery():
    row = {
        "timestamp": "20260408_155825",
        "manual_species_override": "Columba_palumbus",
        "cls_class_name": "Pica_pica",
        "context_only": True,
    }

    with patch("web.blueprints.review._build_review_item") as mock_build_item:
        mock_build_item.return_value = {
            "review_reason": "low_score",
            "reason_label": "Low Score (87%)",
        }

        member = _build_review_event_member(
            row,
            conn=None,
            species_locale="de",
            output_dir="/tmp",
            common_names={"Columba_palumbus": "Ringeltaube"},
            recent_species=[],
        )

    assert member["context_only"] is True
    assert member["candidate_species"] == "Columba_palumbus"
    assert member["candidate_species_common"] == "Ringeltaube"
    assert member["review_reason"] == "context"
    assert member["reason_label"] == "In Gallery"


def test_review_approve_requires_bbox_after_species_selection(client):
    mock_conn = MagicMock()

    with patch("web.blueprints.review.db_service") as mock_db:
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "Bounding box review is required" in data["message"]


def test_review_approve_confirms_after_manual_species_and_bbox(client):
    mock_conn = MagicMock()
    fetchone_results = iter(
        [
            {
                "manual_bbox_review": "correct",
                "manual_species_override": "Parus_major",
                "species_source": "manual",
            },
            None,
            [1],
            [0],
        ]
    )
    mock_conn.execute.return_value.fetchone.side_effect = lambda: next(fetchone_results)

    with (
        patch("core.review_state.update_review_status") as mock_visibility,
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch(
            "web.blueprints.review.gallery_service.invalidate_cache"
        ) as mock_invalidate,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status"] == "confirmed_bird"
    assert data["gallery_visible"] is True
    assert "now visible in the gallery" in data["message"]
    mock_visibility.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "confirmed_bird", commit=False
    )
    answer = _label_recorder(client).call_args.args[1]
    assert answer.object_bird_presence == "present"
    assert answer.bbox_quality == "suitable"
    assert answer.species_key == "Parus_major"
    mock_invalidate.assert_called_once()


def test_review_approve_keeps_image_untagged_when_unresolved_siblings_remain(client):
    mock_conn = MagicMock()
    fetchone_results = iter(
        [
            {
                "manual_bbox_review": "correct",
                "manual_species_override": "Parus_major",
                "species_source": "manual",
            },
            None,
            [3],
            [2],
        ]
    )
    mock_conn.execute.return_value.fetchone.side_effect = lambda: next(fetchone_results)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/approve",
            json={
                "filename": "review-item.jpg",
                "detection_id": 17,
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status"] == "untagged"
    assert data["gallery_visible"] is False
    assert "remains out of the gallery" in data["message"]
    mock_db.update_review_status.assert_not_called()
    assert mock_conn.execute.call_count >= 3


def test_quick_species_entries_include_server_resolved_thumb_urls():
    from web.blueprints.review import _build_review_quick_species

    picker_entries = [
        {
            "scientific": "Sitta_europaea",
            "common": "Kleiber",
            "source": "prediction",
            "score": 0.73,
        }
    ]

    with patch("web.blueprints.review.resolve_species_thumbnail_url") as mock_thumb:
        mock_thumb.return_value = (
            "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
        )
        quick_species = _build_review_quick_species(
            "Sitta_europaea",
            picker_entries,
            [],
            {"Sitta_europaea": "Kleiber"},
            species_thumbnail_map={
                "Sitta_europaea": "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
            },
            thumbnail_cache_key="review:DE",
        )

    assert quick_species[0]["scientific"] == "Sitta_europaea"
    assert (
        quick_species[0]["thumb_url"]
        == "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
    )


def test_quick_species_filters_non_bird_from_every_source():
    """Regression: ``non_bird`` must not appear as a clickable card
    regardless of which source surfaces it (cls top-k, current species
    of a previously mis-relabeled detection, or recent-days
    aggregate).

    Before this fix, all three sources could land ``non_bird`` in the
    picker. Clicking it triggered ``POST /api/review/quick-species`` →
    HTTP 400 ``"unknown species"`` because the validator correctly
    rejected the synthetic class.
    """
    from web.blueprints.review import _build_review_quick_species

    picker_entries = [
        {
            "scientific": "Parus_major",
            "common": "Kohlmeise",
            "source": "prediction",
            "score": 0.82,
        },
        {
            "scientific": "non_bird",
            "common": "non bird",
            "source": "prediction",
            "score": 0.08,
        },
    ]
    recent_species = [
        {"scientific": "non_bird", "common": "non bird"},
        {"scientific": "Cyanistes_caeruleus", "common": "Blaumeise"},
    ]

    def fake_is_known(key, locale="DE"):
        return key in {"Parus_major", "Cyanistes_caeruleus"}

    with (
        patch("web.blueprints.review.resolve_species_thumbnail_url", return_value=""),
        patch("web.blueprints.review.is_known_species", side_effect=fake_is_known),
    ):
        quick_species = _build_review_quick_species(
            current_species="non_bird",
            picker_entries=picker_entries,
            recent_species=recent_species,
            common_names={
                "Parus_major": "Kohlmeise",
                "Cyanistes_caeruleus": "Blaumeise",
            },
        )

    scientifics = [entry["scientific"] for entry in quick_species]
    assert "non_bird" not in scientifics
    assert "Parus_major" in scientifics
    assert "Cyanistes_caeruleus" in scientifics


def test_load_recent_review_species_filters_non_bird():
    """The ``fetch_recent_review_species`` aggregate sums
    ``effective_species_sql`` which coalesces ``manual_species_override``
    first. A single prior accidental relabel to ``non_bird`` would
    otherwise pollute the "Recent days" pill for 7 days.
    """
    from web.blueprints.review import _load_recent_review_species

    fake_rows = [
        {"species_key": "Parus_major", "hit_count": 12, "last_seen": "20260522_120000"},
        {"species_key": "non_bird", "hit_count": 1, "last_seen": "20260522_210000"},
        {
            "species_key": "Cyanistes_caeruleus",
            "hit_count": 5,
            "last_seen": "20260522_110000",
        },
    ]

    def fake_is_known(key, locale="DE"):
        return key in {"Parus_major", "Cyanistes_caeruleus"}

    with (
        patch(
            "web.blueprints.review.db_service.fetch_recent_review_species",
            return_value=fake_rows,
        ),
        patch("web.blueprints.review.is_known_species", side_effect=fake_is_known),
    ):
        recent = _load_recent_review_species(
            conn=MagicMock(),
            common_names={
                "Parus_major": "Kohlmeise",
                "Cyanistes_caeruleus": "Blaumeise",
            },
        )

    assert [r["scientific"] for r in recent] == ["Parus_major", "Cyanistes_caeruleus"]


def test_review_event_approve_confirms_event_and_recomputes_gallery_visibility(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                        "review_status": "untagged",
                    }
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("core.review_state.update_review_status") as mock_visibility,
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch(
            "web.blueprints.review.gallery_service.invalidate_cache"
        ) as mock_invalidate,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["gallery_visible_filenames"] == ["review-item.jpg"]
    answer = _label_recorder(client).call_args.args[1]
    assert answer.detection_id == 17
    assert answer.species_key == "Parus_major"
    assert answer.bbox_quality == "suitable"
    mock_visibility.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "confirmed_bird", commit=False
    )
    mock_invalidate.assert_called_once()


def test_review_event_approve_returns_conflict_when_event_disappears(client):
    mock_conn = MagicMock()

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = None
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "no longer exists" in data["message"]


def test_review_event_approve_keeps_images_hidden_when_open_detections_remain(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                        "review_status": "untagged",
                    }
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[2])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[3])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["gallery_visible_filenames"] == []
    assert data["review_status_by_filename"]["review-item.jpg"] == "untagged"
    mock_db.update_review_status.assert_not_called()


def test_review_event_approve_ignores_gallery_anchors_for_event_payload(client):
    """Anchored events may show Gallery frames, but approval only touches review frames."""
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            assert params == [17]
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                        "review_status": "untagged",
                    }
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17, 42],
            "candidate_species": "Parus_major",
            "eligibility": "event_eligible",
            "members": [
                {"best_detection_id": 17, "context_only": False},
                {"best_detection_id": 42, "context_only": True},
            ],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17, 42],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["detection_ids"] == [17]
    answer = _label_recorder(client).call_args.args[1]
    assert answer.detection_id == 17
    assert answer.species_key == "Parus_major"
    assert answer.bbox_quality == "suitable"


def test_review_event_approve_allows_user_selected_species_override(client):
    """Direct event-surface species correction must survive approval.

    The event payload may still carry the old candidate species from the
    server-side grouping pass, but the operator is allowed to choose a
    different valid species locally before clicking `Approve Event`.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                        "review_status": "untagged",
                    }
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Parus_major"}
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
            "candidate_species": "Cyanistes_caeruleus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
                "species": "Parus_major",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert _label_recorder(client).call_args.args[1].species_key == "Parus_major"


def test_review_event_approve_preserves_per_frame_manual_species(client):
    """Per-frame manual relabels must survive an Approve Event stamp.

    Reproduces the Multi-Select bug: the operator relabels a subset of
    frames (e.g. 3 of 7) to species A via per-cell relabel, then picks
    species B as the event-level species and clicks Approve Event. The
    frames already stamped with A must keep A — only the still-automatic
    frames get B. BBox review is still applied to every frame since it
    is an event-level decision.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "pigeon-frame-1.jpg",
                        "review_status": "untagged",
                        "manual_species_override": "",
                        "species_source": "cls",
                    },
                    {
                        "detection_id": 18,
                        "image_filename": "pigeon-frame-2.jpg",
                        "review_status": "untagged",
                        "manual_species_override": "",
                        "species_source": "cls",
                    },
                    {
                        "detection_id": 19,
                        "image_filename": "cat-frame-1.jpg",
                        "review_status": "untagged",
                        "manual_species_override": "Felis_catus",
                        "species_source": "manual",
                    },
                    {
                        "detection_id": 20,
                        "image_filename": "cat-frame-2.jpg",
                        "review_status": "untagged",
                        "manual_species_override": "Felis_catus",
                        "species_source": "manual",
                    },
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Columba_livia"}
        mock_event.return_value = {
            "event_key": "mixed-event-xyz",
            "detection_ids": [17, 18, 19, 20],
            "candidate_species": "Columba_livia",
            "eligibility": "event_eligible",
            "members": [
                {"best_detection_id": 17, "context_only": False},
                {"best_detection_id": 18, "context_only": False},
                {"best_detection_id": 19, "context_only": False},
                {"best_detection_id": 20, "context_only": False},
            ],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "event_key": "mixed-event-xyz",
                "detection_ids": [17, 18, 19, 20],
                "species": "Columba_livia",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    answers = [call.args[1] for call in _label_recorder(client).call_args_list]
    assert [answer.species_key for answer in answers] == [
        "Columba_livia",
        "Columba_livia",
        "Felis_catus",
        "Felis_catus",
    ]
    assert all(answer.bbox_quality == "suitable" for answer in answers)


def test_review_event_trash_moves_images_without_active_detections_to_trash(client):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                    }
                ]
            )
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[0])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch(
            "web.blueprints.review.gallery_service.invalidate_cache"
        ) as mock_invalidate,
    ):
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-trash",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status_by_filename"]["review-item.jpg"] == "untagged"
    assert data["trash_filenames"] == []
    assert data["review_filenames"] == ["review-item.jpg"]
    assert "explicit full-image answer" in data["message"]
    answer = _label_recorder(client).call_args.args[1]
    assert answer.object_bird_presence == "absent"
    mock_db.reject_detections.assert_called_once_with(mock_conn, [17], commit=False)
    mock_db.update_review_status.assert_not_called()
    mock_invalidate.assert_called_once()


def test_review_event_trash_keeps_images_in_review_when_active_detections_remain(
    client,
):
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                    }
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[1])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[2])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_event.return_value = {
            "event_key": "bird-event-abc123",
            "detection_ids": [17],
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-trash",
            json={
                "event_key": "bird-event-abc123",
                "detection_ids": [17],
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["review_status_by_filename"]["review-item.jpg"] == "untagged"
    assert data["trash_filenames"] == []
    assert data["review_filenames"] == ["review-item.jpg"]
    answer = _label_recorder(client).call_args.args[1]
    assert answer.object_bird_presence == "absent"
    mock_db.reject_detections.assert_called_once_with(mock_conn, [17], commit=False)
    mock_db.update_review_status.assert_not_called()


# ---------------------------------------------------------------------------
# Continuity-batch approval path (no event_key)
# ---------------------------------------------------------------------------


def test_review_event_approve_batch_refuses_gallery_anchor_detection_ids(client):
    """Batch approvals must refuse anchor detections.

    A continuity batch approval POSTs ``detection_ids`` without an
    ``event_key`` because the ids span sibling events. The backend must
    refuse outright if any of those detections belong to an image
    already confirmed in the gallery (``review_status='confirmed_bird'``),
    since those are read-only context anchors.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "review-item.jpg",
                        "review_status": "untagged",
                    },
                    {
                        "detection_id": 42,
                        "image_filename": "gallery-anchor.jpg",
                        "review_status": "confirmed_bird",
                    },
                ]
            )
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
    ):
        mock_allowed.return_value = {"Pica_pica"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                # Batch path: no event_key.
                "detection_ids": [17, 42],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert data["status"] == "error"
    assert "Gallery anchors" in data["message"]
    assert data["anchor_filenames"] == ["gallery-anchor.jpg"]
    mock_db.apply_species_override_many.assert_not_called()
    mock_db.set_manual_bbox_review.assert_not_called()


def test_review_event_approve_batch_confirms_actionable_ids_without_event_key(client):
    """Submitting only review detection ids works for batch approval.

    When the operator clicks `Approve Batch` the client POSTs
    ``detection_ids`` that span multiple sibling events. The backend
    skips the strict ``event_key``/``detection_ids`` parity check and
    confirms every submitted id, as long as none of them belong to a
    Gallery anchor.
    """
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        if "SELECT d.detection_id, d.image_filename" in query:
            return _sql_result(
                fetchall=[
                    {
                        "detection_id": 17,
                        "image_filename": "spatz-frame.jpg",
                        "review_status": "untagged",
                    },
                    {
                        "detection_id": 18,
                        "image_filename": "kleiber-frame.jpg",
                        "review_status": "untagged",
                    },
                ]
            )
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica"}
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-approve",
            json={
                "detection_ids": [17, 18],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["event_key"] is None
    assert sorted(data["detection_ids"]) == [17, 18]
    answers = [call.args[1] for call in _label_recorder(client).call_args_list]
    assert [answer.detection_id for answer in answers] == [17, 18]
    assert all(answer.species_key == "Pica_pica" for answer in answers)


def _detection_dict(
    detection_id,
    timestamp,
    species,
    *,
    context_only=False,
    bbox=(0.20, 0.20, 0.18, 0.18),
):
    return {
        "detection_id": detection_id,
        "active_detection_id": detection_id,
        "filename": f"frame-{detection_id}.jpg",
        "timestamp": timestamp,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "cls_class_name": species,
        "manual_species_override": None,
        "species_key": species,
        "species_source": "classifier",
        "sibling_detection_count": 1,
        "context_only": context_only,
    }


def _make_resolve_conn(rows):
    """Build a MagicMock conn whose SELECT on detections returns `rows`."""
    mock_conn = MagicMock()

    def execute_side_effect(query, params=None):
        normalized = " ".join(query.split())
        if "SELECT d.detection_id, d.image_filename" in normalized:
            return _sql_result(fetchall=rows)
        if "COALESCE(d.decision_state" in query:
            return _sql_result(fetchone=[0])
        if "SELECT COUNT(*)" in query:
            return _sql_result(fetchone=[1])
        return _sql_result()

    mock_conn.execute.side_effect = execute_side_effect
    return mock_conn


def test_review_event_resolve_commits_keep_and_trash_in_one_call(client):
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch(
            "web.blueprints.review.gallery_service.invalidate_cache"
        ) as mock_invalidate,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["keep_detection_ids"] == [17]
    assert data["trash_detection_ids"] == [18]

    answers = [call.args[1] for call in _label_recorder(client).call_args_list]
    assert answers[0].detection_id == 17
    assert answers[0].object_bird_presence == "present"
    assert answers[0].species_key == "Columba_palumbus"
    assert len(answers) == 1
    mock_db.reject_detections.assert_called_once_with(mock_conn, [18], commit=False)
    mock_invalidate.assert_called_once()


def test_review_event_resolve_rejects_empty_keep_list(client):
    response = client.post(
        "/api/review/event-resolve",
        json={
            "event_key": "bird-event-abc",
            "keep_detection_ids": [],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
            "bbox_review": "correct",
        },
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "keep_detection_ids must not be empty" in data["message"]


def test_review_event_resolve_rejects_overlapping_keep_and_trash(client):
    response = client.post(
        "/api/review/event-resolve",
        json={
            "event_key": "bird-event-abc",
            "keep_detection_ids": [17, 18],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
            "bbox_review": "correct",
        },
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "disjoint" in data["message"]
    assert data["overlap"] == [18]


def test_review_event_resolve_rejects_partial_event_coverage(client):
    """keep+trash must cover every detection of the loaded event."""
    mock_conn = _make_resolve_conn([])

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        # Event actually has three detections; the caller only covered two.
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18, 19],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert "changed and must be reloaded" in data["message"]


def test_review_event_resolve_refuses_gallery_anchors(client):
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
        {
            "detection_id": 18,
            "image_filename": "anchor.jpg",
            "review_status": "confirmed_bird",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
    ):
        mock_allowed.return_value = {"Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Columba_palumbus",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17],
                "trash_detection_ids": [18],
                "species": "Columba_palumbus",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 409
    data = response.get_json()
    assert "Gallery anchors" in data["message"]
    assert data["anchor_filenames"] == ["anchor.jpg"]
    mock_db.reject_detections.assert_not_called()


def test_review_event_resolve_preserves_per_frame_manual_override(client):
    """Per-frame wins: a Keep frame already relabelled via bulk/relabel
    must keep its manual species override. The event-level species from
    the right control rail only stamps frames that have no override yet.
    """
    rows = [
        # Frame 17 already relabelled to Columba_palumbus (Tauben) via
        # the per-cell WmSpeciesPicker path.
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
        # Frame 18 is still untouched — should receive the event species.
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": None,
            "species_source": None,
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica", "Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Pica_pica",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17, 18],
                "trash_detection_ids": [],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    answers = [call.args[1] for call in _label_recorder(client).call_args_list]
    assert [answer.species_key for answer in answers] == [
        "Columba_palumbus",
        "Pica_pica",
    ]
    assert all(answer.bbox_quality == "suitable" for answer in answers)


def test_review_event_resolve_preserves_override_when_all_keeps_are_manual(client):
    """Edge case: every Keep frame is already hand-relabelled. The
    bulk `apply_species_override_many` call must not fire at all."""
    rows = [
        {
            "detection_id": 17,
            "image_filename": "frame-a.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
        {
            "detection_id": 18,
            "image_filename": "frame-b.jpg",
            "review_status": "untagged",
            "manual_species_override": "Columba_palumbus",
            "species_source": "manual",
        },
    ]
    mock_conn = _make_resolve_conn(rows)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review._load_single_review_event") as mock_event,
        patch("web.blueprints.review.gallery_service.invalidate_cache"),
    ):
        mock_allowed.return_value = {"Pica_pica", "Columba_palumbus"}
        mock_event.return_value = {
            "event_key": "bird-event-abc",
            "detection_ids": [17, 18],
            "candidate_species": "Pica_pica",
            "eligibility": "event_eligible",
        }
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post(
            "/api/review/event-resolve",
            json={
                "event_key": "bird-event-abc",
                "keep_detection_ids": [17, 18],
                "trash_detection_ids": [],
                "species": "Pica_pica",
                "bbox_review": "correct",
            },
        )

    assert response.status_code == 200
    mock_db.apply_species_override_many.assert_not_called()


def test_review_event_resolve_requires_species_and_bbox(client):
    # Missing species.
    r1 = client.post(
        "/api/review/event-resolve",
        json={
            "keep_detection_ids": [17],
            "trash_detection_ids": [18],
            "bbox_review": "correct",
        },
    )
    assert r1.status_code == 409
    assert "species" in r1.get_json()["message"].lower()

    # Missing bbox_review.
    r2 = client.post(
        "/api/review/event-resolve",
        json={
            "keep_detection_ids": [17],
            "trash_detection_ids": [18],
            "species": "Columba_palumbus",
        },
    )
    assert r2.status_code == 409
    assert "bounding box" in r2.get_json()["message"].lower()


def test_review_thumb_rejects_out_of_root_path(client):
    """A filename resolving outside the media roots must 404, never serve."""
    escaping = Path("/etc/passwd")

    with patch("web.blueprints.review.gallery_service") as mock_gallery:
        mock_gallery.get_image_paths.return_value = {
            "original": escaping,
            "preview": escaping,
        }
        resp = client.get("/api/review-thumb/anything.jpg")

    assert resp.status_code == 404
    mock_gallery.generate_preview_thumbnail.assert_not_called()


def test_review_thumb_serves_cached_preview_inside_root(client, tmp_path):
    """A contained preview that already exists is served without regeneration."""
    from utils.path_manager import PathManager

    pm = PathManager(str(tmp_path))
    preview = pm.thumbs_dir / "2026-01-01" / "20260101_120000_preview.webp"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"webp-bytes")
    original = pm.originals_dir / "2026-01-01" / "20260101_120000.jpg"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"jpg-bytes")

    with (
        patch("web.blueprints.review.path_service") as mock_path_service,
        patch("web.blueprints.review.gallery_service") as mock_gallery,
    ):
        mock_path_service.get_path_manager.return_value = pm
        mock_gallery.get_image_paths.return_value = {
            "original": original,
            "preview": preview,
        }
        resp = client.get("/api/review-thumb/20260101_120000.jpg")

    assert resp.status_code == 200
    mock_gallery.generate_preview_thumbnail.assert_not_called()


def test_removed_analysis_route_returns_404(client: FlaskClient) -> None:
    assert client.post("/api/review/analyze/test.jpg").status_code == 404
