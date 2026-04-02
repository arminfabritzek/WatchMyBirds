"""Tests for review blueprint decision actions."""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.review import review_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(review_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_review_decision_accepts_trash_alias(client):
    mock_conn = MagicMock()

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
    assert data["review_status"] == "no_bird"
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "no_bird"
    )
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
            [0],
        ]
    )
    mock_conn.execute.return_value.fetchone.side_effect = lambda: next(fetchone_results)

    with (
        patch("web.blueprints.review.db_service") as mock_db,
        patch("web.blueprints.review._get_allowed_review_species") as mock_allowed,
        patch("web.blueprints.review.gallery_service.invalidate_cache") as mock_invalidate,
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
    mock_db.update_review_status.assert_called_once_with(
        mock_conn, ["review-item.jpg"], "confirmed_bird"
    )
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
        mock_thumb.return_value = "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
        quick_species = _build_review_quick_species(
            "Sitta_europaea",
            picker_entries,
            [],
            {"Sitta_europaea": "Kleiber"},
            species_thumbnail_map={"Sitta_europaea": "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"},
            thumbnail_cache_key="review:DE",
        )

    assert quick_species[0]["scientific"] == "Sitta_europaea"
    assert quick_species[0]["thumb_url"] == "/uploads/derivatives/optimized/2026-04-02/nuthatch.webp"
