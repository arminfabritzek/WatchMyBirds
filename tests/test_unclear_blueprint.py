"""
Tests for the Unclear blueprint HTTP routes.

Covers the request/response contract — payload validation, auth gates,
and the wiring between the blueprint and db_service. The SQL semantics
themselves are tested in tests/test_unclear_sql.py.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.unclear import unclear_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(unclear_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


@pytest.fixture
def unauth_client(app):
    return app.test_client()


# -- confirm-day -------------------------------------------------------------


def test_confirm_day_requires_day_parameter(client):
    response = client.post("/api/unclear/confirm-day", json={})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_confirm_day_empty_day_with_no_detections_returns_zero(client):
    """A day with no Unclear rows is a success with confirmed=0 — not an error."""
    fake_conn = MagicMock()
    fake_conn.close.return_value = None

    with (
        patch("web.blueprints.unclear.db_service.get_connection", return_value=fake_conn),
        patch(
            "web.blueprints.unclear.db_service.fetch_unclear_detection_ids_for_day",
            return_value=[],
        ),
        patch("web.blueprints.unclear.db_service.confirm_unclear_detections") as confirm_fn,
    ):
        response = client.post("/api/unclear/confirm-day", json={"day": "2026-05-19"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["confirmed"] == 0
    # Don't even bother calling confirm with an empty list
    confirm_fn.assert_not_called()


def test_confirm_day_promotes_ids_and_invalidates_cache(client):
    fake_conn = MagicMock()
    fake_conn.close.return_value = None

    with (
        patch("web.blueprints.unclear.db_service.get_connection", return_value=fake_conn),
        patch(
            "web.blueprints.unclear.db_service.fetch_unclear_detection_ids_for_day",
            return_value=[100, 101, 102],
        ),
        patch(
            "web.blueprints.unclear.db_service.confirm_unclear_detections",
            return_value=3,
        ) as confirm_fn,
        patch("web.blueprints.unclear.gallery_service.invalidate_cache") as cache_fn,
    ):
        response = client.post("/api/unclear/confirm-day", json={"day": "2026-05-19"})

    assert response.status_code == 200
    data = response.get_json()
    assert data == {
        "status": "success",
        "confirmed": 3,
        "day": "2026-05-19",
    }
    confirm_fn.assert_called_once_with(
        fake_conn, [100, 101, 102], source="manual_bulk_confirm_day"
    )
    cache_fn.assert_called_once()


# -- discard-day -------------------------------------------------------------


def test_discard_day_requires_day_parameter(client):
    response = client.post("/api/unclear/discard-day", json={})
    assert response.status_code == 400


def test_discard_day_rejects_via_db_service(client):
    fake_conn = MagicMock()
    fake_conn.close.return_value = None

    with (
        patch("web.blueprints.unclear.db_service.get_connection", return_value=fake_conn),
        patch(
            "web.blueprints.unclear.db_service.fetch_unclear_detection_ids_for_day",
            return_value=[200, 201],
        ),
        patch("web.blueprints.unclear.db_service.reject_detections") as reject_fn,
        patch("web.blueprints.unclear.gallery_service.invalidate_cache"),
    ):
        response = client.post("/api/unclear/discard-day", json={"day": "2026-05-19"})

    assert response.status_code == 200
    data = response.get_json()
    assert data == {
        "status": "success",
        "discarded": 2,
        "day": "2026-05-19",
    }
    reject_fn.assert_called_once_with(fake_conn, [200, 201])


# -- auth --------------------------------------------------------------------


def test_unauth_confirm_day_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/unclear/confirm-day", json={"day": "2026-05-19"}
    )
    # login_required either redirects (302) or 401s — accept both
    assert response.status_code in (302, 401, 403)


def test_unauth_discard_day_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/unclear/discard-day", json={"day": "2026-05-19"}
    )
    assert response.status_code in (302, 401, 403)
