"""Tests for the public login and first-run password setup flow."""

from unittest.mock import patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp, login_required

    app.register_blueprint(auth_bp)

    @app.route("/protected")
    @login_required
    def protected():
        return "ok"

    @app.route("/public")
    def public():
        return "public"

    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client


def test_login_redirects_to_password_setup_when_rpi_uses_default_password(client):
    with patch(
        "web.blueprints.auth.auth_service.should_require_password_setup",
        return_value=True,
    ):
        response = client.get("/login?next=/review", follow_redirects=False)

    assert response.status_code == 302
    assert "/setup/password" in response.headers["Location"]
    assert "next=/review" in response.headers["Location"]


def test_protected_route_redirects_to_password_setup_when_required(client):
    with patch(
        "web.blueprints.auth.auth_service.should_require_password_setup",
        return_value=True,
    ):
        response = client.get("/protected", follow_redirects=False)

    assert response.status_code == 302
    assert "/setup/password" in response.headers["Location"]
    assert "next=/protected" in response.headers["Location"]


def test_setup_password_persists_password_and_authenticates_session(client):
    with (
        patch(
            "web.blueprints.auth.auth_service.should_require_password_setup",
            return_value=True,
        ),
        patch(
            "web.blueprints.auth.settings_service.update_settings",
            return_value=(True, []),
        ) as mock_update,
    ):
        response = client.post(
            "/setup/password",
            data={
                "password": "birdhouse123",
                "password_confirm": "birdhouse123",
                "next": "/settings",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/settings")
    mock_update.assert_called_once_with({"EDIT_PASSWORD": "birdhouse123"})

    with client.session_transaction() as sess:
        assert sess["authenticated"] is True
