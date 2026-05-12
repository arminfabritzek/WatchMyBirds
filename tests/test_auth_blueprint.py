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


def test_setup_password_without_telemetry_checkbox_keeps_telemetry_off(
    client, tmp_path
):
    """First-run setup with NO telemetry_optin field = telemetry stays
    off. No UUID generated, no settings.yaml write to telemetry keys.

    This is the most important behavioral guarantee of the consent
    flow: a user who only sees the password setup and never ticks the
    box must end up in the same state as a user who never visited
    Settings — telemetry off.
    """
    from utils.settings import load_settings_yaml, save_settings_yaml

    # Pre-create empty settings.yaml in tmp dir.
    save_settings_yaml({}, str(tmp_path))

    with (
        patch(
            "web.blueprints.auth.auth_service.should_require_password_setup",
            return_value=True,
        ),
        patch(
            "web.blueprints.auth.settings_service.update_settings",
            return_value=(True, []),
        ),
        patch(
            "config.get_config",
            return_value={"OUTPUT_DIR": str(tmp_path), "telemetry_enabled": False},
        ),
    ):
        response = client.post(
            "/setup/password",
            data={
                "password": "birdhouse123",
                "password_confirm": "birdhouse123",
                "next": "/settings",
                # NO telemetry_optin field — simulates user not ticking
            },
            follow_redirects=False,
        )

    assert response.status_code == 302

    yaml_after = load_settings_yaml(str(tmp_path))
    assert yaml_after.get("telemetry_enabled") is not True, (
        "telemetry_enabled must NOT be true when checkbox unchecked"
    )
    assert "telemetry_installation_id" not in yaml_after or not yaml_after.get(
        "telemetry_installation_id"
    ), "No UUID should be generated when checkbox unchecked"


def test_setup_password_with_telemetry_checkbox_enables_and_wakes(client, tmp_path):
    """First-run setup WITH telemetry_optin=1 = telemetry on, UUID
    generated, scheduler wake-up triggered (first heartbeat fires
    within ~10ms instead of waiting for the next 5min tick).
    """
    from utils.settings import load_settings_yaml, save_settings_yaml

    save_settings_yaml({}, str(tmp_path))

    fake_cfg = {"OUTPUT_DIR": str(tmp_path), "telemetry_enabled": False}

    with (
        patch(
            "web.blueprints.auth.auth_service.should_require_password_setup",
            return_value=True,
        ),
        patch(
            "web.blueprints.auth.settings_service.update_settings",
            return_value=(True, []),
        ),
        patch("config.get_config", return_value=fake_cfg),
        patch("web.services.telemetry_service.wake_now") as mock_wake,
    ):
        response = client.post(
            "/setup/password",
            data={
                "password": "birdhouse123",
                "password_confirm": "birdhouse123",
                "next": "/settings",
                "telemetry_optin": "1",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302

    yaml_after = load_settings_yaml(str(tmp_path))
    assert yaml_after.get("telemetry_enabled") is True, (
        "telemetry_enabled must be true when checkbox checked"
    )
    uuid = yaml_after.get("telemetry_installation_id", "")
    assert len(uuid) == 32, "UUID must be generated on opt-in"
    assert all(c in "0123456789abcdef" for c in uuid)

    mock_wake.assert_called_once()


# ---------------------------------------------------------------------------
# get_redirect_target — open-redirect guard
# ---------------------------------------------------------------------------


class TestGetRedirectTarget:
    """Verify that ?next= cannot be used as an open-redirect vector."""

    def _call(self, next_param):
        from web.services.auth_service import get_redirect_target

        return get_redirect_target(next_param, default="/gallery")

    def test_none_falls_back_to_default(self):
        assert self._call(None) == "/gallery"

    def test_empty_falls_back_to_default(self):
        assert self._call("") == "/gallery"

    def test_relative_path_is_accepted(self):
        assert self._call("/review") == "/review"

    def test_relative_path_with_query_is_accepted(self):
        assert self._call("/review?page=2") == "/review?page=2"

    def test_protocol_relative_url_is_rejected(self):
        # //evil.com/foo would let an attacker redirect off-origin
        assert self._call("//evil.com/foo") == "/gallery"

    def test_absolute_url_is_rejected(self):
        assert self._call("https://evil.com/foo") == "/gallery"

    def test_javascript_scheme_is_rejected(self):
        assert self._call("javascript:alert(1)") == "/gallery"

    def test_backslash_normalisation_is_rejected(self):
        # Browsers normalise ``\\`` to ``/``, so /\\evil.com/x would
        # become //evil.com/x in the address bar. Defense-in-depth.
        assert self._call(r"/\evil.com/x") == "/gallery"

    def test_path_without_leading_slash_is_rejected(self):
        # A bare "review" would be interpreted as relative to whatever
        # the current page is; safer to send the user back to default.
        assert self._call("review") == "/gallery"
