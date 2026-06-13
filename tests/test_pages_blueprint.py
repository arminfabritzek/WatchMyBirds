from pathlib import Path

import pytest
from flask import Flask


@pytest.fixture
def app():
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
    )
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.pages import pages_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_tbwd_pages_render(client):
    assert client.get("/tbwd").status_code == 200
    assert client.get("/tbwd-vision").status_code == 200


def test_privacy_renders(client):
    assert client.get("/privacy").status_code == 200


def test_settings_requires_auth(app):
    with app.test_client() as c:
        resp = c.get("/settings")
    assert resp.status_code in (302, 401)


def test_assets_served_with_public_cache_header(client):
    resp = client.get("/assets/design-system.css")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "public, max-age=604800"


def test_admin_profile_rejects_non_whitelisted_route(client):
    resp = client.get("/admin/_profile?route=/secret")
    assert resp.status_code == 400
