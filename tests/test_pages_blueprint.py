from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from flask import Flask
from flask.testing import FlaskClient


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


def test_removed_presentation_pages_return_not_found(client):
    assert client.get("/tbwd").status_code == 404
    assert client.get("/tbwd-vision").status_code == 404


def test_privacy_renders(client) -> None:
    response = client.get("/privacy")
    assert response.status_code == 200
    content = " ".join(response.get_data(as_text=True).split())
    assert "Training-data exports" in content
    assert "WatchMyBirds does not upload this bundle" in content
    assert "Sharing requires your own manual transfer." in content
    assert "entirely optional" in content


def test_settings_requires_auth(app):
    with app.test_client() as c:
        resp = c.get("/settings")
    assert resp.status_code in (302, 401)


def test_assets_served_with_public_cache_header(client):
    resp = client.get("/assets/design-system.css")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "public, max-age=604800"


def test_page_dependencies_are_available_without_internet(client: FlaskClient) -> None:
    dependencies = []

    class Resources(HTMLParser):
        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            values = dict(attrs)
            if tag == "script" and values.get("src"):
                dependencies.append(values["src"])
            if tag == "link" and values.get("rel") in {
                "stylesheet",
                "preconnect",
                "dns-prefetch",
                "preload",
                "modulepreload",
            }:
                dependencies.append(values["href"])

    response = client.get("/privacy")
    assert response.status_code == 200
    Resources().feed(response.get_data(as_text=True))
    assert dependencies
    for dependency in dependencies:
        parsed = urlsplit(dependency)
        assert not parsed.scheme and not parsed.netloc, dependency
        asset = client.get(dependency)
        assert asset.status_code == 200, dependency
        if parsed.path.endswith(".css"):
            assert asset.mimetype == "text/css"
        elif parsed.path.endswith(".js"):
            assert asset.mimetype in {"text/javascript", "application/javascript"}


def test_admin_profile_rejects_non_whitelisted_route(client):
    resp = client.get("/admin/_profile?route=/secret")
    assert resp.status_code == 400
