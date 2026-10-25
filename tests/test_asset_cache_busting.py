"""Packaged browser assets change URLs together when the build changes."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask
from flask.testing import FlaskClient

from utils import deploy_info
from web.blueprints.auth import auth_bp
from web.blueprints.pages import pages_bp

_ROOT = Path(__file__).resolve().parents[1]
_ASSET_INCLUDE = re.compile(r'(?:src|href)="(/assets/[^"\n]+\.(?:js|css)[^"\n]*)"')


def _client() -> FlaskClient:
    app = Flask(__name__, template_folder=str(_ROOT / "templates"))
    app.config["TESTING"] = True
    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)
    return app.test_client()


@pytest.fixture
def build_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    commit = tmp_path / "BUILD_COMMIT"
    commit.write_text("abcdef1234567890\n")
    monkeypatch.setitem(deploy_info._FILE_SOURCES, "git_commit", [commit])
    return commit


def _asset_urls(client: FlaskClient) -> list[str]:
    response = client.get("/privacy")
    assert response.status_code == 200
    urls = _ASSET_INCLUDE.findall(response.get_data(as_text=True))
    assert len(urls) >= 3
    return urls


def test_all_rendered_assets_use_build_commit(build_commit: Path) -> None:
    client = _client()
    urls = _asset_urls(client)
    for url in urls:
        assert parse_qs(urlsplit(url).query) == {"v": ["abcdef1"]}
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "public, max-age=604800"
    assert _asset_urls(client) == urls
    assert _asset_urls(_client()) == urls


def test_all_asset_urls_change_together_after_update(build_commit: Path) -> None:
    old_client = _client()
    old_urls = _asset_urls(old_client)
    build_commit.write_text("123456789abcdef\n")
    assert _asset_urls(old_client) == old_urls
    new_urls = _asset_urls(_client())
    assert len(new_urls) == len(old_urls)
    for old, new in zip(old_urls, new_urls, strict=True):
        assert urlsplit(old).path == urlsplit(new).path
        assert parse_qs(urlsplit(new).query) == {"v": ["1234567"]}
        assert old != new


def test_every_template_asset_include_uses_shared_version(build_commit: Path) -> None:
    app = _client().application
    includes = []
    with app.test_request_context():
        for template in (_ROOT / "templates").rglob("*.html"):
            for include in _ASSET_INCLUDE.findall(template.read_text()):
                includes.append(include)
                rendered = app.jinja_env.from_string(include).render()
                assert parse_qs(urlsplit(rendered).query) == {"v": ["abcdef1"]}, (
                    template,
                    include,
                )
                assert (_ROOT / urlsplit(rendered).path.lstrip("/")).is_file()
    assert any("station_browse.js" in include for include in includes)
    assert any("design-system.css" in include for include in includes)
    assert any("bird_editor.js" in include for include in includes)
