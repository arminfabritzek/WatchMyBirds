"""The rendered editor URL follows installed content without manual snapshots."""

import hashlib
import os
import re
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from web.blueprints.auth import auth_bp
from web.blueprints.pages import pages_bp

_ROOT = Path(__file__).resolve().parents[1]


def _client() -> FlaskClient:
    app = Flask(__name__, template_folder=str(_ROOT / "templates"))
    app.config["TESTING"] = True
    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)
    return app.test_client()


def _editor_url(client: FlaskClient) -> str:
    response = client.get("/privacy")
    assert response.status_code == 200
    match = re.search(
        r'src="(/assets/js/bird_editor\.js\?v=[0-9a-f]{16})"',
        response.get_data(as_text=True),
    )
    assert match, "Missing content-versioned editor URL"
    return match.group(1)


def test_editor_cache_token_matches_served_content() -> None:
    client = _client()
    url = _editor_url(client)
    response = client.get(url)
    assert response.status_code == 200
    assert url.endswith(hashlib.sha256(response.data).hexdigest()[:16])
    assert _editor_url(client) == url
    assert _editor_url(_client()) == url


def test_editor_url_changes_after_content_update_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("web.blueprints.pages._ASSETS_FOLDER", str(tmp_path))
    editor = tmp_path / "js/bird_editor.js"
    editor.parent.mkdir()
    editor.write_bytes(b"old editor")
    old_stat = editor.stat()
    first_url = _editor_url(_client())
    editor.write_bytes(b"new editor")
    os.utime(editor, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    client = _client()
    second_url = _editor_url(client)
    assert first_url != second_url
    assert client.get(second_url).data == b"new editor"
    assert second_url.endswith(hashlib.sha256(b"new editor").hexdigest()[:16])


def test_missing_editor_fails_at_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("web.blueprints.pages._ASSETS_FOLDER", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        _client()
