"""Tests for inbox upload blueprint behavior."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask
from PIL import Image


class _FakePathManager:
    def __init__(self, pending_dir: Path):
        self._pending_dir = pending_dir

    def get_inbox_pending_dir(self) -> Path:
        return self._pending_dir


@pytest.fixture
def app(tmp_path):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.inbox import inbox_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(inbox_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def _jpeg_bytes() -> io.BytesIO:
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_inbox_upload_saves_valid_image_without_stream_seek_errors(client, tmp_path):
    pending_dir = tmp_path / "pending"
    fake_manager = _FakePathManager(pending_dir)

    with (
        patch(
            "web.blueprints.inbox.backup_restore_service.is_restore_active",
            return_value=False,
        ),
        patch(
            "web.blueprints.inbox.path_service.get_path_manager",
            return_value=fake_manager,
        ),
        patch("web.blueprints.inbox.config", {"OUTPUT_DIR": str(tmp_path)}),
    ):
        response = client.post(
            "/api/inbox",
            data={"files[]": (_jpeg_bytes(), "upload.jpg")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["uploaded"] == ["upload.jpg"]
    assert payload["errors"] == []
    assert (pending_dir / "upload.jpg").exists()
