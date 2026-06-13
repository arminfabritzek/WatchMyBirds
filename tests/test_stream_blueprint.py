from unittest.mock import MagicMock

import numpy as np
import pytest
from flask import Flask


@pytest.fixture
def detection_manager():
    return MagicMock()


@pytest.fixture
def app(detection_manager):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.stream import init_stream_bp, stream_bp

    init_stream_bp(detection_manager=detection_manager)
    app.register_blueprint(stream_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client


def test_snapshot_returns_503_when_no_frame(client, detection_manager):
    detection_manager.get_display_frame.return_value = None
    resp = client.get("/api/snapshot")
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "No frame available"


def test_snapshot_returns_jpeg_when_frame_available(client, detection_manager):
    detection_manager.get_display_frame.return_value = np.zeros(
        (48, 64, 3), dtype=np.uint8
    )
    resp = client.get("/api/snapshot")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.headers["Content-Disposition"].startswith("attachment; filename=snapshot_")
    assert resp.data[:2] == b"\xff\xd8"  # JPEG SOI marker
