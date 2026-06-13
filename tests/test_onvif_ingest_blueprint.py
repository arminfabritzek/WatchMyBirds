from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.onvif_ingest import init_onvif_ingest_bp, onvif_ingest_bp

    init_onvif_ingest_bp(detection_manager=MagicMock())
    app.register_blueprint(auth_bp)
    app.register_blueprint(onvif_ingest_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_onvif_discover_returns_camera_list(client):
    with patch(
        "web.blueprints.onvif_ingest.onvif_service.discover_cameras",
        return_value=[{"ip": "10.0.0.5"}],
    ):
        resp = client.get("/api/onvif/discover")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["cameras"] == [{"ip": "10.0.0.5"}]


def test_onvif_get_stream_uri_requires_ip(client):
    resp = client.post("/api/onvif/get_stream_uri", json={})
    assert resp.status_code == 400
    assert resp.get_json()["status"] == "error"


def test_onvif_get_stream_uri_success(client):
    with patch(
        "web.blueprints.onvif_ingest.onvif_service.get_stream_uri",
        return_value="rtsp://10.0.0.5/stream",
    ):
        resp = client.post(
            "/api/onvif/get_stream_uri",
            json={"ip": "10.0.0.5", "port": 554},
        )
    assert resp.status_code == 200
    assert resp.get_json()["uri"] == "rtsp://10.0.0.5/stream"


def test_ingest_start_launches_user_ingest(client):
    fake_dm = MagicMock()
    with (
        patch("web.blueprints.onvif_ingest._detection_manager", fake_dm),
        patch("os.path.exists", return_value=False),
    ):
        resp = client.post("/api/ingest/start")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_routes_require_authentication():
    app = Flask(__name__)
    app.secret_key = "k"
    from web.blueprints.auth import auth_bp
    from web.blueprints.onvif_ingest import init_onvif_ingest_bp, onvif_ingest_bp

    init_onvif_ingest_bp(detection_manager=MagicMock())
    app.register_blueprint(auth_bp)
    app.register_blueprint(onvif_ingest_bp)
    with app.test_client() as c:
        resp = c.get("/api/onvif/discover")
    # login_required redirects unauthenticated requests
    assert resp.status_code in (302, 401)
