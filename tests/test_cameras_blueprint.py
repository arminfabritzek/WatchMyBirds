from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.cameras import cameras_bp, init_cameras_bp

    init_cameras_bp(detection_manager=MagicMock())
    app.register_blueprint(auth_bp)
    app.register_blueprint(cameras_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_cameras_list(client):
    with patch(
        "web.blueprints.cameras.onvif_service.get_saved_cameras",
        return_value=[{"id": 1, "ip": "10.0.0.9"}],
    ):
        resp = client.get("/api/cameras")
    assert resp.status_code == 200
    assert resp.get_json()["cameras"] == [{"id": 1, "ip": "10.0.0.9"}]


def test_cameras_add_requires_ip(client):
    resp = client.post("/api/cameras", json={})
    assert resp.status_code == 400
    assert resp.get_json()["status"] == "error"


def test_cameras_add_success(client):
    with patch(
        "web.blueprints.cameras.onvif_service.save_camera",
        return_value={"id": 7, "ip": "10.0.0.9"},
    ):
        resp = client.post("/api/cameras", json={"ip": "10.0.0.9", "port": 80})
    assert resp.status_code == 200
    assert resp.get_json()["camera"]["id"] == 7


def test_cameras_delete_not_found(client):
    with patch(
        "web.blueprints.cameras.onvif_service.delete_camera", return_value=False
    ):
        resp = client.delete("/api/cameras/99")
    assert resp.status_code == 404


def test_cameras_update_success(client):
    with patch(
        "web.blueprints.cameras.onvif_service.update_camera", return_value=True
    ):
        resp = client.put("/api/cameras/1", json={"name": "Front"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_cameras_test_connection_success(client):
    with patch(
        "web.blueprints.cameras.onvif_service.get_camera",
        return_value={"ip": "10.0.0.9", "username": "u", "password": "p"},
    ), patch(
        "web.blueprints.cameras.onvif_service.get_device_info",
        return_value={"manufacturer": "Acme", "model": "X", "has_ptz": True},
    ), patch(
        "web.blueprints.cameras.onvif_service.update_test_result"
    ):
        resp = client.post("/api/cameras/1/test")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["details"]["has_ptz"] is True


def test_cameras_use_rejects_missing_credentials(client):
    with patch(
        "web.blueprints.cameras.onvif_service.get_camera",
        return_value={"ip": "10.0.0.9", "name": "Cam"},
    ):
        resp = client.post("/api/cameras/1/use")
    assert resp.status_code == 400
    assert "credentials" in resp.get_json()["message"].lower()


def test_cameras_require_authentication():
    app = Flask(__name__)
    app.secret_key = "k"
    from web.blueprints.auth import auth_bp
    from web.blueprints.cameras import cameras_bp, init_cameras_bp

    init_cameras_bp(detection_manager=MagicMock())
    app.register_blueprint(auth_bp)
    app.register_blueprint(cameras_bp)
    with app.test_client() as c:
        resp = c.get("/api/cameras")
    assert resp.status_code in (302, 401)
