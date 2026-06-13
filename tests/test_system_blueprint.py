from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def detection_manager():
    dm = MagicMock()
    dm.paused = False
    dm.decision_state_counts = {}
    dm.is_deep_scan_active.return_value = False
    return dm


@pytest.fixture
def app(detection_manager):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.system import init_system_bp, system_bp

    init_system_bp(detection_manager=detection_manager)
    app.register_blueprint(auth_bp)
    app.register_blueprint(system_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_healthz_is_public(app):
    with app.test_client() as c:
        resp = c.get("/healthz")
    assert resp.status_code == 200
    assert resp.data == b"ok\n"


def test_status_reports_detection_state(client):
    with patch(
        "web.blueprints.system.backup_restore_service.is_restart_required",
        return_value=False,
    ):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["detection_paused"] is False
    assert body["detection_running"] is True


def test_detection_pause(client, detection_manager):
    resp = client.post("/api/detection/pause")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"
    assert detection_manager.paused is True


def test_detection_resume_blocked_during_restore(client):
    with patch(
        "web.blueprints.system.backup_restore_service.is_restore_active",
        return_value=True,
    ):
        resp = client.post("/api/detection/resume")
    assert resp.status_code == 409


def test_system_stats_shape(client):
    resp = client.get("/api/system/stats")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert "cpu" in body and "ram" in body


def test_system_versions(client):
    resp = client.get("/api/system/versions")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "app_version" in body
    assert "kernel" in body


def test_shutdown_when_power_unavailable(client):
    with patch(
        "web.blueprints.system.is_power_management_available", return_value=False
    ):
        resp = client.post("/api/system/shutdown")
    assert resp.status_code == 400


def test_status_requires_authentication(app):
    with app.test_client() as c:
        resp = c.get("/api/status")
    assert resp.status_code in (302, 401)
