"""Flask API tests for /api/v1/companion/* endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask


@pytest.fixture
def companion_service_double():
    svc = MagicMock()
    svc.state.return_value = {
        "enabled": True,
        "configured": True,
        "model_id": "wmb-companion:1b-q4",
        "pause_detection": True,
        "timeout_s": 60.0,
        "busy": False,
        "lease_holder": None,
        "last_utterance_ts": None,
    }
    svc.recent.return_value = [
        {"trigger_id": "utt_x", "ts": "2026-05-10T10:00:00", "text": "hi"},
    ]
    svc.chat.return_value = {
        "ok": True,
        "status": "ok",
        "text": "Speaker A chirps.",
        "model_id": "wmb-companion:1b-q4",
        "elapsed_ms": 12,
        "trigger_id": "utt_chat_1",
    }
    svc.event.return_value = {
        "ok": True,
        "status": "ok",
        "text": "Speaker B notices it.",
        "model_id": "wmb-companion:1b-q4",
        "elapsed_ms": 12,
        "trigger_id": "utt_event_1",
    }
    svc.feedback.return_value = {
        "trigger_id": "utt_chat_1",
        "feedback": "up",
    }
    return svc


@pytest.fixture
def app(companion_service_double):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    app.register_blueprint(auth_bp)

    from web.blueprints.companion import companion_bp, init_companion_bp
    init_companion_bp(companion_service_double)
    app.register_blueprint(companion_bp)

    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


@pytest.fixture
def unauth_client(app):
    with app.test_client() as client:
        yield client


def test_state_returns_service_payload(client, companion_service_double):
    res = client.get("/api/v1/companion/state")
    assert res.status_code == 200
    body = res.get_json()
    assert body["enabled"] is True
    assert body["model_id"] == "wmb-companion:1b-q4"
    companion_service_double.state.assert_called_once()


def test_state_requires_auth(unauth_client):
    res = unauth_client.get("/api/v1/companion/state")
    # login_required redirects unauth requests.
    assert res.status_code in (302, 401)


def test_recent_with_default_limit(client, companion_service_double):
    res = client.get("/api/v1/companion/recent")
    assert res.status_code == 200
    body = res.get_json()
    assert body["entries"][0]["trigger_id"] == "utt_x"
    companion_service_double.recent.assert_called_once_with(limit=50)


def test_recent_with_custom_limit(client, companion_service_double):
    res = client.get("/api/v1/companion/recent?limit=25")
    assert res.status_code == 200
    companion_service_double.recent.assert_called_once_with(limit=25)


def test_recent_rejects_out_of_range_limit(client):
    assert client.get("/api/v1/companion/recent?limit=0").status_code == 400
    assert client.get("/api/v1/companion/recent?limit=999").status_code == 400


def test_chat_happy_path(client, companion_service_double):
    res = client.post(
        "/api/v1/companion/chat",
        json={"message": "Hallo", "language": "de", "tone": "kid_friendly"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["text"] == "Speaker A chirps."
    call = companion_service_double.chat.call_args
    assert call.kwargs["message"] == "Hallo"
    assert call.kwargs["language"] == "de"
    assert call.kwargs["tone"] == "kid_friendly"


def test_chat_missing_message_is_400(client):
    res = client.post("/api/v1/companion/chat", json={})
    assert res.status_code == 400


def test_chat_unreachable_is_503(client, companion_service_double):
    companion_service_double.chat.return_value = {
        "ok": False,
        "status": "unreachable",
        "reason": "transport: URLError",
    }
    res = client.post("/api/v1/companion/chat", json={"message": "x"})
    assert res.status_code == 503
    body = res.get_json()
    assert body["error"] == "service_unavailable"
    assert body["status"] == "unreachable"


def test_chat_busy_is_409(client, companion_service_double):
    companion_service_double.chat.return_value = {
        "ok": False,
        "status": "busy",
        "reason": "compute lease held by aesthetic_tagger",
    }
    res = client.post("/api/v1/companion/chat", json={"message": "x"})
    assert res.status_code == 409


def test_chat_unauth_is_redirected(unauth_client):
    res = unauth_client.post("/api/v1/companion/chat", json={"message": "x"})
    assert res.status_code in (302, 401)


def test_event_happy_path(client, companion_service_double):
    res = client.post(
        "/api/v1/companion/event",
        json={"species": "kohlmeise", "count": 2, "rare": False},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True


def test_event_requires_species(client):
    res = client.post("/api/v1/companion/event", json={"count": 1})
    assert res.status_code == 400


def test_feedback_happy_path(client, companion_service_double):
    res = client.post(
        "/api/v1/companion/feedback",
        json={"trigger_id": "utt_chat_1", "vote": "up"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    companion_service_double.feedback.assert_called_once_with(
        trigger_id="utt_chat_1", vote="up"
    )


def test_feedback_unknown_id_is_404(client, companion_service_double):
    companion_service_double.feedback.return_value = None
    res = client.post(
        "/api/v1/companion/feedback",
        json={"trigger_id": "no_such", "vote": "up"},
    )
    assert res.status_code == 404


def test_feedback_invalid_vote_is_400(client):
    res = client.post(
        "/api/v1/companion/feedback",
        json={"trigger_id": "utt_x", "vote": "maybe"},
    )
    assert res.status_code == 400


def test_disabled_chat_is_503_with_disabled_marker(client, companion_service_double):
    companion_service_double.chat.return_value = {
        "ok": False,
        "status": "disabled",
        "reason": "COMPANION_ENABLED is false",
    }
    res = client.post("/api/v1/companion/chat", json={"message": "x"})
    assert res.status_code == 503
    body = res.get_json()
    assert body["status"] == "disabled"
