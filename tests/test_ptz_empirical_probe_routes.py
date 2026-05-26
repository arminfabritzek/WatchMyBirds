"""HTTP-route tests for the empirical PTZ probe wizard.

The state-machine itself is tested in test_ptz_empirical_probe.py; these tests verify
that the five HTTP routes correctly delegate to the service layer,
shape JSON responses the wizard expects, and gate on @login_required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# ---------------------------------------------------------------------------
# Flask test app
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp

    app.register_blueprint(auth_bp)
    from web.blueprints.api_v1 import api_v1

    api_v1.detection_manager = MagicMock()
    app.register_blueprint(api_v1)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


@pytest.fixture
def unauth_client(app):
    """Client with no session — used to verify @login_required gates."""
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------


def test_start_returns_session_payload(client):
    from web.blueprints import api_v1 as api_v1_module

    fake_payload = {
        "camera_id": 0,
        "camera_ip": "10.0.0.42",
        "current_index": 0,
        "total_steps": 8,
        "done": False,
        "aborted": False,
        "current_step": {
            "id": "c_pan_right_slow",
            "kind": "continuous",
            "description": "pan right, slow velocity, 300ms burst",
            "expectation": "...",
            "purpose": "...",
            "inputs": {"pan": 0.2, "duration_sec": 0.3},
        },
        "results": {},
        "started_at": 12345.6,
        "finished_at": 0.0,
    }
    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.start_session.return_value = fake_payload

        resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/start")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"]["camera_id"] == 0
    assert body["data"]["current_step"]["kind"] == "continuous"
    # Default invocation (no body) → service called with probe_slot=None.
    mock_service.start_session.assert_called_once_with(0, probe_slot=None)


def test_start_forwards_probe_slot_to_service(client):
    """Body {probe_slot: '20'} is parsed and forwarded as kw-arg to the service."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.start_session.return_value = {
            "camera_id": 0, "current_index": 0, "total_steps": 14,
            "done": False, "aborted": False, "current_step": {"kind": "continuous"},
            "results": {}, "started_at": 0, "finished_at": 0,
            "probe_slot": "20", "overview_preset": "Preset001", "camera_ip": "",
        }
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/start",
            json={"probe_slot": "20"},
        )

    assert resp.status_code == 200
    mock_service.start_session.assert_called_once_with(0, probe_slot="20")


def test_start_rejects_out_of_range_probe_slot(client):
    """Slot 0 / 33 / 'abc' must be rejected with 400 before reaching the service."""
    for bad in ("0", "33", "abc", "-1"):
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/start",
            json={"probe_slot": bad},
        )
        assert resp.status_code == 400, f"slot={bad!r} should 400"
        body = resp.get_json()
        assert "probe_slot" in body["message"]


def test_start_accepts_blank_probe_slot(client):
    """Empty / missing probe_slot must pass through as None (operator opted out)."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.start_session.return_value = {
            "camera_id": 0, "current_index": 0, "total_steps": 14,
            "done": False, "aborted": False, "current_step": {"kind": "continuous"},
            "results": {}, "started_at": 0, "finished_at": 0,
            "probe_slot": "", "overview_preset": "Preset001", "camera_ip": "",
        }
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/start",
            json={"probe_slot": "   "},
        )

    assert resp.status_code == 200
    mock_service.start_session.assert_called_once_with(0, probe_slot=None)


def test_start_returns_400_when_no_overview_preset(client):
    """ValueError from the service surfaces as 400 — the wizard UI
    displays the message verbatim ("set a Home preset first")."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.start_session.side_effect = ValueError(
            "No overview preset configured for this camera."
        )

        resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/start")

    assert resp.status_code == 400
    body = resp.get_json()
    assert "overview preset" in body["message"].lower()


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


def test_status_returns_payload_when_session_in_flight(client):
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.session_status.return_value = {
            "camera_id": 0,
            "current_index": 3,
            "total_steps": 8,
            "done": False,
        }

        resp = client.get("/api/v1/cameras/0/ptz/empirical-probe/status")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"]["current_index"] == 3


def test_status_returns_404_idle_when_no_session(client):
    """The wizard UI uses 404 as the signal 'no probe in flight, hide
    resume prompt'. The body shape ({status: idle}) lets the JS
    distinguish from a network 404."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.session_status.return_value = None

        resp = client.get("/api/v1/cameras/0/ptz/empirical-probe/status")

    assert resp.status_code == 404
    body = resp.get_json()
    assert body["status"] == "idle"


# ---------------------------------------------------------------------------
# /execute
# ---------------------------------------------------------------------------


def test_execute_returns_step_result(client):
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.execute_step.return_value = {
            "step_id": "c_pan_right_slow",
            "kind": "continuous",
            "inputs": {"pan": 0.5, "duration_sec": 0.3},
            "executed_at": 12345.0,
            "onvif_error": "",
            "poll_samples": [],
        }

        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/execute",
            json={"step_id": "c_pan_right_slow"},
        )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data"]["step_id"] == "c_pan_right_slow"
    assert body["data"]["onvif_error"] == ""
    mock_service.execute_step.assert_called_once_with(0, "c_pan_right_slow")


def test_execute_returns_400_when_step_id_missing(client):
    """Free-form route requires step_id in the body."""
    resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/execute")
    assert resp.status_code == 400
    body = resp.get_json()
    assert "step_id" in body["message"]


def test_execute_returns_400_when_no_session(client):
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.execute_step.side_effect = ValueError(
            "No probe session in progress"
        )

        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/execute",
            json={"step_id": "c_pan_right_slow"},
        )

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------


def test_feedback_records_step_verdict(client):
    """Free-form: feedback is addressed by step_id, no advance, no auto-finalize."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.record_feedback.return_value = {
            "step_id": "c_pan_right_slow",
            "feedback": "yes",
            "verdict_count": 1,
            "total_steps": 33,
        }

        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/feedback",
            json={
                "step_id": "c_pan_right_slow",
                "result": "yes",
                "comment": "worked great",
            },
        )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data"]["step_id"] == "c_pan_right_slow"
    assert body["data"]["feedback"] == "yes"
    assert body["data"]["verdict_count"] == 1
    mock_service.record_feedback.assert_called_once_with(
        0, step_id="c_pan_right_slow", feedback="yes", comment="worked great"
    )


def test_feedback_rejects_invalid_result(client):
    """Anything other than yes/no/skip → 400 with a clear message."""
    resp = client.post(
        "/api/v1/cameras/0/ptz/empirical-probe/feedback",
        json={"step_id": "c_pan_right_slow", "result": "maybe"},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "yes" in body["message"]
    assert "no" in body["message"]
    assert "skip" in body["message"]


def test_feedback_rejects_missing_step_id(client):
    """Free-form route requires step_id."""
    resp = client.post(
        "/api/v1/cameras/0/ptz/empirical-probe/feedback",
        json={"result": "yes"},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert "step_id" in body["message"]


def test_feedback_handles_missing_body(client):
    """No body → step_id missing → 400."""
    resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/feedback")
    assert resp.status_code == 400


def test_finalize_returns_cache_path(client):
    """Operator-triggered finalize endpoint writes the YAML and returns
    the cache path + verdict count for the success modal."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.finalize_session.return_value = {
            "cache_path": "/path/to/cam0.yaml",
            "cache_error": "",
            "verdict_count": 33,
            "total_steps": 33,
        }
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/finalize"
        )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data"]["cache_path"].endswith("cam0.yaml")
    assert body["data"]["verdict_count"] == 33


def test_finalize_returns_400_when_no_session(client):
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.finalize_session.side_effect = ValueError(
            "No probe session in progress for camera 0"
        )
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/finalize"
        )

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /abort
# ---------------------------------------------------------------------------


def test_abort_returns_true_when_session_active(client):
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.abort_session.return_value = True

        resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/abort")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data"]["aborted"] is True


def test_abort_returns_false_when_nothing_to_abort(client):
    """Idempotent — calling abort when no session is in flight returns
    aborted=False, not an error. Wizard close-confirm uses this so it
    doesn't error on already-cleaned-up sessions."""
    from web.blueprints import api_v1 as api_v1_module

    with patch.object(
        api_v1_module, "ptz_empirical_probe_service"
    ) as mock_service:
        mock_service.abort_session.return_value = False

        resp = client.post("/api/v1/cameras/0/ptz/empirical-probe/abort")

    assert resp.status_code == 200
    assert resp.get_json()["data"]["aborted"] is False


# ---------------------------------------------------------------------------
# /apply-budget — read-side near-focus discovery
# ---------------------------------------------------------------------------


def test_apply_budget_returns_applied_when_service_succeeds(client):
    from web.blueprints import api_v1 as api_v1_module

    fake_payload = {"applied": True, "value": 0.75, "reason": ""}
    with patch.object(
        api_v1_module.ptz_empirical_probe_service,
        "apply_near_focus_budget",
        return_value=fake_payload,
    ) as mock_apply:
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/apply-budget"
        )
    mock_apply.assert_called_once_with(0)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"] == fake_payload


def test_apply_budget_surfaces_reason_when_nothing_to_apply(client):
    """When the cache YAML has no follow_zoom_max_burst_sec field
    (operator skipped the near-focus step), the route returns 200
    with applied=false + a human-readable reason for the wizard to
    show in the button tooltip."""
    from web.blueprints import api_v1 as api_v1_module

    fake_payload = {
        "applied": False,
        "value": None,
        "reason": "No near-focus budget recorded yet — run the wizard's near-focus step first.",
    }
    with patch.object(
        api_v1_module.ptz_empirical_probe_service,
        "apply_near_focus_budget",
        return_value=fake_payload,
    ):
        resp = client.post(
            "/api/v1/cameras/0/ptz/empirical-probe/apply-budget"
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["data"]["applied"] is False
    assert "near-focus" in body["data"]["reason"]


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


def test_all_routes_require_login(unauth_client):
    """Every empirical-probe route is @login_required. Without a session,
    they must redirect or 401 — never proceed to the service layer."""
    routes = [
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/start"),
        ("GET", "/api/v1/cameras/0/ptz/empirical-probe/status"),
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/execute"),
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/feedback"),
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/finalize"),
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/abort"),
        ("POST", "/api/v1/cameras/0/ptz/empirical-probe/apply-budget"),
    ]
    for method, url in routes:
        if method == "POST":
            resp = unauth_client.post(url)
        else:
            resp = unauth_client.get(url)
        # Either 302 (redirect to login) or 401 are acceptable —
        # what matters is "not 200 with success payload".
        assert resp.status_code in (301, 302, 401, 403), (
            f"{method} {url} returned {resp.status_code}, expected auth gate"
        )
