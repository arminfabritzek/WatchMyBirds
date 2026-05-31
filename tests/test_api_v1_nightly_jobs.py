"""HTTP-level tests for the /api/v1/nightly/* and /api/v1/od/status routes.

Goal: prove the endpoints wire the hub correctly and don't leak
500s when called against an unconfigured/edge-case state.
"""

from __future__ import annotations

import threading

import pytest
from flask import Flask

from web.blueprints.api_v1 import init_api_v1
from web.services import nightly_job_hub
from web.services.nightly_job_hub import JobBase


class _FakeJob(JobBase):
    def __init__(self, slug: str):
        self._slug = slug
        self.started = threading.Event()
        self.finished = threading.Event()

    @property
    def name(self) -> str:
        return self._slug

    @property
    def display_name(self) -> str:
        return "Fake " + self._slug

    def run(self, stop_event, reason):
        self.started.set()
        stop_event.wait(timeout=0.5)
        self.finished.set()
        return 0


class _FakeDetectionManager:
    """Mimics DetectionManager.get_od_status without the full pipeline."""

    def get_od_status(self):
        return {
            "od_active": True,
            "reason": "master-switch-on",
            "next_transition_utc": None,
            "lat": None,
            "lon": None,
            "twilight_mode": "civil",
        }


@pytest.fixture
def app(monkeypatch, tmp_path):
    """A minimal Flask app with the API blueprint and a fake DM.

    Mirrors the setup in tests/test_api_v1.py — auth blueprint
    registered so @login_required can resolve, session
    authenticated via session_transaction in the client fixture.
    """
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    nightly_job_hub._registry.clear()  # type: ignore[attr-defined]
    nightly_job_hub._last_fire_date.clear()  # type: ignore[attr-defined]

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret"

    from web.blueprints.auth import auth_bp
    app.register_blueprint(auth_bp)

    init_api_v1(app, detection_manager=_FakeDetectionManager())
    yield app

    nightly_job_hub._registry.clear()  # type: ignore[attr-defined]
    nightly_job_hub._last_fire_date.clear()  # type: ignore[attr-defined]


@pytest.fixture
def client(app):
    """Authenticated client (login_required passes)."""
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


def test_list_empty_registry(client):
    resp = client.get("/api/v1/nightly/jobs")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"jobs": []}


def test_list_after_register(client):
    nightly_job_hub.register_job(_FakeJob("a"))
    resp = client.get("/api/v1/nightly/jobs")
    assert resp.status_code == 200
    jobs = resp.get_json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["name"] == "a"
    assert jobs[0]["running"] is False


def test_run_now_unknown_returns_404(client):
    resp = client.post(
        "/api/v1/nightly/jobs/does_not_exist/run_now",
        json={"reason": "test"},
    )
    assert resp.status_code == 404
    assert resp.get_json()["status"] == "unknown_job"


def test_run_now_known_job_starts_it(client):
    job = _FakeJob("b")
    nightly_job_hub.register_job(job)

    resp = client.post(
        "/api/v1/nightly/jobs/b/run_now",
        json={"reason": "test trigger"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "started"
    assert job.started.wait(timeout=1.0)

    stop_resp = client.post("/api/v1/nightly/jobs/b/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.get_json()["status"] == "stop_requested"
    assert job.finished.wait(timeout=2.0)


def test_stop_unknown_returns_404(client):
    resp = client.post("/api/v1/nightly/jobs/missing/stop")
    assert resp.status_code == 404


def test_od_status_endpoint_returns_payload(client):
    resp = client.get("/api/v1/od/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["reason"] == "master-switch-on"
    assert body["od_active"] is True
