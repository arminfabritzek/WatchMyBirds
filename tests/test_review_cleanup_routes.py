"""Route contract for "Move Review Queue to Trash" cleanup endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def app():
    project_root = _project_root()
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "assets"),
    )
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.review import review_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(review_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_cleanup_preview_returns_service_payload(client):
    payload = {
        "events": 3,
        "images": 2,
        "detections": 5,
        "favorites": 1,
        "export_relevant": 1,
    }
    with patch("web.blueprints.review.review_cleanup_service") as svc:
        svc.preview.return_value = payload
        resp = client.get("/api/review/cleanup/preview")

    assert resp.status_code == 200
    assert resp.get_json() == payload
    svc.preview.assert_called_once_with()


def test_cleanup_run_returns_moved_counts(client):
    with (
        patch("web.blueprints.review.review_cleanup_service") as svc,
        patch("web.blueprints.review.gallery_service"),
    ):
        svc.run.return_value = {"images_moved": 2, "detections_moved": 5}
        resp = client.post("/api/review/cleanup/run")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["images_moved"] == 2
    assert data["detections_moved"] == 5
    svc.run.assert_called_once_with()


def test_cleanup_routes_require_login(app):
    with app.test_client() as anon:
        assert anon.get("/api/review/cleanup/preview").status_code in (302, 401, 403)
        assert anon.post("/api/review/cleanup/run").status_code in (302, 401, 403)
