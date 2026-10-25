"""Retention API — preview + run routes and missing-original (410) serving."""

import threading

import pytest
from flask import Flask
from flask.testing import FlaskClient

from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection
from utils.db.detections import insert_detection


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


def _seed_deletable(output_dir, filename):
    # Production-shaped artifacts (optimized <stem>.webp + thumb
    # <stem>_crop_1.webp recorded in thumbnail_path).
    with closing_connection() as conn:
        seed_image(conn, filename, output_dir, orig_bytes=1000)


ENABLED_CFG = {"AUTOMATIC_DELETION_ENABLED": True, "OLD_BIRD_PHOTOS": "keep_previews"}


@pytest.fixture
def client(output_dir, monkeypatch):
    from config import update_runtime_settings

    with closing_connection():
        pass
    update_runtime_settings(ENABLED_CFG)
    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG}
    monkeypatch.setattr("config.get_config", lambda: cfg)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"
    from web.blueprints.auth import auth_bp
    from web.blueprints.retention import retention_bp
    from web.services import nightly_job_hub
    from web.services.nightly_jobs.retention_job import RetentionJob

    nightly_job_hub._registry.clear()
    nightly_job_hub.register_job(RetentionJob())
    app.register_blueprint(auth_bp)
    app.register_blueprint(retention_bp)
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c
    nightly_job_hub._registry.clear()


def test_preview_returns_counts_and_is_side_effect_free(client, output_dir):
    fn = "20260101_120000_a.jpg"
    _seed_deletable(output_dir, fn)

    resp = client.get("/api/v1/retention/preview")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["due_files"] == 1
    assert data["due_bytes"] == 1000

    # Side-effect free: original still on disk, DB still present.
    assert (output_dir / "originals" / "2026-01-01" / fn).exists()
    with closing_connection() as conn:
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename=?", (fn,)
        ).fetchone()
    assert row["original_present"] == 1


def test_preview_reports_switch_and_upcoming_files(client: FlaskClient) -> None:
    data = client.get("/api/v1/retention/preview").get_json()
    assert data["enabled"] is True
    assert data["settings"]["old_bird_photos"] == "keep_previews"
    assert "next_14_days" in data


def test_run_starts_shared_job_and_deletes_previewed_set(client, output_dir):
    fn = "20260101_120000_b.jpg"
    _seed_deletable(output_dir, fn)

    resp = client.post("/api/v1/retention/run")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "started"

    from web.services import nightly_job_hub

    runtime = nightly_job_hub._registry["retention"]
    runtime.thread.join(timeout=2)
    status = nightly_job_hub.status("retention")
    assert status["last_rc"] == 0
    assert status["progress"]["deleted"] == 1
    assert status["progress"]["freed_bytes"] == 1000

    assert not (output_dir / "originals" / "2026-01-01" / fn).exists()
    # Derivatives preserved.
    assert (
        output_dir
        / "derivatives"
        / "optimized"
        / "2026-01-01"
        / "20260101_120000_b.webp"
    ).exists()


def test_manual_run_cannot_overlap_existing_retention_job(client, monkeypatch):
    from core import retention_core

    started = threading.Event()
    release = threading.Event()

    def slow_run(**kwargs):
        started.set()
        release.wait(timeout=2)
        return {
            "deleted": 0,
            "freed_bytes": 0,
            "missing": 0,
            "errors": 0,
            "protected": {},
            "processed": 0,
            "total": 0,
            "stopped": False,
        }

    monkeypatch.setattr(retention_core, "run", slow_run)
    first = client.post("/api/v1/retention/run")
    assert first.get_json()["status"] == "started"
    assert started.wait(timeout=1)

    second = client.post("/api/v1/retention/run")
    assert second.get_json()["status"] == "already_running"
    release.set()

    from web.services import nightly_job_hub

    nightly_job_hub._registry["retention"].thread.join(timeout=2)


def test_serve_original_returns_410_when_retention_deleted(output_dir, monkeypatch):
    fn = "20260101_120000_g.jpg"
    # DB row exists but marked deleted; file absent on disk.
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, "
            "original_present, original_deleted_at) "
            "VALUES (?, ?, 'confirmed_bird', 0, '2026-05-01T00:00:00+00:00')",
            (fn, fn[:15]),
        )
        conn.commit()

    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG}
    monkeypatch.setattr("config.get_config", lambda: cfg)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "k"
    import web.blueprints.media as media_mod

    monkeypatch.setattr(media_mod, "config", cfg)
    from web.blueprints.media import media_bp

    app.register_blueprint(media_bp)
    with app.test_client() as c:
        resp = c.get(f"/uploads/originals/2026-01-01/{fn}")
    assert resp.status_code == 410
    assert b"retention" in resp.data.lower()


def test_download_returns_410_when_retention_deleted(output_dir, monkeypatch):
    fn = "20260101_120000_h.jpg"
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, "
            "original_present, original_deleted_at) "
            "VALUES (?, ?, 'confirmed_bird', 0, '2026-05-01T00:00:00+00:00')",
            (fn, fn[:15]),
        )
        det = insert_detection(
            conn,
            {
                "image_filename": fn,
                "bbox_x": 0.1,
                "bbox_y": 0.1,
                "bbox_w": 0.2,
                "bbox_h": 0.2,
                "od_class_name": "bird",
                "od_confidence": 0.9,
            },
        )
        conn.execute(
            "UPDATE detections SET status='active' WHERE detection_id=?", (det,)
        )
        conn.commit()
        det_id = det

    cfg = {
        "OUTPUT_DIR": str(output_dir),
        **ENABLED_CFG,
        "EXPORT_BURN_IN_METADATA": False,
    }
    monkeypatch.setattr("config.get_config", lambda: cfg)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "k"
    import web.blueprints.media as media_mod

    monkeypatch.setattr(media_mod, "config", cfg)
    from web.blueprints.media import media_bp

    app.register_blueprint(media_bp)
    with app.test_client() as c:
        resp = c.get(f"/api/image/download/{det_id}")
    assert resp.status_code == 410


def test_serve_original_410_even_when_retention_disabled(output_dir, monkeypatch):
    # enable -> delete -> disable: the marker persists, so a removed original
    # must still report 410, not 404. AUTOMATIC_DELETION_ENABLED is False here.
    fn = "20260101_120000_i.jpg"
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, "
            "original_present, original_deleted_at) "
            "VALUES (?, ?, 'confirmed_bird', 0, '2026-05-01T00:00:00+00:00')",
            (fn, fn[:15]),
        )
        conn.commit()

    cfg = {
        "OUTPUT_DIR": str(output_dir),
        **ENABLED_CFG,
        "AUTOMATIC_DELETION_ENABLED": False,
    }
    monkeypatch.setattr("config.get_config", lambda: cfg)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "k"
    import web.blueprints.media as media_mod

    monkeypatch.setattr(media_mod, "config", cfg)
    from web.blueprints.media import media_bp

    app.register_blueprint(media_bp)
    with app.test_client() as c:
        resp = c.get(f"/uploads/originals/2026-01-01/{fn}")
    assert resp.status_code == 410


def test_serve_original_genuinely_missing_is_404_not_410(output_dir, monkeypatch):
    # No DB row at all (never existed) and no file -> 404, not 410.
    fn = "20260101_120000_j.jpg"
    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG}
    monkeypatch.setattr("config.get_config", lambda: cfg)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "k"
    import web.blueprints.media as media_mod

    monkeypatch.setattr(media_mod, "config", cfg)
    from web.blueprints.media import media_bp

    app.register_blueprint(media_bp)
    with app.test_client() as c:
        resp = c.get(f"/uploads/originals/2026-01-01/{fn}")
    assert resp.status_code == 404


def test_run_requires_authentication(output_dir, monkeypatch):
    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG}
    monkeypatch.setattr("config.get_config", lambda: cfg)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "k"
    from web.blueprints.auth import auth_bp
    from web.blueprints.retention import retention_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(retention_bp)
    with app.test_client() as c:
        resp = c.post("/api/v1/retention/run")  # no auth session
    assert resp.status_code in (302, 401, 403)
