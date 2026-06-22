"""Retention API — preview + run routes and missing-original (410) serving."""

import pytest
from flask import Flask

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


ENABLED_CFG = {
    "RETENTION_ENABLED": True,
    "RETENTION_DAYS": 90,
    "RETENTION_PROTECT_FAVORITES": True,
    "RETENTION_PROTECT_UNREVIEWED": True,
}


@pytest.fixture
def client(output_dir, monkeypatch):
    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG}
    monkeypatch.setattr("config.get_config", lambda: cfg)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"
    from web.blueprints.auth import auth_bp
    from web.blueprints.retention import retention_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(retention_bp)
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


def test_preview_returns_counts_and_is_side_effect_free(client, output_dir):
    fn = "20260101_120000_a.jpg"
    _seed_deletable(output_dir, fn)

    resp = client.get("/api/v1/retention/preview")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["deletable"]["count"] == 1
    assert data["deletable"]["estimated_bytes"] == 1000

    # Side-effect free: original still on disk, DB still present.
    assert (output_dir / "originals" / "2026-01-01" / fn).exists()
    with closing_connection() as conn:
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename=?", (fn,)
        ).fetchone()
    assert row["original_present"] == 1


def test_preview_reports_posture(client):
    resp = client.get("/api/v1/retention/preview")
    data = resp.get_json()
    # Legacy ENABLED_CFG (no posture, enabled=True) resolves to conservative.
    assert data["posture"] == "conservative"


def test_conservative_protects_unreviewed_but_reclaim_retires_it(
    output_dir, monkeypatch
):
    fn = "20260101_120000_u.jpg"
    with closing_connection() as conn:
        seed_image(conn, fn, output_dir, review_status="untagged", orig_bytes=1000)

    base = {"OUTPUT_DIR": str(output_dir), "RETENTION_DAYS": 90}

    monkeypatch.setattr(
        "config.get_config", lambda: {**base, "RETENTION_POSTURE": "conservative"}
    )
    from core import retention_core

    cons = retention_core.preview()
    assert cons["posture"] == "conservative"
    assert cons["deletable"]["count"] == 0
    assert cons["protected"]["unreviewed"] == 1

    monkeypatch.setattr(
        "config.get_config", lambda: {**base, "RETENTION_POSTURE": "reclaim"}
    )
    rec = retention_core.preview()
    assert rec["posture"] == "reclaim"
    assert rec["deletable"]["count"] == 1
    assert rec["protected"]["unreviewed"] == 0


def test_run_deletes_previewed_set_and_reports_counts(client, output_dir):
    fn = "20260101_120000_b.jpg"
    _seed_deletable(output_dir, fn)

    resp = client.post("/api/v1/retention/run")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["deleted"] == 1
    assert data["freed_bytes"] == 1000

    assert not (output_dir / "originals" / "2026-01-01" / fn).exists()
    # Derivatives preserved.
    assert (
        output_dir
        / "derivatives"
        / "optimized"
        / "2026-01-01"
        / "20260101_120000_b.webp"
    ).exists()


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
    # must still report 410, not 404. RETENTION_ENABLED is False here.
    fn = "20260101_120000_i.jpg"
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, "
            "original_present, original_deleted_at) "
            "VALUES (?, ?, 'confirmed_bird', 0, '2026-05-01T00:00:00+00:00')",
            (fn, fn[:15]),
        )
        conn.commit()

    cfg = {"OUTPUT_DIR": str(output_dir), **ENABLED_CFG, "RETENTION_ENABLED": False}
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
