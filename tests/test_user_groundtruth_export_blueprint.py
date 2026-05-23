"""Tests for the user-groundtruth export blueprint HTTP routes.

Covers the request/response contract — auth gates, payload validation,
empty-batch handling, the wiring between routes and the service.
The deep service semantics (COCO validity, manifest fields, bucket
isolation) are covered in
``tests/test_user_groundtruth_export_service.py`` and
``tests/test_user_groundtruth_queries.py``.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

# Service helpers we re-use to build a real DB fixture for the
# end-to-end blueprint test (rather than mocking everything).
from web.services.user_groundtruth_export_service import (
    build_batch,
    record_batch_exported,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_seeded_db(tmp_path: Path) -> Path:
    """Create a real on-disk SQLite that the blueprint's
    ``closing_connection`` can open. Seeds three buckets so the
    preview + build endpoints have something to operate on."""
    db_path = tmp_path / "test_images.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            review_updated_at TEXT
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            status TEXT DEFAULT 'active',
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_confidence REAL,
            od_class_name TEXT,
            detector_model_version TEXT,
            classifier_model_version TEXT,
            frame_width INTEGER, frame_height INTEGER,
            decision_level TEXT,
            raw_species_name TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            species_updated_at TEXT,
            is_favorite INTEGER DEFAULT 0,
            rating_source TEXT DEFAULT 'auto'
        );
        CREATE TABLE export_batches (
            batch_id TEXT PRIMARY KEY,
            built_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            since_at TEXT,
            until_at TEXT NOT NULL,
            hard_negatives_count INTEGER NOT NULL DEFAULT 0,
            confirmed_positives_count INTEGER NOT NULL DEFAULT 0,
            species_relabels_count INTEGER NOT NULL DEFAULT 0,
            favorites_count INTEGER NOT NULL DEFAULT 0,
            frame_integrity_dropped_count INTEGER NOT NULL DEFAULT 0,
            exporter_version TEXT NOT NULL,
            wmb_app_version TEXT,
            notes TEXT
        );
        CREATE TABLE groundtruth_export_exclusions (
            exclusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            detection_id INTEGER,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT NOT NULL DEFAULT 'dry_run',
            released_at TEXT
        );
        """
    )
    # One row per bucket
    conn.execute(
        "INSERT INTO images (filename, review_status, review_updated_at) "
        "VALUES ('20260522_100000_hn.jpg', 'no_bird', '2026-05-22T10:00:00Z')"
    )
    conn.execute(
        "INSERT INTO detections (image_filename, bbox_x, bbox_y, bbox_w, bbox_h, "
        "frame_width, frame_height, decision_level, raw_species_name) "
        "VALUES ('20260522_100000_hn.jpg', 0.1, 0.1, 0.2, 0.2, 1920, 1080, "
        "'species_review', 'Parus_major')"
    )
    conn.execute("INSERT INTO images (filename) VALUES ('20260522_110000_cp.jpg')")
    # CP requires explicit user action — species_source='manual'
    conn.execute(
        "INSERT INTO detections (image_filename, bbox_x, bbox_y, bbox_w, bbox_h, "
        "frame_width, frame_height, decision_level, raw_species_name, "
        "species_source, species_updated_at) "
        "VALUES ('20260522_110000_cp.jpg', 0.2, 0.2, 0.1, 0.1, 1920, 1080, "
        "'species', 'Cyanistes_caeruleus', 'manual', '2026-05-22T11:00:00Z')"
    )
    conn.execute("INSERT INTO images (filename) VALUES ('20260522_120000_rl.jpg')")
    # Relabel: manual_species_override is the user-action gate
    conn.execute(
        "INSERT INTO detections (image_filename, bbox_x, bbox_y, bbox_w, bbox_h, "
        "frame_width, frame_height, decision_level, raw_species_name, "
        "manual_species_override, species_updated_at) "
        "VALUES ('20260522_120000_rl.jpg', 0.3, 0.3, 0.1, 0.1, 1920, 1080, "
        "'species', 'Parus_major', 'Erithacus_rubecula', '2026-05-22T12:00:00Z')"
    )
    conn.commit()
    conn.close()
    return db_path


def _make_originals(tmp_path: Path) -> Path:
    """Create the date-sharded originals/ directory with fake JPGs so
    the build endpoint's zipping path actually finds files."""
    base = tmp_path / "originals"
    date_dir = base / "2026-05-22"
    date_dir.mkdir(parents=True)
    for fn in (
        "20260522_100000_hn.jpg",
        "20260522_110000_cp.jpg",
        "20260522_120000_rl.jpg",
    ):
        (date_dir / fn).write_bytes(b"FAKEJPGBYTES")
    return base


@pytest.fixture
def app_and_db(tmp_path):
    """Build a Flask app with the blueprint registered against a
    real seeded tmp SQLite. Returns (app, db_path, output_dir).
    """
    db_path = _make_seeded_db(tmp_path)
    _make_originals(tmp_path)  # under tmp_path/originals/

    # PathManager looks for originals/ as a child of output_dir, so
    # output_dir == tmp_path here.
    output_dir = str(tmp_path)

    # Templates: the blueprint renders user_groundtruth_export.html
    # which extends base.html. For HTTP-contract tests we mock out
    # the render to avoid pulling the full template hierarchy.
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.user_groundtruth_export import (
        init_user_groundtruth_export_bp,
        user_groundtruth_export_bp,
    )

    app.register_blueprint(auth_bp)
    init_user_groundtruth_export_bp(
        output_dir=output_dir,
        app_version="0.46.0-test",
    )
    app.register_blueprint(user_groundtruth_export_bp)

    # Point closing_connection at our seeded DB. Two patches:
    # - _get_db_path: route get_connection() to our tmp DB
    # - _init_schema: skip the full app-schema apply that would fail
    #   on our minimal tmp DB (e.g. references to content_hash, ptz_*
    #   columns the blueprint test doesn't need).
    with (
        patch(
            "utils.db.connection._get_db_path",
            return_value=db_path,
        ),
        patch(
            "utils.db.connection._init_schema",
            return_value=None,
        ),
    ):
        yield app, db_path, output_dir


@pytest.fixture
def client(app_and_db):
    app, _, _ = app_and_db
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


@pytest.fixture
def unauth_client(app_and_db):
    app, _, _ = app_and_db
    return app.test_client()


# ---------------------------------------------------------------------------
# Auth gates
# ---------------------------------------------------------------------------


def test_preview_requires_auth(unauth_client):
    r = unauth_client.get("/api/groundtruth-export/preview")
    # auth_required redirects unauthenticated requests
    assert r.status_code in (302, 401, 403)


def test_build_requires_auth(unauth_client):
    r = unauth_client.post("/api/groundtruth-export/build", json={})
    assert r.status_code in (302, 401, 403)


def test_page_requires_auth(unauth_client):
    r = unauth_client.get("/admin/groundtruth-export")
    assert r.status_code in (302, 401, 403)


def test_dry_run_page_requires_auth(unauth_client):
    r = unauth_client.get("/admin/groundtruth-export/dry-run")
    assert r.status_code in (302, 401, 403)


def test_appbar_export_link_points_to_groundtruth_export():
    content = (
        Path(__file__).resolve().parent.parent
        / "templates"
        / "partials"
        / "appbar.html"
    ).read_text(encoding="utf-8")

    assert 'href="/admin/groundtruth-export"' in content
    assert "Export user-reviewed groundtruth for Pipeline-Dev" in content
    assert "current_path in ['/admin/groundtruth-export', '/admin/export']" in content


def test_legacy_training_export_links_to_new_groundtruth_export():
    content = (
        Path(__file__).resolve().parent.parent / "templates" / "training_export.html"
    ).read_text(encoding="utf-8")

    assert 'href="/admin/groundtruth-export"' in content
    assert "New groundtruth export" in content


def test_legacy_review_export_button_points_to_groundtruth_export():
    content = (
        Path(__file__).resolve().parent.parent / "templates" / "orphans.html"
    ).read_text(encoding="utf-8")

    assert 'href="/admin/groundtruth-export"' in content
    assert "Export user-reviewed groundtruth for Pipeline-Dev" in content


def test_groundtruth_export_page_links_to_dry_run():
    content = (
        Path(__file__).resolve().parent.parent
        / "templates"
        / "user_groundtruth_export.html"
    ).read_text(encoding="utf-8")

    assert 'href="/admin/groundtruth-export/dry-run"' in content
    assert "Review dry-run HTML" in content


# ---------------------------------------------------------------------------
# Preview endpoint
# ---------------------------------------------------------------------------


def test_preview_returns_pending_counts(client):
    r = client.get("/api/groundtruth-export/preview")
    assert r.status_code == 200
    data = r.get_json()
    assert "pending" in data
    # Seeded DB: 1 HN, 2 CP (the RL row also counts as CP), 1 RL
    assert data["pending"]["hard_negatives"] == 1
    assert data["pending"]["confirmed_positives"] == 2
    assert data["pending"]["species_relabels"] == 1
    assert data["total_pending"] == 4
    assert data["last_batch"] is None
    assert data["since"] is None


def test_preview_reflects_recorded_batches(client, app_and_db):
    """After a batch is recorded, the next preview's `since` advances
    and the pending counts drop. End-to-end through the DB."""
    _, db_path, output_dir = app_and_db

    # First preview: full pending
    r1 = client.get("/api/groundtruth-export/preview")
    full = r1.get_json()["total_pending"]
    assert full == 4

    # Manually record a batch (simulates a prior build) — directly
    # through the service so we don't depend on the build endpoint
    # for this test.
    from utils.path_manager import PathManager

    pm = PathManager(output_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    fixed_now = datetime(2026, 5, 23, 0, 0, 0, tzinfo=UTC)
    batch = build_batch(
        conn,
        path_resolver=pm.get_original_path,
        wmb_app_version="0.46.0-test",
        now=fixed_now,
    )
    record_batch_exported(conn, batch, notes="seeded")
    conn.close()

    # Second preview: `since` set to first batch's `until`, pending should drop
    r2 = client.get("/api/groundtruth-export/preview")
    data = r2.get_json()
    assert data["since"] == batch.until
    assert data["last_batch"] is not None
    assert data["last_batch"]["batch_id"] == batch.batch_id
    assert data["last_batch"]["counts"]["confirmed_positives"] == 2
    # No new actions happened after the batch, so pending is now 0
    assert data["total_pending"] == 0


def test_preview_accepts_since_override(client):
    """The ``since`` query param overrides the default last-batch `until`."""
    r = client.get("/api/groundtruth-export/preview?since=2099-01-01T00:00:00Z")
    data = r.get_json()
    # Future `since` → nothing is pending
    assert data["since"] == "2099-01-01T00:00:00Z"
    assert data["total_pending"] == 0


# ---------------------------------------------------------------------------
# Dry-run page
# ---------------------------------------------------------------------------


def test_dry_run_page_renders_exact_batch_without_recording(client, app_and_db):
    _, db_path, _ = app_and_db

    r = client.get("/admin/groundtruth-export/dry-run")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Groundtruth Export Dry-run" in body
    assert 'data-dryrun-tab="confirmed_positives"' in body
    assert 'data-dryrun-tab="species_relabels"' in body
    assert 'data-dryrun-tab="favorites"' in body
    assert 'data-dryrun-tab="hard_negatives"' in body
    assert "Confirmed positives (2)" in body
    assert "Species re-labels (1)" in body
    assert "Favorites (0)" in body
    assert "FPs / Kein Vogel (1)" in body
    assert "Kein Vogel (Training Hard-negatives)" in body
    assert "Confirmed positives" in body
    assert "Species re-labels" in body
    assert 'id="dryRunTileSize"' in body
    assert "wmbDryRunTileSize" in body
    assert "dryrun-crop" in body
    assert "dryrun-avatar" in body
    assert "/assets/review_species/Cyanistes_caeruleus.webp" in body
    assert "dryrun-avatar--no_bird" in body
    assert "dryrun-bbox" in body
    assert "Exclude from export" in body
    assert "Move to WMB Trash" in body
    assert 'data-trash-export-image="20260522_110000_cp.jpg"' in body
    assert "20260522_100000_hn.jpg" in body
    assert "20260522_110000_cp.jpg" in body
    assert "20260522_120000_rl.jpg" in body
    assert "Build this dry-run ZIP" in body
    assert "const frozenUntil =" in body

    conn = sqlite3.connect(db_path)
    n_batches = conn.execute("SELECT COUNT(*) FROM export_batches").fetchone()[0]
    conn.close()
    assert n_batches == 0


def test_dry_run_page_renders_frame_integrity_drop_as_visual_tab(client, app_and_db):
    _, db_path, output_dir = app_and_db
    filename = "20260522_130000_mixed.jpg"
    original = Path(output_dir) / "originals" / "2026-05-22" / filename
    original.write_bytes(b"FAKEJPGBYTES")

    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO images (filename) VALUES (?)", (filename,))
    conn.execute(
        "INSERT INTO detections (image_filename, bbox_x, bbox_y, bbox_w, bbox_h, "
        "frame_width, frame_height, decision_level, raw_species_name, "
        "species_source, species_updated_at) "
        "VALUES (?, 0.1, 0.1, 0.2, 0.2, 1920, 1080, "
        "'species', 'Parus_major', 'manual', '2026-05-22T13:00:00Z')",
        (filename,),
    )
    conn.execute(
        "INSERT INTO detections (image_filename, bbox_x, bbox_y, bbox_w, bbox_h, "
        "frame_width, frame_height, decision_level, raw_species_name) "
        "VALUES (?, 0.5, 0.5, 0.2, 0.2, 1920, 1080, "
        "'species_review', 'Cyanistes_caeruleus')",
        (filename,),
    )
    conn.commit()
    conn.close()

    r = client.get("/admin/groundtruth-export/dry-run")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'data-dryrun-tab="frame_integrity_dropped"' in body
    assert "Needs cleanup (1)" in body
    assert "Needs cleanup (not exported)" in body
    assert "dryrun-bbox--positive" in body
    assert "dryrun-bbox--missing" in body
    assert f'data-exclude-image="{filename}"' in body
    assert f'data-trash-export-image="{filename}"' in body
    assert "missing image on disk" not in body


def test_dry_run_exclude_image_quarantines_frame(client, app_and_db):
    _, db_path, _ = app_and_db

    r = client.post(
        "/api/groundtruth-export/exclusions",
        json={
            "scope": "image",
            "image_filename": "20260522_110000_cp.jpg",
            "reason": "bad dry-run row",
        },
    )

    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["inserted"] is True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT scope, image_filename, reason, released_at "
        "FROM groundtruth_export_exclusions"
    ).fetchall()
    conn.close()
    assert [dict(r) for r in rows] == [
        {
            "scope": "image",
            "image_filename": "20260522_110000_cp.jpg",
            "reason": "bad dry-run row",
            "released_at": None,
        }
    ]

    body = client.get("/admin/groundtruth-export/dry-run").get_data(as_text=True)
    assert "20260522_110000_cp.jpg" not in body


def test_dry_run_trash_image_moves_to_wmb_trash_and_excludes(client, app_and_db):
    _, db_path, _ = app_and_db

    r = client.post(
        "/api/groundtruth-export/trash-image",
        json={
            "image_filename": "20260522_110000_cp.jpg",
            "reason": "bad WMB row",
        },
    )

    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["updated_images"] == 1
    assert data["active_detections"] == 1
    assert data["excluded"] is True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    image = conn.execute(
        "SELECT review_status, review_updated_at FROM images WHERE filename = ?",
        ("20260522_110000_cp.jpg",),
    ).fetchone()
    detection = conn.execute(
        "SELECT status FROM detections WHERE image_filename = ?",
        ("20260522_110000_cp.jpg",),
    ).fetchone()
    exclusion = conn.execute(
        "SELECT scope, reason, released_at FROM groundtruth_export_exclusions "
        "WHERE image_filename = ?",
        ("20260522_110000_cp.jpg",),
    ).fetchone()
    conn.close()

    assert image["review_status"] == "no_bird"
    assert image["review_updated_at"]
    assert detection["status"] == "active"
    assert dict(exclusion) == {
        "scope": "image",
        "reason": "bad WMB row",
        "released_at": None,
    }

    body = client.get("/admin/groundtruth-export/dry-run").get_data(as_text=True)
    assert "20260522_110000_cp.jpg" not in body


def test_dry_run_trash_image_requires_valid_filename(client):
    r = client.post(
        "/api/groundtruth-export/trash-image",
        json={"image_filename": "../bad.jpg"},
    )

    assert r.status_code == 400
    assert r.get_json()["message"] == "invalid image filename"


def test_dry_run_page_respects_since_override(client):
    r = client.get("/admin/groundtruth-export/dry-run?since=2099-01-01T00:00:00Z")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "No rows would be exported for this window." in body
    assert "2099-01-01T00:00:00Z" in body


# ---------------------------------------------------------------------------
# Build endpoint
# ---------------------------------------------------------------------------


def test_build_returns_zip_with_correct_mimetype(client):
    r = client.post("/api/groundtruth-export/build", json={})
    assert r.status_code == 200
    assert r.mimetype == "application/zip"
    cd = r.headers.get("Content-Disposition", "")
    assert "attachment" in cd
    assert "user_groundtruth_" in cd
    assert cd.endswith('.zip"') or cd.endswith(".zip")


def test_build_zip_contains_expected_files(client):
    r = client.post("/api/groundtruth-export/build", json={})
    assert r.status_code == 200
    # Verify ZIP structure round-trip
    import io

    buf = io.BytesIO(r.data)
    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
    assert "coco_annotations.json" in names
    assert "batch_metadata.json" in names
    assert "README.md" in names
    assert "manifests/hard_negatives.jsonl" in names
    assert "manifests/confirmed_positives.jsonl" in names
    assert "manifests/species_relabels.jsonl" in names
    # The seeded originals should be present
    assert "images/2026-05-22/20260522_100000_hn.jpg" in names


def test_build_records_batch_in_db(client, app_and_db):
    """A successful build must write a row to export_batches so the
    next preview's `since` advances. The browser-side download is a
    side effect; the DB write is the durable contract."""
    _, db_path, _ = app_and_db
    conn = sqlite3.connect(db_path)
    n_before = conn.execute("SELECT COUNT(*) FROM export_batches").fetchone()[0]
    conn.close()
    assert n_before == 0

    r = client.post(
        "/api/groundtruth-export/build",
        json={"notes": "blueprint test"},
    )
    assert r.status_code == 200

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT batch_id, notes, hard_negatives_count, "
        "confirmed_positives_count, species_relabels_count "
        "FROM export_batches"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["notes"] == "blueprint test"
    assert rows[0]["hard_negatives_count"] == 1
    assert rows[0]["confirmed_positives_count"] == 2
    assert rows[0]["species_relabels_count"] == 1


def test_build_with_empty_window_still_returns_valid_zip(client):
    """Operator can deliberately advance the window even when nothing
    new came in. The ZIP is empty-structured but valid."""
    r = client.post(
        "/api/groundtruth-export/build",
        json={"since": "2099-01-01T00:00:00Z"},
    )
    assert r.status_code == 200
    import io

    buf = io.BytesIO(r.data)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))
        coco = json.loads(zf.read("coco_annotations.json"))
    assert meta["counts"]["hard_negatives"] == 0
    assert meta["counts"]["confirmed_positives"] == 0
    assert meta["counts"]["species_relabels"] == 0
    assert coco["images"] == []


def test_build_advances_window_on_subsequent_call(client, app_and_db):
    """Two consecutive builds chain their `until` -> `since` so no
    row appears in both batches."""
    _, db_path, _ = app_and_db

    # Build 1
    r1 = client.post("/api/groundtruth-export/build", json={})
    assert r1.status_code == 200
    import io

    meta1 = json.loads(zipfile.ZipFile(io.BytesIO(r1.data)).read("batch_metadata.json"))
    assert meta1["counts"]["confirmed_positives"] == 2

    # Build 2 (no new data) — should pick up `since` from build 1
    r2 = client.post("/api/groundtruth-export/build", json={})
    assert r2.status_code == 200
    meta2 = json.loads(zipfile.ZipFile(io.BytesIO(r2.data)).read("batch_metadata.json"))
    # No double-count: total is zero now
    assert meta2["counts"]["confirmed_positives"] == 0
    assert meta2["counts"]["hard_negatives"] == 0
    assert meta2["since"] == meta1["until"]


def test_build_handles_missing_image_files_gracefully(client, app_and_db, tmp_path):
    """Operator deleted a frame after labeling — build should not
    crash, missing files should appear in batch_metadata."""
    # Remove one of the seeded originals
    (tmp_path / "originals" / "2026-05-22" / "20260522_110000_cp.jpg").unlink()

    r = client.post("/api/groundtruth-export/build", json={})
    assert r.status_code == 200
    import io

    with zipfile.ZipFile(io.BytesIO(r.data)) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))
        names = set(zf.namelist())
    assert "20260522_110000_cp.jpg" in meta["missing_images_on_disk"]
    # And the file is NOT in the ZIP
    assert "images/2026-05-22/20260522_110000_cp.jpg" not in names
    # But the other two are
    assert "images/2026-05-22/20260522_100000_hn.jpg" in names
    assert "images/2026-05-22/20260522_120000_rl.jpg" in names


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_build_returns_500_when_blueprint_not_initialized(tmp_path):
    """If init_user_groundtruth_export_bp was never called, the
    path-resolver factory raises and the endpoint must surface a
    clean 500 rather than a 500 ISE traceback."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret"

    from web.blueprints import user_groundtruth_export as bp_mod
    from web.blueprints.auth import auth_bp

    # Wipe shared state so the resolver raises
    bp_mod._shared.clear()

    app.register_blueprint(auth_bp)
    app.register_blueprint(bp_mod.user_groundtruth_export_bp)

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        r = client.post("/api/groundtruth-export/build", json={})
    assert r.status_code == 500
    assert r.get_json()["status"] == "error"
