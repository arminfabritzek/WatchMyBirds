"""Local development isolation, transfer validation and editable-copy boundaries."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest

from scripts import local_dev, local_snapshot_remote
from utils.db.connection import _init_schema

NAME = "20261003_080000_000001.jpg"


@pytest.fixture
def snapshot_case(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with sqlite3.connect(source / "images.db") as conn:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO images(filename,timestamp) VALUES(?,?)",
            (NAME, "20261003_080000"),
        )
        conn.execute(
            "INSERT INTO detections(image_filename,bbox_x,bbox_y,bbox_w,bbox_h,"
            "thumbnail_path) VALUES(?,0.1,0.1,0.3,0.3,?)",
            (NAME, NAME.replace(".jpg", "_crop_1.webp")),
        )
    from PIL import Image

    original = source / "originals/2026-10-03" / NAME
    original.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "green").save(original)
    for folder, name in (
        ("optimized", NAME.replace(".jpg", ".webp")),
        ("thumbs", NAME.replace(".jpg", "_crop_1.webp")),
    ):
        image = source / "derivatives" / folder / "2026-10-03" / name
        image.parent.mkdir(parents=True)
        Image.new("RGB", (50, 50), "green").save(image)
    (source / "settings.yaml").write_text("TELEGRAM_ENABLED: true\n")
    (source / "cameras.yaml").write_text("password: private-camera-test\n")
    root = local_dev.managed_root(tmp_path / "private-dev")
    snapshot = root / "snapshots/test-snapshot"
    output = snapshot / "output"
    output.mkdir(parents=True)
    original_source_hash = local_dev.digest(source / "images.db")
    manifest = local_snapshot_remote.collect(source, output)
    manifest["test_original_source_hash"] = original_source_hash
    for name in manifest["files"]:
        local_dev.copy_file(source / name, output / name)
    local_dev.write_json(snapshot / "manifest.json", manifest)
    (snapshot / "COMPLETED").write_text("verified\n")
    return source, root, snapshot, manifest


def test_snapshot_exports_media_without_station_secrets(snapshot_case):
    source, _, snapshot, manifest = snapshot_case
    assert len(manifest["files"]) == 3
    assert manifest["source_missing_media"] == []
    assert manifest["counts"]["images"] == 1
    assert not (snapshot / "output/settings.yaml").exists()
    assert not (snapshot / "output/cameras.yaml").exists()
    assert "private-camera-test" not in json.dumps(manifest)
    assert (
        local_dev.digest(source / "images.db") == manifest["test_original_source_hash"]
    )
    local_dev.verify_snapshot(snapshot / "output", manifest)


def test_changed_media_rejects_activation_and_preserves_current(snapshot_case):
    _, root, snapshot, manifest = snapshot_case
    work = local_dev.activate(root, "test-snapshot")
    current = (root / "current.json").read_bytes()
    image = snapshot / "output" / next(iter(manifest["files"]))
    image.write_bytes(b"changed during transfer")
    with pytest.raises(ValueError, match="missing or changed"):
        local_dev.activate(root, "test-snapshot")
    assert (root / "current.json").read_bytes() == current
    assert work.is_dir()


def test_reset_creates_independent_copies_without_overwriting_edits(snapshot_case):
    source, root, snapshot, manifest = snapshot_case
    first = local_dev.activate(root, "test-snapshot")
    image_name = next(iter(manifest["files"]))
    first_image = first / "output" / image_name
    first_image.write_bytes(b"local-only edited image")
    with sqlite3.connect(first / "output/images.db") as conn:
        conn.execute("UPDATE detections SET is_favorite=1")
    second = local_dev.activate(root, "test-snapshot")
    assert first != second
    assert first_image.read_bytes() == b"local-only edited image"
    assert (
        local_dev.digest(second / "output" / image_name)
        == manifest["files"][image_name]["sha256"]
    )
    assert (
        local_dev.digest(source / "images.db") == manifest["test_original_source_hash"]
    )
    assert (
        local_dev.digest(snapshot / "output/images.db") == manifest["database_sha256"]
    )
    assert first_image.stat().st_ino != (snapshot / "output" / image_name).stat().st_ino


@pytest.mark.parametrize(
    "name", ["../outside.jpg", "/tmp/outside.jpg", "originals/../../private.jpg"]
)
def test_manifest_path_traversal_is_rejected(snapshot_case, name):
    _, _, snapshot, manifest = snapshot_case
    manifest["files"][name] = {"size": 0, "sha256": "invalid"}
    with pytest.raises(ValueError):
        local_dev.verify_snapshot(snapshot / "output", manifest)


def test_media_symlink_is_not_exported(snapshot_case):
    source, _, _, _ = snapshot_case
    (source / "originals/2026-10-03/link.jpg").symlink_to(source / "settings.yaml")
    destination = source.parent / "second-backup"
    destination.mkdir()
    with pytest.raises(ValueError, match="symlinks"):
        local_snapshot_remote.collect(source, destination)


def test_operations_are_serialized(tmp_path):
    root = local_dev.managed_root(tmp_path / "private-dev")
    with local_dev.exclusive(root), pytest.raises(ValueError, match="running"):
        with local_dev.exclusive(root):
            pytest.fail("lock admitted a second operation")


@pytest.mark.parametrize(
    "target", ["-oProxyCommand=evil", "admin@pi;touch /tmp/evil", "admin@pi$(evil)"]
)
def test_shell_syntax_in_ssh_target_is_rejected(target):
    with pytest.raises(ValueError):
        local_dev.validate_target(target, "/opt/app/data/output")


@pytest.mark.parametrize(
    "event,args",
    [
        ("socket.connect", (None, ("192.168.1.2", 80))),
        ("socket.connect", (None, ("127.0.0.1", 1984))),
        ("socket.sendto", (None, ("239.255.255.250", 3702))),
        ("socket.getaddrinfo", ("example.invalid", 80, 0, 0, 0)),
        ("subprocess.Popen", ("ssh", [], None, None)),
    ],
)
def test_local_process_rejects_external_or_relay_side_effects(event, args):
    with pytest.raises(PermissionError):
        local_dev.network_guard(event, args)


def test_real_factory_auth_csrf_persistence_and_disabled_routes(snapshot_case):
    _, root, snapshot, manifest = snapshot_case
    work = local_dev.activate(root, "test-snapshot")
    code = r"""
import json, sqlite3, sys, threading
from pathlib import Path
from scripts.local_dev import create_local_app, network_guard
work = Path(sys.argv[1])
sys.addaudithook(network_guard)
before = {t.name for t in threading.enumerate()}
app = create_local_app(work)
app.testing = True
client = app.test_client()
assert client.get('/healthz').status_code == 200
assert client.get('/local-dev/status').json['camera_enabled'] is False
assert client.get('/', headers={'Host': 'evil.example'}).status_code == 403
for path in ['/api/v1/cameras', '/api/v1/onvif/discover', '/video_feed', '/api/v1/system/restart', '/api/v1/settings', '/api/v1/telegram/send-report']:
    response = client.post(path) if path == '/api/v1/system/restart' else client.get(path)
    assert response.status_code == 503, (path, response.status_code)
assert client.get('/api/v1/storage/status').status_code == 302
assert client.post('/api/v1/storage/scan').status_code == 403
assert client.post('/api/labels/answer', json={}).status_code == 403
assert client.post('/api/review/event-approve', json={}).status_code == 404
login = client.get('/login')
assert login.status_code == 200
with client.session_transaction() as session:
    token = session['_csrf_token']
password = json.loads((work.parent.parent / 'local-credentials.json').read_text())['edit_password']
assert client.post('/login', data={'password': password, '_csrf_token': token}).status_code == 302
for path in ['/', '/gallery', '/gallery/2026-10-03', '/species', '/admin/review', '/admin/review?day=all&sort=priority', '/storage']:
    response = client.get(path)
    assert response.status_code == 200, (path, response.status_code)
    assert b'Local development' in response.data
    if path == '/':
        assert b'Live camera is off in local development' in response.data
        assert b'id="go2rtc-frame"' not in response.data
        assert b'id="video-feed-fallback"' not in response.data
    assert "connect-src 'self';" in response.headers['Content-Security-Policy']
    assert "frame-src 'none';" in response.headers['Content-Security-Policy']
    assert "script-src 'self' 'unsafe-inline';" in response.headers['Content-Security-Policy']
    assert "style-src 'self' 'unsafe-inline';" in response.headers['Content-Security-Policy']
    assert "font-src 'self';" in response.headers['Content-Security-Policy']
for path, method in [('/api/review/analyze/test.jpg', 'POST'), ('/api/moderation/bulk/rescan', 'POST'), ('/api/moderation/rescan-jobs/test/status', 'GET'), ('/api/moderation/rescan-proposals/1/apply', 'POST')]:
    assert client.open(path, method=method).status_code == 404
assert client.get('/unknown-route').status_code == 404
assert client.get('/admin/review?layout=legacy').status_code == 404
assert client.get('/admin/review?diagnostic=1').status_code == 404
assert client.get('/api/review/modal/bird-event-012345abcdef/999999').status_code == 404
response = client.post('/api/labels/answer', json={'filename': '20261003_080000_000001.jpg', 'detection_id': 1, 'bbox_quality': 'suitable'}, headers={'X-CSRF-Token': token})
assert response.status_code == 200, response.text
with sqlite3.connect(work / 'output/images.db') as conn:
    assert conn.execute("SELECT count(*) FROM human_label_facts WHERE fact_type='bbox_quality' AND answer_value='suitable'").fetchone()[0] == 1
assert {t.name for t in threading.enumerate()} == before
headers = {'X-CSRF-Token': token}
assert client.post('/api/v1/storage/scan', json={'path': '/outside'}, headers=headers).status_code == 400
assert client.post('/api/v1/storage/scan?path=/outside', headers=headers).status_code == 400
assert client.get('/api/v1/storage/status').json['status'] == 'idle'
assert client.post('/api/v1/storage/scan', headers=headers).status_code == 202
app.extensions['storage_accounting'].wait(10)
measured = client.get('/api/v1/storage/status').json
assert measured['status'] == 'complete', measured
assert measured['result']['database']['records']['images'] == 1
assert len(measured['history']) == 1
assert {t.name for t in threading.enumerate()} == before
from config import get_config
cfg = get_config()
assert not cfg['TELEGRAM_ENABLED'] and not cfg['MQTT_ENABLED'] and not cfg['telemetry_enabled']
print('isolated-local-app-pass')
"""
    environment = dict(
        os.environ, PYTHON_DOTENV_DISABLED="1", OUTPUT_DIR=str(work / "output")
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(work)],
        cwd=local_dev.REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated-local-app-pass" in result.stdout
    assert (
        local_dev.digest(snapshot / "output/images.db") == manifest["database_sha256"]
    )


def test_retention_flags_do_not_invent_missing_originals(snapshot_case):
    source, _, _, _ = snapshot_case
    with sqlite3.connect(source / "images.db") as conn:
        conn.execute("UPDATE images SET original_present=0")
    (source / "originals/2026-10-03" / NAME).unlink()
    destination = source.parent / "retained-backup"
    destination.mkdir()
    manifest = local_snapshot_remote.collect(source, destination)
    assert manifest["source_missing_media"] == []
    assert not any(name.startswith("originals/") for name in manifest["files"])


def test_missing_source_derivative_is_explicit(snapshot_case):
    source, _, _, _ = snapshot_case
    missing = "derivatives/optimized/2026-10-03/" + NAME.replace(".jpg", ".webp")
    (source / missing).unlink()
    destination = source.parent / "missing-source-backup"
    destination.mkdir()
    manifest = local_snapshot_remote.collect(source, destination)
    assert manifest["source_missing_media"] == [missing]


def test_incomplete_snapshot_is_not_activated(snapshot_case):
    _, root, snapshot, _ = snapshot_case
    (snapshot / "COMPLETED").unlink()
    with pytest.raises(FileNotFoundError):
        local_dev.activate(root, "test-snapshot")
    assert not (root / "current.json").exists()


def test_unmarked_directory_is_not_adopted(tmp_path):
    (tmp_path / "existing-private.txt").write_text("preserve")
    with pytest.raises(ValueError, match="nonempty"):
        local_dev.managed_root(tmp_path)
    assert (tmp_path / "existing-private.txt").read_text() == "preserve"


def test_remote_snapshot_closes_sqlite_sidecars(snapshot_case):
    _, _, snapshot, _ = snapshot_case
    assert not (snapshot / "output/images.db-wal").exists()
    assert not (snapshot / "output/images.db-shm").exists()
