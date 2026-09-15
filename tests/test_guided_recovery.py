"""Safety and orchestration tests for guided USB recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import recovery_core, usb_backup_core
from scripts import recovery_runner


def _snapshot(root: Path, name: str = "20260915_120000_manual") -> Path:
    snapshot = root / "snapshots" / name
    output = snapshot / "data" / "output"
    original = output / "originals" / "2026-09-15" / "20260915_120000_bird.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"bird")
    (output / "settings.yaml").write_text(
        "EDIT_PASSWORD: old-password\nCAMERA_URL: rtsp://old-camera\n"
        "TELEGRAM_BOT_TOKEN: source-token\nRETENTION_DAYS: 30\n",
        encoding="utf-8",
    )
    (output / "cameras.yaml").write_text("camera: source\n", encoding="utf-8")
    (output / "go2rtc.yaml").write_text("stream: source\n", encoding="utf-8")
    db = snapshot / "data" / "images.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE images (filename TEXT PRIMARY KEY, content_hash TEXT, original_present INTEGER)"
        )
        conn.execute("CREATE TABLE detections (detection_id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE classifications (classification_id INTEGER PRIMARY KEY)"
        )
        conn.execute("CREATE TABLE sources (source_id INTEGER PRIMARY KEY)")
        conn.execute(
            "INSERT INTO images VALUES (?, ?, 1)",
            (original.name, hashlib.sha256(b"bird").hexdigest()),
        )
        conn.execute("INSERT INTO detections VALUES (1)")
        conn.execute("INSERT INTO classifications VALUES (1)")
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    (snapshot / "data" / "images.db.sha256").write_text(f"{digest}  images.db\n")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_name": name,
                "completed_at": "2026-09-15T12:01:00Z",
                "host": "test-source-device",
                "app_version": "0.5.6",
            }
        )
    )
    (snapshot / "COMPLETED").write_text("2026-09-15T12:01:00Z")
    return snapshot


def _point_usb_core(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(usb_backup_core, "MOUNT_POINT", root)
    monkeypatch.setattr(usb_backup_core, "SNAPSHOTS_DIR", root / "snapshots")
    monkeypatch.setattr(usb_backup_core, "LATEST_LINK", root / "latest")
    monkeypatch.setattr(usb_backup_core, "BACKUP_DEVICE", root / "device")
    monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda _path: True)


def test_preview_reports_counts_space_source_and_settings_policy(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "usb")
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "settings.yaml").write_text(
        "EDIT_PASSWORD: current-password\nCAMERA_URL: rtsp://current-camera\n",
        encoding="utf-8",
    )

    preview = recovery_core.inspect_snapshot(snapshot, destination)

    assert preview["compatible"] is True
    assert preview["counts"] == {"images": 1, "detections": 1, "classifications": 1}
    assert preview["source"] == "test-source-device"
    assert preview["required_bytes"] > 0
    assert preview["space_ok"] is True
    assert preview["settings_preserved"] == ["CAMERA_URL", "EDIT_PASSWORD"]
    assert preview["settings_excluded"] == ["TELEGRAM_BOT_TOKEN"]
    assert "RETENTION_DAYS" in preview["settings_restored"]


def test_recovery_preserves_current_auth_and_device_settings(tmp_path):
    snapshot = _snapshot(tmp_path / "usb")
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "settings.yaml").write_text(
        "EDIT_PASSWORD: current-password\nCAMERA_URL: rtsp://current-camera\nRETENTION_DAYS: 90\n",
        encoding="utf-8",
    )
    (destination / "cameras.yaml").write_text("camera: current\n", encoding="utf-8")
    (destination / "go2rtc.yaml").write_text("stream: current\n", encoding="utf-8")

    recovery_core.recover_snapshot(snapshot, destination, mode="recovery", force=True)

    restored = recovery_core._read_settings(destination / "settings.yaml")
    assert restored["EDIT_PASSWORD"] == "current-password"
    assert restored["CAMERA_URL"] == "rtsp://current-camera"
    assert "TELEGRAM_BOT_TOKEN" not in restored
    assert restored["RETENTION_DAYS"] == 30
    assert (destination / "cameras.yaml").read_text() == "camera: current\n"
    assert (destination / "go2rtc.yaml").read_text() == "stream: current\n"


def test_migration_never_imports_source_credentials_or_device_files(tmp_path):
    snapshot = _snapshot(tmp_path / "usb")
    destination = tmp_path / "output"

    recovery_core.recover_snapshot(snapshot, destination, mode="migration")

    restored = recovery_core._read_settings(destination / "settings.yaml")
    assert "EDIT_PASSWORD" not in restored
    assert "CAMERA_URL" not in restored
    assert "TELEGRAM_BOT_TOKEN" not in restored
    assert restored["RETENTION_DAYS"] == 30
    assert not (destination / "cameras.yaml").exists()
    assert not (destination / "go2rtc.yaml").exists()


def test_atomic_journal_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    calls = []
    original_fsync = recovery_core.os.fsync

    def tracked_fsync(descriptor):
        calls.append(stat.S_ISDIR(recovery_core.os.fstat(descriptor).st_mode))
        original_fsync(descriptor)

    monkeypatch.setattr(recovery_core.os, "fsync", tracked_fsync)

    journal = tmp_path / "journal.json"
    recovery_core._write_json_atomic(journal, {"phase": "test"})

    assert calls[0] is False
    assert calls[-1] is True
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600


def test_recovery_orders_durability_barriers_around_renames(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "usb")
    destination = tmp_path / "output"
    destination.mkdir()
    events = []
    original_write = recovery_core._write_json_atomic
    original_rename = Path.rename

    def tracked_write(path, payload):
        events.append(f"journal:{payload['phase']}")
        original_write(path, payload)

    def tracked_rename(path, target):
        events.append(f"rename:{path.name}->{target.name}")
        return original_rename(path, target)

    monkeypatch.setattr(recovery_core, "_write_json_atomic", tracked_write)
    monkeypatch.setattr(
        recovery_core,
        "_fsync_tree",
        lambda path: events.append(f"fsync-tree:{path.name}"),
    )
    monkeypatch.setattr(
        recovery_core,
        "_fsync_directory",
        lambda path: events.append(f"fsync-dir:{path.name}"),
    )
    monkeypatch.setattr(Path, "rename", tracked_rename)

    recovery_core.recover_snapshot(snapshot, destination, mode="recovery", force=True)

    staging_sync = next(
        index
        for index, event in enumerate(events)
        if event.startswith("fsync-tree:.output-restore-")
    )
    verified = events.index("journal:verified")
    prepared = events.index("journal:checkpoint_prepared")
    checkpoint_rename = next(
        index
        for index, event in enumerate(events)
        if event.startswith("rename:output->output-before-restore-")
    )
    moved = events.index("journal:checkpoint_moved")
    publish_rename = next(
        index
        for index, event in enumerate(events)
        if event.startswith("rename:.output-restore-") and event.endswith("->output")
    )
    published = events.index("journal:published")

    assert staging_sync < verified
    assert prepared < checkpoint_rename < moved
    assert "fsync-dir:" + tmp_path.name in events[checkpoint_rename + 1 : moved]
    assert publish_rename < published
    assert "fsync-dir:" + tmp_path.name in events[publish_rename + 1 : published]


def test_low_space_refuses_before_staging(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path / "usb")
    destination = tmp_path / "output"
    destination.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        recovery_core.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(recovery_core.RecoveryError, match="free space") as raised:
        recovery_core.recover_snapshot(snapshot, destination, mode="migration")

    assert raised.value.code == "low_space"
    assert not destination.exists()
    assert not list(tmp_path.glob(".output-restore-*"))


def test_interrupted_checkpoint_move_is_rolled_back(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    staging = tmp_path / ".output-restore-test"
    checkpoint.mkdir()
    staging.mkdir()
    (checkpoint / "old").write_text("safe")
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "checkpoint_moved",
                "destination": str(destination),
                "staging": str(staging),
                "checkpoint": str(checkpoint),
            }
        )
    )

    message = recovery_core.reconcile_interrupted_recovery(destination)

    assert "rolled back" in message
    assert (destination / "old").read_text() == "safe"
    assert not staging.exists()
    assert not journal.exists()


def test_interrupted_checkpoint_prepare_after_rename_is_rolled_back(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    staging = tmp_path / ".output-restore-test"
    checkpoint.mkdir()
    staging.mkdir()
    (checkpoint / "old").write_text("safe")
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "checkpoint_prepared",
                "destination": str(destination),
                "staging": str(staging),
                "checkpoint": str(checkpoint),
            }
        )
    )

    recovery_core.reconcile_interrupted_recovery(destination)

    assert (destination / "old").read_text() == "safe"
    assert not staging.exists()
    assert not journal.exists()


def test_interrupted_after_publication_before_journal_update_is_kept(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    staging = tmp_path / ".output-restore-test"
    destination.mkdir()
    checkpoint.mkdir()
    (destination / "new").write_text("restored")
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "checkpoint_moved",
                "destination": str(destination),
                "staging": str(staging),
                "checkpoint": str(checkpoint),
            }
        )
    )

    recovery_core.reconcile_interrupted_recovery(destination)

    assert (destination / "new").read_text() == "restored"
    assert checkpoint.exists()
    assert not journal.exists()


def test_interrupted_published_swap_keeps_checkpoint(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    staging = tmp_path / ".output-restore-test"
    destination.mkdir()
    checkpoint.mkdir()
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "published",
                "destination": str(destination),
                "staging": str(staging),
                "checkpoint": str(checkpoint),
            }
        )
    )

    recovery_core.reconcile_interrupted_recovery(destination)

    assert destination.exists()
    assert checkpoint.exists()
    assert not journal.exists()


def test_interrupted_rollback_after_current_move_completes_checkpoint(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    failed = tmp_path / ".output-restore-failed-test"
    checkpoint.mkdir()
    failed.mkdir()
    (checkpoint / "state").write_text("previous", encoding="utf-8")
    (failed / "state").write_text("restored", encoding="utf-8")
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "rollback_current_moved",
                "destination": str(destination),
                "staging": str(failed),
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    recovery_core.reconcile_interrupted_recovery(destination)

    assert (destination / "state").read_text() == "previous"
    assert (failed / "state").read_text() == "restored"
    assert not checkpoint.exists()
    assert not journal.exists()


def test_interrupted_rollback_before_first_move_keeps_current_data(tmp_path):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    failed = tmp_path / ".output-restore-failed-test"
    destination.mkdir()
    checkpoint.mkdir()
    (destination / "state").write_text("restored", encoding="utf-8")
    journal = tmp_path / ".output-recovery-journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "rollback_prepared",
                "destination": str(destination),
                "staging": str(failed),
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    recovery_core.reconcile_interrupted_recovery(destination)

    assert (destination / "state").read_text() == "restored"
    assert checkpoint.exists()
    assert not journal.exists()


def test_runner_rejects_unsafe_snapshot_identifier_and_destination(
    tmp_path, monkeypatch
):
    usb = tmp_path / "usb"
    _snapshot(usb)
    _point_usb_core(monkeypatch, usb)
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", tmp_path / "output")
    base = {
        "schema_version": 1,
        "job_id": "a" * 32,
        "token": "t" * 40,
        "snapshot_id": "../escape",
        "destination": str(tmp_path / "output"),
        "mode": "migration",
    }
    with pytest.raises(recovery_core.RecoveryError) as unsafe_snapshot:
        recovery_runner.RecoveryRunner.validate_request(base)
    assert unsafe_snapshot.value.code == "snapshot_missing"

    symlink_name = "20260915_130000_manual"
    (usb / "snapshots" / symlink_name).symlink_to(
        usb / "snapshots" / "20260915_120000_manual", target_is_directory=True
    )
    base["snapshot_id"] = symlink_name
    with pytest.raises(recovery_core.RecoveryError) as unsafe_symlink:
        recovery_runner.RecoveryRunner.validate_request(base)
    assert unsafe_symlink.value.code == "snapshot_missing"

    base["snapshot_id"] = "20260915_120000_manual"
    base["destination"] = str(tmp_path / "other")
    with pytest.raises(recovery_core.RecoveryError) as unsafe_destination:
        recovery_runner.RecoveryRunner.validate_request(base)
    assert unsafe_destination.value.code == "unsafe_destination"


def test_runner_serializes_concurrent_maintenance(tmp_path, monkeypatch):
    lock = tmp_path / "maintenance.lock"
    monkeypatch.setattr(recovery_runner, "MAINTENANCE_LOCK", lock)
    request = {
        "job_id": "a" * 32,
        "token": "t" * 40,
        "snapshot_id": "snapshot",
        "destination": str(tmp_path / "output"),
        "mode": "migration",
    }
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", tmp_path / "jobs")
    runner = recovery_runner.RecoveryRunner(request)
    with lock.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(recovery_core.RecoveryError) as busy:
            runner._acquire_maintenance_lock()
    assert busy.value.code == "operation_busy"


def test_runner_surfaces_restart_failure_with_checkpoint_actions(tmp_path, monkeypatch):
    usb = tmp_path / "usb"
    snapshot = _snapshot(usb)
    _point_usb_core(monkeypatch, usb)
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "images.db").write_bytes(b"old")
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        recovery_runner, "MAINTENANCE_LOCK", tmp_path / "maintenance.lock"
    )
    request = {
        "job_id": "b" * 32,
        "token": "t" * 40,
        "snapshot_id": snapshot.name,
        "destination": str(destination),
        "mode": "recovery",
    }
    commands = []
    runner = recovery_runner.RecoveryRunner(
        request,
        run_command=lambda argv, **_kwargs: commands.append(argv),
        health_probe=lambda: False,
    )
    monkeypatch.setattr(runner, "_start_and_check", lambda: False)

    runner.run()

    assert commands[0] == ["systemctl", "stop", "app.service"]
    assert runner.status["state"] == "failed"
    assert runner.status["error_code"] == "restart_failed"
    assert runner.status["checkpoint_available"] is True
    assert Path(runner.status["checkpoint"]).is_dir()


def test_runner_restarts_safe_installation_after_recovery_failure(
    tmp_path, monkeypatch
):
    usb = tmp_path / "usb"
    snapshot = _snapshot(usb)
    _point_usb_core(monkeypatch, usb)
    destination = tmp_path / "output"
    destination.mkdir()
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        recovery_runner, "MAINTENANCE_LOCK", tmp_path / "maintenance.lock"
    )
    monkeypatch.setattr(
        recovery_core,
        "recover_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            recovery_core.RecoveryError("staging_failed", "Staging failed safely.")
        ),
    )
    commands = []
    runner = recovery_runner.RecoveryRunner(
        {
            "job_id": "c" * 32,
            "token": "t" * 40,
            "snapshot_id": snapshot.name,
            "destination": str(destination),
            "mode": "migration",
        },
        run_command=lambda argv, **_kwargs: commands.append(argv),
        health_probe=lambda: True,
    )

    runner.run()

    assert commands == [
        ["systemctl", "stop", "app.service"],
        ["systemctl", "start", "app.service"],
    ]
    assert runner.status["state"] == "failed"
    assert runner.status["app_available"] is True
    assert "safe data state" in runner.status["message"]


def test_runner_resumes_published_job_with_health_check_only(tmp_path, monkeypatch):
    destination = tmp_path / "output"
    destination.mkdir()
    jobs = tmp_path / "jobs"
    job_id = "d" * 32
    status_dir = jobs / job_id
    status_dir.mkdir(parents=True)
    status = {
        "schema_version": 1,
        "job_id": job_id,
        "state": "running",
        "stage": "published",
        "percent": 90,
        "message": "Published",
        "checkpoint": None,
        "checkpoint_available": False,
        "created_at": "2026-09-15T12:00:00+00:00",
        "updated_at": "2026-09-15T12:01:00+00:00",
    }
    (status_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", jobs)
    monkeypatch.setattr(
        recovery_runner, "MAINTENANCE_LOCK", tmp_path / "maintenance.lock"
    )
    monkeypatch.setattr(
        recovery_core,
        "recover_snapshot",
        lambda *_args, **_kwargs: pytest.fail("published recovery was rerun"),
    )
    commands = []
    runner = recovery_runner.RecoveryRunner(
        {
            "job_id": job_id,
            "token": "t" * 40,
            "snapshot_id": "no-longer-needed",
            "destination": str(destination),
            "mode": "recovery",
        },
        run_command=lambda argv, **_kwargs: commands.append(argv),
        health_probe=lambda: True,
    )

    runner.run()

    assert commands == [["systemctl", "start", "app.service"]]
    assert runner.status["state"] == "succeeded"
    assert runner.status["created_at"] == status["created_at"]


def test_resumed_restart_failure_keeps_retry_visible(tmp_path, monkeypatch):
    destination = tmp_path / "output"
    destination.mkdir()
    jobs = tmp_path / "jobs"
    job_id = "f" * 32
    status_dir = jobs / job_id
    status_dir.mkdir(parents=True)
    (status_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "state": "running",
                "stage": "restarting_app",
                "percent": 94,
                "message": "Restarting",
                "checkpoint": None,
                "checkpoint_available": False,
                "created_at": "2026-09-15T12:00:00+00:00",
                "updated_at": "2026-09-15T12:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", jobs)
    monkeypatch.setattr(
        recovery_runner, "MAINTENANCE_LOCK", tmp_path / "maintenance.lock"
    )
    runner = recovery_runner.RecoveryRunner(
        {
            "job_id": job_id,
            "token": "t" * 40,
            "snapshot_id": "snapshot",
            "destination": str(destination),
            "mode": "recovery",
        },
        health_probe=lambda: False,
    )
    monkeypatch.setattr(runner, "_start_and_check", lambda: False)

    runner.run()

    assert runner.status["state"] == "failed"
    assert runner.status["error_code"] == "restart_failed"
    assert runner.status["app_available"] is False


def test_failed_job_is_reopened_after_reboot_without_usb(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    job_id = "1" * 32
    job_dir = jobs / job_id
    job_dir.mkdir(parents=True)
    destination = tmp_path / "output"
    request = {
        "schema_version": 1,
        "job_id": job_id,
        "token": "t" * 40,
        "snapshot_id": "removed-snapshot",
        "destination": str(destination),
        "mode": "recovery",
    }
    (job_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": "failed",
                "stage": "failed",
                "updated_at": "2026-09-15T12:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recovery_runner, "REQUEST_PATH", tmp_path / "request.json")
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", jobs)
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)

    assert recovery_runner.load_request() == request


def test_rollback_refuses_to_swap_data_when_app_stop_fails(tmp_path, monkeypatch):
    destination = tmp_path / "output"
    checkpoint = tmp_path / "output-before-restore-test"
    destination.mkdir()
    checkpoint.mkdir()
    (destination / "state").write_text("restored", encoding="utf-8")
    (checkpoint / "state").write_text("previous", encoding="utf-8")
    monkeypatch.setattr(recovery_runner, "EXPECTED_DESTINATION", destination)
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        recovery_runner, "MAINTENANCE_LOCK", tmp_path / "maintenance.lock"
    )
    runner = recovery_runner.RecoveryRunner(
        {
            "job_id": "2" * 32,
            "token": "t" * 40,
            "snapshot_id": "snapshot",
            "destination": str(destination),
            "mode": "recovery",
        }
    )
    runner.update(
        state="failed",
        checkpoint=str(checkpoint),
        checkpoint_available=True,
        app_available=False,
    )

    def stop_fails(action):
        assert action == "stop"
        raise recovery_runner.subprocess.CalledProcessError(1, ["systemctl", "stop"])

    monkeypatch.setattr(runner, "_systemctl", stop_fails)

    assert runner.roll_back() is False
    assert (destination / "state").read_text() == "restored"
    assert (checkpoint / "state").read_text() == "previous"
    assert runner.status["error_code"] == "app_stop_failed"


def test_systemd_units_allow_requests_and_gate_app_start_on_reconciliation():
    app_unit = Path("systemd/app.service").read_text(encoding="utf-8")
    recovery_unit = Path("rpi/systemd/wmb-recovery.service").read_text(encoding="utf-8")

    assert "/var/lib/watchmybirds-recovery/incoming" in app_unit
    assert "recovery_runner.py --reconcile-only" in app_unit
    assert "ExecStartPre=" in recovery_unit
    assert "recovery_runner.py --reconcile-only" in recovery_unit


def test_authorized_progress_poll_accepts_browser_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery_runner, "JOBS_DIR", tmp_path / "jobs")
    runner = recovery_runner.RecoveryRunner(
        {
            "job_id": "e" * 32,
            "token": "secret-token-" + "t" * 32,
            "snapshot_id": "snapshot",
            "destination": str(tmp_path / "output"),
            "mode": "migration",
        }
    )
    accepted = threading.Event()
    handler_class = recovery_runner.make_handler(runner, accepted)
    handler = object.__new__(handler_class)
    handler.path = f"/api/status?token={runner.token}"
    sent = []
    handler._send = lambda status, body, content_type: sent.append(
        (status, body, content_type)
    )

    handler.do_GET()

    assert sent[0][0] == recovery_runner.HTTPStatus.OK
    assert accepted.is_set()


def test_web_submission_requires_confirmation_and_stages_fixed_request(
    tmp_path, monkeypatch
):
    from web.services import recovery_service

    incoming = tmp_path / "state" / "incoming"
    monkeypatch.setattr(recovery_service, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(recovery_service, "INCOMING_DIR", incoming)
    monkeypatch.setattr(recovery_service, "JOBS_DIR", tmp_path / "state" / "jobs")
    monkeypatch.setattr(recovery_service, "REQUEST_PATH", incoming / "request.json")
    monkeypatch.setattr(recovery_service, "is_supported", lambda: True)
    monkeypatch.setattr(
        recovery_service,
        "preview_snapshot",
        lambda _name: {"blockers": [], "mode": "migration"},
    )
    monkeypatch.setattr(
        recovery_service, "get_config", lambda: {"OUTPUT_DIR": str(tmp_path / "output")}
    )
    commands = []
    monkeypatch.setattr(
        recovery_service.subprocess,
        "run",
        lambda argv, **_kwargs: commands.append(argv),
    )

    with pytest.raises(recovery_core.RecoveryError) as unconfirmed:
        recovery_service.start_recovery(
            "snapshot", replace_confirmed=False, checkpoint_acknowledged=True
        )
    assert unconfirmed.value.code == "confirmation_required"

    result = recovery_service.start_recovery(
        "snapshot", replace_confirmed=True, checkpoint_acknowledged=True
    )

    request = json.loads((incoming / "request.json").read_text())
    assert request["snapshot_id"] == "snapshot"
    assert request["destination"] == str((tmp_path / "output").resolve())
    assert request["token"] == result["token"]
    assert commands == [["systemctl", "restart", "wmb-recovery.service"]]


def test_recovery_api_requires_authenticated_session(monkeypatch):
    from flask import Flask

    from web.blueprints.api_v1 import api_v1
    from web.blueprints.auth import auth_bp

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_v1)
    client = app.test_client()

    response = client.get("/api/v1/system/recovery/snapshot/preview")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]

    monkeypatch.setattr(
        "web.services.recovery_service.preview_snapshot",
        lambda _name: {"compatible": True, "blockers": []},
    )
    with client.session_transaction() as session:
        session["authenticated"] = True
    response = client.get("/api/v1/system/recovery/snapshot/preview")
    assert response.status_code == 200
    assert response.get_json()["compatible"] is True
