"""Execute backup failure paths with fake hardware and synthetic data."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def backup_harness(tmp_path):
    root = Path(__file__).resolve().parents[1]
    app = tmp_path / "app"
    output = app / "data/output"
    output.mkdir(parents=True)
    (app / ".venv/bin").mkdir(parents=True)
    (app / ".venv/bin/python").symlink_to(sys.executable)
    (app / "scripts").symlink_to(root / "scripts", target_is_directory=True)
    mount = tmp_path / "usb"
    mount.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    stubs = {
        "mountpoint": "exit 0",
        "findmnt": "echo ext4",
        "stat": 'case "$2" in "%U") echo watchmybirds;; *) echo 4096;; esac',
        "df": 'printf "Filesystem 1B-blocks Used Available Use%% Mounted\\nfixture 99999999999 0 99999999999 0 fixture\\n"',
        "flock": "exit 0",
        "du": 'echo "4096 fixture"',
        "sync": 'if [ -f "$READY_MARKER" ]; then exit 1; fi; touch "$READY_MARKER"',
        "rsync": """case " $* " in
            *--dry-run*)
                if [ "$ESTIMATE_FAIL" = 1 ]; then exit 23; fi
                echo "Total transferred file size: ${ESTIMATE_BYTES:-0} bytes";;
            *)
                if [ "$WAIT_FOR_SIGNAL" = 1 ]; then
                    touch "$READY_MARKER"
                    sleep 30
                fi;;
        esac""",
    }
    for name, body in stubs.items():
        path = commands / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    script = tmp_path / "backup.sh"
    script.write_text(
        (root / "rpi/backup.sh")
        .read_text()
        .replace(
            'readonly MOUNT_POINT="/mnt/wmb-backup"', f'readonly MOUNT_POINT="{mount}"'
        )
        .replace('readonly APP_DIR="/opt/app"', f'readonly APP_DIR="{app}"')
        .replace(
            "/run/lock/watchmybirds/maintenance.lock",
            str(tmp_path / "maintenance.lock"),
        )
    )
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "ESTIMATE_FAIL": "0",
        "WAIT_FOR_SIGNAL": "0",
        "READY_MARKER": str(tmp_path / "ready"),
    }
    return script, mount, output, env


def run_backup(harness):
    script, _, _, env = harness
    return subprocess.run(
        ["bash", str(script), "--kind", "manual"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_failed_estimate_aborts_instead_of_assuming_zero(backup_harness):
    _, mount, _, env = backup_harness
    env["ESTIMATE_FAIL"] = "1"
    result = run_backup(backup_harness)
    assert result.returncode == 12
    assert not list(mount.rglob("COMPLETED"))
    assert json.loads((mount / "LAST_RUN_STATUS.json").read_text())["exit_code"] == 12


def test_large_estimate_with_bytes_suffix_is_enforced(backup_harness):
    _, mount, _, env = backup_harness
    env["ESTIMATE_BYTES"] = "999,999,999,999"
    assert run_backup(backup_harness).returncode == 12
    assert not list(mount.rglob("COMPLETED"))


def test_missing_database_cannot_publish_snapshot(backup_harness):
    _, mount, _, _ = backup_harness
    assert run_backup(backup_harness).returncode == 13
    assert not list(mount.rglob("COMPLETED"))


def test_capture_missing_from_copy_cannot_publish_snapshot(backup_harness):
    _, mount, output, _ = backup_harness
    with sqlite3.connect(output / "images.db") as conn:
        conn.execute("CREATE TABLE images(filename TEXT, original_present INTEGER)")
        conn.execute("INSERT INTO images VALUES ('20260915_120000_new.jpg', 1)")
    result = run_backup(backup_harness)
    assert result.returncode == 17, result.stderr
    assert not list(mount.rglob("COMPLETED"))
    assert list(mount.rglob("CORRUPT"))


def test_term_records_failure(backup_harness):
    script, mount, _, env = backup_harness
    env["WAIT_FOR_SIGNAL"] = "1"
    proc = subprocess.Popen(
        ["bash", str(script), "--kind", "manual"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not Path(env["READY_MARKER"]).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert Path(env["READY_MARKER"]).exists()
        status = json.loads((mount / "LAST_RUN_STATUS.json").read_text())
        assert status["status"] == "running"
        assert status["stage"] == "copying_images"
        assert status["finished_at"] is None
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
        assert proc.returncode == 143
        assert (
            json.loads((mount / "LAST_RUN_STATUS.json").read_text())["status"]
            == "failed"
        )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_failed_completion_flush_removes_completed_marker(backup_harness):
    _, mount, output, _ = backup_harness
    with sqlite3.connect(output / "images.db") as conn:
        conn.execute("CREATE TABLE images(filename TEXT, original_present INTEGER)")
    result = run_backup(backup_harness)
    assert result.returncode == 17, result.stderr
    assert "Completion flush failed" in result.stderr
    assert not list(mount.rglob("COMPLETED"))
    assert list(mount.rglob("CORRUPT"))


def test_copy_error_is_retained_on_usb(backup_harness):
    _, mount, _, env = backup_harness
    command = Path(env["PATH"].split(":")[0]) / "rsync"
    command.write_text("""#!/bin/sh
case " $* " in
*--dry-run*) echo 'Total transferred file size: 0 bytes';;
*) echo 'fixture unreadable file' >&2; exit 23;;
esac
""")
    result = run_backup(backup_harness)
    assert result.returncode == 14
    assert "fixture unreadable file" in (mount / "BACKUP_LOG.txt").read_text()
    status = json.loads((mount / "LAST_RUN_STATUS.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "copying_images"
