"""Authenticated web orchestration for the independent Pi recovery runner."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from config import get_config
from core import recovery_core, usb_backup_core
from logging_config import get_logger

logger = get_logger(__name__)

SERVICE_UNIT = "wmb-recovery.service"
RUNNER_PORT = 8051
STATE_ROOT = Path("/var/lib/watchmybirds-recovery")
INCOMING_DIR = STATE_ROOT / "incoming"
JOBS_DIR = STATE_ROOT / "jobs"
REQUEST_PATH = INCOMING_DIR / "request.json"
TRIGGER_LOCK = threading.Lock()


def is_supported() -> bool:
    """Guided orchestration is intentionally advertised for Pi installs only."""
    return Path("/etc/systemd/system/wmb-recovery.service").is_file()


def preview_snapshot(name: str) -> dict[str, Any]:
    snapshot = usb_backup_core.get_snapshot_directory(name)
    if snapshot is None:
        raise recovery_core.RecoveryError(
            "snapshot_missing", "The selected backup was not found."
        )
    destination = Path(str(get_config()["OUTPUT_DIR"]))
    preview = recovery_core.inspect_snapshot(snapshot, destination)
    preview["settings_preserved"] = recovery_core.settings_policy_labels(
        preview["settings_preserved"]
    )
    preview["settings_restored_count"] = len(preview.pop("settings_restored"))
    preview["guided_supported"] = is_supported()
    return preview


def _read_status() -> dict[str, Any] | None:
    try:
        candidates = sorted(
            JOBS_DIR.glob("*/status.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def start_recovery(
    name: str,
    *,
    replace_confirmed: bool,
    checkpoint_acknowledged: bool,
) -> dict[str, Any]:
    """Stage one fixed-format request and start the root-owned runner."""
    if not replace_confirmed or not checkpoint_acknowledged:
        raise recovery_core.RecoveryError(
            "confirmation_required",
            "Confirm that current data will be replaced and a checkpoint retained.",
        )
    if not is_supported():
        raise recovery_core.RecoveryError(
            "unsupported_deployment",
            "Guided recovery is currently available on Raspberry Pi appliances only. Use the documented CLI fallback on this deployment.",
        )
    preview = preview_snapshot(name)
    if preview["blockers"]:
        raise recovery_core.RecoveryError(
            "preflight_failed", "; ".join(preview["blockers"])
        )

    if not TRIGGER_LOCK.acquire(blocking=False):
        raise recovery_core.RecoveryError(
            "operation_busy", "Another maintenance operation is starting."
        )
    try:
        current = _read_status()
        if current and current.get("state") not in {
            "succeeded",
            "failed",
            "rolled_back",
        }:
            raise recovery_core.RecoveryError(
                "operation_busy", "Another recovery is already running."
            )
        if REQUEST_PATH.exists():
            raise recovery_core.RecoveryError(
                "operation_busy",
                "A recovery request is already waiting for the runner.",
            )

        job_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        payload = {
            "schema_version": 1,
            "job_id": job_id,
            "token": token,
            "snapshot_id": name,
            "destination": str(Path(str(get_config()["OUTPUT_DIR"])).resolve()),
            "mode": preview["mode"],
        }
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        temp = REQUEST_PATH.with_name(f".request-{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        temp.chmod(0o640)
        temp.replace(REQUEST_PATH)
        action = (
            "restart"
            if current
            and current.get("state") in {"succeeded", "failed", "rolled_back"}
            else "start"
        )
        try:
            subprocess.run(
                ["systemctl", action, SERVICE_UNIT],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            REQUEST_PATH.unlink(missing_ok=True)
            logger.error("Could not start recovery service: %s", exc)
            raise recovery_core.RecoveryError(
                "runner_start_failed",
                "The recovery runner could not start. Your data was not changed.",
            ) from exc
        return {
            "job_id": job_id,
            "token": token,
            "runner_port": RUNNER_PORT,
            "mode": preview["mode"],
        }
    finally:
        TRIGGER_LOCK.release()
