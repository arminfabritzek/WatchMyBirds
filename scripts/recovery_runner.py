#!/usr/bin/env python3
"""Narrow root runner and independent progress server for Pi recovery.

The unprivileged app may only start the systemd unit and write one constrained
request. This process revalidates the snapshot identifier, destination, mode,
and maintenance lock before stopping app.service.
"""

from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import recovery_core, usb_backup_core  # noqa: E402

STATE_ROOT = Path(
    os.environ.get("WMB_RECOVERY_STATE_DIR", "/var/lib/watchmybirds-recovery")
)
REQUEST_PATH = STATE_ROOT / "incoming" / "request.json"
JOBS_DIR = STATE_ROOT / "jobs"
MAINTENANCE_LOCK = Path("/run/lock/watchmybirds/maintenance.lock")
EXPECTED_DESTINATION = Path(
    os.environ.get("WMB_RECOVERY_DESTINATION", "/opt/app/data/output")
)
APP_UNIT = "app.service"
HEALTH_URL = "http://127.0.0.1:8050/healthz"
CLOSED_STATES = {"succeeded", "rolled_back"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any], mode: int = 0o640) -> None:
    recovery_core.write_json_durable(path, payload, mode=mode)


class RecoveryRunner:
    """One durable recovery job with injectable process and health hooks."""

    def __init__(
        self,
        request: dict[str, Any],
        *,
        run_command=subprocess.run,
        health_probe: Any | None = None,
    ) -> None:
        self.request = request
        self.job_id = request["job_id"]
        self.token = request["token"]
        self.job_dir = JOBS_DIR / self.job_id
        self.status_path = self.job_dir / "status.json"
        self._run_command = run_command
        self._health_probe = health_probe or self._default_health_probe
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._maintenance_handle: Any | None = None
        try:
            existing = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        self.status: dict[str, Any] = (
            existing
            if isinstance(existing, dict) and existing.get("job_id") == self.job_id
            else {
                "schema_version": 1,
                "job_id": self.job_id,
                "state": "queued",
                "stage": "queued",
                "percent": 0,
                "message": "Recovery request accepted.",
                "error_code": None,
                "checkpoint_available": False,
                "checkpoint": None,
                "app_available": False,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        )
        self._write_status()

    @staticmethod
    def validate_request(
        payload: Any, *, require_snapshot: bool = True
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise recovery_core.RecoveryError(
                "invalid_request", "Recovery request format is invalid."
            )
        required = {"job_id", "token", "snapshot_id", "destination", "mode"}
        if set(payload) != required | {"schema_version"}:
            raise recovery_core.RecoveryError(
                "invalid_request", "Recovery request fields are invalid."
            )
        if (
            not isinstance(payload["job_id"], str)
            or len(payload["job_id"]) != 32
            or not payload["job_id"].isalnum()
        ):
            raise recovery_core.RecoveryError(
                "invalid_request", "Recovery job identifier is invalid."
            )
        if not isinstance(payload["token"], str) or len(payload["token"]) < 32:
            raise recovery_core.RecoveryError(
                "invalid_request", "Recovery access token is invalid."
            )
        if payload["mode"] not in {"migration", "recovery"}:
            raise recovery_core.RecoveryError(
                "invalid_request", "Recovery mode is invalid."
            )
        if (
            Path(str(payload["destination"])).resolve()
            != EXPECTED_DESTINATION.resolve()
        ):
            raise recovery_core.RecoveryError(
                "unsafe_destination",
                "Recovery destination is not the appliance data directory.",
            )
        if not usb_backup_core.is_safe_snapshot_identifier(payload["snapshot_id"]):
            raise recovery_core.RecoveryError(
                "snapshot_missing", "The selected backup identifier is unsafe."
            )
        if (
            require_snapshot
            and usb_backup_core.get_snapshot_directory(payload["snapshot_id"]) is None
        ):
            raise recovery_core.RecoveryError(
                "snapshot_missing", "The selected backup is unavailable or unsafe."
            )
        return payload

    def _write_status(self) -> None:
        self.status["updated_at"] = _utc_now()
        _atomic_json(self.status_path, self.status)

    def update(self, **changes: Any) -> None:
        with self._state_lock:
            self.status.update(changes)
            self._write_status()

    def public_status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                key: value for key, value in self.status.items() if key != "checkpoint"
            }

    def authorized(self, token: str) -> bool:
        return hmac.compare_digest(token.encode("utf-8"), self.token.encode("utf-8"))

    def _acquire_maintenance_lock(self) -> None:
        MAINTENANCE_LOCK.parent.mkdir(parents=True, exist_ok=True)
        handle = MAINTENANCE_LOCK.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise recovery_core.RecoveryError(
                "operation_busy",
                "Another backup, format, or recovery operation is running. Wait for it to finish and try again.",
            ) from exc
        self._maintenance_handle = handle

    def _release_maintenance_lock(self) -> None:
        if self._maintenance_handle is not None:
            fcntl.flock(self._maintenance_handle.fileno(), fcntl.LOCK_UN)
            self._maintenance_handle.close()
            self._maintenance_handle = None

    def _systemctl(self, action: str) -> None:
        self._run_command(
            ["systemctl", action, APP_UNIT],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

    @staticmethod
    def _default_health_probe() -> bool:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:  # noqa: S310 -- fixed loopback URL
                return response.status == 200 and response.read(32).strip() == b"ok"
        except (OSError, urllib.error.URLError):
            return False

    def _start_and_check(self) -> bool:
        try:
            self._systemctl("start")
        except (OSError, subprocess.SubprocessError):
            return False
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._health_probe():
                return True
            time.sleep(1)
        return False

    def _latest_checkpoint(self) -> Path | None:
        candidates = sorted(
            EXPECTED_DESTINATION.parent.glob(
                f"{EXPECTED_DESTINATION.name}-before-restore-*"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def run(self) -> None:
        with self._operation_lock:
            self._run()

    def retry_start(self) -> bool:
        if not self._operation_lock.acquire(blocking=False):
            return False
        try:
            self._acquire_maintenance_lock()
            return self._retry_start()
        except (OSError, recovery_core.RecoveryError):
            return False
        finally:
            self._release_maintenance_lock()
            self._operation_lock.release()

    def roll_back(self) -> bool:
        if not self._operation_lock.acquire(blocking=False):
            return False
        try:
            return self._roll_back()
        finally:
            self._operation_lock.release()

    def _run(self) -> None:
        checkpoint: Path | None = None
        app_stopped = False
        try:
            self._acquire_maintenance_lock()
            if self.status.get("stage") in {"published", "restarting_app"}:
                self._systemctl("stop")
                app_stopped = True
                recovery_core.reconcile_interrupted_recovery(EXPECTED_DESTINATION)
                checkpoint_raw = self.status.get("checkpoint")
                checkpoint = (
                    Path(checkpoint_raw)
                    if checkpoint_raw
                    else self._latest_checkpoint()
                )
                self.update(
                    state="running",
                    stage="restarting_app",
                    percent=94,
                    message="Recovered data is safe. Restarting WatchMyBirds and checking it…",
                    checkpoint=str(checkpoint) if checkpoint else None,
                    checkpoint_available=bool(checkpoint),
                )
                if not self._start_and_check():
                    raise recovery_core.RecoveryError(
                        "restart_failed",
                        "The data recovery completed, but WatchMyBirds did not start successfully. Retry startup or roll back to the retained checkpoint.",
                    )
                self.update(
                    state="succeeded",
                    stage="complete",
                    percent=100,
                    message="Recovery complete. WatchMyBirds is healthy and ready.",
                    app_available=True,
                )
                return
            snapshot = usb_backup_core.get_snapshot_directory(
                self.request["snapshot_id"]
            )
            if snapshot is None:
                raise recovery_core.RecoveryError(
                    "snapshot_missing",
                    "The selected backup was removed before recovery began.",
                )
            self.update(
                state="running",
                stage="stopping_app",
                percent=3,
                message="Stopping database users safely…",
            )
            self._systemctl("stop")
            app_stopped = True
            self.update(
                stage="recovering", percent=5, message="Starting verified recovery…"
            )
            result = recovery_core.recover_snapshot(
                snapshot,
                EXPECTED_DESTINATION,
                mode=self.request["mode"],
                force=self.request["mode"] == "recovery",
                progress=lambda stage, percent, message: self.update(
                    state="running", stage=stage, percent=percent, message=message
                ),
            )
            checkpoint = result.checkpoint
            self.update(
                stage="restarting_app",
                percent=94,
                message="Recovered data is safe. Restarting WatchMyBirds and checking it…",
                checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_available=bool(checkpoint),
            )
            if not self._start_and_check():
                raise recovery_core.RecoveryError(
                    "restart_failed",
                    "The data recovery completed, but WatchMyBirds did not start successfully. Retry startup or roll back to the retained checkpoint.",
                )
            self.update(
                state="succeeded",
                stage="complete",
                percent=100,
                message="Recovery complete. WatchMyBirds is healthy and ready.",
                app_available=True,
            )
        except recovery_core.RecoveryError as exc:
            app_available = (
                False
                if exc.code == "restart_failed"
                else self._start_and_check()
                if app_stopped
                else self._health_probe()
            )
            message = str(exc)
            if app_available:
                message += " WatchMyBirds is available with the safe data state."
            self.update(
                state="failed",
                stage="failed",
                message=message,
                error_code=exc.code,
                app_available=app_available,
                checkpoint=str(checkpoint)
                if checkpoint
                else self.status.get("checkpoint"),
                checkpoint_available=bool(checkpoint or self.status.get("checkpoint")),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            app_available = (
                self._start_and_check() if app_stopped else self._health_probe()
            )
            self.update(
                state="failed",
                stage="failed",
                message="Recovery stopped safely because a required system operation failed. Current data or a complete checkpoint was retained.",
                error_code="system_operation_failed",
                technical_error=type(exc).__name__,
                app_available=app_available,
                checkpoint=str(checkpoint)
                if checkpoint
                else self.status.get("checkpoint"),
                checkpoint_available=bool(checkpoint or self.status.get("checkpoint")),
            )
        finally:
            self._release_maintenance_lock()

    def _retry_start(self) -> bool:
        if self.status.get("state") != "failed":
            return False
        self.update(
            state="running",
            stage="restarting_app",
            message="Retrying WatchMyBirds startup…",
        )
        if self._start_and_check():
            self.update(
                state="succeeded",
                stage="complete",
                percent=100,
                message="WatchMyBirds started successfully. Recovery is complete.",
                error_code=None,
                app_available=True,
            )
            return True
        self.update(
            state="failed",
            stage="failed",
            message="WatchMyBirds still could not start. You can try again or roll back to the retained checkpoint.",
            error_code="restart_failed",
            app_available=False,
        )
        return False

    def _roll_back(self) -> bool:
        checkpoint_raw = self.status.get("checkpoint")
        if self.status.get("state") != "failed" or not checkpoint_raw:
            return False
        self.update(
            state="running",
            stage="rolling_back",
            message="Restoring the retained pre-recovery checkpoint…",
        )
        try:
            self._acquire_maintenance_lock()
            try:
                self._systemctl("stop")
            except (OSError, subprocess.SubprocessError) as exc:
                raise recovery_core.RecoveryError(
                    "app_stop_failed",
                    "WatchMyBirds could not be stopped, so rollback was refused and no data was changed.",
                ) from exc
            recovery_core.rollback_checkpoint(
                EXPECTED_DESTINATION, Path(checkpoint_raw)
            )
            if not self._start_and_check():
                raise recovery_core.RecoveryError(
                    "rollback_restart_failed",
                    "The checkpoint was restored, but WatchMyBirds still did not start. Keep the device powered on and use the recovery diagnostics from the SD card image support flow.",
                )
            self.update(
                state="rolled_back",
                stage="complete",
                percent=100,
                message="The previous installation checkpoint was restored and WatchMyBirds is healthy.",
                error_code=None,
                checkpoint_available=False,
                app_available=True,
            )
            return True
        except (OSError, recovery_core.RecoveryError) as exc:
            self.update(
                state="failed",
                stage="failed",
                message=str(exc),
                error_code=getattr(exc, "code", "rollback_failed"),
            )
            return False
        finally:
            self._release_maintenance_lock()


PROGRESS_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WatchMyBirds recovery</title><style>
:root{font-family:system-ui,sans-serif;color:#243235;background:#f2f5f3}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px;box-sizing:border-box}.card{box-sizing:border-box;width:min(680px,100%);background:white;border:1px solid #cad4d0;border-radius:18px;padding:clamp(22px,5vw,42px);box-shadow:0 18px 50px #18352b18}h1{margin-top:0;font-size:clamp(1.6rem,5vw,2.4rem)}progress{width:100%;height:16px}.status{font-size:1.08rem;min-height:3em}.hint{color:#60716d}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}button,a{border:0;border-radius:9px;padding:12px 18px;font:inherit;font-weight:650;cursor:pointer;background:#53676b;color:white;text-decoration:none}.secondary{background:#e5ebe8;color:#243235}.danger{background:#a83b35}button[hidden],a[hidden]{display:none}.error{color:#8b2d28}.ok{color:#267249}@media(max-width:520px){body{padding:0}.card{min-height:100vh;border-radius:0;box-shadow:none}.actions>*{width:100%;text-align:center}}
</style></head><body><main class="card"><p class="hint">WatchMyBirds · Guided USB recovery</p><h1 id="title">Preparing recovery…</h1><progress id="bar" max="100" value="0"></progress><p id="status" class="status" role="status" aria-live="polite">Connecting to the independent recovery runner…</p><p id="hint" class="hint">Keep the Raspberry Pi and USB stick powered. This page remains available while the main app restarts.</p><div class="actions"><button id="retry" hidden>Retry app startup</button><button id="rollback" class="danger" hidden>Restore previous checkpoint</button><a id="open" href="http://watchmybirds.local:8050/gallery" hidden>Open WatchMyBirds</a></div></main><script>
const token=location.hash.slice(1),statusEl=document.getElementById('status'),title=document.getElementById('title'),bar=document.getElementById('bar'),retry=document.getElementById('retry'),rollback=document.getElementById('rollback'),open=document.getElementById('open');open.href=location.protocol+'//'+location.hostname+':8050/gallery';
async function action(name){retry.disabled=rollback.disabled=true;try{await fetch('/api/'+name+'?token='+encodeURIComponent(token),{method:'POST'});}finally{setTimeout(poll,500)}}
retry.onclick=()=>action('retry');rollback.onclick=()=>action('rollback');
async function poll(){try{const r=await fetch('/api/status?token='+encodeURIComponent(token),{cache:'no-store'});if(!r.ok)throw Error();const s=await r.json();bar.value=s.percent||0;statusEl.textContent=s.message||'Working…';statusEl.className='status '+(s.state==='failed'?'error':(s.state==='succeeded'||s.state==='rolled_back'?'ok':''));title.textContent=s.state==='failed'?'Recovery needs attention':s.state==='succeeded'?'Recovery complete':s.state==='rolled_back'?'Previous installation restored':'Recovering your birds…';retry.hidden=!(s.state==='failed'&&!s.app_available);rollback.hidden=!(s.state==='failed'&&s.checkpoint_available);open.hidden=!(s.state==='succeeded'||s.state==='rolled_back'||(s.state==='failed'&&s.app_available));retry.disabled=rollback.disabled=false;if(!['failed','succeeded','rolled_back'].includes(s.state))setTimeout(poll,1000);}catch(e){statusEl.textContent='Reconnecting to the recovery runner…';setTimeout(poll,1200)}}
if(!token){title.textContent='Recovery link unavailable';statusEl.textContent='Return to WatchMyBirds Settings and start recovery again.'}else poll();
</script></body></html>"""


def make_handler(
    runner: RecoveryRunner, accepted: threading.Event | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _token(self) -> str:
            return parse_qs(urlparse(self.path).query).get("token", [""])[0]

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 -- stdlib handler contract
            path = urlparse(self.path).path
            if path == "/":
                self._send(
                    HTTPStatus.OK, PROGRESS_HTML.encode(), "text/html; charset=utf-8"
                )
                return
            if path == "/api/status" and runner.authorized(self._token()):
                if accepted is not None:
                    accepted.set()
                self._send(
                    HTTPStatus.OK,
                    json.dumps(runner.public_status()).encode(),
                    "application/json",
                )
                return
            self._send(
                HTTPStatus.FORBIDDEN, b'{"error":"forbidden"}', "application/json"
            )

        def do_POST(self) -> None:  # noqa: N802 -- stdlib handler contract
            path = urlparse(self.path).path
            if not runner.authorized(self._token()):
                self._send(
                    HTTPStatus.FORBIDDEN, b'{"error":"forbidden"}', "application/json"
                )
                return
            ok = (
                runner.retry_start()
                if path == "/api/retry"
                else runner.roll_back()
                if path == "/api/rollback"
                else False
            )
            self._send(
                HTTPStatus.OK if ok else HTTPStatus.CONFLICT,
                json.dumps(runner.public_status()).encode(),
                "application/json",
            )

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def load_request() -> dict[str, Any] | None:
    """Consume a new request or reopen the newest actionable durable job."""
    if not REQUEST_PATH.exists():
        try:
            requests = sorted(
                JOBS_DIR.glob("*/request.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            requests = []
        for request_path in requests:
            try:
                status = json.loads(
                    (request_path.parent / "status.json").read_text(encoding="utf-8")
                )
                request = RecoveryRunner.validate_request(
                    json.loads(request_path.read_text(encoding="utf-8")),
                    require_snapshot=False,
                )
                return None if status.get("state") in CLOSED_STATES else request
            except (OSError, json.JSONDecodeError, recovery_core.RecoveryError):
                continue
        return None
    if REQUEST_PATH.is_symlink() or not REQUEST_PATH.is_file():
        raise recovery_core.RecoveryError(
            "invalid_request", "No regular recovery request is waiting."
        )
    try:
        payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise recovery_core.RecoveryError(
            "invalid_request", "Recovery request could not be read."
        ) from exc
    validated = RecoveryRunner.validate_request(payload, require_snapshot=False)
    job_dir = JOBS_DIR / validated["job_id"]
    job_dir.mkdir(parents=True, exist_ok=False)
    job_dir.chmod(0o750)
    request_target = job_dir / "request.json"
    REQUEST_PATH.replace(request_target)
    request_target.chmod(0o600)
    with request_target.open("rb") as handle:
        os.fsync(handle.fileno())
    recovery_core._fsync_directory(REQUEST_PATH.parent)
    recovery_core._fsync_directory(job_dir)
    return validated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--reconcile-only", action="store_true")
    args = parser.parse_args()
    if args.reconcile_only:
        recovery_core.reconcile_interrupted_recovery(EXPECTED_DESTINATION)
        return 0
    request = load_request()
    if request is None:
        return 0
    runner = RecoveryRunner(request)
    try:
        recovery_core.reconcile_interrupted_recovery(EXPECTED_DESTINATION)
    except (OSError, recovery_core.RecoveryError) as exc:
        runner.update(
            state="failed",
            stage="failed",
            app_available=False,
            message=str(exc),
            error_code=getattr(exc, "code", "reconciliation_failed"),
        )
    accepted = threading.Event()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner, accepted))
    failed = runner.status.get("state") == "failed"
    resuming = runner.status.get("stage") != "queued" and not failed
    if failed:
        runner.update(app_available=runner._health_probe())

    def run_after_handoff() -> None:
        if resuming or accepted.wait(timeout=60):
            runner.run()
        else:
            runner.update(
                state="failed",
                stage="failed",
                message="The recovery page was not opened in time. Return to Settings and try again; no data was changed.",
                error_code="handoff_timeout",
            )

    if not failed:
        worker = threading.Thread(target=run_after_handoff, daemon=True)
        worker.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
