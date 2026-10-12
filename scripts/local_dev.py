"""One-way Pi data pull and isolated local WatchMyBirds web development.

Run with the repository's Python: scripts/local_dev.py --help.
No operation deploys code, modifies the live DB, or restarts the Pi.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Flask

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path.home() / "Library/Application Support/WatchMyBirds-dev"
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]
MEDIA_ROOTS = {"originals", "derivatives/thumbs", "derivatives/optimized"}
MARKER = ".wmb-local-dev"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def managed_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (
        root == REPO
        or root.is_relative_to(REPO)
        or any((parent / ".git").exists() for parent in (root, *root.parents))
    ):
        raise ValueError("development data must live outside the repository")
    if root in {Path("/"), Path.home(), Path("/tmp"), Path("/private/tmp")}:
        raise ValueError("choose a dedicated development directory")
    marker = root / MARKER
    if root.exists() and not marker.is_file() and any(root.iterdir()):
        raise ValueError("refusing to adopt a nonempty unmarked directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not marker.exists():
        marker.write_text("watchmybirds-local-development-v1\n")
    if (
        marker.is_symlink()
        or marker.read_text() != "watchmybirds-local-development-v1\n"
    ):
        raise ValueError("invalid development directory marker")
    root.chmod(0o700)
    return root


def child_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe relative path")
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path leaves its managed directory")
    return path


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    path = child_path(root, ".operation.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "local app or another data operation is running; stop it first"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, check=True, **kwargs)


def validate_target(target: str, remote_output: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", target):
        raise ValueError("SSH target must be user@host (without shell syntax)")
    if (
        not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_output)
        or ".." in Path(remote_output).parts
    ):
        raise ValueError("remote output must be an absolute path without traversal")


def verify_snapshot(output: Path, manifest: dict) -> None:
    if manifest.get("format") != 1:
        raise ValueError("unsupported manifest")
    database = child_path(output, "images.db")
    if digest(database) != manifest["database_sha256"]:
        raise ValueError("database checksum mismatch")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("database integrity check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("database foreign-key check failed")
        for table in ("images", "detections", "human_label_facts"):
            if table in manifest["counts"]:
                count = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                if count != manifest["counts"][table]:
                    raise ValueError("database row count mismatch")
    for name, metadata in manifest["files"].items():
        if not any(name.startswith(prefix + "/") for prefix in MEDIA_ROOTS):
            raise ValueError("manifest includes files outside bird media")
        path = child_path(output, name)
        if (
            not path.is_file()
            or path.stat().st_size != metadata["size"]
            or digest(path) != metadata["sha256"]
        ):
            raise ValueError(f"missing or changed media: {name}; retry refresh")


def copy_file(source: Path, destination: Path) -> None:
    """APFS copy-on-write on macOS; never hardlink editable media to the snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        import ctypes

        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        clonefile = library.clonefile
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clonefile.restype = ctypes.c_int
        if clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
    shutil.copy2(source, destination)


def activate(root: Path, snapshot_name: str) -> Path:
    snapshot = child_path(root, "snapshots/" + snapshot_name)
    if child_path(snapshot, "COMPLETED").read_text() != "verified\n":
        raise ValueError("snapshot is not verified")
    manifest = json.loads(child_path(snapshot, "manifest.json").read_text())
    verify_snapshot(snapshot / "output", manifest)
    generation = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    work = child_path(root, "work/" + generation)
    output = work / "output"
    output.mkdir(parents=True, mode=0o700)
    for name in ["images.db", *manifest["files"]]:
        copy_file(child_path(snapshot / "output", name), child_path(output, name))
    # Local identity and credentials are independently generated, never fetched.
    credentials = child_path(root, "local-credentials.json")
    if not credentials.exists():
        write_json(credentials, {"edit_password": uuid.uuid4().hex})
        credentials.chmod(0o600)
    write_json(work / "manifest.json", manifest)
    write_json(
        root / "current.json", {"work": "work/" + generation, "snapshot": snapshot_name}
    )
    return work


def refresh(root: Path, target: str, remote_output: str) -> None:
    validate_target(target, remote_output)
    cache = child_path(root, "media-cache")
    cache.mkdir(exist_ok=True)
    print("Pulling bird media only; the live app keeps running.", flush=True)
    transport = shlex.join(["ssh", *SSH_OPTIONS])
    for prefix in sorted(MEDIA_ROOTS):
        local = child_path(cache, prefix)
        local.mkdir(parents=True, exist_ok=True)
        run(
            [
                "rsync",
                "-rt",
                "--timeout=60",
                "--safe-links",
                "--no-links",
                "--rsync-path=sudo -n rsync",
                "-e",
                transport,
                f"{target}:{remote_output}/{prefix}/",
                str(local) + "/",
            ]
        )
    remote_script = Path(__file__).with_name("local_snapshot_remote.py").read_text()
    command = "sudo -n python3 - " + shlex.quote(remote_output)
    remote_tmp = None
    try:
        # Stream diagnostics during the full source-media hash pass.
        with subprocess.Popen(
            ["ssh", *SSH_OPTIONS, target, command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(remote_script)
            process.stdin.close()
            line = process.stdout.readline().strip()
            if not re.fullmatch(r"/tmp/wmb-local-pull-[a-z0-9_]+", line):
                raise ValueError(
                    "remote helper did not report its private temporary directory"
                )
            remote_tmp = line
            print(
                "Creating online SQLite backup and checking source media hashes…",
                flush=True,
            )
            for line in process.stdout:
                print(line.rstrip(), flush=True)
            if process.wait() != 0:
                raise ValueError(
                    "remote snapshot failed; previous local state is preserved"
                )
        generation = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        snapshot = child_path(root, "snapshots/" + generation)
        snapshot.mkdir(parents=True)
        output = snapshot / "output"
        output.mkdir()
        for name in ("images.db", "manifest.json"):
            destination = output / name if name == "images.db" else snapshot / name
            with destination.open("xb") as stream:
                run(
                    [
                        "ssh",
                        *SSH_OPTIONS,
                        target,
                        "sudo -n cat " + shlex.quote(remote_tmp + "/" + name),
                    ],
                    stdout=stream,
                )
        manifest = json.loads((snapshot / "manifest.json").read_text())
        manifest["ssh_target"] = target
        write_json(snapshot / "manifest.json", manifest)
        # A second pass collects arrivals during pre-copy; hashes reject changed/deleted media.
        for prefix in sorted(MEDIA_ROOTS):
            run(
                [
                    "rsync",
                    "-rt",
                    "--timeout=60",
                    "--checksum",
                    "--safe-links",
                    "--no-links",
                    "--rsync-path=sudo -n rsync",
                    "-e",
                    transport,
                    f"{target}:{remote_output}/{prefix}/",
                    str(cache / prefix) + "/",
                ]
            )
        for name in manifest["files"]:
            copy_file(child_path(cache, name), child_path(output, name))
        print("Verifying downloaded database and every image…", flush=True)
        verify_snapshot(output, manifest)
        (snapshot / "COMPLETED").write_text("verified\n")
        write_json(root / "latest.json", {"snapshot": generation})
        print(
            json.dumps(
                {
                    "snapshot": generation,
                    "counts": manifest["counts"],
                    "media_files": len(manifest["files"]),
                    "source_missing_media": len(manifest["source_missing_media"]),
                }
            ),
            flush=True,
        )
        if not (root / "current.json").exists():
            print("First working copy: " + str(activate(root, generation)), flush=True)
        else:
            print(
                "Existing working copy preserved. Use reset to activate the new snapshot.",
                flush=True,
            )
    finally:
        if remote_tmp:
            # Exact generated directory; cleanup only the two owned artifacts, never recursive rm.
            cleanup = (
                "sudo -n rm -f -- "
                + shlex.quote(remote_tmp + "/images.db")
                + " "
                + shlex.quote(remote_tmp + "/manifest.json")
            )
            cleanup += (
                " "
                + shlex.quote(remote_tmp + "/images.db-wal")
                + " "
                + shlex.quote(remote_tmp + "/images.db-shm")
            )
            cleanup += "; sudo -n rmdir -- " + shlex.quote(remote_tmp)
            subprocess.run(
                ["ssh", *SSH_OPTIONS, target, cleanup], check=False, timeout=20
            )


def network_guard(event: str, args: tuple) -> None:
    """Dev-process audit boundary: deny outbound connections and child processes."""
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
        "os.exec",
    }:
        raise PermissionError("child processes are disabled in local web development")
    if event in {"socket.connect", "socket.sendto"}:
        raise PermissionError("outbound connections are disabled in local development")
    if event in {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
    } and (args[0][0] if isinstance(args[0], tuple) else args[0]) not in {
        None,
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise PermissionError("external DNS disabled in local development")


def create_local_app(work: Path) -> Flask:
    """Use the real Flask factory and DB services, without main.py startup."""
    sys.path.insert(0, str(REPO))
    output = work / "output"
    manifest = json.loads((work / "manifest.json").read_text())
    credentials = json.loads(
        (work.parent.parent / "local-credentials.json").read_text()
    )
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["OUTPUT_DIR"] = str(output)
    os.environ["MODEL_BASE_PATH"] = str(work / "models")
    os.environ["INGEST_DIR"] = str(work / "ingest")
    os.environ["EDIT_PASSWORD"] = credentials["edit_password"]
    os.environ.pop("FLASK_SECRET_KEY", None)
    from config import get_config

    cfg = get_config()
    # Reassert after YAML/env merge, before any web module captures the config.
    cfg.update(
        {
            "OUTPUT_DIR": str(output),
            "MODEL_BASE_PATH": str(work / "models"),
            "INGEST_DIR": str(work / "ingest"),
            "EDIT_PASSWORD": credentials["edit_password"],
            "VIDEO_SOURCE": "",
            "CAMERA_URL": "",
            "GO2RTC_API_BASE": "http://127.0.0.1:1984",
            "TELEGRAM_MODE": "off",
            "TELEGRAM_ENABLED": False,
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
            "MQTT_ENABLED": False,
            "MQTT_HOST": "",
            "MQTT_PASSWORD": "",
            "telemetry_enabled": False,
            "telemetry_installation_id": "",
            "AESTHETIC_TAG_ENABLED": False,
            "COMPANION_ENABLED": False,
            "COMPANION_LLAMA_CPP_GGUF_PATH": "",
            "ENABLE_NIGHTLY_DEEP_SCAN": False,
            "RETENTION_AUTO_ENABLED": False,
            "DEBUG_MODE": False,
            "STATION_NAME": "Local development",
        }
    )
    from types import SimpleNamespace

    from flask import jsonify, request

    from web.web_interface import create_web_interface

    app = create_web_interface(SimpleNamespace(paused=True))
    allowed_blueprints = {
        "auth",
        "index",
        "gallery",
        "media",
        "analytics",
        "human_labels",
        "canonical_dataset",
        "trash",
    }
    allowed_endpoints = {
        "pages.serve_asset",
        "pages.settings_route",
        "system.healthz",
        "system.api_status",
        "api_v1.status",
        "api_v1.get_species_thumbnails",
        "review.review_page",
        "review.review_panel_fragment",
        "review.review_modal_fragment",
        "review.review_event_panel_fragment",
        "review.review_thumb",
        "review.review_decision",
        "review.review_cleanup_preview",
        "review.review_cleanup_run",
        "review.update_bbox_review_state",
        "review.review_quick_species",
        "review.review_approve",
        "review.review_event_approve",
        "review.review_event_trash",
        "review.review_event_unresolved",
        "review.review_event_resolve",
    }

    def local_boundary() -> Any:
        if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
            return jsonify(error="Local development accepts loopback hosts only"), 403
        if (
            request.blueprint not in allowed_blueprints
            and request.endpoint not in allowed_endpoints
        ):
            return jsonify(
                error="Unavailable in local development: camera, system and background operations are disabled"
            ), 503
        return None

    # Put the guard before CSRF/auth/route execution. Existing auth and CSRF stay intact.
    app.before_request_funcs.setdefault(None, []).insert(0, local_boundary)

    @app.context_processor
    def local_context() -> dict:
        return {
            "local_development": True,
            "local_snapshot_date": datetime.fromisoformat(manifest["captured_at_utc"])
            .astimezone()
            .strftime("%d %b %Y, %H:%M %Z"),
        }

    @app.get("/local-dev/status")
    def local_status() -> Any:
        return jsonify(
            mode="local-development",
            captured_at_utc=manifest["captured_at_utc"],
            snapshot_counts=manifest["counts"],
            camera_enabled=False,
            background_jobs_enabled=False,
        )

    @app.after_request
    def local_browser_boundary(response: Any) -> Any:
        # Browser-side relay requests must not reach a running local camera bridge.
        policy = response.headers.get("Content-Security-Policy", "")
        policy = re.sub(r"connect-src [^;]*;", "connect-src 'self';", policy)
        policy = re.sub(r"frame-src [^;]*;", "frame-src 'none';", policy)
        response.headers["Content-Security-Policy"] = policy
        response.headers["X-WMB-Mode"] = "local-development"
        return response

    # Flask executes after-request hooks in reverse registration order.
    hooks = app.after_request_funcs[None]
    hooks.insert(0, hooks.pop())
    allowed_endpoints.add("local_status")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser(
        "refresh", help="Pull Pi bird data; preserve existing working edits"
    )
    pull.add_argument("--ssh-target", required=True)
    pull.add_argument("--remote-output", default="/opt/app/data/output")
    commands.add_parser(
        "reset",
        help="Create a new working copy from the latest verified snapshot; keep older copies",
    )
    start = commands.add_parser(
        "start", help="Start the local web UI without camera/background services"
    )
    start.add_argument("--port", type=int, default=8080)
    commands.add_parser(
        "status", help="Show current snapshot and local working directory"
    )
    args = parser.parse_args()
    os.umask(0o077)
    root = managed_root(args.data_root)
    if args.command == "status":
        print(
            (root / "current.json").read_text()
            if (root / "current.json").exists()
            else "No working copy yet"
        )
        return
    with exclusive(root):
        if args.command == "refresh":
            refresh(root, args.ssh_target, args.remote_output)
        elif args.command == "reset":
            latest = json.loads((root / "latest.json").read_text())
            print(activate(root, latest["snapshot"]))
        else:
            if not 1024 <= args.port <= 65535:
                raise ValueError("port must be between 1024 and 65535")
            current = json.loads((root / "current.json").read_text())
            if not current["work"].startswith("work/"):
                raise ValueError("current must identify an editable working generation")
            work = child_path(root, current["work"])
            # Audit hooks apply only to this dedicated development process.
            sys.addaudithook(network_guard)
            app = create_local_app(work)
            print(f"Local UI: http://127.0.0.1:{args.port}", flush=True)
            print(
                "Local login password is stored privately in: "
                + str(root / "local-credentials.json"),
                flush=True,
            )
            app.run(
                host="127.0.0.1",
                port=args.port,
                threaded=True,
                debug=False,
                use_reloader=False,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Local development stopped; existing data is preserved.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Local development stopped: {exc}", file=sys.stderr)
        sys.exit(1)
