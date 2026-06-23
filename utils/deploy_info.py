"""
Shared build metadata reader and deploy type detection.

Provides a single source of truth for app version, git commit,
build date, and deployment type — consumed by both the legacy
``/api/system/versions`` route and the V1 endpoint.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File read order — Docker paths first, RPi compatibility fallback second.
# ---------------------------------------------------------------------------
_FILE_SOURCES: dict[str, list[Path]] = {
    "app_version": [
        Path("/app/APP_VERSION"),
        Path("/opt/app/APP_VERSION"),
    ],
    "git_commit": [
        Path("/app/BUILD_COMMIT"),
        Path("/opt/app/commit.txt"),
    ],
    "build_date": [
        Path("/app/BUILD_DATE"),
        Path("/opt/app/build_date.txt"),
    ],
}

_UNKNOWN = "Unknown"


def _read_first(paths: list[Path]) -> str:
    """Return the trimmed content of the first readable file, or *_UNKNOWN*."""
    for p in paths:
        try:
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001
            continue
    return _UNKNOWN


def _try_local_app_version() -> str:
    """Fallback: read APP_VERSION relative to the project root (dev mode)."""
    try:
        project_root = Path(__file__).resolve().parent.parent
        version_file = project_root / "APP_VERSION"
        if version_file.is_file():
            text = version_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
    except Exception:  # noqa: BLE001
        pass
    return _UNKNOWN


def _try_local_git(args: list[str]) -> str:
    """Run a `git` command at the project root and return its trimmed stdout.

    Dev-mode fallback only — when build-artifact files (BUILD_COMMIT,
    BUILD_DATE) are absent because the app is launched from a checkout
    without going through the Docker/RPi build pipelines. Returns
    ``_UNKNOWN`` on any failure (missing git binary, not a repo,
    shallow clone without history, subprocess timeout).

    Intentionally NOT used on Docker/RPi: on those targets the build
    files are the source of truth, and a missing file means the build
    pipeline forgot to write it. Silently papering over that with a
    subprocess call would hide a real build bug.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            text = result.stdout.strip()
            if text:
                return text
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("git metadata read failed; using _UNKNOWN", exc_info=True)
    return _UNKNOWN


def detect_deploy_type() -> str:
    """Return a simple product-facing deployment label.

    Returns:
        ``"docker"`` — running inside a Docker container.
        ``"rpi"``    — running on the Raspberry Pi appliance.
        ``"dev"``    — local development / unknown.
    """
    # 1. Docker
    if Path("/.dockerenv").exists():
        return "docker"
    try:
        cgroup_text = Path("/proc/1/cgroup").read_text(
            encoding="utf-8", errors="ignore"
        )
        if "docker" in cgroup_text.lower() or "containerd" in cgroup_text.lower():
            return "docker"
    except Exception:  # noqa: BLE001
        pass

    # 2. RPi appliance
    if Path("/opt/app").is_dir():
        try:
            import shutil

            if shutil.which("systemctl") is not None:
                return "rpi"
        except Exception:  # noqa: BLE001
            pass

    # 3. Default
    return "dev"


def read_build_metadata() -> dict[str, str]:
    """Return build metadata from well-known file locations.

    The returned dict always contains:

    - ``app_version``  – semantic version from ``APP_VERSION`` file.
    - ``git_commit``   – short (7-char) commit hash.
    - ``build_date``   – ISO-ish build timestamp.
    - ``deploy_type``  – one of ``"docker"``, ``"rpi"``, ``"dev"``.
    """
    app_version = _read_first(_FILE_SOURCES["app_version"])
    if app_version == _UNKNOWN:
        app_version = _try_local_app_version()

    git_commit = _read_first(_FILE_SOURCES["git_commit"])
    if git_commit != _UNKNOWN:
        git_commit = git_commit[:7]  # short hash

    build_date = _read_first(_FILE_SOURCES["build_date"])
    deploy_type = detect_deploy_type()

    # Dev-mode only: when the build pipeline didn't write the files
    # (i.e. someone launched `python main.py` from a checkout), fall
    # back to live git queries so the footer shows something useful.
    # Not applied on docker/rpi: a missing file there is a real build
    # bug we don't want to mask.
    if deploy_type == "dev":
        if git_commit == _UNKNOWN:
            git_commit = _try_local_git(["rev-parse", "--short=7", "HEAD"])
        if build_date == _UNKNOWN:
            build_date = _try_local_git(["log", "-1", "--format=%cs", "HEAD"])

    return {
        "app_version": app_version,
        "git_commit": git_commit,
        "build_date": build_date,
        "deploy_type": deploy_type,
    }
