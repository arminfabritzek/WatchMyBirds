"""
Update service for WatchMyBirds RPi deployments.

Checks GitHub for new releases/main-branch builds and triggers the
privileged wmb-update.service systemd unit to perform the actual installation.

Status is tracked via a JSON file in the app data directory so the frontend
can poll for progress without needing a persistent WebSocket connection.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_REPO = "arminfabritzek/WatchMyBirds"
GITHUB_API_RELEASES = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# These paths match the systemd service configuration on the RPi.
_DATA_DIR = Path(os.environ.get("OUTPUT_DIR", "/opt/app/data/output")).parent
_STATUS_FILE = _DATA_DIR / "update_status.json"
_REQUEST_FILE = _DATA_DIR / "update_request.txt"

SYSTEMD_UPDATE_UNIT = "wmb-update.service"

_REQUEST_TIMEOUT = 8  # seconds for GitHub API calls


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def _github_get(url: str) -> Any:
    """Perform a GET request to the GitHub API and return parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "WatchMyBirds-Updater/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_latest_release() -> dict[str, Any] | None:
    """Return info about the latest GitHub release, or None on error."""
    try:
        data = _github_get(GITHUB_API_LATEST)
        return _parse_release(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info("No releases found on GitHub.")
        else:
            logger.warning("GitHub API error (latest): %s", e)
    except Exception as e:
        logger.warning("Failed to fetch latest release: %s", e)
    return None


def list_releases(limit: int = 10) -> list[dict[str, Any]]:
    """Return a list of the most recent GitHub releases."""
    try:
        data = _github_get(f"{GITHUB_API_RELEASES}?per_page={limit}")
        return [_parse_release(r) for r in data if isinstance(r, dict)]
    except Exception as e:
        logger.warning("Failed to fetch releases list: %s", e)
    return []


def _parse_release(data: dict) -> dict[str, Any]:
    """Extract the fields we care about from a GitHub release object."""
    return {
        "tag_name": data.get("tag_name", ""),
        "name": data.get("name") or data.get("tag_name", ""),
        "published_at": data.get("published_at", ""),
        "prerelease": bool(data.get("prerelease")),
        "draft": bool(data.get("draft")),
        "body": (data.get("body") or "")[:500],  # truncate changelog
        "tarball_url": data.get("tarball_url", ""),
        "html_url": data.get("html_url", ""),
    }


# ---------------------------------------------------------------------------
# Status file helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_update_status() -> dict[str, Any]:
    """Read and return the current update status from disk."""
    try:
        if _STATUS_FILE.is_file():
            text = _STATUS_FILE.read_text(encoding="utf-8")
            return json.loads(text)
    except Exception as e:
        logger.debug("Could not read update status: %s", e)
    return {"state": "idle", "message": "", "target": "", "timestamp": ""}


def _write_status(state: str, message: str, target: str = "") -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _STATUS_FILE.write_text(
            json.dumps(
                {
                    "state": state,
                    "message": message,
                    "target": target,
                    "timestamp": _now_iso(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("Failed to write update status: %s", e)


# ---------------------------------------------------------------------------
# Update trigger
# ---------------------------------------------------------------------------

# Valid target values: a release tag like "v0.2.0", or the special string
# "main" to install the latest commit from the main branch.
_VALID_TARGET_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/"
)


def _validate_target(target: str) -> bool:
    """Basic allowlist validation to prevent command injection."""
    if not target or len(target) > 64:
        return False
    return all(c in _VALID_TARGET_CHARS for c in target)


def request_update(target: str) -> tuple[bool, str]:
    """
    Write an update request and trigger the wmb-update.service unit.

    Returns (success, message).
    Only available when running on the RPi (systemd + /opt/app).
    """
    if not _validate_target(target):
        return False, "Invalid target version string."

    if not _is_update_available():
        return False, "Update management is only available on RPi deployments."

    current_status = get_update_status()
    if current_status.get("state") in ("downloading", "installing", "restarting"):
        return False, "An update is already in progress."

    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _REQUEST_FILE.write_text(target, encoding="utf-8")
        _write_status("pending", f"Update to {target} requested.", target)
    except Exception as e:
        logger.error("Failed to write update request: %s", e)
        return False, f"Could not write update request: {e}"

    # Trigger the privileged systemd service.
    try:
        subprocess.run(
            ["systemctl", "start", SYSTEMD_UPDATE_UNIT],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.info("wmb-update.service started for target: %s", target)
        return True, f"Update to {target} started."
    except subprocess.CalledProcessError as e:
        details = (e.stderr or e.stdout or "").strip()
        logger.error("Failed to start wmb-update.service: %s", details)
        _write_status("error", f"Could not start update service: {details}", target)
        return False, f"Could not start update service: {details}"
    except FileNotFoundError:
        _write_status("error", "systemctl not found.", target)
        return False, "systemctl not found — not an RPi deployment?"
    except Exception as e:
        logger.error("Unexpected error starting update service: %s", e)
        _write_status("error", str(e), target)
        return False, str(e)


def _is_update_available() -> bool:
    """Return True when running in an environment that supports OTA updates."""
    return (
        os.path.isdir("/run/systemd/system") and shutil.which("systemctl") is not None
    )


def is_update_supported() -> bool:
    """Public accessor used by the API layer."""
    return _is_update_available()
