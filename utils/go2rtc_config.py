"""
Minimal safe API for reading/writing go2rtc YAML configuration.

Handles atomic writes with backup to avoid corrupting the go2rtc config
on crash or power-loss.
"""

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from logging_config import get_logger

logger = get_logger(__name__)


def ensure_go2rtc_config_exists(path: str, template_path: str = "") -> bool:
    """
    Ensures a go2rtc config file exists at *path*.

    If the file doesn't exist and *template_path* is provided and points to
    an existing file, copies the template.  Otherwise creates a minimal
    default config.

    Returns True if the file already existed or was created successfully.
    """
    target = Path(path)
    if target.exists():
        return True

    try:
        target.parent.mkdir(parents=True, exist_ok=True)

        if template_path:
            tpl = Path(template_path)
            if tpl.exists():
                shutil.copy2(tpl, target)
                logger.info("go2rtc config copied from template: %s → %s", tpl, target)
                return True

        # Minimal default – empty source until user sets CAMERA_URL.
        # NEVER use a fake/fallback URL here: go2rtc reads config only at
        # startup, and a stale default would persist until next restart.
        default_config = {
            "streams": {"camera": []},
            "api": {"listen": ":1984"},
            "rtsp": {"listen": ":8554"},
        }
        _write_yaml_atomic(target, default_config)
        logger.info("go2rtc config created with defaults: %s", target)
        return True

    except Exception as exc:
        logger.error("Failed to create go2rtc config at %s: %s", path, exc)
        return False


def set_camera_stream_source(
    path: str, camera_url: str, stream_name: str = "camera"
) -> bool:
    """
    Updates the ``streams.<stream_name>`` entry in the go2rtc config to
    point at *camera_url*.

    Preserves all other config sections.  Creates a ``.bak`` backup before
    writing.

    Returns True on success.
    """
    target = Path(path)
    if not target.exists():
        logger.warning(
            "go2rtc config not found at %s – cannot update stream source", path
        )
        return False

    try:
        with open(target, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

        # Ensure structure
        if "streams" not in config or not isinstance(config["streams"], dict):
            config["streams"] = {}

        config["streams"][stream_name] = [camera_url]

        # Backup + atomic write
        _backup(target)
        _write_yaml_atomic(target, config)
        logger.info(
            "go2rtc stream '%s' updated to: %s",
            stream_name,
            _mask_credentials(camera_url),
        )
        return True

    except Exception as exc:
        logger.error("Failed to update go2rtc config at %s: %s", path, exc)
        return False


def sync_camera_stream_source(
    path: str,
    camera_url: str,
    stream_name: str = "camera",
    template_path: str = "",
) -> bool:
    """
    Ensures go2rtc config exists and updates ``streams.<stream_name>``.

    If *template_path* is empty, tries ``<repo-root>/go2rtc.yaml.example``.
    Falls back to a minimal generated config when template is unavailable.
    """
    resolved_template = template_path
    if not resolved_template:
        # utils/go2rtc_config.py -> repo root is parent of utils/
        resolved_template = str(
            Path(__file__).resolve().parent.parent / "go2rtc.yaml.example"
        )

    if not ensure_go2rtc_config_exists(path, template_path=resolved_template):
        logger.warning("go2rtc config sync skipped - config create failed: %s", path)
        return False

    return set_camera_stream_source(path, camera_url, stream_name)


def read_camera_stream_source(path: str, stream_name: str = "camera") -> str:
    """
    Reads the first source URL for ``streams.<stream_name>`` from the
    go2rtc config.

    Returns the URL string, or empty string if not found / unreadable.
    """
    target = Path(path)
    if not target.exists():
        return ""

    try:
        with open(target, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

        streams = config.get("streams", {})
        if not isinstance(streams, dict):
            return ""

        sources = streams.get(stream_name, [])
        if isinstance(sources, list) and sources:
            return str(sources[0])
        elif isinstance(sources, str):
            return sources

        return ""

    except Exception as exc:
        logger.warning("Failed to read go2rtc config at %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# Runtime reload via go2rtc REST API
# ---------------------------------------------------------------------------


def reload_go2rtc_stream(
    api_base: str = "http://127.0.0.1:1984",
    stream_name: str = "camera",
    camera_url: str = "",
    timeout_sec: float = 2.0,
) -> bool:
    """
    Tells the running go2rtc instance to update its stream source at runtime.

    Uses ``PUT /api/streams?name=<stream_name>&src=<camera_url>`` which is the
    format expected by go2rtc's API handler.  The handler reads stream sources
    from the ``src`` query parameter(s), NOT from a JSON request body.

    Returns True on success, False on any failure (logged but never raised).
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    if not camera_url:
        logger.debug("reload_go2rtc_stream: skipped – camera_url is empty")
        return False

    # go2rtc PUT handler: query.Get("name") → stream name,
    #                     query["src"]      → source URL list.
    # Both must be query params; JSON body is ignored by go2rtc.
    params = urllib.parse.urlencode({"name": stream_name, "src": camera_url})
    url = f"{api_base.rstrip('/')}/api/streams?{params}"

    try:
        req = urllib.request.Request(url, method="PUT")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            ok = resp.status in (200, 201, 204)
            if ok:
                logger.info(
                    "event=reload_result status=success stream=%s source=%s",
                    stream_name,
                    _mask_credentials(camera_url),
                )
            else:
                logger.warning(
                    "event=reload_result status=unexpected_code "
                    "http_status=%s stream=%s",
                    resp.status,
                    stream_name,
                )
            return ok
    except urllib.error.HTTPError as exc:
        logger.warning(
            "event=reload_result status=http_error http_status=%s stream=%s reason=%s",
            exc.code,
            stream_name,
            exc.reason,
        )
        return False
    except Exception as exc:
        logger.warning(
            "event=reload_result status=error stream=%s error=%s",
            stream_name,
            exc,
        )
        return False


def reload_go2rtc_stream_with_retry(
    api_base: str = "http://127.0.0.1:1984",
    stream_name: str = "camera",
    camera_url: str = "",
    max_attempts: int = 3,
    backoff_steps: tuple[float, ...] = (1.0, 2.0, 4.0),
    timeout_sec: float = 2.0,
) -> bool:
    """
    Startup-safe wrapper around :func:`reload_go2rtc_stream` with bounded
    retry and exponential backoff.

    Intended for the boot path where go2rtc may not be fully ready yet.
    Each attempt is logged with a structured ``event=reload_result`` line
    including the attempt index.

    Returns True as soon as any attempt succeeds, False if all fail.
    """
    import time

    if not camera_url:
        logger.debug("reload_go2rtc_stream_with_retry: skipped – camera_url is empty")
        return False

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "event=relay_reload_attempt attempt=%d/%d stream=%s source=%s",
            attempt,
            max_attempts,
            stream_name,
            _mask_credentials(camera_url),
        )
        ok = reload_go2rtc_stream(
            api_base=api_base,
            stream_name=stream_name,
            camera_url=camera_url,
            timeout_sec=timeout_sec,
        )
        if ok:
            logger.info(
                "event=relay_reload_success attempt=%d/%d stream=%s",
                attempt,
                max_attempts,
                stream_name,
            )
            return True

        # Backoff before next attempt (skip after last attempt)
        if attempt < max_attempts:
            delay = (
                backoff_steps[attempt - 1]
                if attempt - 1 < len(backoff_steps)
                else backoff_steps[-1]
            )
            logger.info(
                "event=relay_reload_backoff attempt=%d/%d next_delay=%.1fs stream=%s",
                attempt,
                max_attempts,
                delay,
                stream_name,
            )
            time.sleep(delay)

    logger.warning(
        "event=relay_reload_exhausted attempts=%d stream=%s source=%s",
        max_attempts,
        stream_name,
        _mask_credentials(camera_url),
    )
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_yaml_atomic(path: Path, data: dict) -> None:
    """Writes YAML atomically via temp file + rename."""
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _backup(path: Path) -> None:
    """Creates a single .bak backup copy."""
    bak = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, bak)
    except Exception as exc:
        logger.debug("Could not create backup %s: %s", bak, exc)


def _mask_credentials(url: str) -> str:
    """Masks password in RTSP/HTTP URLs for safe logging."""
    if "://" not in url:
        return url
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            masked_netloc = f"{parsed.username}:*****@{parsed.hostname}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=masked_netloc))
    except Exception:
        pass
    return url
