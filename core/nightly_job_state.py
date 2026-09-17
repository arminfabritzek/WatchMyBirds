"""Durable lifecycle snapshots for background jobs."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from config import get_config
from utils.path_manager import PathManager


def load_status(name: str) -> dict[str, Any]:
    """Read a snapshot, tolerating missing or damaged status files."""
    path = PathManager(str(get_config()["OUTPUT_DIR"])).get_nightly_job_status_path(
        name
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_status(name: str, payload: dict[str, Any]) -> None:
    """Replace a snapshot atomically so readers never see partial JSON."""
    path = PathManager(str(get_config()["OUTPUT_DIR"])).get_nightly_job_status_path(
        name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
