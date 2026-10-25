"""One manually started storage scan and a bounded local measurement history."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.retention_core import activation_date, settings_for_plan
from core.storage_accounting import inspect_storage
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)
HISTORY_LIMIT = 32
MAX_STATE_BYTES = 2 * 1024 * 1024


def growth_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        return dict(status="insufficient_history")
    previous, current = samples[-2:]
    if previous["scope"] != current["scope"]:
        return dict(status="scope_changed")
    seconds = (
        datetime.fromisoformat(current["at"]) - datetime.fromisoformat(previous["at"])
    ).total_seconds()
    if seconds <= 0:
        return dict(status="insufficient_interval")
    delta = current["bytes"] - previous["bytes"]
    return dict(
        status="measured",
        interval_seconds=seconds,
        delta_bytes=delta,
        bytes_per_day=delta * 86400 / seconds if seconds >= 3600 else None,
        collection_gap=seconds > 48 * 3600,
        approximate=previous.get("approximate", True)
        or current.get("approximate", True),
        basis="output regular files only; symlinks excluded; manual samples; observed rate, not a forecast",
    )


class StorageScan:
    def __init__(
        self,
        path_mgr: PathManager | str,
        settings_provider: Callable[[], dict[str, Any]],
        *,
        inspector: Callable[..., dict[str, Any]] = inspect_storage,
    ) -> None:
        self.path_mgr = PathManager(path_mgr) if isinstance(path_mgr, str) else path_mgr
        self.settings_provider = settings_provider
        self.inspector = inspector
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = dict(
            status="idle",
            progress={},
            result=None,
            history=[],
            growth=dict(status="insufficient_history"),
        )
        path = self.path_mgr.contained_path(
            self.path_mgr.get_storage_usage_path(), self.path_mgr.base_dir
        )
        if path is not None:
            try:
                if path.stat().st_size <= MAX_STATE_BYTES:
                    saved = json.loads(path.read_text())
                    if saved.get("schema_version") == 1 and isinstance(
                        saved.get("history"), list
                    ):
                        self._state.update(
                            result=saved.get("result"),
                            history=saved["history"][-HISTORY_LIMIT:],
                        )
                        self._state["growth"] = growth_summary(self._state["history"])
                        self._state["status"] = (
                            self._state["result"].get("status", "complete")
                            if self._state["result"]
                            else "idle"
                        )
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning(
                    "Storage history unavailable; no previous measurement loaded"
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self._state)
        report = state.get("result") or {}
        preview = report.get("deletion_preview") or {}
        if report.get("forecast"):
            settings = self.settings_provider()
            enabled_at = activation_date(settings)
            saved_at = preview.get("enabled_at")
            if (
                preview.get("settings") != asdict(settings_for_plan(settings))
                or (enabled_at.isoformat() if enabled_at else None) != saved_at
            ):
                report["forecast"] = dict(status="settings_changed")
        return state

    def start(self) -> bool:
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state.update(
                status="running", progress=dict(phase="starting"), error=None
            )
            self._thread = threading.Thread(
                target=self._run, name="storage-accounting", daemon=True
            )
            self._thread.start()
            return True

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _progress(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._state["progress"].update(value)

    def _run(self) -> None:
        try:
            result = self.inspector(
                self.path_mgr, self.settings_provider(), progress=self._progress
            )
            with self._lock:
                history = list(self._state["history"])
            output = next(
                (
                    scope
                    for scope in result["inventory"]["scopes"]
                    if scope["scope"] == "output" and scope["state"] == "measured"
                ),
                None,
            )
            if output and not result["inventory"]["errors"]:
                # Inventory roots can share a device. Output growth is measured
                # separately so attaching backups cannot masquerade as bird growth.
                output_bytes = result["inventory"]["output_bytes"]
                history.append(
                    dict(
                        at=result["finished_at"],
                        bytes=output_bytes,
                        scope=f"{hashlib.sha256(str(self.path_mgr.base_dir.resolve()).encode()).hexdigest()[:16]}:{output['device']}",
                        approximate=True,
                    )
                )
            history = history[-HISTORY_LIMIT:]
            self._persist(result, history)
            with self._lock:
                self._state.update(
                    status=result["status"],
                    result=result,
                    history=history,
                    growth=growth_summary(history),
                    progress=dict(phase="finished"),
                )
        except Exception:
            logger.exception("Storage accounting failed")
            with self._lock:
                self._state.update(
                    status="failed",
                    error="Storage scan failed; see local logs",
                    progress=dict(phase="failed"),
                )

    def _persist(self, result: dict[str, Any], history: list[dict[str, Any]]) -> None:
        destination = self.path_mgr.contained_path(
            self.path_mgr.get_storage_usage_path(), self.path_mgr.base_dir
        )
        if destination is None:
            raise ValueError("Storage history path is outside output")
        payload = json.dumps(
            dict(schema_version=1, result=result, history=history),
            separators=(",", ":"),
        ).encode()
        if len(payload) > MAX_STATE_BYTES:
            raise ValueError("Storage report exceeds local history bound")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".storage-usage-", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
