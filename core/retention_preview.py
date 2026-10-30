"""One background deletion preview with coalesced requests and saved-rule checks."""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)
Progress = Callable[[dict[str, int]], None]


class RetentionPreview:
    def __init__(
        self,
        builder: Callable[[dict[str, Any], Progress], dict[str, Any]],
        settings_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self._builder = builder
        self._settings_provider = settings_provider
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._key: tuple[object, ...] | None = None
        self._requested_key: tuple[object, ...] | None = None
        self._requested_settings: dict[str, Any] = {}
        self._state: dict[str, Any] = dict(
            status="idle", progress={}, result=None, calculated_at=None, error=None
        )

    def request(self, *, refresh: bool = False) -> dict[str, Any]:
        settings = copy.deepcopy(self._settings_provider())
        key = tuple(
            settings.get(name)
            for name in (
                "AUTOMATIC_DELETION_ENABLED",
                "AUTOMATIC_DELETION_ENABLED_AT",
                "OLD_BIRD_PHOTOS",
            )
        )
        with self._lock:
            self._requested_key = key
            self._requested_settings = settings
            if self._state["status"] != "running" and (
                refresh or self._state["status"] == "idle" or self._key != key
            ):
                self._state.update(
                    status="running",
                    progress={},
                    result=None,
                    calculated_at=None,
                    error=None,
                )
                self._thread = threading.Thread(
                    target=self._run, name="retention-preview", daemon=True
                )
                self._thread.start()
            state = copy.deepcopy(self._state)
            if state["status"] == "running":
                state["result"] = None
            return state

    def wait(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _progress(self, progress: dict[str, int]) -> None:
        with self._lock:
            self._state["progress"].update(progress)

    def _run(self) -> None:
        while True:
            with self._lock:
                key = self._requested_key
                settings = self._requested_settings
                self._state["progress"] = {}
            try:
                result = self._builder(settings, self._progress)
            except Exception:
                logger.exception("Deletion preview failed")
                with self._lock:
                    if self._requested_key != key:
                        continue
                    self._key = key
                    self._state.update(
                        status="failed",
                        result=None,
                        error="Preview failed; see local logs",
                    )
                return
            with self._lock:
                # A rule change during calculation requires a new preview, while
                # capture activity alone must not keep restarting a long scan.
                if self._requested_key != key:
                    continue
                self._key = key
                self._state.update(
                    status="complete",
                    result=result,
                    calculated_at=datetime.now(UTC).isoformat(),
                    error=None,
                )
                return
