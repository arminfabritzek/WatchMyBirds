"""Cached save admission and background recovery through the retention executor."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from core.deletion_plan import DEFAULT_SETTINGS, deletion_preview
from core.retention_core import activation_date, reclaim_space, settings_for_plan
from utils.db import closing_connection
from utils.path_manager import PathManager
from utils.settings import load_settings_yaml

logger = logging.getLogger(__name__)


class StoragePressure:
    def __init__(
        self,
        output_dir: str,
        *,
        free_bytes: Callable[[], int] | None = None,
        settings_provider: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        reserve_bytes: int = DEFAULT_SETTINGS.reserve_bytes,
    ) -> None:
        self.paths = PathManager(output_dir)
        self._free_bytes = free_bytes or (lambda: shutil.disk_usage(output_dir).free)
        self._settings = settings_provider or (lambda: load_settings_yaml(output_dir))
        self._clock = clock
        self.reserve_bytes = reserve_bytes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._checked_at = float("-inf")
        self._retry_at = float("-inf")
        self._free = 0
        self._paused = False
        self._message = ""
        self._last_cleanup: dict[str, Any] = {}

    def allow_save(self, budget_bytes: int) -> bool:
        with self._lock:
            # A new image has no objects until the processing frame completes.
            # Keep it out of a concurrent rule-6 pressure pass during that gap.
            if self._worker is not None and self._worker.is_alive():
                self._paused = True
                self._message = "Photo saving paused: reclaiming unprotected photos. Live view continues."
                return False
            now = self._clock()
            if now - self._checked_at >= 2:
                try:
                    self._free = self._free_bytes()
                    self._checked_at = now
                except OSError:
                    self._paused = True
                    self._message = "Photo saving paused: free space is unavailable. Live view continues."
                    return False
            target = self.reserve_bytes + 2 * budget_bytes
            if self._free >= target:
                self._paused = False
                self._message = ""
                # Account for intervening captures before the next filesystem check.
                self._free -= budget_bytes
                return True
            self._paused = True
            if not self._message:
                self._message = "Photo saving paused: card nearly full. Checking unprotected photos. Live view continues."
            if (
                not self._stop.is_set()
                and (self._worker is None or not self._worker.is_alive())
                and now >= self._retry_at
            ):
                self._retry_at = now + 60
                self._worker = threading.Thread(
                    target=self.recover,
                    args=(self.reserve_bytes + 4 * budget_bytes,),
                    name="StorageRecovery",
                    daemon=True,
                )
                self._worker.start()
            return False

    def recover(self, target_bytes: int) -> dict[str, Any]:
        """Recover outside the capture path; callable synchronously for diagnostics."""
        result: dict[str, Any] = {}
        try:
            settings = self._settings()
            with closing_connection() as conn:
                if activation_date(settings) is not None:
                    result = reclaim_space(
                        conn,
                        str(self.paths.base_dir),
                        settings,
                        free_bytes=self._free_bytes,
                        target_bytes=target_bytes,
                        settings_provider=self._settings,
                        stop_requested=self._stop.is_set,
                    )
                free = self._free_bytes()
                message = ""
                if free < target_bytes:
                    if activation_date(self._settings()) is None:
                        reason = "automatic deletion is off"
                    else:
                        plan = deletion_preview(
                            conn,
                            settings_for_plan(settings),
                            datetime.now(UTC),
                            paths=self.paths,
                        )
                        training = plan["classes"].get("3", {}).get("bytes", 0)
                        reason = (
                            f"{training / 1024**3:.2f} GiB of training photos awaiting export or confirmation"
                            if training
                            else "remaining photos are protected or previews are being kept; free space or change storage settings"
                        )
                    message = f"Photo saving paused: {reason}. Live view continues; saving resumes automatically when space is available."
            with self._lock:
                self._free = free
                self._checked_at = self._clock()
                self._paused = free < target_bytes
                self._message = message
                if result.get("freed_bytes"):
                    self._last_cleanup = result
            return result
        except Exception:
            logger.exception("Storage recovery failed")
            with self._lock:
                self._message = "Photo saving paused: storage recovery failed. Check storage and logs. Live view continues."
            return {"errors": 1}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paused": self._paused,
                "message": self._message,
                "free_bytes": self._free,
                "reserve_bytes": self.reserve_bytes,
                "last_cleanup": dict(self._last_cleanup),
            }

    def close(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)
