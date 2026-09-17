"""Nightly-job adapter for full-resolution original retention."""

from __future__ import annotations

import threading
from typing import Any

from config import get_config
from core import retention_core
from web.services.nightly_job_hub import JobBase, update_progress


class RetentionJob(JobBase):
    @property
    def name(self) -> str:
        return "retention"

    @property
    def display_name(self) -> str:
        return "Storage Retention"

    @property
    def requires_night_pause(self) -> bool:
        return False

    def load_last_status(self) -> dict[str, Any]:
        return retention_core.load_last_run_status()

    def should_run_in_daily_loop(self) -> bool:
        """Automatic deletion requires its own explicit opt-in."""
        cfg = get_config()
        if not bool(cfg.get("RETENTION_AUTO_ENABLED", False)):
            return False
        settings = retention_core.resolve_posture_settings(cfg)
        return bool(settings.get("RETENTION_ENABLED", False))

    def run(self, stop_event: threading.Event, reason: str) -> int:
        result = retention_core.run(
            stop_requested=stop_event.is_set,
            progress_callback=lambda progress: update_progress(self.name, progress),
        )
        update_progress(self.name, result)
        return 1 if result["errors"] else 0
