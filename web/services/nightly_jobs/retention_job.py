"""Nightly adapter for the shared photo deletion plan."""

from __future__ import annotations

import threading

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

    def should_run_in_daily_loop(self) -> bool:
        return bool(get_config().get("AUTOMATIC_DELETION_ENABLED", False))

    def run(self, stop_event: threading.Event, reason: str) -> int:
        result = retention_core.run(
            stop_requested=stop_event.is_set,
            progress_callback=lambda progress: update_progress(self.name, progress),
        )
        update_progress(self.name, result)
        return 1 if result["errors"] else 0
