"""Nightly adapter that brings older bird crops up to the current crop size."""

from __future__ import annotations

import threading

from core import crop_resolution
from web.services.nightly_job_hub import JobBase, update_progress


class CropResolutionJob(JobBase):
    @property
    def name(self) -> str:
        return "crop_resolution"

    @property
    def display_name(self) -> str:
        return "Crop Resolution Upgrade"

    def run(self, stop_event: threading.Event, reason: str) -> int:
        result = crop_resolution.run(
            stop_requested=stop_event.is_set,
            progress_callback=lambda progress: update_progress(self.name, progress),
        )
        update_progress(self.name, result)
        return 1 if result["errors"] else 0
