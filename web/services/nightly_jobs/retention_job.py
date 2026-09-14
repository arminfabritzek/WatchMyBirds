"""Nightly-job adapter for full-resolution original retention."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from config import get_config
from core import retention_core
from web.services.nightly_job_hub import JobBase, update_progress

logger = logging.getLogger(__name__)


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
        started_at = datetime.now(tz=UTC)
        try:
            result = retention_core.run(
                stop_requested=stop_event.is_set,
                progress_callback=lambda progress: update_progress(self.name, progress),
            )
        except Exception as exc:
            self._persist_status(
                started_at=started_at,
                reason=reason,
                rc=1,
                progress={},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        update_progress(self.name, result)
        rc = 1 if result["errors"] else 0
        self._persist_status(
            started_at=started_at,
            reason=reason,
            rc=rc,
            progress=result,
            error=None,
        )
        return rc

    def _persist_status(
        self,
        *,
        started_at: datetime,
        reason: str,
        rc: int,
        progress: dict[str, Any],
        error: str | None,
    ) -> None:
        try:
            previous = retention_core.load_last_run_status()
            last_daily_fire_date = previous.get("last_daily_fire_date")
            if reason == "nightly auto":
                last_daily_fire_date = started_at.date().isoformat()
            retention_core.save_last_run_status(
                {
                    "last_started_at": started_at.isoformat(),
                    "last_finished_at": datetime.now(tz=UTC).isoformat(),
                    "last_reason": reason,
                    "last_rc": rc,
                    "last_error": error,
                    "last_daily_fire_date": last_daily_fire_date,
                    "progress": progress,
                }
            )
        except OSError:
            logger.exception("RetentionJob: failed to persist last-run status")
