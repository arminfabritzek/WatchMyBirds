"""Hub adapter for the aesthetic-tagger.

The CLIP-based aesthetic-tagger has its own scheduler module
(``web/services/aesthetic_tag_scheduler.py``) with elaborate
internals: HF cache redirect, compute lease, throttle, per-species
cap, today-vs-yesterday SQL window. The Telegram pre-report bridge
calls ``run_now(...)`` directly with very specific arguments.

This adapter exposes the same job to the Settings UI via the hub.
A "Run now" click here fires the tagger with the same defaults the
nightly daily-loop uses (since=None → tagger's own "yesterday
00:00 UTC" default; throttle 0; no per-species cap). Operators who
need the Telegram-bridge variant still go through the report
scheduler.
"""

from __future__ import annotations

import logging
import threading

from web.services.nightly_job_hub import JobBase

logger = logging.getLogger(__name__)


class AestheticTaggerJob(JobBase):
    @property
    def name(self) -> str:
        return "aesthetic_tagger"

    @property
    def display_name(self) -> str:
        return "Aesthetic Tagger (CLIP)"

    def should_run_in_daily_loop(self) -> bool:
        """Decline if optional CLIP deps aren't installed (slim image)."""
        try:
            from web.services.aesthetic_tag_scheduler import (
                _check_dependencies_available,
            )

            return _check_dependencies_available()
        except Exception:
            logger.exception("AestheticTaggerJob: dependency check failed")
            return False

    def run(self, stop_event: threading.Event, reason: str) -> int:
        """Delegate to aesthetic_tag_scheduler.run_now with hub defaults.

        Note: the underlying CLIP worker does not currently honour a
        stop_event mid-run. A Stop click here flips the hub state,
        but the actual interruption only takes effect at the worker's
        next natural break (per-image loop). This is a known
        limitation; a follow-up plan can thread stop_event into the
        worker.
        """
        from web.services.aesthetic_tag_scheduler import run_now

        ok = run_now(
            f"hub:{reason}",
            since=None,
            today_only=False,
            throttle_ms=0,
            per_species_cap=None,
        )
        if stop_event.is_set():
            logger.info(
                "AestheticTaggerJob: stop requested mid-run (effective at "
                "next worker checkpoint)"
            )
        return 0 if ok else 1
