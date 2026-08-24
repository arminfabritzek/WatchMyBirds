"""Periodic local runtime sampling for defensible observation effort."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from logging_config import get_logger
from utils.db.connection import closing_connection

logger = get_logger(__name__)


class StationEffortMonitor:
    """Record bounded runtime samples without depending on web requests."""

    def __init__(self, detection_manager: Any, interval_seconds: float = 60.0):
        self.detection_manager = detection_manager
        self.interval_seconds = max(10.0, float(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="StationEffortMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval_seconds, 5.0))
        self._write_sample(app_online=False)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._write_sample(app_online=True)
            self._stop_event.wait(self.interval_seconds)

    def _write_sample(self, *, app_online: bool) -> None:
        try:
            sample = self._snapshot(app_online=app_online)
            with closing_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO station_runtime_samples (
                        sampled_at, sample_interval_seconds, app_online,
                        camera_online, stream_online, detector_ready,
                        detector_active, od_active, od_reason, ptz_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample["sampled_at"],
                        self.interval_seconds,
                        int(sample["app_online"]),
                        int(sample["camera_online"]),
                        int(sample["stream_online"]),
                        int(sample["detector_ready"]),
                        int(sample["detector_active"]),
                        int(sample["od_active"]),
                        sample["od_reason"],
                        sample["ptz_state"],
                    ),
                )
        except Exception:
            logger.warning("Station effort sample failed", exc_info=True)

    def _snapshot(self, *, app_online: bool) -> dict[str, object]:
        manager = self.detection_manager
        now_epoch = time.time()
        latest_frame_at = float(getattr(manager, "latest_raw_timestamp", 0.0) or 0.0)
        stream_online = app_online and (now_epoch - latest_frame_at) <= 5.0
        camera_online = app_online and getattr(manager, "video_capture", None) is not None

        detection_service = getattr(manager, "detection_service", None)
        try:
            detector_ready = bool(
                app_online and detection_service and detection_service.is_ready()
            )
        except Exception:
            detector_ready = False

        try:
            od_status = manager.get_od_status() if app_online else {}
        except Exception:
            od_status = {}
        od_active = bool(od_status.get("od_active", False))
        detector_active = bool(
            detector_ready
            and stream_online
            and od_active
            and not bool(getattr(manager, "paused", False))
        )
        ptz_state = "unavailable"
        ptz_controller = getattr(manager, "auto_ptz_controller", None)
        if app_online and ptz_controller is not None:
            try:
                ptz_state = str(ptz_controller.status().get("state") or "unknown")
            except Exception:
                ptz_state = "unknown"
        return {
            "sampled_at": datetime.now(UTC).isoformat(),
            "app_online": app_online,
            "camera_online": camera_online,
            "stream_online": stream_online,
            "detector_ready": detector_ready,
            "detector_active": detector_active,
            "od_active": od_active,
            "od_reason": str(od_status.get("reason") or "offline"),
            "ptz_state": ptz_state,
        }


__all__ = ["StationEffortMonitor"]
