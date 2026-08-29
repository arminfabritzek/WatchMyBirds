"""Ephemeral click-to-aim sessions for visually verified PTZ movement."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from core.visual_ptz_core import (
    ClosedLoopVisualPtz,
    VisualPtzIteration,
    VisualPtzLimits,
)
from utils.visual_alignment import FeatureAligner

logger = logging.getLogger(__name__)


class ExternalPauseController(Protocol):
    def pause_for_external(self, reason: str) -> bool: ...

    def resume_from_external(self) -> bool: ...


class AimBusyError(RuntimeError):
    """Raised when another owner already controls the camera."""


class AimUnavailableError(RuntimeError):
    """Raised when a fresh reference frame cannot be captured."""


class AimController:
    """Run one bounded visual aim session in a background thread."""

    _PAUSE_REASON = "Click-to-Aim"

    def __init__(
        self,
        *,
        frame_supplier: Callable[[], np.ndarray | None],
        mover: Callable[[int, float, float, int], None],
        stopper: Callable[[int], None],
        external_pause: ExternalPauseController | None = None,
        aligner: FeatureAligner | None = None,
        limits: VisualPtzLimits | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._frame_supplier = frame_supplier
        self._mover = mover
        self._stopper = stopper
        self._external_pause = external_pause
        self._aligner = aligner or FeatureAligner(method="auto")
        self._limits = limits or VisualPtzLimits(
            max_iterations=5,
            deadband=0.06,
            min_quality=0.45,
            gain=1.2,
            min_step=0.25,
            max_step=0.45,
            max_total_motion=4.0,
            move_duration_ms=300,
            max_move_duration_ms=700,
            move_duration_gain_ms=4000.0,
            settle_sec=0.25,
            coarse_threshold=0.18,
            coarse_gain=2.0,
            coarse_min_step=0.85,
            coarse_max_step=1.0,
            coarse_move_duration_ms=650,
            coarse_max_move_duration_ms=1500,
            coarse_duration_gain_ms=3000.0,
            coarse_settle_sec=0.25,
            serialize_axes=True,
            min_progress=0.005,
            max_stagnant_iterations=2,
        )
        self._sleep = sleep
        self._start_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._status: dict[str, Any] = self._idle_status()

    def start(self, camera_id: int, target_x: float, target_y: float) -> dict[str, Any]:
        if not 0.0 <= target_x <= 1.0 or not 0.0 <= target_y <= 1.0:
            raise ValueError("target coordinates must be between 0 and 1")

        with self._start_lock:
            return self._start_serialized(camera_id, target_x, target_y)

    def _start_serialized(
        self, camera_id: int, target_x: float, target_y: float
    ) -> dict[str, Any]:
        with self._lock:
            if self._status["active"]:
                raise AimBusyError("A click-to-aim session is already active")

        pause_acquired = False
        try:
            if self._external_pause is not None:
                try:
                    self._external_pause.pause_for_external(self._PAUSE_REASON)
                except RuntimeError as exc:
                    raise AimBusyError(str(exc)) from exc
                pause_acquired = True
            reference = self._frame_supplier()
            if reference is None or reference.size == 0:
                raise AimUnavailableError("No fresh camera frame is available")

            session_id = uuid.uuid4().hex
            self._cancel.clear()
            with self._lock:
                if self._status["active"]:
                    raise AimBusyError("A click-to-aim session is already active")
                self._status = {
                    "active": True,
                    "session_id": session_id,
                    "camera_id": int(camera_id),
                    "state": "matching",
                    "target_x": float(target_x),
                    "target_y": float(target_y),
                    "mapped_x": float(target_x),
                    "mapped_y": float(target_y),
                    "iteration": 0,
                    "max_iterations": self._limits.max_iterations,
                    "quality": None,
                    "inliers": 0,
                    "error": None,
                    "pan_command": 0.0,
                    "tilt_command": 0.0,
                    "message": "Matching the selected point",
                }
            worker = threading.Thread(
                target=self._run,
                args=(session_id, int(camera_id), reference, (target_x, target_y)),
                kwargs={"pause_acquired": pause_acquired},
                daemon=True,
                name=f"ptz-aim-{camera_id}",
            )
            worker.start()
            pause_acquired = False
            return self.status()
        finally:
            if pause_acquired and self._external_pause is not None:
                self._external_pause.resume_from_external()

    def cancel(self, camera_id: int) -> dict[str, Any]:
        with self._lock:
            if not self._status["active"]:
                return dict(self._status)
            if self._status["camera_id"] != int(camera_id):
                raise AimBusyError("Another camera owns the active aim session")
            self._cancel.set()
            self._status["state"] = "cancelling"
            self._status["message"] = "Stopping click-to-aim"
        self._stopper(int(camera_id))
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _run(
        self,
        session_id: str,
        camera_id: int,
        reference: np.ndarray,
        reference_anchor: tuple[float, float],
        *,
        pause_acquired: bool,
    ) -> None:
        final_state = "target_lost"
        final_message = "The selected point could not be verified"
        try:
            loop = ClosedLoopVisualPtz(
                aligner=self._aligner,
                frame_supplier=self._frame_supplier_required,
                mover=lambda pan, tilt, duration: self._mover(
                    camera_id, pan, tilt, duration
                ),
                stopper=lambda: self._stopper(camera_id),
                limits=self._limits,
                sleep=self._sleep,
            )
            run = loop.align_to(
                reference,
                reference_anchor=reference_anchor,
                should_cancel=self._cancel.is_set,
                on_event=lambda state, iteration: self._on_event(
                    session_id, state, iteration
                ),
            )
            final_state = {
                "aligned": "centered",
                "cancelled": "cancelled",
                "low_quality": "target_lost",
                "no_progress": "target_lost",
                "motion_budget": "safety_stop",
                "max_iterations": "safety_stop",
            }.get(run.status, "target_lost")
            final_message = run.message
        except Exception:
            logger.exception("Click-to-aim session failed")
        finally:
            try:
                self._stopper(camera_id)
            except Exception:
                logger.exception("Click-to-aim final PTZ stop failed")
            finally:
                if pause_acquired and self._external_pause is not None:
                    try:
                        self._external_pause.resume_from_external()
                    except Exception:
                        logger.exception("Click-to-aim Auto-PTZ resume failed")
                self._finish(session_id, final_state, final_message)

    def _frame_supplier_required(self) -> np.ndarray:
        frame = self._frame_supplier()
        if frame is None or frame.size == 0:
            raise AimUnavailableError("No fresh camera frame is available")
        return frame

    def _on_event(
        self,
        session_id: str,
        state: str,
        iteration: VisualPtzIteration,
    ) -> None:
        alignment = iteration.alignment
        error_x = alignment.get("error_x")
        error_y = alignment.get("error_y")
        error = None
        if error_x is not None and error_y is not None:
            error = float((error_x**2 + error_y**2) ** 0.5)
        with self._lock:
            if self._status.get("session_id") != session_id:
                return
            self._status.update(
                {
                    "state": state,
                    "iteration": iteration.index,
                    "quality": alignment.get("quality"),
                    "inliers": alignment.get("inliers", 0),
                    "error": error,
                    "mapped_x": alignment.get("mapped_x"),
                    "mapped_y": alignment.get("mapped_y"),
                    "pan_command": iteration.pan_command,
                    "tilt_command": iteration.tilt_command,
                    "message": (
                        "Moving camera" if state == "moving" else "Verifying target"
                    ),
                }
            )

    def _finish(self, session_id: str, state: str, message: str) -> None:
        with self._lock:
            if self._status.get("session_id") != session_id:
                return
            self._status["active"] = False
            self._status["state"] = state
            self._status["message"] = message

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "active": False,
            "session_id": None,
            "camera_id": None,
            "state": "idle",
            "target_x": None,
            "target_y": None,
            "mapped_x": None,
            "mapped_y": None,
            "iteration": 0,
            "max_iterations": 5,
            "quality": None,
            "inliers": 0,
            "error": None,
            "pan_command": 0.0,
            "tilt_command": 0.0,
            "message": "Click Aim, then select a point in the live view",
        }
