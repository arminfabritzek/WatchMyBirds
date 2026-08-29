"""Visual keyframe storage and bounded closed-loop PTZ alignment."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.path_manager import PathManager, get_path_manager
from utils.visual_alignment import AlignmentResult, FeatureAligner


@dataclass(frozen=True)
class VisualPtzLimits:
    max_iterations: int = 8
    deadband: float = 0.025
    min_quality: float = 0.45
    gain: float = 1.2
    min_step: float = 0.06
    max_step: float = 0.22
    max_total_motion: float = 1.2
    move_duration_ms: int = 180
    max_move_duration_ms: int = 180
    move_duration_gain_ms: float = 1000.0
    settle_sec: float = 1.0
    coarse_threshold: float = 0.18
    coarse_gain: float = 1.5
    coarse_min_step: float = 0.45
    coarse_max_step: float = 0.65
    coarse_move_duration_ms: int = 400
    coarse_max_move_duration_ms: int = 400
    coarse_duration_gain_ms: float = 1000.0
    coarse_settle_sec: float = 0.45
    serialize_axes: bool = False
    min_progress: float = 0.005
    max_stagnant_iterations: int = 2
    invert_pan: bool = False
    invert_tilt: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 30:
            raise ValueError("max_iterations must be between 1 and 30")
        if not 0.001 <= self.deadband <= 0.25:
            raise ValueError("deadband must be between 0.001 and 0.25")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be between 0 and 1")
        if not 0.0 < self.min_step <= self.max_step <= 1.0:
            raise ValueError("move steps must satisfy 0 < min_step <= max_step <= 1")
        if not self.deadband < self.coarse_threshold <= 1.0:
            raise ValueError("coarse_threshold must be above deadband and at most 1")
        if self.coarse_gain <= 0:
            raise ValueError("coarse_gain must be positive")
        if not 0.0 < self.coarse_min_step <= self.coarse_max_step <= 1.0:
            raise ValueError(
                "coarse steps must satisfy 0 < coarse_min_step <= coarse_max_step <= 1"
            )
        if self.move_duration_ms <= 0 or self.coarse_move_duration_ms <= 0:
            raise ValueError("move durations must be positive")
        if not self.move_duration_ms <= self.max_move_duration_ms:
            raise ValueError("fine move duration range is invalid")
        if not self.coarse_move_duration_ms <= self.coarse_max_move_duration_ms:
            raise ValueError("coarse move duration range is invalid")
        if self.move_duration_gain_ms <= 0 or self.coarse_duration_gain_ms <= 0:
            raise ValueError("move duration gains must be positive")
        if self.settle_sec < 0 or self.coarse_settle_sec < 0:
            raise ValueError("settle durations must not be negative")
        if self.max_total_motion <= 0:
            raise ValueError("max_total_motion must be positive")


@dataclass(frozen=True)
class VisualPtzIteration:
    index: int
    alignment: dict[str, Any]
    pan_command: float = 0.0
    tilt_command: float = 0.0
    progress: float | None = None
    debug_path: str = ""


@dataclass(frozen=True)
class VisualPtzRun:
    status: str
    aligned: bool
    iterations: list[VisualPtzIteration] = field(default_factory=list)
    total_motion: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionObservation:
    """Visual evidence that one commanded camera axis moved as intended."""

    axis: str
    direction: int
    response: float | None
    unit: str
    movement_detected: bool
    direction_correct: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_motion_response(
    alignment: AlignmentResult,
    *,
    axis: str,
    direction: int,
    min_translation: float = 0.004,
    min_zoom_scale: float = 0.005,
) -> MotionObservation:
    """Interpret reference→current geometry as physical camera motion."""
    if axis not in {"pan", "tilt", "zoom"}:
        raise ValueError("axis must be pan, tilt, or zoom")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if not alignment.success:
        return MotionObservation(
            axis,
            direction,
            None,
            "normalized_frame" if axis != "zoom" else "scale_delta",
            False,
            False,
            alignment.reason or "visual alignment failed",
        )

    if axis == "pan":
        response = -alignment.error_x if alignment.error_x is not None else None
        threshold = min_translation
        unit = "normalized_frame"
    elif axis == "tilt":
        response = alignment.error_y
        threshold = min_translation
        unit = "normalized_frame"
    else:
        response = alignment.scale - 1.0 if alignment.scale is not None else None
        threshold = min_zoom_scale
        unit = "scale_delta"
    if response is None:
        return MotionObservation(
            axis,
            direction,
            None,
            unit,
            False,
            False,
            "alignment did not expose the required axis measurement",
        )
    movement_detected = abs(response) >= threshold
    direction_correct = movement_detected and response * direction > 0
    reason = ""
    if not movement_detected:
        reason = "observed movement is below the visual noise threshold"
    elif not direction_correct:
        reason = "observed movement is opposite to the commanded direction"
    return MotionObservation(
        axis,
        direction,
        response,
        unit,
        movement_detected,
        direction_correct,
        reason,
    )


class VisualKeyframeStore:
    """Persist visual anchors as disposable derivative assets."""

    def __init__(self, path_manager: PathManager | None = None) -> None:
        self.path_manager = path_manager or get_path_manager()

    def save(
        self,
        camera_id: int,
        keyframe_id: str,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        image_path = self.path_manager.get_visual_ptz_keyframe_path(
            camera_id, keyframe_id
        )
        if frame is None or frame.size == 0:
            raise ValueError("frame must not be empty")
        if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
            raise OSError(f"Could not write keyframe: {image_path}")
        payload = {
            "camera_id": int(camera_id),
            "keyframe_id": image_path.stem,
            "captured_at": datetime.now(UTC).isoformat(),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "image_path": str(image_path),
        }
        return payload

    def load(self, camera_id: int, keyframe_id: str) -> np.ndarray:
        image_path = self.path_manager.get_visual_ptz_keyframe_path(
            camera_id, keyframe_id
        )
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Visual PTZ keyframe not found: {image_path}")
        return frame

    def list(self, camera_id: int) -> list[dict[str, Any]]:
        keyframes_dir = self.path_manager.get_visual_ptz_keyframe_path(
            camera_id, "placeholder"
        ).parent
        items: list[dict[str, Any]] = []
        for image_path in sorted(keyframes_dir.glob("*.jpg")):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            items.append(
                {
                    "camera_id": int(camera_id),
                    "keyframe_id": image_path.stem,
                    "captured_at": datetime.fromtimestamp(
                        image_path.stat().st_mtime, UTC
                    ).isoformat(),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "image_path": str(image_path),
                }
            )
        return items


class OpenCvFrameSource:
    """Read fresh frames from RTSP/HTTP while keeping one connection open."""

    def __init__(
        self, stream_url: str, *, timeout_sec: float = 8.0, fresh_reads: int = 3
    ) -> None:
        self.stream_url = stream_url
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.fresh_reads = max(1, int(fresh_reads))
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> OpenCvFrameSource:
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.timeout_sec * 1000))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.timeout_sec * 1000))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.open(self.stream_url):
            capture.release()
            raise RuntimeError("Could not open camera stream")
        self._capture = capture
        return self

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Frame source is not open")
        frame = None
        for _ in range(self.fresh_reads):
            ok, candidate = self._capture.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            raise RuntimeError("Could not read a fresh camera frame")
        return frame

    def __exit__(self, *_args: Any) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class ClosedLoopVisualPtz:
    """Iteratively reduce image-space error using bounded PTZ bursts."""

    def __init__(
        self,
        *,
        aligner: FeatureAligner,
        frame_supplier: Callable[[], np.ndarray],
        mover: Callable[[float, float, int], None],
        stopper: Callable[[], None],
        limits: VisualPtzLimits | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.aligner = aligner
        self.frame_supplier = frame_supplier
        self.mover = mover
        self.stopper = stopper
        self.limits = limits or VisualPtzLimits()
        self.sleep = sleep

    def align_to(
        self,
        reference: np.ndarray,
        *,
        reference_anchor: tuple[float, float] = (0.5, 0.5),
        should_cancel: Callable[[], bool] | None = None,
        on_event: Callable[[str, VisualPtzIteration], None] | None = None,
        debug_dir: Path | None = None,
    ) -> VisualPtzRun:
        history: list[VisualPtzIteration] = []
        total_motion = 0.0
        previous_error: float | None = None
        stagnant = 0
        try:
            for index in range(1, self.limits.max_iterations + 1):
                if should_cancel and should_cancel():
                    return VisualPtzRun(
                        "cancelled", False, history, total_motion, "Aim cancelled"
                    )
                debug_path = (
                    debug_dir / f"iteration_{index:02d}.jpg" if debug_dir else None
                )
                current = self.frame_supplier()
                alignment = self.aligner.align(
                    reference,
                    current,
                    reference_anchor=reference_anchor,
                    debug_path=debug_path,
                )
                magnitude = alignment.error_magnitude
                progress = (
                    previous_error - magnitude
                    if previous_error is not None and magnitude is not None
                    else None
                )
                base_iteration = {
                    "index": index,
                    "alignment": alignment.to_dict(),
                    "progress": progress,
                    "debug_path": str(debug_path or ""),
                }
                if not alignment.success or alignment.quality < self.limits.min_quality:
                    iteration = VisualPtzIteration(**base_iteration)
                    history.append(iteration)
                    if on_event:
                        on_event("verifying", iteration)
                    return VisualPtzRun(
                        "low_quality",
                        False,
                        history,
                        total_motion,
                        alignment.reason
                        or "Visual match quality is below the safety threshold",
                    )
                assert alignment.error_x is not None and alignment.error_y is not None
                if (
                    abs(alignment.error_x) <= self.limits.deadband
                    and abs(alignment.error_y) <= self.limits.deadband
                ):
                    iteration = VisualPtzIteration(**base_iteration)
                    history.append(iteration)
                    if on_event:
                        on_event("verifying", iteration)
                    return VisualPtzRun(
                        "aligned",
                        True,
                        history,
                        total_motion,
                        "Target view reached inside the visual deadband",
                    )

                if progress is not None and progress < self.limits.min_progress:
                    stagnant += 1
                else:
                    stagnant = 0
                if stagnant >= self.limits.max_stagnant_iterations:
                    iteration = VisualPtzIteration(**base_iteration)
                    history.append(iteration)
                    if on_event:
                        on_event("verifying", iteration)
                    return VisualPtzRun(
                        "no_progress",
                        False,
                        history,
                        total_motion,
                        "Visual error did not improve; stopped before further motion",
                    )

                pan, tilt, move_duration_ms, settle_sec = self._motion_command(
                    alignment.error_x,
                    alignment.error_y,
                )
                if self.limits.invert_pan:
                    pan = -pan
                if self.limits.invert_tilt:
                    tilt = -tilt
                move_commands = [(pan, tilt)]
                if self.limits.serialize_axes and pan != 0.0 and tilt != 0.0:
                    move_commands = [(pan, 0.0), (0.0, tilt)]
                proposed = sum(
                    math.hypot(command_pan, command_tilt) * move_duration_ms / 1000.0
                    for command_pan, command_tilt in move_commands
                )
                if total_motion + proposed > self.limits.max_total_motion:
                    iteration = VisualPtzIteration(**base_iteration)
                    history.append(iteration)
                    if on_event:
                        on_event("verifying", iteration)
                    return VisualPtzRun(
                        "motion_budget",
                        False,
                        history,
                        total_motion,
                        "Cumulative PTZ safety budget exhausted",
                    )
                iteration = VisualPtzIteration(
                    **base_iteration,
                    pan_command=pan,
                    tilt_command=tilt,
                )
                history.append(iteration)
                if on_event:
                    on_event("moving", iteration)
                for command_pan, command_tilt in move_commands:
                    if should_cancel and should_cancel():
                        return VisualPtzRun(
                            "cancelled",
                            False,
                            history,
                            total_motion,
                            "Aim cancelled",
                        )
                    self.mover(command_pan, command_tilt, move_duration_ms)
                total_motion += proposed
                previous_error = magnitude
                self.sleep(settle_sec)
        except BaseException:
            self.stopper()
            raise

        if should_cancel and should_cancel():
            return VisualPtzRun(
                "cancelled", False, history, total_motion, "Aim cancelled"
            )
        current = self.frame_supplier()
        alignment = self.aligner.align(
            reference,
            current,
            reference_anchor=reference_anchor,
            debug_path=None,
        )
        final_iteration = VisualPtzIteration(
            index=self.limits.max_iterations,
            alignment=alignment.to_dict(),
            progress=(
                previous_error - alignment.error_magnitude
                if previous_error is not None and alignment.error_magnitude is not None
                else None
            ),
        )
        history.append(final_iteration)
        if on_event:
            on_event("verifying", final_iteration)
        if not alignment.success or alignment.quality < self.limits.min_quality:
            self.stopper()
            return VisualPtzRun(
                "low_quality",
                False,
                history,
                total_motion,
                alignment.reason
                or "Visual match quality is below the safety threshold",
            )
        assert alignment.error_x is not None and alignment.error_y is not None
        if (
            abs(alignment.error_x) <= self.limits.deadband
            and abs(alignment.error_y) <= self.limits.deadband
        ):
            self.stopper()
            return VisualPtzRun(
                "aligned",
                True,
                history,
                total_motion,
                "Target view reached inside the visual deadband",
            )
        self.stopper()
        return VisualPtzRun(
            "max_iterations",
            False,
            history,
            total_motion,
            "Maximum iteration count reached",
        )

    def _motion_command(
        self,
        error_x: float,
        error_y: float,
    ) -> tuple[float, float, int, float]:
        pan_error = error_x if abs(error_x) > self.limits.deadband else 0.0
        tilt_error = -error_y if abs(error_y) > self.limits.deadband else 0.0
        dominant_error = max(abs(pan_error), abs(tilt_error))

        if dominant_error >= self.limits.coarse_threshold:
            gain = self.limits.coarse_gain
            min_step = self.limits.coarse_min_step
            max_step = self.limits.coarse_max_step
            duration_ms = min(
                self.limits.coarse_max_move_duration_ms,
                max(
                    self.limits.coarse_move_duration_ms,
                    round(dominant_error * self.limits.coarse_duration_gain_ms),
                ),
            )
            settle_sec = self.limits.coarse_settle_sec
        else:
            gain = self.limits.gain
            min_step = self.limits.min_step
            max_step = self.limits.max_step
            duration_ms = min(
                self.limits.max_move_duration_ms,
                max(
                    self.limits.move_duration_ms,
                    round(dominant_error * self.limits.move_duration_gain_ms),
                ),
            )
            settle_sec = self.limits.settle_sec

        dominant_speed = min(max_step, max(min_step, dominant_error * gain))
        scale = dominant_speed / dominant_error
        return (
            pan_error * scale,
            tilt_error * scale,
            duration_ms,
            settle_sec,
        )
