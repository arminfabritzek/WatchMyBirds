"""
Auto PTZ tracking state machine.

The controller consumes lightweight detection signals and queues PTZ commands
for a background worker so object detection never waits on ONVIF I/O.
"""

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from core import ptz_core
from core.ptz_grid import cell_for_center, cell_preset_name
from detectors.od_classes import is_bird_od_class
from logging_config import get_logger

logger = get_logger(__name__)

# Goto-retry policy for cheap ONVIF PTZ cameras.
#
# Observed live on 2026-05-16: a cheap camera will sometimes reject
# GotoPreset with a generic "Preset token does not exist" SOAP fault
# even though GetPresets lists the same token — typically when a
# previous goto is still mid-flight, when the camera is busy with an
# internal task, or as a transient SOAP/network blip. A short retry
# converts most of those into eventual success without flooding the
# camera. Only goto commands are retried; move/stop are time-critical
# (they reflect operator joystick state) and must not be replayed.
_GOTO_RETRY_ATTEMPTS = 2  # additional attempts after the first → 3 total
_GOTO_RETRY_BACKOFF_SEC = 0.8

PtzState = Literal[
    "idle", "overview", "settling", "acquiring", "tracking", "lost_grace", "returning"
]


@dataclass(frozen=True)
class PtzCommand:
    action: Literal["goto", "move", "stop"]
    camera_id: int
    preset_token: str = ""
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    duration_ms: int = 250
    # Speculative state committed under _lock at enqueue time so the
    # cooldown gate sees the in-flight target. If the worker later
    # reports the goto failed, the rollback uses these fields to undo
    # the commit — but only if _last_preset still matches (CAS check),
    # so a newer enqueue is never clobbered by a stale failure.
    rollback_preset: str = ""
    rollback_zone: str = ""


class AutoPtzController:
    """Preset-first auto PTZ controller with optional hybrid move tracking."""

    def __init__(
        self,
        *,
        camera_provider: Callable[[], dict[str, Any] | None] | None = None,
        command_runner: Callable[[PtzCommand], None] | None = None,
        clock: Callable[[], float] | None = None,
        worker_enabled: bool = True,
    ) -> None:
        self._camera_provider = camera_provider or ptz_core.find_auto_ptz_camera
        self._command_runner = command_runner or self._run_command
        self._clock = clock or time.monotonic
        self._worker_enabled = worker_enabled
        self._queue: queue.Queue[PtzCommand] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state: PtzState = "idle"
        self._last_seen_mono = 0.0
        self._last_command_mono = 0.0
        self._last_error = ""
        self._last_zone = ""
        self._last_preset = ""
        self._acquire_count = 0
        self._last_target_center: tuple[float, float] | None = None
        self._manual_view_until: float = 0.0  # 0 = no manual-view override active
        self._acquiring_preset: str = ""  # token currently being acquired
        self._grid_current_cell: tuple[int, int] | None = (
            None  # last cell tracked in grid mode
        )
        # Lost-detection cooldown: when a frame has no Bird above the
        # confidence threshold, we set this deadline and reject new
        # detection-driven moves until it elapses. Lets the cam actually
        # complete its return-to-overview goto before the next stray
        # high-confidence false-positive yanks it back into tracking.
        # 0.0 = no cooldown active.
        self._lost_cooldown_until: float = 0.0
        # External pause (e.g., the in-UI empirical probe wizard) takes
        # exclusive control of the camera. While set, handle_detections /
        # handle_no_detection become no-ops so the controller cannot
        # fight the wizard's commands. Empty string = not paused.
        self._external_pause_reason: str = ""
        # Follow-mode zoom-in budget: total seconds of zoom-in burst
        # accumulated since the last overview goto. Bounds the lens at
        # its near-focus limit on cams without absolute zoom feedback.
        # Reset to 0 by every return_to_overview / lost-timeout goto.
        self._zoom_in_budget_used_sec: float = 0.0
        # Set when the operator took manual control (joystick or non-
        # overview preset goto) so we have no idea where the lens
        # currently sits. Follow-mode refuses to zoom in until an
        # overview goto re-establishes a known wide-angle baseline.
        # Safer than guessing; clears on overview goto.
        self._zoom_in_locked_until_overview: bool = False

        self._worker: threading.Thread | None = None
        if worker_enabled:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="auto-ptz-worker",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def handle_detections(
        self,
        *,
        frame_shape: tuple[int, ...],
        detections: list[dict[str, Any]],
        min_confidence: float = 0.0,
    ) -> None:
        """Process a frame's detections and optionally move the camera.

        ``min_confidence`` filters out weak detections **before** anything
        else happens — phantom Bird boxes (leaves, shadows, low-confidence
        false positives between the detection floor and the save threshold)
        used to keep the cam endlessly chasing nothing because the tracker
        saw any class=bird, conf>floor entry as a target. Calling code
        passes the effective save-threshold so the tracker reacts to the
        same detections the gallery would persist.

        When all detections are below ``min_confidence`` the call falls
        through to ``handle_no_detection``: the cam halts the in-flight
        burst (follow-mode) and counts down to the overview return.
        """
        # External pause has the highest priority: when the in-UI empirical
        # probe wizard (or any future external owner) has the camera,
        # detection-driven moves must NOT fire. We do not even update
        # state here — paused means strictly invisible to the operator's
        # current task.
        with self._lock:
            if self._external_pause_reason:
                return

        # Lost-detection cooldown: handle_no_detection set a deadline
        # so the cam can actually reach the overview preset before the
        # next stray high-confidence false-positive yanks it back into
        # tracking. While the cooldown is active, ignore detections
        # entirely — no state mutation, no _last_seen_mono refresh.
        # The countdown to overview keeps ticking off the original
        # last_seen value, exactly as the operator's UI shows.
        now = self._clock()
        with self._lock:
            if self._lost_cooldown_until > 0 and now < self._lost_cooldown_until:
                return
            if self._lost_cooldown_until > 0 and now >= self._lost_cooldown_until:
                # Cooldown elapsed — clear it so future no-detect cycles
                # can re-arm fresh ones.
                self._lost_cooldown_until = 0.0

        camera = self._camera_provider()
        if not camera:
            self._set_idle("No enabled PTZ camera matches the active stream")
            return

        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        if not config.get("enabled"):
            self._set_idle("Auto PTZ disabled")
            return

        # While the camera is still flying to a freshly-issued preset,
        # the detector keeps seeing mid-flight frames that paint the
        # bird into arbitrary zones. Acting on them would chain-fire
        # counter-gotos. Refresh the lost-grace anchor so the bird is
        # not declared lost during the settle window, but otherwise
        # ignore the frame — the settle worker will flip us back to
        # "tracking" once wait_until_idle resolves.
        with self._lock:
            if self._state == "settling" and self._last_zone != "manual":
                self._last_seen_mono = self._clock()
                return

        # Phantom-detection guard: filter out weak detections BEFORE
        # picking a target. Leaves/shadows/low-confidence false positives
        # between the detection floor and the save threshold used to
        # keep the cam chasing nothing because the tracker treated any
        # bird-class detection as a target regardless of confidence.
        # The caller (detection_manager) passes the same effective
        # save-threshold the gallery uses so the cam reacts to exactly
        # the detections the operator would have persisted anyway.
        bird_dets = [
            d for d in detections
            if is_bird_od_class(str(d.get("class_name") or "bird"))
        ]
        bird_confidences = [
            float(d.get("confidence") or 0.0) for d in bird_dets
        ]
        if bird_confidences:
            # Log every frame's bird confidences so we can correlate the
            # PTZ moves below with the actual detection strengths. Helps
            # debug "cam keeps moving even though no bird is visible"
            # reports: usually a false-positive at conf >= min_confidence.
            logger.info(
                "Auto PTZ frame conf=[%s] min=%.2f n_bird=%d",
                ", ".join(f"{c:.2f}" for c in sorted(bird_confidences, reverse=True)),
                min_confidence,
                len(bird_dets),
            )
        if min_confidence > 0.0:
            detections = [
                d for d in detections
                if float(d.get("confidence") or 0.0) >= min_confidence
            ]
            if not detections:
                self.handle_no_detection()
                return

        target = self._select_target(frame_shape=frame_shape, detections=detections)
        if not target:
            self.handle_no_detection()
            return

        # Grid mode is a separate dispatch — different routing math,
        # different cooldown, different "zone" semantics. Kept out of
        # the preset/hybrid path so the legacy logic stays untouched.
        if config.get("mode") == "grid":
            self._handle_detections_grid(
                config=config,
                camera=camera,
                target=target,
            )
            return

        # Follow mode is preset-free: continuous pan/tilt + zoom keep
        # the bbox in the centre at the target size. No "zone" concept,
        # no acquire/settling state machine — every accepted detection
        # frame is a steering input gated only by deadband + cooldown.
        if config.get("mode") == "follow":
            self._handle_detections_follow(
                config=config,
                camera=camera,
                target=target,
                frame_shape=frame_shape,
                detections=detections,
            )
            return

        now = self._clock()
        center_x, center_y, confidence = target
        zone = self._zone_for_center(config, center_x, center_y)
        if not zone or not zone.get("preset"):
            self._update_status(
                state="acquiring",
                error="Detected bird is outside configured PTZ zones",
                target_center=(center_x, center_y),
            )
            return

        zone_name = str(zone.get("name") or "")
        zone_preset = str(zone.get("preset") or "")
        with self._lock:
            self._last_seen_mono = now
            self._last_target_center = (center_x, center_y)
            # Bird detection always reverts to the detection-driven timeout,
            # even if a manual-view override was active from an earlier click.
            self._manual_view_until = 0.0
            zone_changed = (
                self._state in {"acquiring", "tracking"}
                and zone_preset
                and self._acquiring_preset
                and zone_preset != self._acquiring_preset
            )
            if self._state not in {"acquiring", "tracking"} or zone_changed:
                # Fresh acquire window for a new target box so a flapping
                # bird hopping between boxes does not chain-trigger gotos.
                self._acquire_count = 0
            self._acquiring_preset = zone_preset
            self._acquire_count += 1

            if self._acquire_count < int(config["acquire_frames"]):
                self._state = "acquiring"
                self._last_error = ""
                return

        camera_id = int(camera["id"])
        preset_token = zone_preset
        issued = self._maybe_goto_zone(
            camera_id=camera_id,
            preset_token=preset_token,
            zone_name=zone_name,
            config=config,
            now=now,
        )

        if config["mode"] == "hybrid":
            self._maybe_move_to_center(
                camera_id=camera_id,
                center_x=center_x,
                center_y=center_y,
                config=config,
                now=now,
            )

        with self._lock:
            # When a goto fired, park in "settling" until the camera
            # actually arrives. The settle worker (kicked below) flips
            # us back to "tracking" once wait_until_idle resolves —
            # otherwise mid-flight detection frames would route the
            # bird into whatever zone the wide-angle frame paints it
            # in, and the controller would fire a counter-goto as soon
            # as the (shorter) command cooldown expires. Cheap PTZ
            # cameras take 2–6 s to traverse, far longer than the
            # cooldown alone protects against.
            self._state = "settling" if issued else "tracking"
            self._last_error = ""
            self._last_target_center = (center_x, center_y)
            logger.debug(
                "Auto PTZ tracking target zone=%s conf=%.3f center=(%.3f, %.3f)",
                zone_name,
                confidence,
                center_x,
                center_y,
            )

        if issued:
            self._spawn_detection_settle_worker(camera_id, config)

    def _handle_detections_grid(
        self,
        *,
        config: dict[str, Any],
        camera: dict[str, Any],
        target: tuple[float, float, float],
    ) -> None:
        """Grid-mode dispatch: route bbox center to a (row, col) cell.

        Reuses the same acquire_frames / state-machine semantics as the
        preset path, but with grid-specific routing, naming, and a
        separate cooldown. Hysteresis on cell selection is the design's
        answer to flap between adjacent cells when the bird sits on a
        boundary — see core.ptz_grid.cell_for_center.
        """
        now = self._clock()
        center_x, center_y, confidence = target

        shape = config.get("grid_shape") or [3, 3]
        try:
            rows, cols = int(shape[0]), int(shape[1])
        except (IndexError, TypeError, ValueError):
            rows, cols = 3, 3

        hysteresis = float(config.get("grid_hysteresis_margin") or 0.05)
        with self._lock:
            current = self._grid_current_cell
        row, col = cell_for_center(
            center_x, center_y, rows, cols, current_cell=current, hysteresis=hysteresis
        )
        cell_key = f"r{row}_c{col}"
        grid_cells = config.get("grid_cells") or {}
        preset_token = str(grid_cells.get(cell_key) or "").strip()
        if not preset_token:
            # Operator did not finish the setup wizard for this cell.
            self._update_status(
                state="acquiring",
                error=f"Grid cell {cell_key} has no preset configured",
                target_center=(center_x, center_y),
            )
            return

        cell_name = cell_preset_name(row, col)
        with self._lock:
            self._last_seen_mono = now
            self._last_target_center = (center_x, center_y)
            self._manual_view_until = 0.0
            cell_changed = (
                self._state in {"acquiring", "tracking"}
                and self._grid_current_cell is not None
                and self._grid_current_cell != (row, col)
            )
            if self._state not in {"acquiring", "tracking"} or cell_changed:
                self._acquire_count = 0
            self._grid_current_cell = (row, col)
            self._acquiring_preset = preset_token
            self._acquire_count += 1

            if self._acquire_count < int(config["grid_acquire_frames"]):
                self._state = "acquiring"
                self._last_error = ""
                return

        camera_id = int(camera["id"])
        issued = self._maybe_goto_cell(
            camera_id=camera_id,
            preset_token=preset_token,
            cell_name=cell_name,
            config=config,
            now=now,
        )

        with self._lock:
            # See preset-mode handle_detections for the rationale —
            # settle worker prevents counter-gotos from mid-flight frames.
            self._state = "settling" if issued else "tracking"
            self._last_error = ""
            self._last_target_center = (center_x, center_y)
            logger.debug(
                "Auto PTZ grid tracking cell=%s conf=%.3f center=(%.3f, %.3f)",
                cell_name,
                confidence,
                center_x,
                center_y,
            )

        if issued:
            self._spawn_detection_settle_worker(camera_id, config)

    def _handle_detections_follow(
        self,
        *,
        config: dict[str, Any],
        camera: dict[str, Any],
        target: tuple[float, float, float],
        frame_shape: tuple[int, ...],
        detections: list[dict[str, Any]],
    ) -> None:
        """Follow-mode dispatch: continuous pan/tilt + zoom keep the bbox
        centred at the target size. No presets, no acquire window — each
        accepted frame steers, gated only by deadband + cooldown.

        Requires the cam's empirical probe to report continuous_works.
        Continuous-zoom is gated separately: if declared and empirical OK,
        zoom corrections fire; otherwise pan/tilt only.
        """
        now = self._clock()
        center_x, center_y, confidence = target

        # Look up the winning detection's bbox area for the zoom signal.
        # Re-finding it here keeps _select_target's contract unchanged.
        area_pct = self._bbox_area_pct_for_target(
            frame_shape=frame_shape, detections=detections, center=(center_x, center_y)
        )

        with self._lock:
            self._last_seen_mono = now
            self._last_target_center = (center_x, center_y)
            self._manual_view_until = 0.0
            self._state = "tracking"
            self._last_error = ""
            self._last_zone = "follow"
            self._last_preset = ""

        camera_id = int(camera["id"])
        cooldown_sec = int(config["command_cooldown_ms"]) / 1000.0
        with self._lock:
            if now - self._last_command_mono < cooldown_sec:
                return

        offset_x = center_x - 0.5
        offset_y = center_y - 0.5
        deadband = float(config["deadband"])
        max_speed = float(config["max_speed"])

        # P-gain is deliberately low (0.8 vs hybrid-mode's 2.0). Reason:
        # most cheap PTZ firmware ignores the ONVIF duration_sec parameter
        # and runs each ContinuousMove for its own ~800-1000ms regardless.
        # A high P-gain combined with the long actual burst causes overshoot
        # past the centre, then the next detection frame sees the bird now
        # ABOVE the centre, fires a tilt-down burst that overshoots again
        # — classic limit-cycle oscillation. With 0.8 the cam approaches
        # the centre asymptotically over a few frames instead.
        pan = 0.0
        tilt = 0.0
        if abs(offset_x) > deadband:
            pan = max(-max_speed, min(max_speed, offset_x * max_speed * 0.8))
        if abs(offset_y) > deadband:
            tilt = max(-max_speed, min(max_speed, -offset_y * max_speed * 0.8))

        zoom = 0.0
        if area_pct is not None:
            target_pct = float(config.get("follow_zoom_target_pct", 0.18))
            zoom_deadband = float(config.get("follow_zoom_deadband_pct", 0.05))
            zoom_speed = float(config.get("follow_zoom_speed", 0.3))
            delta = target_pct - area_pct
            if abs(delta) > zoom_deadband:
                # area too small (delta > 0) → zoom IN (positive zoom);
                # area too big (delta < 0) → zoom OUT (negative zoom).
                zoom = zoom_speed if delta > 0 else -zoom_speed

        # Near-focus lens guard. On cams without absolute zoom feedback
        # (GetStatus stub), the area-driven loop can drive the lens
        # past its near-focus distance for close subjects and produce
        # blurred frames. The operator declares the per-cam budget in
        # follow_zoom_max_burst_sec; once spent, further zoom-in is
        # suppressed. Zoom-out always passes — it's the direction that
        # *releases* the lens, and we want that available even after
        # the budget is exhausted. 0.0 disables the guard entirely.
        #
        # Additionally, if the operator took manual control (joystick
        # or non-overview goto) we lock zoom-in entirely until an
        # overview goto re-establishes a known wide-angle baseline.
        # The exact post-manual zoom position is unknown without
        # absolute feedback, so refusing to zoom is the only honest
        # option.
        budget_sec = float(config.get("follow_zoom_max_burst_sec", 0.0) or 0.0)
        if zoom > 0.0:
            with self._lock:
                if self._zoom_in_locked_until_overview:
                    zoom = 0.0
                elif budget_sec > 0.0 and self._zoom_in_budget_used_sec >= budget_sec:
                    zoom = 0.0

        if pan == 0.0 and tilt == 0.0 and zoom == 0.0:
            return

        with self._lock:
            self._last_command_mono = now
            if zoom > 0.0:
                # Charge a zoom-in burst against the budget. We use
                # move_duration_ms as the unit (not the firmware's
                # actual ~800-1000 ms burst length) so the budget
                # stays a function of what we *commanded*, which is
                # what the operator's setting can reason about.
                self._zoom_in_budget_used_sec += (
                    float(config.get("move_duration_ms", 250)) / 1000.0
                )

        self._enqueue(
            PtzCommand(
                action="move",
                camera_id=camera_id,
                pan=pan,
                tilt=tilt,
                zoom=zoom,
                duration_ms=int(config["move_duration_ms"]),
            )
        )
        logger.debug(
            "Auto PTZ follow conf=%.3f center=(%.3f, %.3f) area=%s "
            "→ pan=%.2f tilt=%.2f zoom=%.2f",
            confidence, center_x, center_y,
            f"{area_pct:.3f}" if area_pct is not None else "n/a",
            pan, tilt, zoom,
        )

    def _bbox_area_pct_for_target(
        self,
        *,
        frame_shape: tuple[int, ...],
        detections: list[dict[str, Any]],
        center: tuple[float, float],
    ) -> float | None:
        """Find the bbox whose centre matches `center` (within 1px) and
        return its area as a fraction of the frame. Returns None when
        the bbox cannot be located — caller treats that as 'skip zoom'.
        """
        if not frame_shape or len(frame_shape) < 2:
            return None
        frame_h = max(1, int(frame_shape[0]))
        frame_w = max(1, int(frame_shape[1]))
        target_cx_px = center[0] * frame_w
        target_cy_px = center[1] * frame_h
        for det in detections:
            try:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if abs(cx - target_cx_px) <= 1.5 and abs(cy - target_cy_px) <= 1.5:
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                return (w * h) / float(frame_w * frame_h)
        return None

    def _maybe_goto_cell(
        self,
        *,
        camera_id: int,
        preset_token: str,
        cell_name: str,
        config: dict[str, Any],
        now: float,
    ) -> bool:
        """Like _maybe_goto_zone but uses the grid-specific cooldown.

        Shorter cooldown (default 4s vs preset-mode's 10s) because
        adjacent-cell switching is the *normal* flow, not the exception.
        Hysteresis already filters the boundary-flap case upstream.

        Returns True if a goto was issued.
        """
        cooldown_sec = int(config["grid_command_cooldown_ms"]) / 1000.0
        with self._lock:
            same_target = (
                self._last_preset == preset_token and self._last_zone == cell_name
            )
            if same_target:
                return False
            if now - self._last_command_mono < cooldown_sec:
                return False
            prev_preset = self._last_preset
            prev_zone = self._last_zone
            self._last_command_mono = now
            self._last_preset = preset_token
            self._last_zone = cell_name

        logger.info(
            "AutoPTZ grid trigger cell=%s preset=%s camera_id=%s",
            cell_name,
            preset_token,
            camera_id,
        )
        self._enqueue(
            PtzCommand(
                action="goto",
                camera_id=camera_id,
                preset_token=preset_token,
                rollback_preset=prev_preset,
                rollback_zone=prev_zone,
            )
        )
        return True

    def handle_no_detection(self) -> None:
        with self._lock:
            # External pause owns the camera — no auto-return either.
            if self._external_pause_reason:
                return
            if self._state not in {"acquiring", "tracking", "lost_grace"}:
                return
            was_tracking = self._state == "tracking"

        camera = self._camera_provider()
        if not camera:
            self._set_idle("No enabled PTZ camera matches the active stream")
            return

        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        if not config.get("enabled"):
            self._set_idle("Auto PTZ disabled")
            return

        # Follow-mode specific: halt any in-flight continuous burst the
        # moment we lose the bird. Without this, the cam runs out its
        # firmware-dictated ~800-1000ms burst even though the target is
        # gone — looks like the cam is "still following the old bbox"
        # to the operator. Stop() is idempotent and harmless on cams
        # that already finished the burst. Only fire on the tracking→
        # lost_grace transition, not on every subsequent no-detection
        # frame, to avoid hammering the cam with Stop()s.
        if was_tracking and config.get("mode") == "follow":
            self._enqueue(
                PtzCommand(
                    action="stop",
                    camera_id=int(camera["id"]),
                )
            )

        now = self._clock()
        with self._lock:
            if self._last_seen_mono <= 0:
                return
            # Manual goto sets _manual_view_until as an explicit deadline
            # (longer than lost_timeout_sec, starting after camera settle).
            # Detection-driven lost_grace falls back to last_seen + timeout.
            if self._manual_view_until > 0:
                deadline = self._manual_view_until
            else:
                deadline = self._last_seen_mono + float(config["lost_timeout_sec"])
            if now < deadline:
                self._state = "lost_grace"
                return
            # Deadline reached — clear the manual override before firing.
            self._manual_view_until = 0.0

        overview = str(config.get("overview_preset") or "")
        if not overview:
            self._update_status(state="idle", error="Overview preset is not configured")
            return

        self._enqueue(
            PtzCommand(
                action="goto",
                camera_id=int(camera["id"]),
                preset_token=overview,
            )
        )
        # Arm a cooldown so the next few detection frames don't yank
        # the cam back into tracking before it reaches the overview.
        # Without this, persistent false-positives keep undoing every
        # return: handle_no_detection fires goto, next detection frame
        # ~2s later resets state to "tracking", cam never arrives.
        # Cooldown duration is lost_timeout_sec — same window the
        # operator already configured as "how long without a bird
        # before I give up." Reusing it keeps the model simple.
        lost_cd = float(config.get("lost_timeout_sec") or 6.0)
        with self._lock:
            self._state = "returning"
            self._last_preset = overview
            self._last_zone = "overview"
            self._acquire_count = 0
            self._lost_cooldown_until = now + lost_cd
            # Overview is the one absolute reference point the lens has;
            # clearing the zoom-in budget here gives the next bird a
            # fresh budget without needing GetStatus feedback. Also
            # clears the manual-control lock so follow-mode can zoom
            # again now that we're definitively at a known position.
            self._zoom_in_budget_used_sec = 0.0
            self._zoom_in_locked_until_overview = False

    def notify_external_goto(self, preset_token: str) -> None:
        """Record a preset goto that was triggered outside this controller.

        Manual UI clicks bypass the controller's own goto path, leaving
        last_preset stale. Callers in the web layer pass the token here
        so the /ptz/auto/status response stays accurate for the UI.

        When auto-PTZ is enabled and the manual goto sent the camera to
        a non-overview preset, we kick off a background settle-then-park
        worker so the manual_view_sec countdown only starts after the
        camera has actually arrived.
        """
        if not preset_token:
            return

        camera = self._camera_provider()
        config = ptz_core.normalize_ptz_config((camera or {}).get("ptz"))
        overview = str(config.get("overview_preset") or "")
        auto_enabled = bool(config.get("enabled"))
        seeds_manual_grace = (
            auto_enabled
            and overview
            and preset_token != overview
            and camera is not None
        )

        now = self._clock()
        with self._lock:
            self._last_preset = str(preset_token)
            self._last_zone = "manual"
            # Honor the command cooldown: a manual goto IS a command,
            # so subsequent detection-driven gotos must wait the same
            # cooldown_sec before they can fire. Without this, the
            # detection loop happily issues a counter-goto on the very
            # next frame while the camera is still flying — and because
            # the still-Home/wide-angle frame paints the bird in some
            # neighbouring zone, the camera lurches to that zone first
            # before lost_grace eventually returns it to overview.
            self._last_command_mono = now
            if seeds_manual_grace:
                # Park the controller in "settling" — countdown does not
                # start yet. The background settle-worker flips us to
                # lost_grace once the camera reports IDLE (or fallback).
                self._state = "settling"
                self._last_seen_mono = 0.0
                self._manual_view_until = 0.0
                self._acquire_count = 0
                # Operator sent the cam to a non-overview preset — each
                # preset has its own baked-in zoom level we cannot read
                # back. Lock follow-mode zoom-in until an overview goto
                # re-establishes the known wide-angle baseline.
                self._zoom_in_locked_until_overview = True
            else:
                # Manual goto to overview cancels any pending manual return.
                self._manual_view_until = 0.0
                # Overview is the wide-angle reference; fresh budget,
                # lock cleared.
                self._zoom_in_budget_used_sec = 0.0
                self._zoom_in_locked_until_overview = False

        if seeds_manual_grace:
            assert camera is not None
            cam_id = int(camera["id"])
            settle_max = float(config.get("settle_max_sec") or 8.0)
            view_sec = float(config.get("manual_view_sec") or 15.0)
            t = threading.Thread(
                target=self._settle_then_park_manual,
                name="auto-ptz-settle",
                args=(cam_id, settle_max, view_sec),
                daemon=True,
            )
            t.start()

    def notify_manual_drive(self) -> None:
        """Record a manual joystick move from the stream-page buttons.

        Unlike ``notify_external_goto``, manual drive uses
        ``continuous_move`` which auto-stops after ``duration_ms`` — no
        settle worker needed. The countdown is *refreshable*: every
        Heartbeat-frequency move call pushes the auto-return deadline
        back another ``manual_view_sec`` seconds. So the camera stays
        quiet while the operator is steering, and the deadline only
        starts ticking down after the operator releases the button.

        No-op when auto-PTZ is disabled or no overview preset exists
        (without an overview to return to, there's nothing to gate).
        """
        camera = self._camera_provider()
        if not camera:
            return
        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        if not config.get("enabled"):
            return
        if not str(config.get("overview_preset") or ""):
            return

        now = self._clock()
        view_sec = float(config.get("manual_view_sec") or 15.0)
        with self._lock:
            # Manual drive overrides the lost-detection cooldown — the
            # operator's intent always wins. Without this, a recently
            # triggered auto-return would silently swallow the manual
            # joystick presses for several seconds.
            self._lost_cooldown_until = 0.0
            # Operator-controlled motion: zoom position is now unknown
            # without absolute feedback. Lock follow-mode zoom-in until
            # an overview goto re-establishes the wide-angle baseline.
            self._zoom_in_locked_until_overview = True
            # Refresh the deadline on every heartbeat. last_seen_mono is
            # set so handle_no_detection's fallback path (when manual_view
            # is not active) still has a sensible anchor if the manual
            # session ends without a clean release.
            self._last_seen_mono = now
            self._manual_view_until = now + view_sec
            # Seed _last_command_mono so the detection-driven cooldown
            # gate also applies after manual joystick drive — mirroring
            # the same race-protection commit 2e15f32 added for
            # notify_external_goto and return_to_overview. Without this,
            # mid-fly detection frames (still showing the pre-drive view)
            # paint the bird into a neighbouring zone, and the controller
            # fires a counter-goto with no cooldown gate.
            self._last_command_mono = now
            self._last_zone = "manual_drive"
            self._last_preset = ""  # no preset — operator is freely steering
            self._state = "lost_grace"
            self._acquire_count = 0

    # ------------------------------------------------------------------
    # External-pause API
    #
    # The in-UI empirical probe wizard (active plan
    # 2026-05-18_PTZ_probe-ui-integration) takes exclusive control of
    # the camera while the operator walks through move tests. While
    # paused, handle_detections and handle_no_detection are no-ops so
    # the controller cannot fight the wizard's commands.
    #
    # The pause is *cooperative* — anyone calling pause_for_external
    # must guarantee resume_from_external is called in a finally-block,
    # or the user is locked out of auto-PTZ until app restart.
    # ------------------------------------------------------------------

    def pause_for_external(self, reason: str) -> bool:
        """Acquire exclusive external control of the camera.

        Returns True if the pause was newly acquired, False if the SAME
        reason already holds the pause (idempotent re-pause from the
        same owner is a no-op success).

        Raises ``RuntimeError`` when a DIFFERENT reason is already
        holding the pause — refusing a second owner is the safety
        guarantee: two probes against the same controller would race,
        and one resume would silently release the other's lock.

        ``reason`` is a short human-readable label surfaced to the UI
        ("empirical probe", "firmware update", etc) so the Stream-page
        banner can explain what's holding the camera.
        """
        reason_str = str(reason or "external").strip() or "external"
        with self._lock:
            current = self._external_pause_reason
            if current and current != reason_str:
                raise RuntimeError(
                    f"AutoPtzController already paused by {current!r}; "
                    f"refusing to overwrite with {reason_str!r}"
                )
            was_paused = bool(current)
            self._external_pause_reason = reason_str
        if not was_paused:
            logger.info("AutoPtzController paused (reason=%s)", reason_str)
        return not was_paused

    def resume_from_external(self) -> bool:
        """Release the external-pause lock.

        Returns True if the controller was paused (and is now resumed),
        False if it wasn't paused (no-op). Safe to call from finally
        blocks regardless of pause state.
        """
        with self._lock:
            was_paused = bool(self._external_pause_reason)
            self._external_pause_reason = ""
        if was_paused:
            logger.info("AutoPtzController resumed from external pause")
        return was_paused

    def is_paused(self) -> tuple[bool, str]:
        """Return (paused, reason). Used by status endpoint + UI banner."""
        with self._lock:
            reason = self._external_pause_reason
        return (bool(reason), reason)

    def _spawn_detection_settle_worker(
        self, camera_id: int, config: dict[str, Any]
    ) -> None:
        """Start a background settler for a detection-driven goto.

        Cheap PTZ cameras take 2–6 s to traverse between presets; if we
        let the detection loop act on frames captured during the flight,
        the bird's bbox lands in arbitrary zones and chain-fires
        counter-gotos. The settle worker waits for ONVIF MoveStatus to
        report IDLE (or a fixed-time fallback when the camera does not
        expose it) and only then flips _state back to "tracking" so
        detections are honored again.

        Skipped when the controller's command worker is disabled — the
        same flag tests use to keep behaviour synchronous and avoid
        background ONVIF I/O that would fail without a real camera.
        Synchronous callers flip straight to "tracking" instead.
        """
        if not self._worker_enabled:
            with self._lock:
                if self._state == "settling":
                    self._state = "tracking"
            return
        settle_max = float(config.get("settle_max_sec") or 8.0)
        t = threading.Thread(
            target=self._settle_then_resume_tracking,
            name="auto-ptz-detection-settle",
            args=(camera_id, settle_max),
            daemon=True,
        )
        t.start()

    def _settle_then_resume_tracking(
        self, camera_id: int, settle_max_sec: float
    ) -> None:
        """Block until the camera arrives, then unlock the tracking gate.

        Sibling of _settle_then_park_manual: same wait semantics, but
        the post-settle state transition is "tracking" (detection loop
        resumes) instead of "lost_grace" (manual countdown).
        """
        try:
            client = ptz_core._client_for_camera(camera_id)
            arrived = False
            try:
                arrived = client.wait_until_idle(max_wait_sec=settle_max_sec)
            except Exception as exc:
                logger.debug("wait_until_idle failed, using fallback: %s", exc)
            if not arrived:
                # Fixed-time fallback for cameras that do not expose
                # MoveStatus. Wait the same 5 s the manual settler uses
                # so behaviour stays consistent across both paths.
                time.sleep(5.0)
        except Exception as exc:
            logger.warning("Detection settle worker error: %s", exc)
            time.sleep(5.0)

        # Resume tracking only if nothing else superseded us. If the
        # worker raced against a manual click or the controller went
        # idle, leave that newer state alone.
        with self._lock:
            if self._state != "settling":
                return
            self._state = "tracking"

    def _settle_then_park_manual(
        self, camera_id: int, settle_max_sec: float, view_sec: float
    ) -> None:
        """Wait for the camera to finish moving, then arm the auto-return."""
        try:
            client = ptz_core._client_for_camera(camera_id)
            arrived = False
            try:
                arrived = client.wait_until_idle(max_wait_sec=settle_max_sec)
            except Exception as exc:
                logger.debug("wait_until_idle failed, using fallback: %s", exc)
            if not arrived:
                # Fixed-time fallback for cameras that do not expose MoveStatus.
                time.sleep(5.0)
        except Exception as exc:
            logger.warning("Manual settle worker error: %s", exc)
            time.sleep(5.0)

        # Arm the manual-view deadline so handle_no_detection returns
        # after view_sec instead of the (shorter) lost_timeout_sec.
        with self._lock:
            if self._state != "settling":
                # Another action superseded us; do nothing.
                return
            now = self._clock()
            self._last_seen_mono = now
            self._manual_view_until = now + float(view_sec)
            self._state = "lost_grace"

    def return_to_overview(self) -> bool:
        camera = self._camera_provider()
        if not camera:
            self._set_idle("No enabled PTZ camera matches the active stream")
            return False

        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        overview = str(config.get("overview_preset") or "")
        if not overview:
            self._update_status(state="idle", error="Overview preset is not configured")
            return False

        self._enqueue(
            PtzCommand(
                action="goto", camera_id=int(camera["id"]), preset_token=overview
            )
        )
        now = self._clock()
        with self._lock:
            self._state = "returning"
            self._last_preset = overview
            self._last_zone = "overview"
            self._acquire_count = 0
            # Same race-protection as notify_external_goto: an active
            # return-to-overview is a command, so block detection-driven
            # gotos for the cooldown window while the camera is flying
            # back. Otherwise a freshly-detected bird in a wide-angle
            # frame can hijack the return mid-fly.
            self._last_command_mono = now
            self._zoom_in_budget_used_sec = 0.0
            self._zoom_in_locked_until_overview = False
        return True

    def status(self) -> dict[str, Any]:
        # Fall back to any PTZ-capable camera (even with auto disabled) so
        # the UI can still expose a toggle to turn auto-return back on.
        camera = self._camera_provider()
        if camera is None:
            try:
                camera = ptz_core.find_any_ptz_camera()
            except Exception:
                camera = None
        with self._lock:
            state = self._state
            last_seen = self._last_seen_mono
            manual_until = self._manual_view_until
            external_pause = self._external_pause_reason
            status = {
                "state": state,
                "last_error": self._last_error,
                "last_zone": self._last_zone,
                "last_preset": self._last_preset,
                "acquire_count": self._acquire_count,
                "last_target_center": self._last_target_center,
                # Surfaced for the Stream-page banner: when set, the UI
                # shows "Auto-PTZ paused — <reason>" so it's clear why
                # detections aren't triggering moves.
                "external_pause_reason": external_pause,
            }
        if camera:
            config = ptz_core.normalize_ptz_config(camera.get("ptz"))
            configured_enabled = bool(config.get("enabled"))
            seconds_until_return: float | None = None
            if configured_enabled:
                if state == "settling":
                    # Camera still flying to the target; countdown not armed yet.
                    seconds_until_return = None
                elif manual_until > 0:
                    remaining = manual_until - self._clock()
                    seconds_until_return = max(0.0, round(remaining, 1))
                elif state in {"tracking", "acquiring", "lost_grace"} and last_seen > 0:
                    elapsed = self._clock() - last_seen
                    remaining = float(config["lost_timeout_sec"]) - elapsed
                    seconds_until_return = max(0.0, round(remaining, 1))
            status.update(
                {
                    # `configured_enabled` is the persisted operator intent
                    # (cameras.yaml). `enabled` is kept as an alias so older
                    # API consumers keep working; new code should read
                    # `configured_enabled` to make the meaning explicit.
                    "configured_enabled": configured_enabled,
                    "enabled": configured_enabled,
                    "mode": config.get("mode"),
                    "camera_id": int(camera["id"]),
                    "camera_name": camera.get(
                        "name", f"Camera {int(camera['id']) + 1}"
                    ),
                    "lost_timeout_sec": float(config["lost_timeout_sec"]),
                    "manual_view_sec": float(config.get("manual_view_sec") or 15.0),
                    "seconds_until_return": seconds_until_return,
                }
            )
        else:
            status.update(
                {
                    "configured_enabled": False,
                    "enabled": False,
                    "mode": "",
                    "camera_id": None,
                    "camera_name": "",
                    "lost_timeout_sec": None,
                    "manual_view_sec": None,
                    "seconds_until_return": None,
                }
            )
        return status

    def _maybe_goto_zone(
        self,
        *,
        camera_id: int,
        preset_token: str,
        zone_name: str,
        config: dict[str, Any],
        now: float,
    ) -> bool:
        """Enqueue a goto if cooldown allows. Returns True if a goto was issued."""
        cooldown_sec = int(config["command_cooldown_ms"]) / 1000.0
        with self._lock:
            same_target = (
                self._last_preset == preset_token and self._last_zone == zone_name
            )
            if same_target:
                return False
            if now - self._last_command_mono < cooldown_sec:
                return False
            prev_preset = self._last_preset
            prev_zone = self._last_zone
            self._last_command_mono = now
            self._last_preset = preset_token
            self._last_zone = zone_name

        logger.info(
            "AutoPTZ trigger zone=%s preset=%s camera_id=%s",
            zone_name,
            preset_token,
            camera_id,
        )
        self._enqueue(
            PtzCommand(
                action="goto",
                camera_id=camera_id,
                preset_token=preset_token,
                rollback_preset=prev_preset,
                rollback_zone=prev_zone,
            )
        )
        return True

    def _maybe_move_to_center(
        self,
        *,
        camera_id: int,
        center_x: float,
        center_y: float,
        config: dict[str, Any],
        now: float,
    ) -> None:
        cooldown_sec = int(config["command_cooldown_ms"]) / 1000.0
        with self._lock:
            if now - self._last_command_mono < cooldown_sec:
                return

        offset_x = center_x - 0.5
        offset_y = center_y - 0.5
        deadband = float(config["deadband"])
        if abs(offset_x) <= deadband and abs(offset_y) <= deadband:
            return

        max_speed = float(config["max_speed"])
        pan = max(-max_speed, min(max_speed, offset_x * max_speed * 2.0))
        tilt = max(-max_speed, min(max_speed, -offset_y * max_speed * 2.0))
        if abs(offset_x) <= deadband:
            pan = 0.0
        if abs(offset_y) <= deadband:
            tilt = 0.0

        with self._lock:
            self._last_command_mono = now

        self._enqueue(
            PtzCommand(
                action="move",
                camera_id=camera_id,
                pan=pan,
                tilt=tilt,
                duration_ms=int(config["move_duration_ms"]),
            )
        )

    def _select_target(
        self,
        *,
        frame_shape: tuple[int, ...],
        detections: list[dict[str, Any]],
    ) -> tuple[float, float, float] | None:
        if not frame_shape or len(frame_shape) < 2:
            return None
        frame_h = max(1, int(frame_shape[0]))
        frame_w = max(1, int(frame_shape[1]))

        best: tuple[float, float, float] | None = None
        best_score = -1.0
        for det in detections:
            od_class = str(det.get("class_name") or "bird")
            if not is_bird_od_class(od_class):
                continue

            try:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
                confidence = float(det.get("confidence") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            score = confidence + (area / float(frame_w * frame_h))
            if score <= best_score:
                continue
            center_x = max(0.0, min(1.0, ((x1 + x2) / 2.0) / frame_w))
            center_y = max(0.0, min(1.0, ((y1 + y2) / 2.0) / frame_h))
            best = (center_x, center_y, confidence)
            best_score = score
        return best

    def _zone_for_center(
        self, config: dict[str, Any], center_x: float, center_y: float
    ) -> dict[str, Any] | None:
        # Preferred path: operator-placed preset overlay boxes are the
        # source of truth for detection-zone mapping. Pick the smallest
        # containing box so a tightly-framed preset wins over a wider one.
        overview = str(config.get("overview_preset") or "")
        preset_meta = config.get("preset_metadata") or {}
        if isinstance(preset_meta, dict) and preset_meta:
            best: tuple[float, dict[str, Any]] | None = None
            for token, meta in preset_meta.items():
                if not isinstance(meta, dict):
                    continue
                if overview and token == overview:
                    continue
                w = float(meta.get("box_w_pct") or 0.0)
                h = float(meta.get("box_h_pct") or 0.0)
                if w <= 0 or h <= 0:
                    continue
                cx = float(meta.get("center_x_pct") or 0.0)
                cy = float(meta.get("center_y_pct") or 0.0)
                if (
                    cx - w / 2 <= center_x < cx + w / 2
                    and cy - h / 2 <= center_y < cy + h / 2
                ):
                    area = w * h
                    if best is None or area < best[0]:
                        best = (
                            area,
                            {
                                "name": str(meta.get("label") or token),
                                "preset": str(token),
                            },
                        )
            if best is not None:
                return best[1]
            # No box matched and operator opted into the new model
            # (at least one preset has a real box) → no goto.
            if any(
                isinstance(m, dict)
                and float(m.get("box_w_pct") or 0.0) > 0
                and float(m.get("box_h_pct") or 0.0) > 0
                for m in preset_meta.values()
            ):
                return None

        # Legacy fallback: the old 3-zone horizontal map. Only reached
        # when no preset metadata boxes are configured at all.
        for zone in config.get("zones", []):
            if float(zone.get("x_min", 0.0)) <= center_x < float(
                zone.get("x_max", 1.0)
            ) and float(zone.get("y_min", 0.0)) <= center_y < float(
                zone.get("y_max", 1.0)
            ):
                return zone
        return None

    def _enqueue(self, command: PtzCommand) -> None:
        if not self._worker_enabled:
            # Worker-less path (tests, in-process callers). Mirror the
            # worker's exception semantics so rollback behaviour is
            # uniform: if the runner raises, undo the optimistic state
            # commit via the same CAS check as _worker_loop.
            try:
                self._command_runner(command)
            except Exception as exc:
                logger.error("Auto PTZ command failed: %s", exc)
                self._on_command_failed(command, exc)
            return
        try:
            self._queue.put_nowait(command)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(command)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                command = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._run_with_retry(command)

    def _run_with_retry(self, command: PtzCommand) -> None:
        """Execute a command, retrying transient goto failures.

        Goto-only: move/stop are time-critical operator inputs and must
        never be replayed (an out-of-date move replay would jerk the
        camera against the operator's current heading).

        Retry exits early when:
        * the command succeeds (most retries land on attempt 2)
        * the stop event fires (worker shutdown — no point retrying)
        * a fresher command is already waiting on the queue (the new
          goto supersedes the in-flight one; replaying would waste
          camera bandwidth on a stale target)
        """
        retries = _GOTO_RETRY_ATTEMPTS if command.action == "goto" else 0
        for attempt in range(retries + 1):
            try:
                self._command_runner(command)
                if attempt > 0:
                    logger.info(
                        "Auto PTZ goto succeeded on retry attempt=%d preset=%s",
                        attempt + 1,
                        command.preset_token,
                    )
                return
            except Exception as exc:
                if attempt >= retries:
                    logger.error(
                        "Auto PTZ command failed after %d attempt(s): %s",
                        attempt + 1,
                        exc,
                    )
                    self._on_command_failed(command, exc)
                    return
                logger.warning(
                    "Auto PTZ goto attempt=%d failed (will retry): %s",
                    attempt + 1,
                    exc,
                )
                if self._stop_event.wait(timeout=_GOTO_RETRY_BACKOFF_SEC):
                    return  # shutdown during backoff — drop the command
                if not self._queue.empty():
                    logger.info(
                        "Auto PTZ goto attempt=%d superseded by newer queued "
                        "command; abandoning retry for preset=%s",
                        attempt + 1,
                        command.preset_token,
                    )
                    self._on_command_failed(command, exc)
                    return

    def _on_command_failed(self, command: PtzCommand, exc: Exception) -> None:
        """Record the failure and undo the optimistic state commit.

        For a `goto` whose ONVIF call was rejected (cheap camera says
        "Preset token does not exist", network drop, transient SOAP
        fault), the controller had already set `_last_preset` /
        `_last_zone` to the intended target so the cooldown gate would
        see the in-flight command. With the goto refused, that
        committed state lies: the camera never moved. Roll it back so
        the next `snapshot_for_image_persistence()` reflects what the
        camera is actually showing, not what we hoped it would show.

        CAS check: only restore if `_last_preset` still equals the
        token we just failed on. A newer enqueue (manual click,
        next-frame detection) may have already overwritten the state
        — that newer attempt may yet succeed, and clobbering it would
        re-introduce the same lie in reverse.
        """
        with self._lock:
            self._last_error = str(exc)
            if (
                command.action == "goto"
                and command.preset_token
                and self._last_preset == command.preset_token
            ):
                self._last_preset = command.rollback_preset
                self._last_zone = command.rollback_zone

    def _run_command(self, command: PtzCommand) -> None:
        if command.action == "goto":
            ptz_core.goto_preset(command.camera_id, command.preset_token)
        elif command.action == "move":
            self._run_move_with_burst(command)
        elif command.action == "stop":
            ptz_core.stop(command.camera_id)

    def _run_move_with_burst(self, command: PtzCommand) -> None:
        """Issue a follow-mode move N times back-to-back.

        Same purpose as the joystick's frontend burst (see stream.html):
        cheap PTZ cams that ignore ContinuousMove velocity step a fixed
        firmware-internal distance per call. If pan/tilt steps are too
        small relative to zoom, this enqueues exactly one logical move
        per detection-loop tick but fires it N times.

        Burst counts come from the per-camera ptz config. The "axis" of
        a move is determined by which fields are non-zero: a pure-zoom
        command (zoom!=0, pan==tilt==0) uses manual_zoom_burst; anything
        else uses manual_pan_tilt_burst. Mixed pan+zoom commands aren't
        produced by the current follow logic (zoom-only when bbox-area
        wants correction, pan/tilt-only when offset wants correction),
        so the simple axis classifier is sufficient.

        Between calls the worker sleeps BURST_SPACING_SEC so the
        backend's ContinuousMove + sleep + Stop sequence on call K is
        complete before call K+1 starts — otherwise call K+1's stop
        races call K's start and the visible motion is lost.

        Abort path: if a new command lands on the worker queue mid-
        burst (stop, return-to-overview, fresher move), we bail out
        early. The new command's enqueue means the old burst's tail
        is stale by definition.
        """
        burst = self._burst_for_command(command)
        # Apply the per-call duration multiplier *after* burst lookup
        # so the operator can combine: 3 bursts × 2× duration =
        # 3 × 600 ms ContinuousMove calls per follow correction. Read
        # from the same per-cam config as burst.
        effective_duration_ms = self._effective_duration_ms(command)
        if burst <= 1:
            ptz_core.continuous_move(
                command.camera_id,
                pan=command.pan,
                tilt=command.tilt,
                zoom=command.zoom,
                duration_ms=effective_duration_ms,
            )
            return

        # Spacing must be ≥ duration_ms so each call's Stop completes
        # before the next ContinuousMove starts. The +20 ms is a small
        # safety margin for ONVIF round-trip jitter on cheap cams.
        # Uses effective_duration (post-multiplier) so a 2× duration
        # also waits 2× between bursts.
        spacing_sec = max(0.05, (effective_duration_ms + 20) / 1000.0)

        for i in range(burst):
            ptz_core.continuous_move(
                command.camera_id,
                pan=command.pan,
                tilt=command.tilt,
                zoom=command.zoom,
                duration_ms=effective_duration_ms,
            )
            if i == burst - 1:
                break
            # Sleep BUT honour shutdown: bail if the worker is shutting
            # down. Do NOT bail on a non-empty queue — the detection
            # loop enqueues a new follow-correction every ~250 ms, so
            # any active burst would always see queue.empty() == False
            # after call 1 and abort there. The result would be a
            # silent regression to burst=1, which is exactly what the
            # 2026-05-25 live test was reporting. A fresher `stop` or
            # `goto overview` IS allowed to preempt — see the peek
            # below — but a fresher `move` just means "do another
            # burst next" and should not amputate the current one.
            if self._stop_event.wait(timeout=spacing_sec):
                return
            if self._worker_enabled and self._next_command_supersedes_burst():
                return

    def _next_command_supersedes_burst(self) -> bool:
        """True iff the next queued command is `stop` or `goto`.

        Peeks without consuming. The Queue.maxsize=1 invariant means
        at most one command is waiting, so an internal-list peek is
        safe (no race with a second producer because the detection
        loop is the only writer and we hold the worker thread).
        """
        try:
            with self._queue.mutex:
                if not self._queue.queue:
                    return False
                next_cmd = self._queue.queue[0]
        except Exception:
            return False
        return getattr(next_cmd, "action", None) in ("stop", "goto")

    def _burst_for_command(self, command: PtzCommand) -> int:
        """Pick the right per-axis burst count for this move command.

        Pure-zoom (zoom != 0, pan == tilt == 0) → manual_zoom_burst.
        Everything else (pan/tilt with or without zoom) → manual_pan_tilt_burst.
        """
        try:
            config = ptz_core.get_ptz_config(command.camera_id) or {}
        except Exception:
            return 1
        is_pure_zoom = (
            abs(command.zoom) > 0.001
            and abs(command.pan) < 0.001
            and abs(command.tilt) < 0.001
        )
        key = "manual_zoom_burst" if is_pure_zoom else "manual_pan_tilt_burst"
        try:
            value = int(config.get(key, 1))
        except (TypeError, ValueError):
            return 1
        return max(1, min(6, value))

    def _effective_duration_ms(self, command: PtzCommand) -> int:
        """Apply the per-cam manual_move_duration_multiplier.

        Default 1.0 = command.duration_ms unchanged. Clamped to
        [0.5, 5.0] via normalize_ptz_config, so the effective
        duration is in [duration_ms/2, duration_ms*5]. The
        underlying PtzClient.continuous_move further clamps to
        [50, 2000] ms before issuing the ONVIF call, so a 5×
        multiplier on a 250 ms base saturates at 1250 ms — well
        within the client cap.
        """
        try:
            config = ptz_core.get_ptz_config(command.camera_id) or {}
            mult = float(config.get("manual_move_duration_multiplier", 1.0))
        except (TypeError, ValueError, Exception):
            mult = 1.0
        return max(1, int(round(command.duration_ms * mult)))

    def _set_idle(self, error: str = "") -> None:
        with self._lock:
            self._state = "idle"
            self._last_error = error
            self._acquire_count = 0

    def _update_status(
        self,
        *,
        state: PtzState,
        error: str = "",
        target_center: tuple[float, float] | None = None,
    ) -> None:
        with self._lock:
            self._state = state
            self._last_error = error
            if target_center is not None:
                self._last_target_center = target_center

    def snapshot_for_image_persistence(self) -> dict[str, Any]:
        """Return the PTZ context to record on a captured frame.

        Frame-level (not detection-level): every detection in the same
        frame shares this state. Reads under `_lock`. Returns column-
        ready keys for `insert_image`. v2 absolute-coordinate slots
        stay None until `GetStatus` is wired in.

        State mapping:
          tracking / acquiring / settling / lost_grace → 'preset'
              (camera is at, or flying to, a non-overview zone preset;
              the frame is a close-up regardless of mid-fly motion)
          overview / returning → 'overview'
              (camera is at, or flying to, the overview preset;
              the frame is wide-view)
          idle → 'none'
              (auto-PTZ off or no active camera; we cannot claim the
              camera is at overview because the operator may have left
              it on a manual preset — we just don't know)
        """
        with self._lock:
            state = self._state
            last_preset = self._last_preset
            last_zone = self._last_zone

        # last_zone discriminates between auto-tracking lost_grace and
        # manual-drive lost_grace at the same state value — both keep
        # the camera physically close to the subject, but only the
        # auto case is "preset-targeted" in the literal sense.
        if last_zone == "manual_drive":
            origin = "manual_drive"
        elif state in ("tracking", "acquiring", "settling", "lost_grace"):
            origin = "preset"
        elif state in ("overview", "returning"):
            origin = "overview"
        else:
            origin = "none"

        camera = self._camera_provider()
        camera_id = int(camera["id"]) if camera and "id" in camera else None

        return {
            "ptz_origin": origin,
            "ptz_preset_token": last_preset or None,
            "ptz_zone": last_zone or None,
            "ptz_state": state,
            "ptz_camera_id": camera_id,
            "ptz_pan": None,
            "ptz_tilt": None,
            "ptz_zoom": None,
            "ptz_position_at": None,
        }


def empty_ptz_snapshot() -> dict[str, Any]:
    """PTZ snapshot to record when no controller is available.

    Same key set as `AutoPtzController.snapshot_for_image_persistence`
    so callers can hand the dict straight to `insert_image` without a
    None-check on every key.
    """
    return {
        "ptz_origin": None,
        "ptz_preset_token": None,
        "ptz_zone": None,
        "ptz_state": None,
        "ptz_camera_id": None,
        "ptz_pan": None,
        "ptz_tilt": None,
        "ptz_zoom": None,
        "ptz_position_at": None,
    }
