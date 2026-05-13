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
from detectors.od_classes import is_bird_od_class
from logging_config import get_logger

logger = get_logger(__name__)

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
    ) -> None:
        camera = self._camera_provider()
        if not camera:
            self._set_idle("No enabled PTZ camera matches the active stream")
            return

        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        if not config.get("enabled"):
            self._set_idle("Auto PTZ disabled")
            return

        target = self._select_target(frame_shape=frame_shape, detections=detections)
        if not target:
            self.handle_no_detection()
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
        self._maybe_goto_zone(
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
            self._state = "tracking"
            self._last_error = ""
            self._last_target_center = (center_x, center_y)
            logger.debug(
                "Auto PTZ tracking target zone=%s conf=%.3f center=(%.3f, %.3f)",
                zone_name,
                confidence,
                center_x,
                center_y,
            )

    def handle_no_detection(self) -> None:
        with self._lock:
            if self._state not in {"acquiring", "tracking", "lost_grace"}:
                return

        camera = self._camera_provider()
        if not camera:
            self._set_idle("No enabled PTZ camera matches the active stream")
            return

        config = ptz_core.normalize_ptz_config(camera.get("ptz"))
        if not config.get("enabled"):
            self._set_idle("Auto PTZ disabled")
            return

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
        with self._lock:
            self._state = "returning"
            self._last_preset = overview
            self._last_zone = "overview"
            self._acquire_count = 0

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

        with self._lock:
            self._last_preset = str(preset_token)
            self._last_zone = "manual"
            if seeds_manual_grace:
                # Park the controller in "settling" — countdown does not
                # start yet. The background settle-worker flips us to
                # lost_grace once the camera reports IDLE (or fallback).
                self._state = "settling"
                self._last_seen_mono = 0.0
                self._manual_view_until = 0.0
                self._acquire_count = 0
            else:
                # Manual goto to overview cancels any pending manual return.
                self._manual_view_until = 0.0

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
        with self._lock:
            self._state = "returning"
            self._last_preset = overview
            self._last_zone = "overview"
            self._acquire_count = 0
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
            status = {
                "state": state,
                "last_error": self._last_error,
                "last_zone": self._last_zone,
                "last_preset": self._last_preset,
                "acquire_count": self._acquire_count,
                "last_target_center": self._last_target_center,
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
    ) -> None:
        cooldown_sec = int(config["command_cooldown_ms"]) / 1000.0
        with self._lock:
            same_target = (
                self._last_preset == preset_token and self._last_zone == zone_name
            )
            if same_target:
                return
            if now - self._last_command_mono < cooldown_sec:
                return
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
            PtzCommand(action="goto", camera_id=camera_id, preset_token=preset_token)
        )

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
            self._command_runner(command)
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
            try:
                self._command_runner(command)
            except Exception as exc:
                logger.error("Auto PTZ command failed: %s", exc)
                with self._lock:
                    self._last_error = str(exc)

    def _run_command(self, command: PtzCommand) -> None:
        if command.action == "goto":
            ptz_core.goto_preset(command.camera_id, command.preset_token)
        elif command.action == "move":
            ptz_core.continuous_move(
                command.camera_id,
                pan=command.pan,
                tilt=command.tilt,
                zoom=command.zoom,
                duration_ms=command.duration_ms,
            )
        elif command.action == "stop":
            ptz_core.stop(command.camera_id)

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
