"""Web-layer construction for ephemeral click-to-aim control."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core import ptz_core
from core.ptz_aim_core import AimController, ExternalPauseController


def create_controller(
    frame_supplier: Callable[[], Any],
    *,
    external_pause: ExternalPauseController | None = None,
) -> AimController:
    return AimController(
        frame_supplier=frame_supplier,
        mover=lambda camera_id, pan, tilt, duration: ptz_core.continuous_move(
            camera_id,
            pan=pan,
            tilt=tilt,
            duration_ms=duration,
        ),
        stopper=ptz_core.stop,
        external_pause=external_pause,
    )


def start(
    controller: AimController,
    camera_id: int,
    target_x: float,
    target_y: float,
) -> dict[str, Any]:
    return controller.start(camera_id, target_x, target_y)


def status(controller: AimController) -> dict[str, Any]:
    return controller.status()


def cancel(controller: AimController, camera_id: int) -> dict[str, Any]:
    return controller.cancel(camera_id)
