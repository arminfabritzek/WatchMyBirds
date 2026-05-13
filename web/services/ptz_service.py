"""
PTZ Service - thin web-layer wrapper over core PTZ operations.
"""

from typing import Any

from core import ptz_core


def get_config(camera_id: int) -> dict[str, Any] | None:
    return ptz_core.get_ptz_config(camera_id)


def update_config(camera_id: int, config: dict[str, Any]) -> dict[str, Any] | None:
    return ptz_core.update_ptz_config(camera_id, config)


def list_presets(camera_id: int) -> list[dict[str, str]]:
    return ptz_core.list_presets(camera_id)


def goto_preset(camera_id: int, preset_token: str) -> None:
    ptz_core.goto_preset(camera_id, preset_token)


def move(
    camera_id: int,
    *,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    duration_ms: int = 250,
) -> None:
    ptz_core.continuous_move(
        camera_id,
        pan=pan,
        tilt=tilt,
        zoom=zoom,
        duration_ms=duration_ms,
    )


def stop(camera_id: int) -> None:
    ptz_core.stop(camera_id)


def capture_overview_snapshot(camera_id: int) -> dict[str, Any] | None:
    return ptz_core.capture_overview_snapshot(camera_id)


def set_auto_enabled(camera_id: int, enabled: bool) -> dict[str, Any] | None:
    return ptz_core.set_auto_enabled(camera_id, enabled)


def list_presets_with_metadata(
    camera_id: int, show_all: bool = False
) -> list[dict[str, Any]]:
    return ptz_core.list_presets_with_metadata(camera_id, show_all=show_all)


def set_preset_at_current_position(
    camera_id: int,
    name: str,
    *,
    preset_token: str | None = None,
    center_x_pct: float | None = None,
    center_y_pct: float | None = None,
    box_w_pct: float | None = None,
    box_h_pct: float | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    return ptz_core.set_preset_at_current_position(
        camera_id,
        name,
        preset_token=preset_token,
        center_x_pct=center_x_pct,
        center_y_pct=center_y_pct,
        box_w_pct=box_w_pct,
        box_h_pct=box_h_pct,
        label=label,
    )


def update_preset_metadata_only(
    camera_id: int,
    preset_token: str,
    *,
    center_x_pct: float | None = None,
    center_y_pct: float | None = None,
    box_w_pct: float | None = None,
    box_h_pct: float | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    return ptz_core.update_preset_metadata_only(
        camera_id,
        preset_token,
        center_x_pct=center_x_pct,
        center_y_pct=center_y_pct,
        box_w_pct=box_w_pct,
        box_h_pct=box_h_pct,
        label=label,
    )


def remove_preset(camera_id: int, preset_token: str) -> bool:
    return ptz_core.remove_preset(camera_id, preset_token)
