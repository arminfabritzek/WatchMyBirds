from unittest.mock import patch

from core.ptz_core import (
    clear_auto_ptz_camera_cache,
    find_auto_ptz_camera,
    normalize_ptz_config,
)


def test_normalize_ptz_config_clamps_tracking_values():
    config = normalize_ptz_config(
        {
            "enabled": True,
            "mode": "hybrid",
            "acquire_frames": 99,
            "lost_timeout_sec": -4,
            "command_cooldown_ms": 1,
            "deadband": 2,
            "max_speed": 4,
            "move_duration_ms": 9999,
            "zones": [
                {
                    "name": "left",
                    "preset": "preset-left",
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 0.5,
                    "y_max": 1,
                }
            ],
        }
    )

    assert config["enabled"] is True
    assert config["mode"] == "hybrid"
    assert config["acquire_frames"] == 10
    assert config["lost_timeout_sec"] == 1.0
    assert config["command_cooldown_ms"] == 100
    assert config["deadband"] == 0.4
    assert config["max_speed"] == 1.0
    assert config["move_duration_ms"] == 2000
    assert config["zones"][0]["preset"] == "preset-left"


def test_find_auto_ptz_camera_strips_password():
    """The cached auto-PTZ camera dict must never carry the raw password.

    The status route and the 2 s controller cache both surface the dict
    that find_auto_ptz_camera() returns; a password leak here would land
    in the API response and in process RAM.
    """

    class _FakeStorage:
        def _load_cameras(self):
            return [
                {
                    "ip": "192.168.1.50",
                    "username": "admin",
                    "password": "s3cret",
                    "ptz": {"enabled": True, "overview_preset": "home"},
                }
            ]

    clear_auto_ptz_camera_cache()
    try:
        with (
            patch(
                "core.ptz_core.get_camera_storage", return_value=_FakeStorage()
            ),
            patch(
                "core.ptz_core.get_config",
                return_value={"VIDEO_SOURCE": "rtsp://192.168.1.50/stream"},
            ),
        ):
            camera = find_auto_ptz_camera()
    finally:
        clear_auto_ptz_camera_cache()

    assert camera is not None
    assert "password" not in camera
    assert camera["ip"] == "192.168.1.50"
    assert camera["id"] == 0
