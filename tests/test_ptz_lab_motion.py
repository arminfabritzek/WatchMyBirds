"""Safety tests for the first physical PTZ laboratory movement probe."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from scripts.ptz_lab.motion import (
    MotionLimits,
    rewrite_uri_host,
    run_continuous_probe,
)


def test_rewrite_uri_host_preserves_camera_port_and_path():
    rewritten = rewrite_uri_host(
        "http://camera-user:redacted@192.0.2.88:80/tmpfs/auto.jpg?channel=1",
        "198.51.100.92",
    )

    assert rewritten == "http://198.51.100.92:80/tmpfs/auto.jpg?channel=1"


def test_motion_limits_reject_aggressive_first_probe():
    for values in (
        {"speed": 0.3, "duration_ms": 100},
        {"speed": 0.1, "duration_ms": 300},
        {"speed": 0.0, "duration_ms": 100},
    ):
        try:
            MotionLimits(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe limits accepted: {values}")


def test_motion_limits_allow_explicit_large_jump_with_absolute_cap():
    limits = MotionLimits(speed=1.0, duration_ms=2000, large_jump=True)

    assert limits.speed == 1.0
    assert limits.duration_ms == 2000

    for values in (
        {"speed": 1.01, "duration_ms": 100, "large_jump": True},
        {"speed": 1.0, "duration_ms": 2001, "large_jump": True},
    ):
        try:
            MotionLimits(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"absolute safety cap bypassed: {values}")


def test_continuous_probe_always_stops_and_returns_to_preset():
    client = MagicMock()
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    capture = MagicMock(side_effect=[frame, frame, frame])
    align = MagicMock(
        return_value={
            "success": True,
            "quality": 1.0,
            "pixel_dx": 2.0,
            "error_magnitude": 0.0,
        }
    )

    report = run_continuous_probe(
        client,
        capture=capture,
        align=align,
        return_preset="Preset023",
        axis="pan",
        direction=1,
        limits=MotionLimits(speed=0.1, duration_ms=100),
        settle=lambda _seconds: None,
    )

    client.continuous_move.assert_called_once_with(
        pan=0.1, tilt=0.0, zoom=0.0, duration_ms=100
    )
    client.emergency_stop.assert_called_once()
    assert client.goto_preset.call_count == 4
    assert report["movement"]["status"] == "SUPPORTED"
    assert report["return"]["status"] == "SUPPORTED"


def test_continuous_probe_returns_even_when_post_move_capture_fails():
    client = MagicMock()
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    capture = MagicMock(side_effect=[frame, RuntimeError("snapshot failed"), frame])

    report = run_continuous_probe(
        client,
        capture=capture,
        align=MagicMock(return_value={"error_magnitude": 0.0}),
        return_preset="Preset023",
        axis="tilt",
        direction=-1,
        limits=MotionLimits(speed=0.1, duration_ms=100),
        settle=lambda _seconds: None,
    )

    client.emergency_stop.assert_called_once()
    assert client.goto_preset.call_count == 4
    assert report["movement"]["status"] == "ERROR"
    assert report["return"]["status"] == "SUPPORTED"
