#!/usr/bin/env python3
"""Exercise adaptive follow and lost-target recovery without moving a camera."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ptz_tracking_core import AutoPtzController, PtzCommand


class SimulationClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _detection(center_x: float, center_y: float, size: float = 0.1) -> dict[str, float]:
    half = size * 50.0
    return {
        "x1": center_x * 100.0 - half,
        "y1": center_y * 100.0 - half,
        "x2": center_x * 100.0 + half,
        "y2": center_y * 100.0 + half,
        "confidence": 0.9,
    }


def simulate(fps: float) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    clock = SimulationClock()
    commands: list[PtzCommand] = []
    camera = {
        "id": 0,
        "name": "simulation",
        "enabled": True,
        "ptz": {
            "enabled": True,
            "overview_preset": "overview",
            "lost_timeout_sec": 10.0,
            "command_cooldown_ms": 800,
            "deadband": 0.04,
            "max_speed": 0.35,
            "follow_pan_rate_per_sec": 0.075,
            "follow_tilt_rate_per_sec": 0.15,
            "follow_tilt_max_duration_ms": 500,
            "follow_zoom_duration_ms": 750,
            "follow_zoom_target_pct": 0.18,
            "follow_zoom_deadband_pct": 0.05,
            "follow_zoom_speed": 0.3,
            "follow_zoom_max_burst_sec": 0.75,
            "follow_lost_hold_sec": 2.0,
            "follow_search_sec": 8.0,
            "follow_search_burst_ms": 400,
            "follow_search_zoom_out_ms": 250,
            "manual_pan_tilt_burst": 6,
            "manual_zoom_burst": 6,
            "manual_move_duration_multiplier": 5.0,
        },
    }
    controller = AutoPtzController(
        camera_provider=lambda: camera,
        command_runner=commands.append,
        clock=clock,
        worker_enabled=False,
    )
    frame_shape = (100, 100, 3)

    controller.handle_detections(
        frame_shape=frame_shape,
        detections=[_detection(0.8, 0.5)],
    )
    clock.advance(1.0 / fps)
    controller.handle_detections(
        frame_shape=frame_shape,
        detections=[_detection(0.5, 0.5)],
    )

    for _ in range(int(14 * fps) + 1):
        clock.advance(1.0 / fps)
        controller.handle_no_detection()

    moves = [command for command in commands if command.action == "move"]
    gotos = [command for command in commands if command.action == "goto"]
    assert moves[0].pan > 0 and moves[0].duration_ms == 2000
    assert moves[0].use_manual_tuning is False
    assert any(command.zoom > 0 and command.duration_ms == 750 for command in moves)
    assert any(command.pan > 0 and command.duration_ms == 400 for command in moves)
    zoom_out = [command for command in moves if command.zoom < 0]
    assert len(zoom_out) == 1 and zoom_out[0].duration_ms == 250
    assert gotos and gotos[-1].preset_token == "overview"

    return {
        "fps": fps,
        "status": "SUPPORTED",
        "commands": [asdict(command) for command in commands],
        "final_state": controller.status()["state"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", nargs="+", type=float, default=[0.5, 1.0])
    args = parser.parse_args()
    print(json.dumps([simulate(fps) for fps in args.fps], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
