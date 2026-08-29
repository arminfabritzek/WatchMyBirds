#!/usr/bin/env python3
"""Center a known image point with bounded PTZ pulses, then return to a preset."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.ptz_lab.common import (
    _password,
    connect_lab,
    report_header,
    sanitize,
    write_report,
)
from scripts.ptz_lab.motion import SnapshotCapture, _live_alignment


def _goto_stable(client: Any, preset: str, attempts: int = 3) -> None:
    for _attempt in range(attempts):
        client.goto_preset(preset)
        time.sleep(2.0)


def center_target(
    client: Any,
    capture: SnapshotCapture,
    *,
    target_name: str,
    target_x: float,
    target_y: float,
    return_preset: str,
    tolerance_px: float = 15.0,
    initial_pan_rate_px_s: float = 56.0,
    initial_tilt_rate_px_s: float = 66.0,
    max_pulses: int = 8,
) -> dict[str, Any]:
    """Center one baseline image point using measured pan/tilt displacement."""
    started = time.monotonic()
    control_started: float | None = None
    report: dict[str, Any] = {
        "target": {"name": target_name, "x": target_x, "y": target_y},
        "tolerance_px": tolerance_px,
        "pulses": [],
        "result": {"status": "PENDING"},
        "return": {"status": "PENDING"},
    }
    baseline = None
    try:
        _goto_stable(client, return_preset)
        baseline = capture()
        control_started = time.monotonic()
        height, width = baseline.shape[:2]
        if not (0 <= target_x < width and 0 <= target_y < height):
            raise ValueError("target point lies outside the baseline image")

        rates_px_s = {"pan": initial_pan_rate_px_s, "tilt": initial_tilt_rate_px_s}
        current = baseline
        for pulse_number in range(1, max_pulses + 1):
            alignment = _live_alignment(baseline, current)
            shifted_x = target_x + float(alignment["pixel_dx"])
            shifted_y = target_y + float(alignment["pixel_dy"])
            error_x = shifted_x - width / 2
            error_y = shifted_y - height / 2
            if abs(error_x) <= tolerance_px and abs(error_y) <= tolerance_px:
                report["result"] = {
                    "status": "SUPPORTED",
                    "center_error_x_px": error_x,
                    "center_error_y_px": error_y,
                    "elapsed_s": time.monotonic() - started,
                    "control_elapsed_s": time.monotonic() - control_started,
                    "pulse_count": len(report["pulses"]),
                }
                break

            pan_duration_ms = abs(error_x) / max(rates_px_s["pan"], 1.0) * 1000
            tilt_duration_ms = abs(error_y) / max(rates_px_s["tilt"], 1.0) * 1000
            if abs(error_x) <= tolerance_px:
                pan_duration_ms = 0.0
            if abs(error_y) <= tolerance_px:
                tilt_duration_ms = 0.0
            axis = "pan" if pan_duration_ms >= tilt_duration_ms else "tilt"
            axis_error = error_x if axis == "pan" else error_y
            duration_ms = round(max(pan_duration_ms, tilt_duration_ms))
            duration_ms = min(500 if axis == "tilt" else 2000, max(50, duration_ms))
            direction = 1 if axis_error > 0 else -1
            if axis == "tilt":
                direction *= -1
            before_shift = float(
                alignment["pixel_dx"] if axis == "pan" else alignment["pixel_dy"]
            )
            pulse_started = time.monotonic()
            client.continuous_move(
                pan=float(direction) if axis == "pan" else 0.0,
                tilt=float(direction) if axis == "tilt" else 0.0,
                zoom=0.0,
                duration_ms=duration_ms,
            )
            current = capture()
            elapsed = time.monotonic() - pulse_started
            after_alignment = _live_alignment(baseline, current)
            after_shift = float(
                after_alignment["pixel_dx"]
                if axis == "pan"
                else after_alignment["pixel_dy"]
            )
            displacement = abs(after_shift - before_shift)
            if displacement >= 2.0 and duration_ms >= 50:
                measured_rate = displacement / (duration_ms / 1000)
                rates_px_s[axis] = 0.65 * rates_px_s[axis] + 0.35 * measured_rate
            report["pulses"].append(
                {
                    "number": pulse_number,
                    "axis": axis,
                    "direction": direction,
                    "duration_ms": duration_ms,
                    "wall_time_s": elapsed,
                    "error_before_px": axis_error,
                    "displacement_px": displacement,
                    "estimated_rate_px_s": rates_px_s[axis],
                }
            )
        else:
            alignment = _live_alignment(baseline, current)
            report["result"] = {
                "status": "INCONCLUSIVE",
                "center_error_x_px": target_x
                + float(alignment["pixel_dx"])
                - width / 2,
                "center_error_y_px": target_y
                + float(alignment["pixel_dy"])
                - height / 2,
                "elapsed_s": time.monotonic() - started,
                "control_elapsed_s": time.monotonic() - control_started,
                "pulse_count": len(report["pulses"]),
            }
    except Exception as exc:
        report["result"] = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": time.monotonic() - started,
        }
    finally:
        client.emergency_stop()
        try:
            if baseline is None:
                report["return"] = {"status": "INCONCLUSIVE"}
            else:
                attempts: list[dict[str, Any]] = []
                error = math.inf
                alignment: dict[str, Any] = {}
                for attempt in range(1, 4):
                    client.goto_preset(return_preset)
                    time.sleep(2.0)
                    returned = capture()
                    alignment = _live_alignment(baseline, returned)
                    error = math.hypot(
                        float(alignment["pixel_dx"]),
                        float(alignment["pixel_dy"]),
                    )
                    attempts.append({"attempt": attempt, "error_px": error})
                    if error <= 4.0:
                        break
                report["return"] = {
                    "status": "SUPPORTED" if error <= 4.0 else "INCONCLUSIVE",
                    "error_px": error,
                    "alignment": alignment,
                    "attempts": attempts,
                }
        except Exception as exc:
            report["return"] = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return sanitize(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-env", default="WMB_CAM_PASSWORD")
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--return-preset", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--tolerance-px", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ptz_lab"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("physical movement requires --execute")

    password = _password(args.password_env)
    conn = connect_lab(args.ip, args.port, args.user, password, args.profile_index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture = SnapshotCapture(
        conn.client,
        args.ip,
        args.output_dir / f"center_{args.target_name}_{timestamp}",
    )
    report = report_header("center_target", args.ip, args.port)
    report["read_only"] = False
    report.update(
        center_target(
            conn.client,
            capture,
            target_name=args.target_name,
            target_x=args.target_x,
            target_y=args.target_y,
            return_preset=args.return_preset,
            tolerance_px=args.tolerance_px,
        )
    )
    report["artifacts"] = capture.paths
    json_path, yaml_path = write_report(report, args.output_dir, "ptz_center_target")
    print(f"Centering probe complete. Reports: {json_path} {yaml_path}")
    return 0 if report["return"]["status"] == "SUPPORTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
