#!/usr/bin/env python3
"""Measure pan/tilt vs. zoom step-asymmetry on a PTZ camera.

Cheap PTZ cams often map ONVIF ``ContinuousMove`` velocity to deg/s
with very different scaling per axis — and some ignore velocity entirely
(an internal step is fired on any non-zero velocity). This script tells
you which behaviour your cam has by firing a small matrix of moves and
asking you to grade each as small / medium / large.

The result is the input for picking between two corrective strategies:

  - **Velocity scales** (cam respects speed):  multiply ``JOY_SPEED``
    differently per axis (Option A in the discussion).
  - **Velocity is binary** (cam ignores speed): fire the same move N
    times instead of scaling the magnitude (Option B in the discussion).

Usage on the RPi (run while the live app is up — the script bypasses
the app's ``ptz_core`` cache and talks ONVIF directly, but the running
``AutoPtzController`` will fight you if a bird shows up mid-sweep, so
**switch Auto-PTZ off in Settings before starting**):

    ssh admin@<rpi-host>
    sudo -u watchmybirds /opt/app/.venv/bin/python \\
        /opt/app/scripts/diagnose_ptz_step_asymmetry.py --camera-id 1

The camera does NOT need to be on the same network as your laptop —
the script connects to whichever IP/credentials the WMB DB has for the
named camera. Output is written to stdout and a timestamped YAML
report under ``OUTPUT_DIR/ptz_step_asymmetry/``.

A single full sweep takes ~3 minutes. The Home preset is required and
the cam returns there between every move so each grade is independent.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from core import ptz_core  # noqa: E402
from utils.path_manager import get_path_manager  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ptz_step_asymmetry")

# The sweep matrix. Three speeds per axis, one direction (we don't need
# both — asymmetry across speeds tells us the same story regardless of
# direction, and halving the matrix halves operator boredom). Duration
# is fixed at 300ms to match the joystick's MOVE_DURATION_MS so what we
# measure is what the live UI feels.
SPEEDS = [0.2, 0.5, 0.9]
DURATION_MS = 300
SETTLE_AFTER_HOME_SEC = 3.0
SETTLE_AFTER_MOVE_SEC = 1.0
# Watch-only mode: how long the operator gets to observe each move
# before the next sweep starts. Long enough to mentally note "small",
# "medium", or "large", short enough that 9 sweeps stay under 4 minutes.
WATCH_PAUSE_SEC = 5.0

GRADE_OPTIONS = {
    "0": "no_movement",
    "s": "small",
    "m": "medium",
    "l": "large",
    "x": "endstop",  # cam ran into a physical limit — discard this sample
}


def _prompt_grade(label: str) -> str:
    """Read a single-character grade from stdin. Loops until valid."""
    while True:
        raw = input(f"  Grade {label} [0/s/m/l/x] > ").strip().lower()
        if raw in GRADE_OPTIONS:
            return GRADE_OPTIONS[raw]
        print(f"    not understood. Pick one of: {', '.join(sorted(GRADE_OPTIONS))}")


def _return_home(client, overview_preset: str) -> None:
    """Send the cam to the overview preset and wait for it to settle.

    We do NOT depend on ``wait_until_idle`` here because that's exactly
    one of the behaviours under test — on a cam that doesn't report
    MoveStatus, the helper returns False after one poll and we'd race
    the next move. Fixed sleep is the conservative choice for a probe.
    """
    print(f"    ↩  Returning to overview preset {overview_preset!r}…", flush=True)
    try:
        client.goto_preset(overview_preset)
    except Exception as exc:  # noqa: BLE001
        print(f"    ⚠  goto_preset failed: {exc}")
        return
    time.sleep(SETTLE_AFTER_HOME_SEC)


def _fire_move(
    client, axis: str, direction: int, speed: float
) -> tuple[bool, str | None]:
    """Issue one ContinuousMove burst on the given axis.

    Returns (ok, error_message). On ok=False we still grade the row as
    'no_movement' so the operator sees the ONVIF failure in context.
    """
    pan = direction * speed if axis == "pan" else 0.0
    tilt = direction * speed if axis == "tilt" else 0.0
    zoom = direction * speed if axis == "zoom" else 0.0
    try:
        client.continuous_move(
            pan=pan, tilt=tilt, zoom=zoom, duration_ms=DURATION_MS
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _run_sweep(camera_id: int, *, auto: bool = False) -> list[dict]:
    """Walk the speed × axis matrix once.

    When ``auto`` is False (default), prompts the operator for a grade
    after each move. When True, fires every move with a fixed watch-
    pause between them and skips the prompt — the operator notes grades
    on paper and feeds them back to the analyst afterwards.
    """
    config = ptz_core.get_ptz_config(camera_id) or {}
    overview = str(config.get("overview_preset") or "").strip()
    if not overview:
        raise SystemExit(
            f"Camera {camera_id} has no overview preset configured. Set one "
            "in Settings → Auto-PTZ before running this probe."
        )

    client = ptz_core._client_for_camera(camera_id)

    print()
    print("=" * 64)
    print(f" PTZ step-asymmetry probe — camera {camera_id}"
          + ("  [AUTO / watch-only]" if auto else ""))
    print("=" * 64)
    print()
    print(" Pre-flight checklist:")
    print(" 1. Auto-PTZ is OFF in Settings (otherwise it fights you).")
    print(" 2. Live stream is visible so you can observe each step.")
    print(f" 3. Overview preset {overview!r} points somewhere SAFE — the")
    print("    cam returns there between every move.")
    print()
    if auto:
        print(" Sweep order (9 moves, ~4 minutes total):")
        for axis in ("pan", "tilt", "zoom"):
            for speed in SPEEDS:
                print(f"   - {axis:5} dir=+1 speed={speed:.2f}")
        print()
        print(" After each move you get ~5s to watch. Note the size in")
        print(" your head: small / medium / large / no movement.")
        print()
        print(" Starting in 10 seconds — open the live stream now.")
        for remaining in range(10, 0, -1):
            print(f"   {remaining}…", end="\r", flush=True)
            time.sleep(1.0)
        print("   GO!   ")
    else:
        print(" Grades:")
        print("   0 = no visible movement")
        print("   s = small step  (a notch, barely noticeable)")
        print("   m = medium step (clearly visible, manageable)")
        print("   l = large step  (overshoots, jumps too far)")
        print("   x = endstop hit (sample discarded, will retry once)")
        print()
        input(" Press Enter when ready… ")

    rows: list[dict] = []
    # Axis order: pan first (lateral motion is most familiar), then tilt,
    # zoom last (most surprising / most asymmetric on cheap cams).
    # Tilt sweeps DOWN, not up — the safe-headroom direction when the
    # home preset sits high enough that "up" risks the upper endstop
    # on a 300ms burst at 0.9.
    axis_direction = {"pan": +1, "tilt": -1, "zoom": +1}
    for axis in ("pan", "tilt", "zoom"):
        for speed in SPEEDS:
            direction = axis_direction[axis]
            label = f"{axis} dir=+1 speed={speed:.2f} duration={DURATION_MS}ms"
            print()
            print(f"▶  {label}")

            _return_home(client, overview)

            print("    firing…", flush=True)
            ok, err = _fire_move(client, axis, direction, speed)
            if not ok:
                print(f"    ✖  ONVIF call failed: {err}")
                rows.append({
                    "axis": axis,
                    "speed": speed,
                    "direction": direction,
                    "duration_ms": DURATION_MS,
                    "grade": "onvif_error",
                    "error": err,
                })
                continue

            time.sleep(SETTLE_AFTER_MOVE_SEC)

            if auto:
                # Watch-only: hold for WATCH_PAUSE_SEC so the operator
                # can mentally note the step size, then move on. No
                # grade recorded — that's reported back verbally.
                print(f"    → WATCH (axis={axis} speed={speed:.2f})  "
                      f"{int(WATCH_PAUSE_SEC)}s …", flush=True)
                time.sleep(WATCH_PAUSE_SEC)
                grade = "unrated_auto"
            else:
                grade = _prompt_grade(label)

                # Endstop retry: discard one endstop sample by re-firing
                # once. If the second attempt also endstops, accept it —
                # there's something about the cam's home position that
                # doesn't leave headroom for a 300ms burst at that speed.
                if grade == "endstop":
                    print("    endstop — retrying once after a longer settle.")
                    _return_home(client, overview)
                    time.sleep(SETTLE_AFTER_HOME_SEC)  # double settle
                    ok, err = _fire_move(client, axis, direction, speed)
                    if ok:
                        time.sleep(SETTLE_AFTER_MOVE_SEC)
                        grade = _prompt_grade(label + " (retry)")

            rows.append({
                "axis": axis,
                "speed": speed,
                "direction": direction,
                "duration_ms": DURATION_MS,
                "grade": grade,
            })

    # Final return home before we leave the operator hanging.
    _return_home(client, overview)
    return rows


def _summarize(rows: list[dict]) -> dict:
    """Boil the matrix into a verdict.

    Two checks:
      - Velocity sensitivity per axis (does the grade change between
        speed=0.2 and speed=0.9?).
      - Inter-axis asymmetry (is zoom's medium-grade speed in a wildly
        different position from pan/tilt's?).
    """
    by_axis: dict[str, dict[float, str]] = {"pan": {}, "tilt": {}, "zoom": {}}
    for row in rows:
        by_axis[row["axis"]][row["speed"]] = row["grade"]

    rank = {"no_movement": 0, "small": 1, "medium": 2, "large": 3,
            "endstop": 4, "onvif_error": -1}

    velocity_sensitive: dict[str, bool] = {}
    for axis, samples in by_axis.items():
        ranks = [rank.get(samples.get(s, "no_movement"), 0) for s in SPEEDS]
        # "Sensitive" = the grade actually moves up as speed increases.
        # We accept any monotonic-ish rise; flatlines mean the cam
        # ignores velocity on that axis.
        velocity_sensitive[axis] = max(ranks) > min(ranks)

    # Find each axis's "smallest-step speed" — the lowest speed that
    # still produces a visible move. That's the calibration anchor.
    smallest_visible: dict[str, float | None] = {}
    for axis, samples in by_axis.items():
        chosen: float | None = None
        for s in SPEEDS:
            if rank.get(samples.get(s, "no_movement"), 0) >= 1:
                chosen = s
                break
        smallest_visible[axis] = chosen

    return {
        "by_axis": by_axis,
        "velocity_sensitive": velocity_sensitive,
        "smallest_visible_speed_per_axis": smallest_visible,
    }


def _print_verdict(summary: dict) -> None:
    print()
    print("=" * 64)
    print(" Verdict")
    print("=" * 64)
    print()

    print("  Grade matrix (axis × speed):")
    print(f"    {'axis':6} | " + " | ".join(f" {s:>4.2f} " for s in SPEEDS))
    print(f"    {'-'*6}-+-" + "-+-".join("-" * 6 for _ in SPEEDS))
    for axis in ("pan", "tilt", "zoom"):
        row = summary["by_axis"][axis]
        cells = " | ".join(f" {row.get(s, '—'):>5} " for s in SPEEDS)
        print(f"    {axis:6} | {cells}")
    print()

    print("  Velocity sensitivity per axis:")
    for axis, sensitive in summary["velocity_sensitive"].items():
        marker = "✓ scales with speed" if sensitive else "✗ ignores speed (binary on/off)"
        print(f"    {axis:6}: {marker}")
    print()

    sv = summary["smallest_visible_speed_per_axis"]
    print("  Smallest speed that still produces a visible move:")
    for axis in ("pan", "tilt", "zoom"):
        v = sv.get(axis)
        print(f"    {axis:6}: {v if v is not None else 'none in tested range'}")
    print()

    # Recommendation logic.
    any_binary = any(not v for v in summary["velocity_sensitive"].values())
    if any_binary:
        print("  → Recommendation: Option B (Burst-Multiplier).")
        print("    At least one axis ignores velocity, so scaling JOY_SPEED")
        print("    has no effect there. Fix is to fire pan/tilt 2-3× per")
        print("    heartbeat while keeping zoom at 1×.")
    else:
        print("  → Recommendation: Option A (per-axis speed slider).")
        print("    All three axes respect velocity. Set pan/tilt to a")
        print("    higher JOY_SPEED than zoom (or vice versa) to equalize")
        print("    perceived step size. Store the two values per-camera.")
    print()


def _write_report(camera_id: int, rows: list[dict], summary: dict) -> Path:
    out_root = get_path_manager().base_dir / "ptz_step_asymmetry"
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_root / f"cam{camera_id}_{stamp}.yaml"

    payload = {
        "camera_id": camera_id,
        "timestamp": stamp,
        "duration_ms": DURATION_MS,
        "speeds_tested": SPEEDS,
        "rows": rows,
        "summary": summary,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "PTZ step-asymmetry probe").splitlines()[0]
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        required=True,
        help="ID of the PTZ camera in the WMB DB.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Watch-only mode: no grade prompts, fixed pause between moves.",
    )
    args = parser.parse_args()

    try:
        rows = _run_sweep(args.camera_id, auto=args.auto)
    except KeyboardInterrupt:
        print("\n  Aborted by operator. No report written.")
        return 130

    if args.auto:
        print()
        print("=" * 64)
        print(" Sweep complete — fill this matrix in based on what you saw:")
        print("=" * 64)
        print()
        print(f"    {'axis':6} | " + " | ".join(f" {s:>4.2f} " for s in SPEEDS))
        print(f"    {'-'*6}-+-" + "-+-".join("-" * 6 for _ in SPEEDS))
        for axis in ("pan", "tilt", "zoom"):
            print(f"    {axis:6} | " + " | ".join("  ???  " for _ in SPEEDS))
        print()
        print("  Grades: 0=none / s=small / m=medium / l=large / x=endstop.")
        print("  Report the matrix back to the analyst.")
        return 0

    summary = _summarize(rows)
    _print_verdict(summary)

    report_path = _write_report(args.camera_id, rows, summary)
    print(f"  Report written: {report_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
