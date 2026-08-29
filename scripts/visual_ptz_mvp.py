#!/usr/bin/env python3
"""Capture, locate, and visually re-align an ONVIF PTZ camera without presets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.visual_ptz_core import (
    ClosedLoopVisualPtz,
    OpenCvFrameSource,
    VisualKeyframeStore,
    VisualPtzLimits,
    verify_motion_response,
)
from utils.path_manager import get_path_manager
from utils.visual_alignment import FeatureAligner


def _stream_url(camera_id: int, explicit: str | None) -> str:
    if explicit:
        return explicit
    from core import onvif_core

    resolved = onvif_core.get_camera_uri(camera_id)
    if not resolved:
        raise RuntimeError(
            "Could not resolve an RTSP URI via ONVIF; pass --stream-url explicitly"
        )
    return resolved


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _capture(args: argparse.Namespace) -> int:
    with OpenCvFrameSource(_stream_url(args.camera_id, args.stream_url)) as source:
        payload = VisualKeyframeStore().save(
            args.camera_id,
            args.keyframe,
            source.read(),
        )
    _print_json(payload)
    return 0


def _list(args: argparse.Namespace) -> int:
    _print_json({"keyframes": VisualKeyframeStore().list(args.camera_id)})
    return 0


def _locate(args: argparse.Namespace) -> int:
    store = VisualKeyframeStore()
    reference = store.load(args.camera_id, args.keyframe)
    debug_dir = get_path_manager().get_visual_ptz_debug_dir(args.camera_id)
    debug_path = debug_dir / f"locate_{args.keyframe}.jpg"
    if args.current_image:
        current = cv2.imread(args.current_image)
        if current is None:
            raise FileNotFoundError(
                f"Could not read current image: {args.current_image}"
            )
        result = FeatureAligner(method=args.method).align(
            reference, current, debug_path=debug_path
        )
    else:
        with OpenCvFrameSource(_stream_url(args.camera_id, args.stream_url)) as source:
            result = FeatureAligner(method=args.method).align(
                reference, source.read(), debug_path=debug_path
            )
    payload = result.to_dict()
    payload["debug_path"] = str(debug_path)
    _print_json(payload)
    return 0 if result.success else 2


def _recognize(args: argparse.Namespace) -> int:
    """Rank all stored visual places against one current frame."""
    store = VisualKeyframeStore()
    keyframes = store.list(args.camera_id)
    if not keyframes:
        raise RuntimeError("No visual keyframes stored for this camera")
    if args.current_image:
        current = cv2.imread(args.current_image)
        if current is None:
            raise FileNotFoundError(
                f"Could not read current image: {args.current_image}"
            )
    else:
        with OpenCvFrameSource(_stream_url(args.camera_id, args.stream_url)) as source:
            current = source.read()
    debug_dir = get_path_manager().get_visual_ptz_debug_dir(args.camera_id)
    aligner = FeatureAligner(method=args.method)
    candidates = []
    for keyframe in keyframes:
        keyframe_id = keyframe["keyframe_id"]
        result = aligner.align(
            store.load(args.camera_id, keyframe_id),
            current,
            debug_path=debug_dir / f"recognize_{keyframe_id}.jpg",
        )
        candidates.append({"keyframe_id": keyframe_id, **result.to_dict()})
    candidates.sort(
        key=lambda item: (bool(item["success"]), float(item["quality"])),
        reverse=True,
    )
    accepted = candidates[0]["success"] and candidates[0]["quality"] >= args.min_quality
    _print_json(
        {
            "recognized": bool(accepted),
            "best_keyframe": candidates[0]["keyframe_id"] if accepted else None,
            "candidates": candidates,
        }
    )
    return 0 if accepted else 2


def _align(args: argparse.Namespace) -> int:
    from core import ptz_core

    config = ptz_core.get_ptz_config(args.camera_id) or {}
    if config.get("enabled"):
        raise RuntimeError(
            "Auto-PTZ is enabled and could fight this controller. Disable it in "
            "Settings before running hardware alignment."
        )
    store = VisualKeyframeStore()
    reference = store.load(args.camera_id, args.keyframe)
    limits = VisualPtzLimits(
        max_iterations=args.max_iterations,
        deadband=args.deadband,
        min_quality=args.min_quality,
        max_step=args.max_step,
        max_total_motion=args.max_total_motion,
        move_duration_ms=args.move_duration_ms,
        settle_sec=args.settle_sec,
        invert_pan=args.invert_pan,
        invert_tilt=args.invert_tilt,
    )
    root = get_path_manager().get_visual_ptz_debug_dir(args.camera_id)
    run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with OpenCvFrameSource(
        _stream_url(args.camera_id, args.stream_url), fresh_reads=args.fresh_reads
    ) as source:
        if not args.execute:
            result = FeatureAligner(method=args.method).align(
                reference, source.read(), debug_path=run_dir / "dry_run.jpg"
            )
            payload = result.to_dict()
            payload.update(
                {
                    "dry_run": True,
                    "message": "No PTZ command sent; add --execute to close the loop",
                    "debug_path": str(run_dir / "dry_run.jpg"),
                }
            )
            _print_json(payload)
            return 0 if result.success else 2

        controller = ClosedLoopVisualPtz(
            aligner=FeatureAligner(method=args.method),
            frame_supplier=source.read,
            mover=lambda pan, tilt, duration: ptz_core.continuous_move(
                args.camera_id,
                pan=pan,
                tilt=tilt,
                duration_ms=duration,
            ),
            stopper=lambda: ptz_core.stop(args.camera_id),
            limits=limits,
        )
        run = controller.align_to(reference, debug_dir=run_dir)
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    payload = run.to_dict()
    payload["report_path"] = str(report_path)
    _print_json(payload)
    return 0 if run.aligned else 3


def _probe_motion(args: argparse.Namespace) -> int:
    """Command one axis and verify every pulse from live video geometry."""
    from core import ptz_core

    config = ptz_core.get_ptz_config(args.camera_id) or {}
    if config.get("enabled"):
        raise RuntimeError(
            "Auto-PTZ is enabled and could fight this probe. Disable it first."
        )
    if not 1 <= args.pulses <= 6:
        raise ValueError("pulses must be between 1 and 6")
    if not 0.05 <= args.speed <= 1.0:
        raise ValueError("speed must be between 0.05 and 1.0")
    if not 50 <= args.duration_ms <= 500:
        raise ValueError("duration-ms must be between 50 and 500")

    root = get_path_manager().get_visual_ptz_debug_dir(args.camera_id)
    run_dir = root / datetime.now().strftime("motion_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    aligner = FeatureAligner(method=args.method)
    with OpenCvFrameSource(
        _stream_url(args.camera_id, args.stream_url), fresh_reads=args.fresh_reads
    ) as source:
        before = source.read()
        if not args.execute:
            time.sleep(min(0.5, args.settle_sec))
            after = source.read()
            baseline = aligner.align(before, after, debug_path=run_dir / "baseline.jpg")
            _print_json(
                {
                    "dry_run": True,
                    "message": "No PTZ command sent; baseline visual drift measured",
                    "baseline": baseline.to_dict(),
                    "debug_path": str(run_dir / "baseline.jpg"),
                }
            )
            return 0 if baseline.success else 2

        samples = []
        for index in range(1, args.pulses + 1):
            command = {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}
            command[args.axis] = args.direction * args.speed
            ptz_core.continuous_move(
                args.camera_id,
                pan=command["pan"],
                tilt=command["tilt"],
                zoom=command["zoom"],
                duration_ms=args.duration_ms,
            )
            time.sleep(args.settle_sec)
            after = source.read()
            debug_path = run_dir / f"pulse_{index:02d}.jpg"
            alignment = aligner.align(before, after, debug_path=debug_path)
            observation = verify_motion_response(
                alignment,
                axis=args.axis,
                direction=args.direction,
                min_translation=args.min_response,
                min_zoom_scale=args.min_response,
            )
            samples.append(
                {
                    "index": index,
                    "command": {**command, "duration_ms": args.duration_ms},
                    "alignment": alignment.to_dict(),
                    "observation": observation.to_dict(),
                    "debug_path": str(debug_path),
                }
            )
            before = after
            if not alignment.success or alignment.quality < args.min_quality:
                break

    correct_responses = [
        abs(sample["observation"]["response"])
        for sample in samples
        if sample["observation"]["direction_correct"]
        and sample["observation"]["response"] is not None
    ]
    median_response = (
        statistics.median(correct_responses) if correct_responses else None
    )
    recommended_burst = None
    if median_response and median_response > 0:
        recommended_burst = max(
            1, min(6, math.ceil(args.target_response / median_response))
        )
    verified = len(correct_responses) == len(samples) and len(samples) == args.pulses
    payload = {
        "axis": args.axis,
        "direction": args.direction,
        "verified": verified,
        "pulses_requested": args.pulses,
        "pulses_measured": len(samples),
        "median_response_per_pulse": median_response,
        "target_response": args.target_response,
        "recommended_burst": recommended_burst,
        "samples": samples,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    payload["report_path"] = str(report_path)
    _print_json(payload)
    return 0 if verified else 3


def _demo(args: argparse.Namespace) -> int:
    """Create an offline visual proof with a known image displacement."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    reference = np.full((540, 960, 3), (30, 45, 30), dtype=np.uint8)
    for index in range(180):
        x = int(rng.integers(20, 940))
        y = int(rng.integers(20, 520))
        radius = int(rng.integers(3, 13))
        color = tuple(int(v) for v in rng.integers(65, 245, size=3))
        cv2.circle(reference, (x, y), radius, color, -1)
        if index % 9 == 0:
            cv2.line(reference, (x - 12, y), (x + 12, y), color, 2)
    cv2.rectangle(reference, (290, 180), (650, 430), (160, 110, 55), 8)
    cv2.putText(
        reference,
        "VISUAL PTZ TARGET",
        (315, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (245, 245, 245),
        3,
    )
    transform = cv2.getRotationMatrix2D((480, 270), 1.5, 1.0)
    transform[:, 2] += (args.dx, args.dy)
    current = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    reference_path = output_dir / "reference.jpg"
    current_path = output_dir / "current.jpg"
    debug_path = output_dir / "matches.jpg"
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(current_path), current)
    result = FeatureAligner(method=args.method).align(
        reference, current, debug_path=debug_path
    )
    payload = result.to_dict()
    payload.update(
        {
            "expected_dx": args.dx,
            "expected_dy": args.dy,
            "reference_path": str(reference_path),
            "current_path": str(current_path),
            "debug_path": str(debug_path),
        }
    )
    _print_json(payload)
    return 0 if result.success else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def camera_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--camera-id", type=int, required=True)
        command.add_argument("--stream-url")
        command.add_argument(
            "--method", choices=("auto", "sift", "orb"), default="auto"
        )

    capture = subparsers.add_parser(
        "capture", help="Store the current view as a keyframe"
    )
    camera_arguments(capture)
    capture.add_argument("--keyframe", required=True)
    capture.set_defaults(handler=_capture)

    list_command = subparsers.add_parser("list", help="List stored visual keyframes")
    list_command.add_argument("--camera-id", type=int, required=True)
    list_command.set_defaults(handler=_list)

    locate = subparsers.add_parser("locate", help="Match a keyframe without moving")
    camera_arguments(locate)
    locate.add_argument("--keyframe", required=True)
    locate.add_argument("--current-image")
    locate.set_defaults(handler=_locate)

    recognize = subparsers.add_parser(
        "recognize", help="Rank the current view against every stored keyframe"
    )
    camera_arguments(recognize)
    recognize.add_argument("--current-image")
    recognize.add_argument("--min-quality", type=float, default=0.45)
    recognize.set_defaults(handler=_recognize)

    align = subparsers.add_parser(
        "align", help="Iteratively align to a visual keyframe"
    )
    camera_arguments(align)
    align.add_argument("--keyframe", required=True)
    align.add_argument(
        "--execute", action="store_true", help="Allow bounded real PTZ movement"
    )
    align.add_argument("--max-iterations", type=int, default=8)
    align.add_argument("--deadband", type=float, default=0.025)
    align.add_argument("--min-quality", type=float, default=0.45)
    align.add_argument("--max-step", type=float, default=0.22)
    align.add_argument("--max-total-motion", type=float, default=1.2)
    align.add_argument("--move-duration-ms", type=int, default=180)
    align.add_argument("--settle-sec", type=float, default=1.0)
    align.add_argument("--fresh-reads", type=int, default=3)
    align.add_argument("--invert-pan", action="store_true")
    align.add_argument("--invert-tilt", action="store_true")
    align.set_defaults(handler=_align)

    probe = subparsers.add_parser(
        "probe-motion",
        help="Move one PTZ axis and verify each pulse from before/after frames",
    )
    camera_arguments(probe)
    probe.add_argument("--axis", choices=("pan", "tilt", "zoom"), required=True)
    probe.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    probe.add_argument("--execute", action="store_true")
    probe.add_argument("--pulses", type=int, default=3)
    probe.add_argument("--speed", type=float, default=0.25)
    probe.add_argument("--duration-ms", type=int, default=180)
    probe.add_argument("--settle-sec", type=float, default=1.0)
    probe.add_argument("--fresh-reads", type=int, default=3)
    probe.add_argument("--min-quality", type=float, default=0.45)
    probe.add_argument("--min-response", type=float, default=0.004)
    probe.add_argument("--target-response", type=float, default=0.08)
    probe.set_defaults(handler=_probe_motion)

    demo = subparsers.add_parser("demo", help="Generate an offline match visualization")
    demo.add_argument("--output-dir", default="output/visual_ptz_demo")
    demo.add_argument("--method", choices=("auto", "sift", "orb"), default="auto")
    demo.add_argument("--dx", type=float, default=85.0)
    demo.add_argument("--dy", type=float, default=-45.0)
    demo.set_defaults(handler=_demo)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        if getattr(args, "camera_id", None) is not None:
            try:
                from core import ptz_core

                ptz_core.stop(args.camera_id)
            except Exception:
                pass
        print("Aborted; emergency PTZ stop requested.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"visual-ptz: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
