#!/usr/bin/env python3
"""Isolate the source of the post-PTZ detection-latency regression.

Runs on the RPi as the watchmybirds user. Stops the live app service
for the duration so the probe owns the CPU, then runs three phases
back-to-back against the SAME warmed-up DetectionService:

  Phase A — Detection only, no PTZ controller present.
  Phase B — Detection + idle AutoPtzController (worker thread alive,
            handle_no_detection() called every iteration).
  Phase C — PTZ overhead isolated: only handle_detections() and
            handle_no_detection(), no detector inference at all.

A printed comparison tells us where the 700-to-3700 ms regression sits:

  - A ~= 3000 ms                  -> not the PTZ commit; something else
                                     regressed (env, lib, thermal).
  - A ~= 700 ms, B ~= 3000 ms     -> PTZ-controller side effect on the
                                     detection thread (GIL / import).
  - A == B ~= 700 ms, C tiny      -> regression is somewhere else in
                                     the live loop (motion, draw, etc).

Usage on the RPi:

    sudo systemctl stop app
    sudo -u watchmybirds /opt/app/.venv/bin/python \\
        /opt/app/scripts/diagnose_ptz_overhead.py
    sudo systemctl start app

The script does NOT touch the DB, does NOT write to OUTPUT_DIR, and
does NOT issue real PTZ commands (command_runner is a no-op stub).
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

WARMUP_FRAMES = 5
MEASURE_FRAMES = 30
FRAME_SHAPE = (1080, 1920, 3)


def _make_frame(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=FRAME_SHAPE, dtype=np.uint8)


def _stats(samples_ms: list[float]) -> str:
    if not samples_ms:
        return "no samples"
    return (
        f"n={len(samples_ms)} "
        f"avg={statistics.mean(samples_ms):.1f}ms "
        f"min={min(samples_ms):.1f}ms "
        f"max={max(samples_ms):.1f}ms "
        f"p50={statistics.median(samples_ms):.1f}ms"
    )


def _build_detection_service():
    from detectors.services.detection_service import DetectionService

    service = DetectionService()
    if not service._ensure_initialized():
        raise SystemExit("Detector failed to initialize")
    return service


def _save_threshold(service) -> float:
    from config import effective_save_threshold, get_config

    detector_obj = service._detector
    underlying = getattr(detector_obj, "model", None)
    detector_conf = getattr(underlying, "conf_threshold_default", None)
    return effective_save_threshold(get_config(), detector_conf)


def phase_a_detection_only(service, frames: list[np.ndarray]) -> list[float]:
    print("\n=== Phase A: Detection only, no PTZ controller ===")
    save_thr = _save_threshold(service)
    samples: list[float] = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        service.detect(frame=frame, save_threshold=save_thr)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP_FRAMES:
            samples.append(dt_ms)
    print(_stats(samples))
    return samples


def phase_b_detection_with_idle_ptz(
    service, frames: list[np.ndarray]
) -> list[float]:
    print("\n=== Phase B: Detection + idle AutoPtzController ===")
    from core.ptz_tracking_core import AutoPtzController

    controller = AutoPtzController(
        camera_provider=lambda: None,
        command_runner=lambda cmd: None,
        worker_enabled=True,
    )
    try:
        save_thr = _save_threshold(service)
        samples: list[float] = []
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            result = service.detect(frame=frame, save_threshold=save_thr)
            if result.detected:
                controller.handle_detections(
                    frame_shape=frame.shape,
                    detections=result.detections,
                )
            else:
                controller.handle_no_detection()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if i >= WARMUP_FRAMES:
                samples.append(dt_ms)
        print(_stats(samples))
        return samples
    finally:
        controller.stop()


def phase_c_ptz_overhead_only(frames: list[np.ndarray]) -> dict[str, list[float]]:
    print("\n=== Phase C: PTZ overhead isolated ===")
    from core.ptz_tracking_core import AutoPtzController

    fake_detections = [
        {
            "x1": 100,
            "y1": 100,
            "x2": 400,
            "y2": 400,
            "confidence": 0.9,
            "class_name": "bird",
        }
    ]

    no_det_disabled: list[float] = []
    handle_det_disabled: list[float] = []

    controller = AutoPtzController(
        camera_provider=lambda: None,
        command_runner=lambda cmd: None,
        worker_enabled=True,
    )
    try:
        for frame in frames:
            t0 = time.perf_counter()
            controller.handle_no_detection()
            no_det_disabled.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            controller.handle_detections(
                frame_shape=frame.shape, detections=fake_detections
            )
            handle_det_disabled.append((time.perf_counter() - t0) * 1000.0)
    finally:
        controller.stop()

    print(
        "handle_no_detection (no PTZ camera enabled):    "
        + _stats(no_det_disabled[WARMUP_FRAMES:])
    )
    print(
        "handle_detections   (no PTZ camera enabled):    "
        + _stats(handle_det_disabled[WARMUP_FRAMES:])
    )
    return {
        "no_det_disabled": no_det_disabled[WARMUP_FRAMES:],
        "handle_det_disabled": handle_det_disabled[WARMUP_FRAMES:],
    }


def phase_d_camera_lookup_cache() -> None:
    print("\n=== Phase D: find_auto_ptz_camera() cache behavior ===")
    from core import ptz_core

    # Cold call
    t0 = time.perf_counter()
    ptz_core.find_auto_ptz_camera()
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"cold call (cache miss): {cold_ms:.2f} ms")

    # Hot calls — should all hit the 2 s TTL cache.
    hot_samples: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        ptz_core.find_auto_ptz_camera()
        hot_samples.append((time.perf_counter() - t0) * 1000.0)
    print("hot calls (cache hit): " + _stats(hot_samples))


def main() -> int:
    print("WatchMyBirds — PTZ overhead diagnostic")
    print(f"Python: {sys.version.split()[0]}  PID: {os.getpid()}")
    print(f"Repo:   {REPO_ROOT}")
    print(
        f"Frames: {MEASURE_FRAMES} measured + {WARMUP_FRAMES} warmup "
        f"at {FRAME_SHAPE[1]}x{FRAME_SHAPE[0]}"
    )

    gc.collect()

    rng = np.random.default_rng(42)
    frames = [_make_frame(rng) for _ in range(WARMUP_FRAMES + MEASURE_FRAMES)]

    print("\nLoading detector...")
    t0 = time.perf_counter()
    service = _build_detection_service()
    print(f"Detector ready in {(time.perf_counter() - t0):.2f} s")

    a = phase_a_detection_only(service, frames)
    b = phase_b_detection_with_idle_ptz(service, frames)
    _ = phase_c_ptz_overhead_only(frames)
    phase_d_camera_lookup_cache()

    print("\n=== Verdict ===")
    if not a or not b:
        print("Insufficient samples; rerun.")
        return 1

    a_avg = statistics.mean(a)
    b_avg = statistics.mean(b)
    delta = b_avg - a_avg
    delta_pct = (delta / a_avg) * 100 if a_avg > 0 else 0.0
    print(f"Phase A (no PTZ):   avg {a_avg:.1f} ms")
    print(f"Phase B (with PTZ): avg {b_avg:.1f} ms")
    print(f"Delta:              {delta:+.1f} ms ({delta_pct:+.1f}%)")

    if a_avg > 2000:
        print(
            "\n--> Phase A alone is already >2 s. The PTZ commit is NOT the "
            "primary cause. Look elsewhere: environment, thermal throttling, "
            "library bump, SD-card I/O."
        )
    elif delta_pct > 50:
        print(
            "\n--> Phase B is >50% slower than Phase A. The AutoPtzController "
            "is implicated. Likely GIL pressure from the worker thread or an "
            "import-time side effect."
        )
    else:
        print(
            "\n--> PTZ controller overhead is small. The live regression must "
            "come from somewhere outside Phase A/B coverage (e.g. the "
            "motion-detection path, web-server traffic on shared cores)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
