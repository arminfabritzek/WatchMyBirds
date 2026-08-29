"""Bounded physical movement helpers for the PTZ laboratory."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from scripts.ptz_lab.common import (
    _password,
    connect_lab,
    report_header,
    sanitize,
    write_report,
)
from utils.visual_alignment import FeatureAligner


@dataclass(frozen=True)
class MotionLimits:
    """Hard limits for the first physical camera probe."""

    speed: float = 0.1
    duration_ms: int = 100
    large_jump: bool = False

    def __post_init__(self) -> None:
        max_speed = 1.0 if self.large_jump else 0.25
        max_duration_ms = 2000 if self.large_jump else 250
        if not 0.01 <= self.speed <= max_speed:
            raise ValueError(f"speed must be between 0.01 and {max_speed}")
        if not 50 <= self.duration_ms <= max_duration_ms:
            raise ValueError(f"duration_ms must be between 50 and {max_duration_ms}")


def rewrite_uri_host(uri: str, host: str) -> str:
    """Replace stale advertised host/userinfo while preserving port and path."""
    parsed = urlsplit(uri)
    clean_host = host
    if ":" in clean_host and not clean_host.startswith("["):
        clean_host = f"[{clean_host}]"
    netloc = clean_host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _alignment_payload(
    align: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    try:
        return {"status": "SUPPORTED", "value": sanitize(align(reference, current))}
    except Exception as exc:
        return {
            "status": "INCONCLUSIVE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "value": None,
        }


def run_continuous_probe(
    client: Any,
    *,
    capture: Callable[[], np.ndarray],
    align: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    return_preset: str,
    axis: str,
    direction: int,
    limits: MotionLimits,
    settle: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one tiny burst and guarantee stop plus return-to-preset."""
    if axis not in {"pan", "tilt", "zoom"}:
        raise ValueError("axis must be pan, tilt, or zoom")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if not return_preset:
        raise ValueError("return_preset is required")

    report: dict[str, Any] = {
        "command": {
            "axis": axis,
            "direction": direction,
            "speed": limits.speed,
            "duration_ms": limits.duration_ms,
        },
        "return_preset": return_preset,
        "initial_positioning": {"status": "PENDING"},
        "movement": {"status": "PENDING"},
        "return": {"status": "PENDING"},
    }
    baseline: np.ndarray | None = None
    try:
        for _attempt in range(3):
            client.goto_preset(return_preset)
            settle(2.0)
        report["initial_positioning"] = {"status": "SUPPORTED", "attempts": 3}
        baseline = capture()

        command = {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}
        command[axis] = direction * limits.speed
        client.continuous_move(
            **command,
            duration_ms=limits.duration_ms,
        )
        settle(1.0)
        moved = capture()
        report["movement"] = {
            "status": "SUPPORTED",
            "alignment": _alignment_payload(align, baseline, moved),
        }
    except Exception as exc:
        report["movement"] = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        client.emergency_stop()
        try:
            return_attempts: list[dict[str, Any]] = []
            returned_close = False
            for attempt in range(1, 4):
                client.goto_preset(return_preset)
                settle(2.0)
                returned = capture()
                alignment = (
                    _alignment_payload(align, baseline, returned)
                    if baseline is not None
                    else {"status": "INCONCLUSIVE", "value": None}
                )
                return_attempts.append({"attempt": attempt, "alignment": alignment})
                value = alignment.get("value") or {}
                error = value.get("error_magnitude")
                if alignment.get("status") == "SUPPORTED" and (
                    error is None or float(error) <= 0.005
                ):
                    returned_close = True
                    break
            report["return"] = {
                "status": "SUPPORTED" if returned_close else "INCONCLUSIVE",
                "attempts": return_attempts,
                "alignment": return_attempts[-1]["alignment"],
            }
        except Exception as exc:
            report["return"] = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return sanitize(report)


class SnapshotCapture:
    """Fetch fresh HTTP snapshots and retain local visual evidence."""

    def __init__(self, client: Any, ip: str, artifact_dir: Path) -> None:
        self.client = client
        self.ip = ip
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.paths: list[str] = []
        self._index = 0

    def __call__(self) -> np.ndarray:
        uri = rewrite_uri_host(self.client.get_snapshot_uri(), self.ip)
        response = None
        for auth in (
            HTTPDigestAuth(self.client.username, self.client.password),
            HTTPBasicAuth(self.client.username, self.client.password),
        ):
            candidate = requests.get(
                uri,
                auth=auth,
                headers={"Cache-Control": "no-cache"},
                timeout=8,
            )
            response = candidate
            if candidate.status_code == 200:
                break
        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else "no response"
            raise RuntimeError(f"snapshot request failed: HTTP {status}")
        frame = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None or frame.size == 0:
            raise RuntimeError("snapshot response is not a decodable image")
        self._index += 1
        path = self.artifact_dir / f"frame_{self._index:02d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"could not write snapshot artifact: {path}")
        self.paths.append(str(path))
        return frame


def _live_alignment(reference: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    return FeatureAligner(method="auto").align(reference, current).to_dict()


def main_continuous(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one bounded PTZ burst with guaranteed return"
    )
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--password-env", default="WMB_CAM_PASSWORD")
    parser.add_argument("--return-preset", required=True)
    parser.add_argument("--axis", choices=("pan", "tilt", "zoom"), default="pan")
    parser.add_argument("--direction", choices=(-1, 1), type=int, default=1)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument(
        "--large-jump",
        action="store_true",
        help="Allow up to speed 1.0 and 2000 ms; conservative limits stay default",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/ptz_lab"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("physical movement requires --execute")
    try:
        limits = MotionLimits(
            speed=args.speed,
            duration_ms=args.duration_ms,
            large_jump=args.large_jump,
        )
    except ValueError as exc:
        parser.error(str(exc))

    password = _password(args.password_env)
    conn = connect_lab(args.ip, args.port, args.user, password, args.profile_index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture = SnapshotCapture(
        conn.client,
        args.ip,
        args.output_dir / f"continuous_{timestamp}",
    )
    report = report_header("continuous", args.ip, args.port)
    report["read_only"] = False
    report.update(
        run_continuous_probe(
            conn.client,
            capture=capture,
            align=_live_alignment,
            return_preset=args.return_preset,
            axis=args.axis,
            direction=args.direction,
            limits=limits,
        )
    )
    report["artifacts"] = capture.paths
    json_path, yaml_path = write_report(report, args.output_dir, "ptz_continuous")
    print(f"Bounded movement probe complete. Reports: {json_path} {yaml_path}")
    return 0 if report["return"]["status"] == "SUPPORTED" else 3
