"""Shared read-only ONVIF helpers for the PTZ laboratory scripts."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml
from zeep.helpers import serialize_object

from camera.ptz_client import PtzClient

READ_ONLY_OPERATIONS = frozenset(
    {
        "GetCapabilities",
        "GetConfigurationOptions",
        "GetConfigurations",
        "GetDeviceInformation",
        "GetImagingSettings",
        "GetMoveOptions",
        "GetNodes",
        "GetOptions",
        "GetPresets",
        "GetProfiles",
        "GetServiceCapabilities",
        "GetServices",
        "GetSnapshotUri",
        "GetStatus",
        "GetStreamUri",
        "GetSystemDateAndTime",
        "GetVideoSources",
    }
)

_SENSITIVE_KEYS = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
    "username",
    "user",
}
_UNSUPPORTED_MARKERS = (
    "actionnotsupported",
    "not implemented",
    "not supported",
    "optional action not implemented",
)
_CREDENTIAL_IN_URL = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)
_SENSITIVE_QUERY_MARKERS = ("auth", "credential", "pass", "secret", "token", "user")


@dataclass
class LabConnection:
    """Live handles used only by the read-only laboratory."""

    client: PtzClient
    device: Any
    media: Any
    ptz: Any
    imaging: Any | None
    profiles: list[Any]
    active_profile_token: str


def _sanitize_text(value: str) -> str:
    """Remove URL userinfo without hiding useful camera addresses or paths."""
    if "://" not in value:
        return _CREDENTIAL_IN_URL.sub(r"\g<scheme><redacted>@", value)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _CREDENTIAL_IN_URL.sub(r"\g<scheme><redacted>@", value)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query = urlencode(
        [
            (
                key,
                "<redacted>"
                if any(marker in key.lower() for marker in _SENSITIVE_QUERY_MARKERS)
                else item,
            )
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def sanitize(value: Any, *, _depth: int = 0) -> Any:
    """Convert ONVIF/zeep values to JSON data and remove credentials."""
    if _depth > 20:
        return "<maximum depth reached>"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize(asdict(value), _depth=_depth + 1)

    converted = value
    if not isinstance(value, Mapping | list | tuple | set):
        try:
            converted = serialize_object(value)
        except Exception:
            converted = value

    if isinstance(converted, Mapping):
        output: dict[str, Any] = {}
        for key, item in converted.items():
            key_text = str(key)
            if key_text.lower() in _SENSITIVE_KEYS:
                output[key_text] = "<redacted>"
            else:
                output[key_text] = sanitize(item, _depth=_depth + 1)
        return dict(sorted(output.items()))
    if isinstance(converted, list | tuple | set):
        return [sanitize(item, _depth=_depth + 1) for item in converted]
    if hasattr(converted, "__dict__"):
        public = {
            key: item
            for key, item in vars(converted).items()
            if not key.startswith("_")
        }
        return sanitize(public, _depth=_depth + 1)
    return _sanitize_text(str(converted))


def call_readonly(
    operation: str,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute one allow-listed read operation and retain its diagnostic fault."""
    if operation not in READ_ONLY_OPERATIONS:
        raise ValueError(f"Operation is not read-only: {operation}")
    started = time.perf_counter()
    try:
        value = function(*args, **kwargs)
    except Exception as exc:  # vendor SOAP faults vary by camera firmware
        message = _sanitize_text(str(exc))
        lowered = message.lower()
        status = (
            "UNSUPPORTED"
            if any(marker in lowered for marker in _UNSUPPORTED_MARKERS)
            else "ERROR"
        )
        return {
            "operation": operation,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "error": message,
            "value": None,
        }
    return {
        "operation": operation,
        "status": "SUPPORTED",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "error_type": None,
        "error": None,
        "value": sanitize(value),
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _token(value: Any) -> str:
    return str(_get(value, "token") or _get(value, "Token") or "")


def connect_lab(
    ip: str,
    port: int,
    username: str,
    password: str,
    profile_index: int,
) -> LabConnection:
    """Connect through WMB's hardened client without issuing camera mutations."""
    client = PtzClient(ip, port, username, password, profile_index)
    ptz, active_profile_token = client._ensure_services()
    if client._camera is None or client._media is None:
        raise RuntimeError("ONVIF camera did not expose device/media services")
    profiles = list(client._media.GetProfiles() or [])
    try:
        imaging = client._camera.create_imaging_service()
    except Exception:
        imaging = None
    return LabConnection(
        client=client,
        device=client._camera.create_devicemgmt_service(),
        media=client._media,
        ptz=ptz,
        imaging=imaging,
        profiles=profiles,
        active_profile_token=active_profile_token,
    )


def _profile_summary(profile: Any, index: int) -> dict[str, Any]:
    ptz_config = _get(profile, "PTZConfiguration")
    source_config = _get(profile, "VideoSourceConfiguration")
    return {
        "index": index,
        "name": str(_get(profile, "Name") or ""),
        "token": _token(profile),
        "ptz_configuration_token": _token(ptz_config),
        "ptz_node_token": str(_get(ptz_config, "NodeToken") or ""),
        "video_source_token": str(_get(source_config, "SourceToken") or ""),
    }


def _fingerprint(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def report_header(kind: str, ip: str, port: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": kind,
        "read_only": True,
        "captured_at": datetime.now(UTC).isoformat(),
        "camera": {"ip": ip, "port": int(port)},
    }


def run_identity(conn: LabConnection, ip: str, port: int) -> dict[str, Any]:
    report = report_header("identity", ip, port)
    device_info = call_readonly(
        "GetDeviceInformation", conn.device.GetDeviceInformation
    )
    if device_info["status"] == "SUPPORTED" and isinstance(device_info["value"], dict):
        info = device_info["value"]
        info["SerialNumberFingerprint"] = _fingerprint(info.pop("SerialNumber", ""))
    report["device_information"] = device_info
    report["system_time"] = call_readonly(
        "GetSystemDateAndTime", conn.device.GetSystemDateAndTime
    )
    report["services"] = call_readonly(
        "GetServices", conn.device.GetServices, {"IncludeCapability": True}
    )
    report["device_capabilities"] = call_readonly(
        "GetCapabilities", conn.device.GetCapabilities, {"Category": "All"}
    )
    report["profiles"] = [
        _profile_summary(profile, index) for index, profile in enumerate(conn.profiles)
    ]
    report["active_profile_token"] = conn.active_profile_token

    uris: list[dict[str, Any]] = []
    for profile in conn.profiles:
        profile_token = _token(profile)
        if not profile_token:
            continue
        stream = call_readonly(
            "GetStreamUri",
            conn.media.GetStreamUri,
            {
                "StreamSetup": {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                },
                "ProfileToken": profile_token,
            },
        )
        snapshot = call_readonly(
            "GetSnapshotUri",
            conn.media.GetSnapshotUri,
            {"ProfileToken": profile_token},
        )
        uris.append(
            {"profile_token": profile_token, "stream": stream, "snapshot": snapshot}
        )
    report["profile_uris"] = uris
    return report


def _range(range_value: Any) -> dict[str, Any] | None:
    if range_value is None:
        return None
    minimum = _get(range_value, "Min")
    maximum = _get(range_value, "Max")
    if minimum is None and maximum is None:
        return None
    return {"min": minimum, "max": maximum}


def extract_space_ranges(options: Any) -> list[dict[str, Any]]:
    """Flatten ONVIF PTZ Spaces while preserving the exact advertised URI."""
    clean = sanitize(options)
    spaces = _get(clean, "Spaces", clean)
    if not isinstance(spaces, Mapping):
        return []
    output: list[dict[str, Any]] = []
    for kind, raw_entries in spaces.items():
        if "Space" not in str(kind):
            continue
        entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            uri = str(entry.get("URI") or "")
            output.append(
                {
                    "kind": str(kind),
                    "uri": uri,
                    "is_fov": "translationspacefov" in uri.lower(),
                    "x": _range(entry.get("XRange")),
                    "y": _range(entry.get("YRange")),
                }
            )
    return output


def _unique_profile_tokens(
    profiles: list[Any], configuration_name: str, token_name: str
) -> list[str]:
    tokens: list[str] = []
    for profile in profiles:
        config = _get(profile, configuration_name)
        token = (
            _token(config)
            if token_name == "token"
            else str(_get(config, token_name) or "")
        )
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def run_capabilities(conn: LabConnection, ip: str, port: int) -> dict[str, Any]:
    report = report_header("capabilities", ip, port)
    report["ptz_service"] = call_readonly(
        "GetServiceCapabilities", conn.ptz.GetServiceCapabilities
    )
    report["ptz_nodes"] = call_readonly("GetNodes", conn.ptz.GetNodes)
    report["ptz_configurations"] = call_readonly(
        "GetConfigurations", conn.ptz.GetConfigurations
    )

    option_results: list[dict[str, Any]] = []
    for config_token in _unique_profile_tokens(
        conn.profiles, "PTZConfiguration", "token"
    ):
        result = call_readonly(
            "GetConfigurationOptions",
            conn.ptz.GetConfigurationOptions,
            {"ConfigurationToken": config_token},
        )
        result["configuration_token"] = config_token
        result["spaces"] = (
            extract_space_ranges(result["value"])
            if result["status"] == "SUPPORTED"
            else []
        )
        option_results.append(result)
    report["ptz_configuration_options"] = option_results

    preset_results: list[dict[str, Any]] = []
    for profile in conn.profiles:
        profile_token = _token(profile)
        if not profile_token:
            continue
        result = call_readonly(
            "GetPresets", conn.ptz.GetPresets, {"ProfileToken": profile_token}
        )
        result["profile_token"] = profile_token
        preset_results.append(result)
    report["presets"] = preset_results

    imaging_results: list[dict[str, Any]] = []
    source_tokens = _unique_profile_tokens(
        conn.profiles, "VideoSourceConfiguration", "SourceToken"
    )
    for source_token in source_tokens:
        item: dict[str, Any] = {"video_source_token": source_token}
        if conn.imaging is None:
            item["status"] = "UNSUPPORTED"
            item["error"] = "Imaging service is unavailable"
        else:
            request = {"VideoSourceToken": source_token}
            item["options"] = call_readonly(
                "GetOptions", conn.imaging.GetOptions, request
            )
            item["move_options"] = call_readonly(
                "GetMoveOptions", conn.imaging.GetMoveOptions, request
            )
            item["settings"] = call_readonly(
                "GetImagingSettings", conn.imaging.GetImagingSettings, request
            )
        imaging_results.append(item)
    report["imaging"] = imaging_results
    return report


def _status_position(value: Any) -> tuple[Any, Any, Any] | None:
    position = _get(value, "Position")
    if not position:
        return None
    pan_tilt = _get(position, "PanTilt")
    zoom = _get(position, "Zoom")
    coordinates = (
        _get(pan_tilt, "x"),
        _get(pan_tilt, "y"),
        _get(zoom, "x"),
    )
    return coordinates if any(item is not None for item in coordinates) else None


def analyse_status_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in samples if item.get("status") == "SUPPORTED"]
    positions = [
        position
        for item in successful
        if (position := _status_position(item.get("value"))) is not None
    ]
    move_values: set[str] = set()
    for item in successful:
        move_status = _get(item.get("value"), "MoveStatus")
        for axis in ("PanTilt", "Zoom"):
            value = _get(move_status, axis)
            if value is not None:
                move_values.add(str(value))
    unique_positions = sorted(set(positions), key=repr)
    all_positions_zero = bool(positions) and all(
        all(coordinate in {None, 0} for coordinate in position)
        for position in positions
    )
    if not positions:
        position_assessment = "NOT_REPORTED"
    elif len(unique_positions) > 1:
        position_assessment = "OBSERVED_CHANGING"
    elif all_positions_zero:
        position_assessment = "INCONCLUSIVE_ZERO_STUB"
    else:
        position_assessment = "OBSERVED_STATIC"
    return {
        "requested_samples": len(samples),
        "successful_samples": len(successful),
        "position_reported": bool(positions),
        "position_changed": len(unique_positions) > 1,
        "all_positions_zero": all_positions_zero,
        "position_assessment": position_assessment,
        "unique_positions": [list(position) for position in unique_positions],
        "move_status_values": sorted(move_values),
    }


def run_status(
    conn: LabConnection,
    ip: str,
    port: int,
    *,
    samples: int,
    interval_sec: float,
) -> dict[str, Any]:
    report = report_header("status", ip, port)
    readings: list[dict[str, Any]] = []
    for index in range(samples):
        result = call_readonly(
            "GetStatus",
            conn.ptz.GetStatus,
            {"ProfileToken": conn.active_profile_token},
        )
        result["sample"] = index + 1
        result["captured_at"] = datetime.now(UTC).isoformat()
        readings.append(result)
        if index + 1 < samples:
            time.sleep(interval_sec)
    report["active_profile_token"] = conn.active_profile_token
    report["samples"] = readings
    report["summary"] = analyse_status_samples(readings)
    return report


def write_report(
    report: dict[str, Any], output_dir: Path, stem: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = sanitize(report)
    json_path = output_dir / f"{stem}_{timestamp}.json"
    yaml_path = output_dir / f"{stem}_{timestamp}.yaml"
    json_path.write_text(
        json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    yaml_path.write_text(
        yaml.safe_dump(clean, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return json_path, yaml_path


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ip", required=True, help="Camera IP or hostname")
    parser.add_argument("--port", type=int, default=80, help="ONVIF port")
    parser.add_argument("--user", required=True, help="ONVIF username")
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ptz_lab"))
    parser.add_argument(
        "--password-env",
        default="WMB_CAM_PASSWORD",
        help="Environment variable containing the password; prompts if unset",
    )
    return parser


def _password(environment_name: str) -> str:
    password = os.getenv(environment_name)
    if password is not None:
        return password
    return getpass.getpass("ONVIF password: ")


def _execute(
    args: argparse.Namespace,
    runner: Callable[[LabConnection, str, int], dict[str, Any]],
    stem: str,
) -> int:
    password = _password(args.password_env)
    try:
        conn = connect_lab(args.ip, args.port, args.user, password, args.profile_index)
        report = runner(conn, args.ip, args.port)
    except Exception as exc:
        report = report_header(stem, args.ip, args.port)
        report["connection"] = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": _sanitize_text(str(exc)),
        }
        json_path, yaml_path = write_report(report, args.output_dir, stem)
        print(f"Connection failed. Reports: {json_path} {yaml_path}")
        return 2
    json_path, yaml_path = write_report(report, args.output_dir, stem)
    print(f"Read-only probe complete. Reports: {json_path} {yaml_path}")
    return 0


def main_identity(argv: list[str] | None = None) -> int:
    parser = _parser("Read-only ONVIF camera identity probe")
    return _execute(parser.parse_args(argv), run_identity, "ptz_identity")


def main_capabilities(argv: list[str] | None = None) -> int:
    parser = _parser("Read-only ONVIF PTZ and imaging capability probe")
    return _execute(parser.parse_args(argv), run_capabilities, "ptz_capabilities")


def main_status(argv: list[str] | None = None) -> int:
    parser = _parser("Read-only ONVIF PTZ status stability probe")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not 1 <= args.samples <= 100:
        parser.error("--samples must be between 1 and 100")
    if not 0.0 <= args.interval <= 10.0:
        parser.error("--interval must be between 0 and 10 seconds")
    return _execute(
        args,
        lambda conn, ip, port: run_status(
            conn,
            ip,
            port,
            samples=args.samples,
            interval_sec=args.interval,
        ),
        "ptz_status",
    )
