"""
PTZ Probe — Core ONVIF operations.

Pure, isolated, zero side-effects beyond the ONVIF call itself.
No prints, no inputs, no file I/O. All results returned as dataclasses
or plain dicts so the CLI layer can serialise them to YAML/JSON.

This module does NOT import from the surrounding WatchMyBirds tree.
It is meant to be lifted out of agent_handoff/ and dropped anywhere
with `pip install onvif-zeep pyyaml` and still work.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from onvif import ONVIFCamera, ONVIFError
except ImportError as exc:
    raise ImportError(
        "onvif-zeep is required. Install with: pip install onvif-zeep"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    serial_number: str = ""
    hardware_id: str = ""


@dataclass
class ProfileInfo:
    index: int = 0
    name: str = ""
    token: str = ""
    has_ptz: bool = False
    stream_uri: str | None = None


@dataclass
class PresetInfo:
    token: str = ""
    name: str = ""


@dataclass
class NodeCapabilities:
    """What the camera *claims* to support via ONVIF GetNode."""

    node_token: str = ""
    name: str = ""
    supports_continuous_pan_tilt: bool = False
    supports_continuous_zoom: bool = False
    supports_relative_pan_tilt: bool = False
    supports_relative_zoom: bool = False
    supports_absolute_pan_tilt: bool = False
    supports_absolute_zoom: bool = False
    supports_home_position: bool = False
    fixed_home_position: bool = False
    maximum_number_of_presets: int = 0
    pan_tilt_spaces: list[str] = field(default_factory=list)
    zoom_spaces: list[str] = field(default_factory=list)
    raw_xml: str = ""  # pretty-printed for the log file


@dataclass
class StatusSample:
    """One GetStatus reading at a moment in time."""

    pan: float | None = None
    tilt: float | None = None
    zoom: float | None = None
    move_status_pan_tilt: str | None = None
    move_status_zoom: str | None = None
    utc_time: str | None = None
    error: str | None = None


@dataclass
class MoveResult:
    success: bool = False
    command: str = ""
    error: str | None = None
    duration_sec: float = 0.0
    status_before: StatusSample | None = None
    status_after: StatusSample | None = None


@dataclass
class Connection:
    """Live ONVIF handle. Not serialisable on purpose."""

    camera: Any = None
    device_service: Any = None
    media_service: Any = None
    ptz_service: Any = None
    profiles: list[ProfileInfo] = field(default_factory=list)
    active_profile_token: str = ""
    ip: str = ""
    port: int = 80
    username: str = ""


# ---------------------------------------------------------------------------
# WSDL resolution
# ---------------------------------------------------------------------------


def _resolve_wsdl_dir() -> str | None:
    """Locate onvif-zeep's bundled WSDL directory.

    Order: env override, site-packages, sibling assets/. None if not found —
    the ONVIFCamera constructor will then fall back to its internal default.
    """
    candidates: list[Path] = []

    env_wsdl = os.getenv("ONVIF_WSDL_DIR", "").strip()
    if env_wsdl:
        candidates.append(Path(env_wsdl))

    try:
        import onvif as onvif_module

        onvif_file = getattr(onvif_module, "__file__", None)
        if onvif_file:
            candidates.append(Path(onvif_file).resolve().parent.parent / "wsdl")
            candidates.append(Path(onvif_file).resolve().parent / "wsdl")
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    candidates.append(here / "assets" / "onvif_wsdl")

    for cand in candidates:
        try:
            if (cand / "devicemgmt.wsdl").exists():
                return str(cand)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect(
    ip: str,
    port: int,
    username: str,
    password: str,
) -> Connection:
    """Connect to an ONVIF camera and pre-load profiles."""
    wsdl_dir = _resolve_wsdl_dir()
    try:
        if wsdl_dir:
            cam = ONVIFCamera(ip, port, username, password, wsdl_dir=wsdl_dir)
        else:
            cam = ONVIFCamera(ip, port, username, password)
    except ONVIFError as exc:
        raise ConnectionError(f"ONVIF connect to {ip}:{port} failed: {exc}") from exc

    conn = Connection(camera=cam, ip=ip, port=int(port), username=username)
    conn.device_service = cam.devicemgmt

    try:
        conn.media_service = cam.create_media_service()
    except Exception as exc:
        logger.warning("media service unavailable: %s", exc)

    try:
        conn.ptz_service = cam.create_ptz_service()
    except Exception as exc:
        logger.warning("PTZ service unavailable: %s", exc)

    if conn.media_service:
        try:
            raw = conn.media_service.GetProfiles()
            for idx, p in enumerate(raw):
                has_ptz = bool(
                    hasattr(p, "PTZConfiguration") and p.PTZConfiguration
                )
                pi = ProfileInfo(
                    index=idx,
                    name=str(getattr(p, "Name", f"profile_{idx}")),
                    token=str(p.token),
                    has_ptz=has_ptz,
                )
                try:
                    res = conn.media_service.GetStreamUri(
                        {
                            "StreamSetup": {
                                "Stream": "RTP-Unicast",
                                "Transport": {"Protocol": "RTSP"},
                            },
                            "ProfileToken": p.token,
                        }
                    )
                    pi.stream_uri = str(res.Uri)
                except Exception:
                    pass
                conn.profiles.append(pi)
            ptz_profiles = [p for p in conn.profiles if p.has_ptz]
            conn.active_profile_token = (
                ptz_profiles[0].token if ptz_profiles
                else (conn.profiles[0].token if conn.profiles else "")
            )
        except Exception as exc:
            logger.warning("could not load profiles: %s", exc)

    return conn


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------


def get_device_info(conn: Connection) -> DeviceInfo:
    info = DeviceInfo()
    if not conn.device_service:
        return info
    try:
        raw = conn.device_service.GetDeviceInformation()
        info.manufacturer = str(getattr(raw, "Manufacturer", ""))
        info.model = str(getattr(raw, "Model", ""))
        info.firmware_version = str(getattr(raw, "FirmwareVersion", ""))
        info.serial_number = str(getattr(raw, "SerialNumber", ""))
        info.hardware_id = str(getattr(raw, "HardwareId", ""))
    except Exception as exc:
        logger.error("GetDeviceInformation failed: %s", exc)
    return info


def get_snapshot_uri(conn: Connection) -> str | None:
    """Return the cam's ONVIF GetSnapshotUri for the active profile.

    Read-only — does not change cam state. Returns None if the service is
    unavailable or the cam refuses. The URL typically requires HTTP basic
    or digest auth with the same credentials as the ONVIF connection.
    """
    if not conn.media_service or not conn.active_profile_token:
        return None
    try:
        res = conn.media_service.GetSnapshotUri(
            {"ProfileToken": conn.active_profile_token}
        )
        return str(getattr(res, "Uri", "")) or None
    except Exception as exc:
        logger.warning("GetSnapshotUri failed: %s", exc)
        return None


def get_services(conn: Connection) -> list[dict[str, Any]]:
    """ONVIF GetServices — what services does the cam advertise?"""
    if not conn.device_service:
        return []
    try:
        services = conn.device_service.GetServices({"IncludeCapability": False})
    except Exception as exc:
        logger.error("GetServices failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for svc in services or []:
        out.append(
            {
                "namespace": str(getattr(svc, "Namespace", "")),
                "xaddr": str(getattr(svc, "XAddr", "")),
                "version": _format_version(getattr(svc, "Version", None)),
            }
        )
    return out


def _format_version(version: Any) -> str:
    if version is None:
        return ""
    major = getattr(version, "Major", "")
    minor = getattr(version, "Minor", "")
    return f"{major}.{minor}"


# ---------------------------------------------------------------------------
# PTZ capability discovery (the new bit)
# ---------------------------------------------------------------------------


def get_ptz_nodes(conn: Connection) -> list[NodeCapabilities]:
    """Query PTZ nodes and decode their declared capabilities."""
    if not conn.ptz_service:
        return []
    try:
        nodes = conn.ptz_service.GetNodes()
    except Exception as exc:
        logger.error("GetNodes failed: %s", exc)
        return []

    out: list[NodeCapabilities] = []
    for node in nodes or []:
        nc = NodeCapabilities(
            node_token=str(getattr(node, "token", "")),
            name=str(getattr(node, "Name", "")),
        )

        # SupportedPTZSpaces enumerate which coordinate spaces this node accepts.
        spaces = getattr(node, "SupportedPTZSpaces", None)
        if spaces is not None:
            pt_spaces = getattr(spaces, "AbsolutePanTiltPositionSpace", None) or []
            for sp in pt_spaces:
                nc.pan_tilt_spaces.append(f"absolute:{getattr(sp, 'URI', '')}")
                nc.supports_absolute_pan_tilt = True
            pt_rel = getattr(spaces, "RelativePanTiltTranslationSpace", None) or []
            for sp in pt_rel:
                nc.pan_tilt_spaces.append(f"relative:{getattr(sp, 'URI', '')}")
                nc.supports_relative_pan_tilt = True
            pt_cont = getattr(spaces, "ContinuousPanTiltVelocitySpace", None) or []
            for sp in pt_cont:
                nc.pan_tilt_spaces.append(f"continuous:{getattr(sp, 'URI', '')}")
                nc.supports_continuous_pan_tilt = True

            z_abs = getattr(spaces, "AbsoluteZoomPositionSpace", None) or []
            for sp in z_abs:
                nc.zoom_spaces.append(f"absolute:{getattr(sp, 'URI', '')}")
                nc.supports_absolute_zoom = True
            z_rel = getattr(spaces, "RelativeZoomTranslationSpace", None) or []
            for sp in z_rel:
                nc.zoom_spaces.append(f"relative:{getattr(sp, 'URI', '')}")
                nc.supports_relative_zoom = True
            z_cont = getattr(spaces, "ContinuousZoomVelocitySpace", None) or []
            for sp in z_cont:
                nc.zoom_spaces.append(f"continuous:{getattr(sp, 'URI', '')}")
                nc.supports_continuous_zoom = True

        nc.maximum_number_of_presets = int(
            getattr(node, "MaximumNumberOfPresets", 0) or 0
        )
        nc.supports_home_position = bool(
            getattr(node, "HomeSupported", False)
        )
        nc.fixed_home_position = bool(
            getattr(node, "FixedHomePosition", False)
        )

        try:
            nc.raw_xml = str(node)
        except Exception:
            nc.raw_xml = ""
        out.append(nc)

    return out


def get_service_capabilities(conn: Connection) -> dict[str, Any]:
    """ONVIF GetServiceCapabilities on the PTZ service."""
    if not conn.ptz_service:
        return {}
    try:
        caps = conn.ptz_service.GetServiceCapabilities()
    except Exception as exc:
        logger.error("GetServiceCapabilities (PTZ) failed: %s", exc)
        return {"error": str(exc)}
    return {
        "eflip": bool(getattr(caps, "EFlip", False)),
        "reverse": bool(getattr(caps, "Reverse", False)),
        "get_compatible_configurations": bool(
            getattr(caps, "GetCompatibleConfigurations", False)
        ),
        "move_status": bool(getattr(caps, "MoveStatus", False)),
        "status_position": bool(getattr(caps, "StatusPosition", False)),
        "raw": str(caps),
    }


# ---------------------------------------------------------------------------
# Profiles & presets
# ---------------------------------------------------------------------------


def select_profile(conn: Connection, profile_index: int) -> ProfileInfo | None:
    if 0 <= profile_index < len(conn.profiles):
        p = conn.profiles[profile_index]
        conn.active_profile_token = p.token
        return p
    return None


def get_presets(conn: Connection) -> list[PresetInfo]:
    if not conn.ptz_service or not conn.active_profile_token:
        return []
    try:
        raw = conn.ptz_service.GetPresets({"ProfileToken": conn.active_profile_token})
    except Exception as exc:
        logger.error("GetPresets failed: %s", exc)
        return []
    return [
        PresetInfo(token=str(p.token), name=str(getattr(p, "Name", "")))
        for p in raw or []
    ]


def set_preset(
    conn: Connection,
    preset_name: str,
    preset_token: str | None = None,
) -> tuple[MoveResult, str | None]:
    """Save the cam's current position as a preset.

    Returns (MoveResult, returned_token). The token is what the camera
    actually assigned, which on some cheap firmware does NOT match the
    PresetToken we requested in the body. Callers must read back the
    returned_token, NOT trust their own preset_token argument.
    """
    if not conn.ptz_service:
        return (
            MoveResult(success=False, command="SetPreset", error="no PTZ service"),
            None,
        )
    try:
        req = conn.ptz_service.create_type("SetPreset")
        req.ProfileToken = conn.active_profile_token
        req.PresetName = preset_name
        if preset_token:
            req.PresetToken = preset_token
        result = conn.ptz_service.SetPreset(req)

        # The ONVIF response is a SetPresetResponse object (or, on some
        # cams, just a string). Pull PresetToken out properly.
        returned = None
        if result is None:
            returned = preset_token
        elif isinstance(result, str):
            returned = result
        else:
            returned = getattr(result, "PresetToken", None)
            if returned is None:
                # zeep sometimes returns the raw token without wrapping
                returned = str(result)
            else:
                returned = str(returned)

        return (
            MoveResult(
                success=True,
                command=(
                    f"SetPreset(name={preset_name}, "
                    f"requested_token={preset_token!r}, "
                    f"returned_token={returned!r})"
                ),
            ),
            returned,
        )
    except Exception as exc:
        return (
            MoveResult(success=False, command="SetPreset", error=str(exc)),
            None,
        )


def goto_preset(
    conn: Connection,
    preset_token: str,
    settle_sec: float = 2.0,
) -> MoveResult:
    if not conn.ptz_service:
        return MoveResult(
            success=False,
            command=f"GotoPreset({preset_token})",
            error="no PTZ service",
        )
    before = get_status(conn)
    try:
        req = conn.ptz_service.create_type("GotoPreset")
        req.ProfileToken = conn.active_profile_token
        req.PresetToken = preset_token
        t0 = time.monotonic()
        conn.ptz_service.GotoPreset(req)
        time.sleep(max(0.0, settle_sec))
        after = get_status(conn)
        return MoveResult(
            success=True,
            command=f"GotoPreset({preset_token})",
            duration_sec=time.monotonic() - t0,
            status_before=before,
            status_after=after,
        )
    except Exception as exc:
        emergency_stop(conn)
        return MoveResult(
            success=False,
            command=f"GotoPreset({preset_token})",
            error=str(exc),
            status_before=before,
        )


# ---------------------------------------------------------------------------
# Status (GetStatus)
# ---------------------------------------------------------------------------


def get_status(conn: Connection) -> StatusSample:
    """One GetStatus reading. Returns a sample with .error set on failure."""
    s = StatusSample()
    if not conn.ptz_service or not conn.active_profile_token:
        s.error = "no PTZ service or profile"
        return s
    try:
        req = conn.ptz_service.create_type("GetStatus")
        req.ProfileToken = conn.active_profile_token
        st = conn.ptz_service.GetStatus(req)

        pos = getattr(st, "Position", None)
        if pos is not None:
            pt = getattr(pos, "PanTilt", None)
            if pt is not None:
                s.pan = _as_float(getattr(pt, "x", None))
                s.tilt = _as_float(getattr(pt, "y", None))
            z = getattr(pos, "Zoom", None)
            if z is not None:
                s.zoom = _as_float(getattr(z, "x", None))

        move = getattr(st, "MoveStatus", None)
        if move is not None:
            s.move_status_pan_tilt = _as_str(getattr(move, "PanTilt", None))
            s.move_status_zoom = _as_str(getattr(move, "Zoom", None))

        ut = getattr(st, "UtcTime", None)
        if ut is not None:
            s.utc_time = str(ut)
    except Exception as exc:
        s.error = str(exc)
    return s


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


# ---------------------------------------------------------------------------
# Move primitives
# ---------------------------------------------------------------------------


def continuous_move(
    conn: Connection,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    duration_sec: float = 0.3,
) -> MoveResult:
    """ContinuousMove + sleep + Stop. Captures status before & after."""
    label = _move_label("continuous", pan=pan, tilt=tilt, zoom=zoom)
    if not conn.ptz_service:
        return MoveResult(success=False, command=label, error="no PTZ service")
    before = get_status(conn)
    t0 = time.monotonic()
    try:
        req = conn.ptz_service.create_type("ContinuousMove")
        req.ProfileToken = conn.active_profile_token
        req.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        conn.ptz_service.ContinuousMove(req)
        time.sleep(max(0.05, duration_sec))
        stop_req = conn.ptz_service.create_type("Stop")
        stop_req.ProfileToken = conn.active_profile_token
        stop_req.PanTilt = bool(pan != 0.0 or tilt != 0.0)
        stop_req.Zoom = bool(zoom != 0.0)
        conn.ptz_service.Stop(stop_req)
        time.sleep(0.2)  # let the cam settle before reading position
        after = get_status(conn)
        return MoveResult(
            success=True,
            command=label,
            duration_sec=time.monotonic() - t0,
            status_before=before,
            status_after=after,
        )
    except Exception as exc:
        emergency_stop(conn)
        return MoveResult(
            success=False,
            command=label,
            error=str(exc),
            duration_sec=time.monotonic() - t0,
            status_before=before,
        )


def relative_move(
    conn: Connection,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    speed: float = 0.5,
    settle_sec: float = 1.5,
) -> MoveResult:
    """RelativeMove with Translation values + optional Speed hint."""
    label = _move_label("relative", pan=pan, tilt=tilt, zoom=zoom)
    if not conn.ptz_service:
        return MoveResult(success=False, command=label, error="no PTZ service")
    before = get_status(conn)
    t0 = time.monotonic()
    try:
        req = conn.ptz_service.create_type("RelativeMove")
        req.ProfileToken = conn.active_profile_token
        req.Translation = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        req.Speed = {
            "PanTilt": {"x": speed, "y": speed},
            "Zoom": {"x": speed},
        }
        conn.ptz_service.RelativeMove(req)
        time.sleep(max(0.05, settle_sec))
        after = get_status(conn)
        return MoveResult(
            success=True,
            command=label,
            duration_sec=time.monotonic() - t0,
            status_before=before,
            status_after=after,
        )
    except Exception as exc:
        return MoveResult(
            success=False,
            command=label,
            error=str(exc),
            duration_sec=time.monotonic() - t0,
            status_before=before,
        )


def absolute_move(
    conn: Connection,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    speed: float = 0.5,
    settle_sec: float = 2.0,
) -> MoveResult:
    """AbsoluteMove to (pan, tilt, zoom) in the camera's default space."""
    label = _move_label("absolute", pan=pan, tilt=tilt, zoom=zoom)
    if not conn.ptz_service:
        return MoveResult(success=False, command=label, error="no PTZ service")
    before = get_status(conn)
    t0 = time.monotonic()
    try:
        req = conn.ptz_service.create_type("AbsoluteMove")
        req.ProfileToken = conn.active_profile_token
        req.Position = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        req.Speed = {
            "PanTilt": {"x": speed, "y": speed},
            "Zoom": {"x": speed},
        }
        conn.ptz_service.AbsoluteMove(req)
        time.sleep(max(0.05, settle_sec))
        after = get_status(conn)
        return MoveResult(
            success=True,
            command=label,
            duration_sec=time.monotonic() - t0,
            status_before=before,
            status_after=after,
        )
    except Exception as exc:
        return MoveResult(
            success=False,
            command=label,
            error=str(exc),
            duration_sec=time.monotonic() - t0,
            status_before=before,
        )


def emergency_stop(conn: Connection) -> bool:
    """Best-effort Stop on the active profile."""
    if not conn.ptz_service or not conn.active_profile_token:
        return False
    try:
        conn.ptz_service.Stop(
            {
                "ProfileToken": conn.active_profile_token,
                "PanTilt": True,
                "Zoom": True,
            }
        )
        return True
    except Exception as exc:
        logger.warning("emergency_stop failed: %s", exc)
        return False


def _move_label(kind: str, *, pan: float, tilt: float, zoom: float) -> str:
    return f"{kind}(pan={pan:+.2f}, tilt={tilt:+.2f}, zoom={zoom:+.2f})"


# ---------------------------------------------------------------------------
# MoveStatus poll — does the cam actually report MOVING → IDLE?
# ---------------------------------------------------------------------------


def poll_move_status(
    conn: Connection,
    max_wait_sec: float = 5.0,
    poll_interval_sec: float = 0.2,
) -> list[StatusSample]:
    """Sample GetStatus every poll_interval until max_wait_sec.

    Returns the full series so the CLI layer can inspect whether
    MoveStatus actually transitions or is always-IDLE.
    """
    samples: list[StatusSample] = []
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        samples.append(get_status(conn))
        time.sleep(max(0.05, poll_interval_sec))
    return samples
