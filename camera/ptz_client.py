"""
ONVIF PTZ client helpers.

This module owns the low-level camera protocol calls. Higher layers should use
``core.ptz_core`` instead of importing this module directly from web code.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onvif import ONVIFCamera
from zeep.cache import InMemoryCache
from zeep.transports import Transport

from utils.log_safety import safe_log_value as _slv

logger = logging.getLogger(__name__)

# See camera/network_scanner.py for the rationale: zeep's default
# SqliteCache fails on hardened containers where /tmp/<parent> is
# root-owned. InMemoryCache sidesteps the filesystem entirely.
_ZEEP_TRANSPORT = Transport(cache=InMemoryCache())


@dataclass(frozen=True)
class PtzPreset:
    token: str
    name: str


@dataclass(frozen=True)
class StatusSample:
    """One ``GetStatus`` reading from the camera.

    Fields are best-effort: cams that don't expose position or move-status
    return None for those. Used by the in-UI empirical probe wizard to
    diagnose MoveStatus-stub firmware (cam reports IDLE during active moves).
    """

    pan: float | None
    tilt: float | None
    zoom: float | None
    move_status_pan_tilt: str | None  # "IDLE" | "MOVING" | None
    move_status_zoom: str | None
    utc_time: str | None
    error: str | None = None


@dataclass(frozen=True)
class DeviceInfo:
    """``GetDeviceInformation`` snapshot. Empty strings on missing fields."""

    manufacturer: str
    model: str
    firmware_version: str
    serial_number: str
    hardware_id: str


def _resolve_onvif_wsdl_dir() -> str | None:
    candidates: list[Path] = []

    env_wsdl = os.getenv("ONVIF_WSDL_DIR", "").strip()
    if env_wsdl:
        candidates.append(Path(env_wsdl))

    try:
        import onvif as onvif_module

        candidates.append(Path(onvif_module.__file__).resolve().parent.parent / "wsdl")
    except (ImportError, AttributeError, OSError):
        pass

    project_root = Path(__file__).resolve().parents[1]
    candidates.append(project_root / "assets" / "onvif_wsdl")

    for candidate in candidates:
        try:
            if (candidate / "ptz.wsdl").exists():
                return str(candidate)
        except Exception:
            continue
    return None


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PtzClient:
    """Small ONVIF PTZ adapter for one camera/profile."""

    def __init__(
        self,
        ip: str,
        port: int,
        username: str,
        password: str,
        profile_index: int = 0,
    ) -> None:
        self.ip = ip
        self.port = int(port or 80)
        self.username = username or ""
        self.password = password or ""
        self.profile_index = max(0, int(profile_index or 0))
        self._camera: ONVIFCamera | None = None
        self._media: Any | None = None
        self._ptz: Any | None = None
        self._profile_token: str | None = None

    def _create_camera(self) -> ONVIFCamera:
        wsdl_dir = _resolve_onvif_wsdl_dir()
        if wsdl_dir:
            return ONVIFCamera(
                self.ip,
                self.port,
                self.username,
                self.password,
                wsdl_dir=wsdl_dir,
                transport=_ZEEP_TRANSPORT,
            )
        return ONVIFCamera(
            self.ip,
            self.port,
            self.username,
            self.password,
            transport=_ZEEP_TRANSPORT,
        )

    def _ensure_services(self) -> tuple[Any, str]:
        if self._ptz is not None and self._profile_token:
            return self._ptz, self._profile_token

        logger.debug("Connecting PTZ client for %s:%s", _slv(self.ip), _slv(self.port))
        self._camera = self._create_camera()
        self._media = self._camera.create_media_service()
        self._ptz = self._camera.create_ptz_service()

        profiles = self._media.GetProfiles()
        if not profiles:
            raise RuntimeError("Camera returned no media profiles")

        index = min(self.profile_index, len(profiles) - 1)
        profile = profiles[index]
        token = _get_value(profile, "token")
        if not token:
            raise RuntimeError("Selected media profile has no token")

        self._profile_token = str(token)
        return self._ptz, self._profile_token

    def list_presets(self) -> list[PtzPreset]:
        ptz, profile_token = self._ensure_services()
        request = ptz.create_type("GetPresets")
        request.ProfileToken = profile_token
        raw_presets = ptz.GetPresets(request) or []

        presets: list[PtzPreset] = []
        for preset in raw_presets:
            token = _get_value(preset, "token") or _get_value(preset, "PresetToken")
            name = _get_value(preset, "Name") or token
            if token:
                presets.append(PtzPreset(token=str(token), name=str(name or token)))
        return presets

    def set_preset(self, name: str, preset_token: str | None = None) -> str:
        """Create or overwrite a preset at the current camera position.

        Returns the preset token assigned by the camera. If preset_token
        is given, the camera updates that slot in place (most cameras
        accept this; some create a new slot).
        """
        ptz, profile_token = self._ensure_services()
        request = ptz.create_type("SetPreset")
        request.ProfileToken = profile_token
        request.PresetName = str(name or "")
        if preset_token:
            request.PresetToken = str(preset_token)
        response = ptz.SetPreset(request)
        returned = _get_value(response, "PresetToken")
        if returned:
            return str(returned)
        return str(preset_token or name)

    def remove_preset(self, preset_token: str) -> None:
        if not preset_token:
            raise ValueError("preset_token is required")
        ptz, profile_token = self._ensure_services()
        request = ptz.create_type("RemovePreset")
        request.ProfileToken = profile_token
        request.PresetToken = str(preset_token)
        ptz.RemovePreset(request)

    def set_home_position(self) -> bool:
        """Mark the current PTZ position as the camera's ONVIF home.

        Returns True on success, False if the camera/firmware refuses
        SetHomePosition. Caller decides whether to treat failure as fatal.
        """
        ptz, profile_token = self._ensure_services()
        try:
            request = ptz.create_type("SetHomePosition")
            request.ProfileToken = profile_token
            ptz.SetHomePosition(request)
            return True
        except Exception as exc:
            logger.warning("SetHomePosition refused by camera: %s", exc)
            return False

    def goto_preset(self, preset_token: str, speed: float | None = None) -> None:
        if not preset_token:
            raise ValueError("preset_token is required")

        ptz, profile_token = self._ensure_services()
        request = ptz.create_type("GotoPreset")
        request.ProfileToken = profile_token
        request.PresetToken = str(preset_token)

        if speed is not None:
            value = max(0.0, min(1.0, float(speed)))
            request.Speed = {
                "PanTilt": {"x": value, "y": value},
                "Zoom": {"x": value},
            }

        ptz.GotoPreset(request)

    def continuous_move(
        self,
        *,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
        duration_ms: int = 250,
    ) -> None:
        ptz, profile_token = self._ensure_services()
        pan = max(-1.0, min(1.0, float(pan)))
        tilt = max(-1.0, min(1.0, float(tilt)))
        zoom = max(-1.0, min(1.0, float(zoom)))
        duration_sec = max(0.05, min(2.0, int(duration_ms or 250) / 1000.0))

        request = ptz.create_type("ContinuousMove")
        request.ProfileToken = profile_token
        request.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        ptz.ContinuousMove(request)
        time.sleep(duration_sec)
        self.stop(pan_tilt=True, zoom=abs(zoom) > 0.001)

    def wait_until_idle(
        self, *, max_wait_sec: float = 8.0, poll_interval_sec: float = 0.5
    ) -> bool:
        """Poll ONVIF GetStatus until PTZ MoveStatus reports IDLE.

        Returns True when both PanTilt and Zoom report IDLE within the
        budget. Returns False on timeout, error, or when the camera
        does not expose MoveStatus at all. Caller is expected to apply
        a fixed settle fallback when False is returned.
        """
        ptz, profile_token = self._ensure_services()
        deadline = time.monotonic() + max(0.5, float(max_wait_sec))
        while time.monotonic() < deadline:
            try:
                request = ptz.create_type("GetStatus")
                request.ProfileToken = profile_token
                status = ptz.GetStatus(request)
                move = _get_value(status, "MoveStatus")
                if move is None:
                    return False  # camera does not report MoveStatus
                pan_tilt = _get_value(move, "PanTilt")
                zoom = _get_value(move, "Zoom")
                pt_idle = pan_tilt is None or str(pan_tilt).upper() == "IDLE"
                zm_idle = zoom is None or str(zoom).upper() == "IDLE"
                if pt_idle and zm_idle:
                    return True
            except Exception as exc:
                logger.debug("GetStatus during wait_until_idle failed: %s", exc)
                return False
            time.sleep(max(0.1, float(poll_interval_sec)))
        return False

    def stop(self, *, pan_tilt: bool = True, zoom: bool = True) -> None:
        ptz, profile_token = self._ensure_services()
        request = ptz.create_type("Stop")
        request.ProfileToken = profile_token
        request.PanTilt = bool(pan_tilt)
        request.Zoom = bool(zoom)
        ptz.Stop(request)

    def emergency_stop(self) -> None:
        """Stop both pan/tilt and zoom immediately, logged at WARNING.

        Documented contract: must be safe to call from signal handlers
        (Ctrl-C in CLI tool) and from web error-paths (wizard abort).
        Wraps the standard stop() with extra logging — failure is
        swallowed (best-effort) because the alternative is leaving the
        camera spinning into an endstop.
        """
        try:
            self.stop(pan_tilt=True, zoom=True)
            logger.warning("Emergency stop fired for camera %s", _slv(self.ip))
        except Exception as exc:  # noqa: BLE001 — must not raise from emergency path
            logger.error(
                "Emergency stop FAILED for camera %s: %s", _slv(self.ip), exc
            )

    def relative_move(
        self,
        *,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
        speed: float | None = None,
    ) -> None:
        """Issue a single ONVIF ``RelativeMove`` with the given translation.

        Inputs are clamped to ``[-1, 1]`` per the generic translation space.
        No auto-stop — Relative is a one-shot translation by ONVIF spec;
        the cam decides when it has arrived. Cheap cams sometimes interpret
        Translation as velocity (the empirical probe detects exactly this).

        ``speed`` is optional. When set, clamped to ``[0, 1]`` and passed
        as PanTilt + Zoom speed (same scalar for both axes — most cheap
        cams collapse to one anyway).
        """
        ptz, profile_token = self._ensure_services()
        pan = max(-1.0, min(1.0, float(pan)))
        tilt = max(-1.0, min(1.0, float(tilt)))
        zoom = max(-1.0, min(1.0, float(zoom)))

        request = ptz.create_type("RelativeMove")
        request.ProfileToken = profile_token
        request.Translation = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        if speed is not None:
            speed_clamped = max(0.0, min(1.0, float(speed)))
            request.Speed = {
                "PanTilt": {"x": speed_clamped, "y": speed_clamped},
                "Zoom": {"x": speed_clamped},
            }
        ptz.RelativeMove(request)

    def absolute_move(
        self,
        *,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
        speed: float | None = None,
    ) -> None:
        """Issue a single ONVIF ``AbsoluteMove`` to the given position.

        Inputs clamped to ``[-1, 1]`` per the generic position space.
        WARNING: on cheap firmware, ``AbsoluteMove(0, 0, 0)`` sometimes
        drives to an endstop rather than centre. The empirical probe
        wizard surfaces this; production code should prefer
        ``goto_preset`` (a stored absolute position) over raw absolute
        moves unless the cam has been probed and confirmed compliant.
        """
        ptz, profile_token = self._ensure_services()
        pan = max(-1.0, min(1.0, float(pan)))
        tilt = max(-1.0, min(1.0, float(tilt)))
        zoom = max(-1.0, min(1.0, float(zoom)))

        request = ptz.create_type("AbsoluteMove")
        request.ProfileToken = profile_token
        request.Position = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        if speed is not None:
            speed_clamped = max(0.0, min(1.0, float(speed)))
            request.Speed = {
                "PanTilt": {"x": speed_clamped, "y": speed_clamped},
                "Zoom": {"x": speed_clamped},
            }
        ptz.AbsoluteMove(request)

    def get_status(self) -> StatusSample:
        """Return a single ``GetStatus`` reading as a structured sample.

        Unlike ``wait_until_idle`` (which reads status internally and
        discards the sample), this returns position + move-status + UTC
        time so callers can record the snapshot. Errors are captured in
        the ``error`` field rather than raised — the empirical probe
        wants partial data, not exceptions.
        """
        try:
            ptz, profile_token = self._ensure_services()
            request = ptz.create_type("GetStatus")
            request.ProfileToken = profile_token
            status = ptz.GetStatus(request)
        except Exception as exc:  # noqa: BLE001 — defensive against malformed firmware
            return StatusSample(
                pan=None, tilt=None, zoom=None,
                move_status_pan_tilt=None, move_status_zoom=None,
                utc_time=None, error=str(exc),
            )

        position = _get_value(status, "Position")
        pan_tilt = _get_value(position, "PanTilt") if position else None
        zoom_pos = _get_value(position, "Zoom") if position else None

        def _as_float(obj: Any, attr: str) -> float | None:
            if obj is None:
                return None
            val = _get_value(obj, attr)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        move = _get_value(status, "MoveStatus")
        move_pt = _get_value(move, "PanTilt") if move else None
        move_zoom = _get_value(move, "Zoom") if move else None

        return StatusSample(
            pan=_as_float(pan_tilt, "x"),
            tilt=_as_float(pan_tilt, "y"),
            zoom=_as_float(zoom_pos, "x"),
            move_status_pan_tilt=str(move_pt) if move_pt is not None else None,
            move_status_zoom=str(move_zoom) if move_zoom is not None else None,
            utc_time=str(_get_value(status, "UtcTime") or "") or None,
            error=None,
        )

    def poll_move_status(
        self,
        *,
        duration_sec: float,
        interval_sec: float = 0.2,
    ) -> list[StatusSample]:
        """Repeatedly call ``GetStatus`` for the given duration.

        Returns the sequence of samples. Each sample is a complete
        :class:`StatusSample` (including errors). The empirical probe
        uses this to detect MoveStatus-stub firmware — cams that always
        report ``IDLE`` even during an active move.

        ``interval_sec`` is the gap between samples (the call itself
        takes ~10-50ms per round-trip).
        """
        samples: list[StatusSample] = []
        if duration_sec <= 0:
            return samples
        interval = max(0.05, float(interval_sec))
        deadline = time.monotonic() + float(duration_sec)
        while time.monotonic() < deadline:
            samples.append(self.get_status())
            time.sleep(interval)
        return samples

    def get_device_info(self) -> DeviceInfo:
        """Return the cam's identity (manufacturer/model/firmware/serial).

        Empty strings on missing fields. Used by the empirical probe to
        record what hardware was tested, so a future operator looking at
        the cache file knows whether the result applies to their cam.
        """
        # GetDeviceInformation lives on the device service, not the PTZ
        # service. Build it lazily and only here, since most callers
        # don't need device info.
        try:
            self._ensure_services()  # ensures self._camera is built
            assert self._camera is not None
            device = self._camera.create_devicemgmt_service()
            info = device.GetDeviceInformation()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GetDeviceInformation failed: %s", _slv(str(exc)))
            return DeviceInfo("", "", "", "", "")
        return DeviceInfo(
            manufacturer=str(_get_value(info, "Manufacturer") or ""),
            model=str(_get_value(info, "Model") or ""),
            firmware_version=str(_get_value(info, "FirmwareVersion") or ""),
            serial_number=str(_get_value(info, "SerialNumber") or ""),
            hardware_id=str(_get_value(info, "HardwareId") or ""),
        )

    def get_snapshot_uri(self) -> str:
        """Return the ONVIF snapshot HTTP URL for the active profile."""
        _ptz, profile_token = self._ensure_services()
        assert self._media is not None
        request = self._media.create_type("GetSnapshotUri")
        request.ProfileToken = profile_token
        response = self._media.GetSnapshotUri(request)
        uri = _get_value(response, "Uri")
        if not uri:
            raise RuntimeError("Camera did not return a snapshot URI")
        return str(uri)

    def get_capabilities(self) -> dict[str, Any]:
        """Probe the camera's declared PTZ capabilities via ONVIF.

        Returns a plain dict matching the shape that
        ``agent_handoff/lab/experiments/ptz_probe/ptz_probe_core.py``
        produces for the equivalent operations — declared support
        for continuous / relative / absolute pan-tilt and zoom plus
        the maximum preset count. This is a READ-ONLY ONVIF probe
        (``GetServiceCapabilities`` + ``GetNodes``); it issues no
        move commands and is safe to call against any reachable cam.

        Failure mode: if a sub-call raises (malformed firmware, lost
        network), the affected sub-dict is None and ``error`` carries
        the exception text. The caller must tolerate partial dicts.
        """
        ptz, _profile_token = self._ensure_services()
        result: dict[str, Any] = {
            "service_capabilities": None,
            "nodes": [],
            "declared": {
                "continuous_pan_tilt": False,
                "continuous_zoom": False,
                "relative_pan_tilt": False,
                "relative_zoom": False,
                "absolute_pan_tilt": False,
                "absolute_zoom": False,
                "home_position": False,
                "move_status": False,
                "status_position": False,
                "max_presets": 0,
            },
            "error": None,
        }

        # GetServiceCapabilities — coarse flags (MoveStatus, StatusPosition, EFlip, Reverse).
        try:
            caps = ptz.GetServiceCapabilities()
            result["service_capabilities"] = {
                "move_status": bool(_get_value(caps, "MoveStatus", False)),
                "status_position": bool(_get_value(caps, "StatusPosition", False)),
                "eflip": bool(_get_value(caps, "EFlip", False)),
                "reverse": bool(_get_value(caps, "Reverse", False)),
            }
            result["declared"]["move_status"] = result["service_capabilities"]["move_status"]
            result["declared"]["status_position"] = result["service_capabilities"][
                "status_position"
            ]
        except Exception as exc:  # noqa: BLE001 — defensive: vendor firmwares vary wildly
            logger.warning("GetServiceCapabilities failed: %s", _slv(str(exc)))
            result["error"] = f"service_capabilities: {exc}"

        # GetNodes — per-node declared support spaces (this is the authoritative source).
        try:
            raw_nodes = ptz.GetNodes() or []
            nodes_summary: list[dict[str, Any]] = []
            for node in raw_nodes:
                spaces = _get_value(node, "SupportedPTZSpaces") or {}
                abs_pt = _get_value(spaces, "AbsolutePanTiltPositionSpace") or []
                abs_zoom = _get_value(spaces, "AbsoluteZoomPositionSpace") or []
                rel_pt = _get_value(spaces, "RelativePanTiltTranslationSpace") or []
                rel_zoom = _get_value(spaces, "RelativeZoomTranslationSpace") or []
                cont_pt = _get_value(spaces, "ContinuousPanTiltVelocitySpace") or []
                cont_zoom = _get_value(spaces, "ContinuousZoomVelocitySpace") or []
                max_presets = int(_get_value(node, "MaximumNumberOfPresets", 0) or 0)
                home_supported = bool(_get_value(node, "HomeSupported", False))
                token = _get_value(node, "token") or _get_value(node, "Name")

                nodes_summary.append(
                    {
                        "token": str(token) if token else "",
                        "name": str(_get_value(node, "Name") or ""),
                        "max_presets": max_presets,
                        "home_supported": home_supported,
                        "continuous_pan_tilt": bool(cont_pt),
                        "continuous_zoom": bool(cont_zoom),
                        "relative_pan_tilt": bool(rel_pt),
                        "relative_zoom": bool(rel_zoom),
                        "absolute_pan_tilt": bool(abs_pt),
                        "absolute_zoom": bool(abs_zoom),
                    }
                )

                # Union across nodes — if any node declares a capability, surface it.
                d = result["declared"]
                d["continuous_pan_tilt"] = d["continuous_pan_tilt"] or bool(cont_pt)
                d["continuous_zoom"] = d["continuous_zoom"] or bool(cont_zoom)
                d["relative_pan_tilt"] = d["relative_pan_tilt"] or bool(rel_pt)
                d["relative_zoom"] = d["relative_zoom"] or bool(rel_zoom)
                d["absolute_pan_tilt"] = d["absolute_pan_tilt"] or bool(abs_pt)
                d["absolute_zoom"] = d["absolute_zoom"] or bool(abs_zoom)
                d["home_position"] = d["home_position"] or home_supported
                if max_presets > d["max_presets"]:
                    d["max_presets"] = max_presets

            result["nodes"] = nodes_summary
        except Exception as exc:  # noqa: BLE001 — defensive: GetNodes is optional on some firmwares
            logger.warning("GetNodes failed: %s", _slv(str(exc)))
            prior = result.get("error")
            result["error"] = f"{prior}; nodes: {exc}" if prior else f"nodes: {exc}"

        return result
