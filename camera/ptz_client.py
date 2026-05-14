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
