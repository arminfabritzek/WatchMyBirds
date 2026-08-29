"""
Camera Storage Module.
Manages persistent storage of camera credentials and configuration.
Uses YAML for simple, human-readable storage.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

from utils.log_safety import safe_log_value as _slv

logger = logging.getLogger(__name__)

# Default storage location (in output/ with other runtime data, excluded from sync)
DEFAULT_CAMERAS_FILE = "output/cameras.yaml"


DEFAULT_PTZ_CONFIG: dict = {
    "enabled": False,
    "profile_index": 0,
    "overview_preset": "",
    "overview_snapshot_path": "",
    "lost_timeout_sec": 10.0,
    "manual_view_sec": 15.0,
    "settle_max_sec": 8.0,
    "command_cooldown_ms": 800,
    "deadband": 0.04,
    "max_speed": 0.35,
    "move_duration_ms": 250,
    # Detection-driven continuous pan/tilt centres the bbox;
    # continuous zoom-in/out keeps the bbox area near
    # `follow_zoom_target_pct` of the frame.
    # No presets are used for the active tracking — only the overview
    # preset on lost-timeout return. Designed for cams where the probe
    # confirmed continuous_works + continuous_zoom.
    "follow_zoom_target_pct": 0.18,
    "follow_zoom_deadband_pct": 0.05,
    "follow_zoom_speed": 0.3,
    "follow_pan_rate_per_sec": 0.075,
    "follow_tilt_rate_per_sec": 0.15,
    "follow_tilt_max_duration_ms": 500,
    "follow_zoom_duration_ms": 750,
    "follow_lost_hold_sec": 2.0,
    "follow_search_sec": 8.0,
    "follow_search_burst_ms": 400,
    "follow_search_zoom_out_ms": 250,
    # Lens-protection knob for follow-mode. The controller has no
    # absolute zoom feedback on most cheap PTZ cams (GetStatus returns
    # 0.0), so we cap the total *time* spent driving zoom-in since the
    # last return-to-overview. Once the budget is spent, further
    # zoom-in commands are suppressed (zoom-out and pan/tilt still
    # fire). Reset happens every time the controller goes back to
    # the overview preset. 0.0 = disabled (legacy behaviour).
    "follow_zoom_max_burst_sec": 0.0,
    # Manual-joystick burst multipliers (operator-facing, stream-page).
    # Cheap PTZ cams often ignore ContinuousMove velocity and emit one
    # firmware-internal step per call regardless of speed. When pan/tilt
    # steps are too small but zoom steps are too big, we can't equalise
    # by scaling speed — we have to fire pan/tilt N times per heartbeat
    # while keeping zoom at 1. The joystick frontend reads these values
    # and emits that many /ptz/move calls back-to-back per hold-tick.
    # 1.0 = legacy single-call behaviour, the default for fresh installs.
    "manual_pan_tilt_burst": 1,
    "manual_zoom_burst": 1,
    # Manual-joystick / Auto-PTZ move-duration multiplier.
    # Cheap PTZ cams that ignore ContinuousMove velocity AND throttle
    # back-to-back calls (= burst multiplier doesn't help) sometimes
    # honour a single longer ContinuousMove instead. A multiplier of
    # 2.0 extends move_duration_ms from 250 ms → 500 ms per call.
    # Independent from manual_pan_tilt_burst — operator can drive
    # both at the same time, the effective movement per move-command
    # is `burst × (duration × multiplier)`. 1.0 = legacy behaviour.
    "manual_move_duration_multiplier": 1.0,
    "preset_metadata": {},
}


class CameraStorage:
    """Manages CRUD operations for camera configurations."""

    def __init__(self, storage_path: str | None = None):
        """
        Initialize camera storage.

        Args:
            storage_path: Path to cameras.yaml file. Defaults to OUTPUT_DIR/cameras.yaml.
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Use OUTPUT_DIR from config (consistent with settings.yaml location)
            from config import get_config

            output_dir = get_config().get("OUTPUT_DIR", "./output")
            self.storage_path = Path(output_dir) / "cameras.yaml"

        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Ensure storage file and directory exist."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._save_cameras([])

    def _load_cameras(self) -> list[dict]:
        """Load cameras from YAML file."""
        try:
            with open(self.storage_path) as f:
                data = yaml.safe_load(f) or {}
                return data.get("cameras", [])
        except Exception as e:
            logger.error(f"Failed to load cameras: {e}")
            return []

    def _save_cameras(self, cameras: list[dict]) -> bool:
        """Save cameras to YAML file."""
        try:
            # Holds ONVIF credentials in clear text, so the file is owner-only
            # regardless of umask. chmod covers stores written before this.
            with open(self.storage_path, "w") as f:
                yaml.dump({"cameras": cameras}, f, default_flow_style=False)
            Path(self.storage_path).chmod(0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to save cameras: {e}")
            return False

    def list_cameras(self) -> list[dict]:
        """
        Returns all stored cameras.
        Masks passwords in the response.
        """
        cameras = self._load_cameras()
        # Mask passwords for API response
        return [
            {
                "id": i,
                "name": cam.get("name", f"Camera {i + 1}"),
                "ip": cam.get("ip"),
                "port": cam.get("port", 80),
                "username": cam.get("username", ""),
                "has_password": bool(cam.get("password")),
                "supports_onvif": cam.get("supports_onvif", True),
                "manufacturer": cam.get("manufacturer", ""),
                "model": cam.get("model", ""),
                "last_tested": cam.get("last_tested"),
                "last_test_success": cam.get("last_test_success"),
                "ptz": self._public_ptz_config(cam.get("ptz")),
            }
            for i, cam in enumerate(cameras)
        ]

    def get_camera(self, camera_id: int, include_password: bool = False) -> dict | None:
        """Get a single camera by ID."""
        cameras = self._load_cameras()
        if 0 <= camera_id < len(cameras):
            cam = cameras[camera_id].copy()
            cam["id"] = camera_id
            if not include_password:
                cam.pop("password", None)
                cam["has_password"] = bool(cameras[camera_id].get("password"))
            cam["ptz"] = self._merged_ptz_config(cam.get("ptz"))
            return cam
        return None

    def add_camera(
        self,
        ip: str,
        port: int = 80,
        username: str = "",
        password: str = "",
        name: str = "",
        manufacturer: str = "",
        model: str = "",
    ) -> dict:
        """
        Add a new camera.

        Returns:
            The created camera dict with its ID.
        """
        cameras = self._load_cameras()

        # Check for duplicate IP:port
        for existing in cameras:
            if existing.get("ip") == ip and existing.get("port", 80) == port:
                raise ValueError(f"Camera at {ip}:{port} already exists")

        new_camera = {
            "name": name or f"Camera {len(cameras) + 1}",
            "ip": ip,
            "port": port,
            "username": username,
            "password": password,
            "supports_onvif": True,
            "manufacturer": manufacturer,
            "model": model,
            "created_at": datetime.now().isoformat(),
        }

        cameras.append(new_camera)
        self._save_cameras(cameras)

        camera_id = len(cameras) - 1
        logger.info(f"Added camera: {_slv(ip)}:{_slv(port)} (ID: {camera_id})")

        return {
            "id": camera_id,
            "name": new_camera["name"],
            "ip": ip,
            "port": port,
        }

    def update_camera(
        self,
        camera_id: int,
        ip: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        name: str | None = None,
    ) -> bool:
        """Update an existing camera."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False

        cam = cameras[camera_id]
        if ip is not None:
            cam["ip"] = ip
        if port is not None:
            cam["port"] = port
        if username is not None:
            cam["username"] = username
        if password is not None:
            cam["password"] = password
        if name is not None:
            cam["name"] = name

        cam["updated_at"] = datetime.now().isoformat()

        return self._save_cameras(cameras)

    def update_ptz_config(self, camera_id: int, ptz_config: dict) -> bool:
        """Replace the stored auto-PTZ config for a camera."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False

        cameras[camera_id]["ptz"] = self._merged_ptz_config(ptz_config)
        cameras[camera_id]["updated_at"] = datetime.now().isoformat()
        return self._save_cameras(cameras)

    def delete_camera(self, camera_id: int) -> bool:
        """Delete a camera by ID."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False

        deleted = cameras.pop(camera_id)
        logger.info(f"Deleted camera: {deleted.get('ip')}:{deleted.get('port')}")
        return self._save_cameras(cameras)

    def update_test_result(
        self,
        camera_id: int,
        success: bool,
        manufacturer: str = "",
        model: str = "",
        has_ptz: bool | None = None,
    ) -> bool:
        """Update test result for a camera."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False

        cam = cameras[camera_id]
        cam["last_tested"] = datetime.now().isoformat()
        cam["last_test_success"] = success
        if manufacturer:
            cam["manufacturer"] = manufacturer
        if model:
            cam["model"] = model
        if has_ptz is not None:
            cam["has_ptz"] = bool(has_ptz)

        return self._save_cameras(cameras)

    def update_overview_snapshot_path(self, camera_id: int, relative_path: str) -> bool:
        """Persist the relative path of the PTZ overview snapshot."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False
        cam = cameras[camera_id]
        ptz = dict(cam.get("ptz") or {})
        ptz["overview_snapshot_path"] = str(relative_path or "")
        cam["ptz"] = ptz
        return self._save_cameras(cameras)

    def update_preset_metadata(
        self, camera_id: int, preset_token: str, metadata: dict
    ) -> bool:
        """Persist per-preset metadata (click position, box, label)."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False
        cam = cameras[camera_id]
        ptz = dict(cam.get("ptz") or {})
        bucket = dict(ptz.get("preset_metadata") or {})
        bucket[str(preset_token)] = {str(k): v for k, v in (metadata or {}).items()}
        ptz["preset_metadata"] = bucket
        cam["ptz"] = ptz
        return self._save_cameras(cameras)

    def delete_preset_metadata(self, camera_id: int, preset_token: str) -> bool:
        """Remove per-preset metadata; safe no-op if the entry is absent."""
        cameras = self._load_cameras()
        if camera_id < 0 or camera_id >= len(cameras):
            return False
        cam = cameras[camera_id]
        ptz = dict(cam.get("ptz") or {})
        bucket = dict(ptz.get("preset_metadata") or {})
        if str(preset_token) in bucket:
            del bucket[str(preset_token)]
            ptz["preset_metadata"] = bucket
            cam["ptz"] = ptz
            return self._save_cameras(cameras)
        return True

    def get_credentials(self, camera_id: int) -> tuple[str, str]:
        """Get stored credentials for a camera."""
        cam = self.get_camera(camera_id, include_password=True)
        if cam:
            return cam.get("username", ""), cam.get("password", "")
        return "", ""

    def _merged_ptz_config(self, ptz_config: dict | None) -> dict:
        """Return the supported PTZ config and discard retired mode fields."""
        config = DEFAULT_PTZ_CONFIG.copy()
        config["preset_metadata"] = {}

        if not isinstance(ptz_config, dict):
            return config

        for key in config:
            if key in ptz_config:
                config[key] = ptz_config[key]

        return config

    def _public_ptz_config(self, ptz_config: dict | None) -> dict:
        config = self._merged_ptz_config(ptz_config)
        return {
            "enabled": bool(config.get("enabled")),
            "overview_preset": config.get("overview_preset", ""),
        }


# Singleton instance
_storage_instance: CameraStorage | None = None


def get_camera_storage() -> CameraStorage:
    """Get the singleton CameraStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = CameraStorage()
    return _storage_instance
