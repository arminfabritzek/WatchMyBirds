"""
Camera Storage Module.
Manages persistent storage of camera credentials and configuration.
Uses YAML for simple, human-readable storage.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default storage location (in output/ with other runtime data, excluded from sync)
DEFAULT_CAMERAS_FILE = "output/cameras.yaml"


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
            with open(self.storage_path, "w") as f:
                yaml.dump({"cameras": cameras}, f, default_flow_style=False)
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
        logger.info(f"Added camera: {ip}:{port} (ID: {camera_id})")

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

        return self._save_cameras(cameras)

    def get_credentials(self, camera_id: int) -> tuple[str, str]:
        """Get username and password for a camera (internal use only)."""
        cam = self.get_camera(camera_id, include_password=True)
        if cam:
            return cam.get("username", ""), cam.get("password", "")
        return "", ""


# Singleton instance
_storage_instance: CameraStorage | None = None


def get_camera_storage() -> CameraStorage:
    """Get the singleton CameraStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = CameraStorage()
    return _storage_instance
