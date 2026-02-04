"""
ONVIF Core - ONVIF Camera Operations.

Provides ONVIF camera discovery and stream URI retrieval.
"""

import logging
from typing import Any

from camera.network_scanner import NetworkScanner
from utils.camera_storage import get_camera_storage

logger = logging.getLogger(__name__)


def discover_cameras(fast: bool = False) -> list[dict[str, Any]]:
    """
    Scans the network for ONVIF cameras.

    Args:
        fast: If True, skips aggressive subnet scan (WS-Discovery only)

    Returns:
        List of discovered camera dictionaries
    """
    try:
        scanner = NetworkScanner()
        cameras = scanner.scan(fast=fast)
        return cameras
    except Exception as e:
        logger.error(f"ONVIF discovery failed: {e}")
        raise


def get_stream_uri(
    camera_ip: str, port: int, username: str, password: str, profile_index: int = 0
) -> str | None:
    """
    Retrieves the RTSP stream URI for a camera with credentials.

    Args:
        camera_ip: Camera IP address
        port: ONVIF port (typically 80 or 8080)
        username: ONVIF username
        password: ONVIF password
        profile_index: Media profile index (default 0)

    Returns:
        RTSP stream URI or None on failure
    """
    try:
        scanner = NetworkScanner()
        uri = scanner.get_stream_uri(camera_ip, port, username, password, profile_index)
        return uri
    except Exception as e:
        logger.error(f"Failed to get stream URI for {camera_ip}:{port}: {e}")
        return None


def get_saved_cameras() -> list[dict[str, Any]]:
    """
    Retrieves all saved cameras from storage.

    Returns:
        List of camera configurations
    """
    storage = get_camera_storage()
    return storage.list_cameras()


def save_camera(
    ip: str,
    port: int = 80,
    username: str = "",
    password: str = "",
    name: str = "",
    manufacturer: str = "",
    model: str = "",
) -> dict[str, Any]:
    """
    Saves a camera configuration to storage.

    Args:
        ip: Camera IP address
        port: Camera port (default 80)
        username: Optional username
        password: Optional password
        name: Optional display name
        manufacturer: Optional manufacturer info
        model: Optional model info

    Returns:
        The saved camera data with assigned ID
    """
    storage = get_camera_storage()
    return storage.add_camera(
        ip=ip,
        port=port,
        username=username,
        password=password,
        name=name,
        manufacturer=manufacturer,
        model=model,
    )


def update_camera(
    camera_id: int,
    ip: str | None = None,
    port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    name: str | None = None,
) -> bool:
    """
    Updates an existing camera configuration.

    Args:
        camera_id: ID of the camera to update
        ip: Optional new IP
        port: Optional new port
        username: Optional new username
        password: Optional new password
        name: Optional new name

    Returns:
        True if successful, False if camera not found
    """
    storage = get_camera_storage()
    return storage.update_camera(
        camera_id=camera_id,
        ip=ip,
        port=port,
        username=username,
        password=password,
        name=name,
    )


def delete_camera(camera_id: int) -> bool:
    """
    Deletes a camera configuration.

    Args:
        camera_id: ID of the camera to delete

    Returns:
        True if deleted, False if not found
    """
    storage = get_camera_storage()
    return storage.delete_camera(camera_id)


def test_camera_connection(stream_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """
    Tests a camera connection by attempting to open the stream.

    Args:
        stream_url: RTSP or HTTP stream URL
        timeout: Connection timeout in seconds

    Returns:
        Dictionary with test results (success, message, etc.)
    """
    import cv2

    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))

        if not cap.isOpened():
            return {"success": False, "message": "Could not open stream"}

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"success": False, "message": "Could not read frame"}

        return {
            "success": True,
            "message": "Connection successful",
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_device_info(
    ip: str, port: int, username: str, password: str
) -> dict[str, Any] | None:
    """
    Gets device information from an ONVIF camera.

    Args:
        ip: Camera IP address
        port: ONVIF port
        username: ONVIF username
        password: ONVIF password

    Returns:
        Dictionary with device info or None on failure
    """
    try:
        scanner = NetworkScanner()
        return scanner.get_device_info(ip, port, username, password)
    except Exception as e:
        logger.error(f"Failed to get device info for {ip}:{port}: {e}")
        return None


def get_camera(camera_id: int, include_password: bool = False) -> dict[str, Any] | None:
    """
    Gets a single camera by ID.

    Args:
        camera_id: Camera ID
        include_password: Whether to include password in response

    Returns:
        Camera dict or None if not found
    """
    storage = get_camera_storage()
    return storage.get_camera(camera_id, include_password=include_password)


def update_test_result(
    camera_id: int, success: bool, manufacturer: str = "", model: str = ""
) -> bool:
    """
    Updates the test result for a camera.

    Args:
        camera_id: Camera ID
        success: Whether test was successful
        manufacturer: Manufacturer info from test
        model: Model info from test

    Returns:
        True if successful
    """
    storage = get_camera_storage()
    return storage.update_test_result(camera_id, success, manufacturer, model)


def get_camera_credentials(camera_id: int) -> tuple[str, str]:
    """
    Gets credentials for a camera.

    Args:
        camera_id: Camera ID

    Returns:
        Tuple of (username, password)
    """
    storage = get_camera_storage()
    return storage.get_credentials(camera_id)
