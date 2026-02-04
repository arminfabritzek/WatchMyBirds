"""
ONVIF Service - Web Layer Service for ONVIF Camera Operations.

Thin wrapper over core.onvif_core for web-specific concerns.
"""

from typing import Any

from core import onvif_core


def discover_cameras(fast: bool = False) -> list[dict[str, Any]]:
    """
    Discover ONVIF cameras on the network.

    Args:
        fast: If True, skips aggressive subnet scan

    Delegates to core.onvif_core.
    """
    return onvif_core.discover_cameras(fast=fast)


def get_stream_uri(
    camera_ip: str, port: int, username: str, password: str, profile_index: int = 0
) -> str | None:
    """
    Get RTSP stream URI for a camera.

    Delegates to core.onvif_core.
    """
    return onvif_core.get_stream_uri(camera_ip, port, username, password, profile_index)


def get_saved_cameras() -> list[dict[str, Any]]:
    """
    Get all saved camera configurations.

    Delegates to core.onvif_core.
    """
    return onvif_core.get_saved_cameras()


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
    Save a new camera configuration.

    Delegates to core.onvif_core.
    """
    return onvif_core.save_camera(
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
    Update an existing camera configuration.

    Delegates to core.onvif_core.
    """
    return onvif_core.update_camera(
        camera_id=camera_id,
        ip=ip,
        port=port,
        username=username,
        password=password,
        name=name,
    )


def delete_camera(camera_id: int) -> bool:
    """
    Delete a camera configuration.

    Delegates to core.onvif_core.
    """
    return onvif_core.delete_camera(camera_id)


def test_camera_connection(stream_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """
    Test a camera connection.

    Delegates to core.onvif_core.
    """
    return onvif_core.test_camera_connection(stream_url, timeout)


def get_device_info(
    ip: str, port: int, username: str, password: str
) -> dict[str, Any] | None:
    """
    Get ONVIF device information.

    Delegates to core.onvif_core.
    """
    return onvif_core.get_device_info(ip, port, username, password)


def get_camera(camera_id: int, include_password: bool = False) -> dict[str, Any] | None:
    """
    Get a single camera by ID.

    Delegates to core.onvif_core.
    """
    return onvif_core.get_camera(camera_id, include_password=include_password)


def update_test_result(
    camera_id: int, success: bool, manufacturer: str = "", model: str = ""
) -> bool:
    """
    Update test result for a camera.

    Delegates to core.onvif_core.
    """
    return onvif_core.update_test_result(camera_id, success, manufacturer, model)


def test_camera(camera_id: int) -> bool:
    """
    Test a camera by ID.

    Gets the camera, retrieves its stream URI, and tests the connection.
    Returns True if connection successful, False otherwise.

    Tries in order:
    1. Saved stream_url (if exists)
    2. ONVIF-retrieved stream URI
    3. Standard RTSP URL pattern
    """
    camera = get_camera(camera_id, include_password=True)
    if not camera:
        raise ValueError(f"Camera {camera_id} not found")

    ip = camera.get("ip", "")
    port = camera.get("port", 80)
    username = camera.get("username", "")
    password = camera.get("password", "")

    stream_uri = None

    # 1. Try saved stream_url first
    if camera.get("stream_url"):
        stream_uri = camera.get("stream_url")

    # 2. Try ONVIF
    if not stream_uri:
        stream_uri = get_stream_uri(
            camera_ip=ip,
            port=port,
            username=username,
            password=password,
        )

    # 3. Fallback: Build standard RTSP URL
    if not stream_uri and ip:
        # Common RTSP patterns
        auth = f"{username}:{password}@" if username else ""
        stream_uri = f"rtsp://{auth}{ip}:554/stream1"

    if not stream_uri:
        return False

    # Test the connection
    result = test_camera_connection(stream_uri, timeout=5.0)
    success = result.get("success", False)

    # Update test result in storage
    update_test_result(
        camera_id=camera_id,
        success=success,
        manufacturer=result.get("manufacturer", ""),
        model=result.get("model", ""),
    )

    return success


def get_camera_uri(camera_id: int) -> str | None:
    """
    Get the stream URI for a saved camera.

    Returns the RTSP stream URI or None if not available.
    """
    camera = get_camera(camera_id, include_password=True)
    if not camera:
        return None

    return get_stream_uri(
        camera_ip=camera.get("ip", ""),
        port=camera.get("port", 80),
        username=camera.get("username", ""),
        password=camera.get("password", ""),
    )
