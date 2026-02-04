"""
Settings Core - Settings Management.

Provides settings read/write operations separated from the web layer.
"""

import logging
from typing import Any

from config import (
    get_config,
    get_settings_payload,
    update_runtime_settings,
    validate_runtime_updates,
)

logger = logging.getLogger(__name__)


def get_current_settings() -> dict[str, Any]:
    """
    Returns the current settings as a payload for the UI.

    Returns:
        Dictionary with all UI-relevant settings
    """
    return get_settings_payload()


def update_settings(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Updates runtime settings with the provided payload.

    Args:
        payload: Dictionary of settings to update

    Returns:
        Tuple of (success, list of error messages)
    """
    if not isinstance(payload, dict):
        return False, ["Invalid payload format"]

    valid, errors = validate_runtime_updates(payload)

    if errors:
        return False, errors

    update_runtime_settings(valid)
    return True, []


def get_setting(key: str, default: Any = None) -> Any:
    """
    Gets a single setting value.

    Args:
        key: Setting key
        default: Default value if key not found

    Returns:
        Setting value or default
    """
    config = get_config()
    return config.get(key, default)


def get_save_threshold() -> float:
    """
    Returns the configured save threshold.

    Returns:
        Save threshold value
    """
    config = get_config()
    return config.get("SAVE_THRESHOLD", 0.65)


def get_gallery_display_threshold() -> float:
    """
    Returns the configured gallery display threshold.

    Returns:
        Gallery display threshold value
    """
    config = get_config()
    return config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)


def get_edit_password() -> str | None:
    """
    Returns the configured edit password (for auth purposes).

    Returns:
        Edit password or None if not set
    """
    config = get_config()
    password = config.get("EDIT_PASSWORD")
    if not password or password in ["SECRET_PASSWORD", "default_pass"]:
        return None
    return password
