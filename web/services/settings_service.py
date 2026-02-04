"""
Settings Service - Web Layer Service for Settings Operations.

Thin wrapper over core.settings_core for web-specific concerns.
"""

from typing import Any

from core import settings_core


def get_settings() -> dict[str, Any]:
    """
    Get current settings for UI.

    Delegates to core.settings_core.
    """
    return settings_core.get_current_settings()


def update_settings(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Update runtime settings.

    Args:
        payload: Settings to update

    Returns:
        Tuple of (success, error messages)
    """
    return settings_core.update_settings(payload)


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a single setting value.

    Delegates to core.settings_core.
    """
    return settings_core.get_setting(key, default)


def get_save_threshold() -> float:
    """
    Get the configured save threshold.

    Delegates to core.settings_core.
    """
    return settings_core.get_save_threshold()


def get_gallery_display_threshold() -> float:
    """
    Get the gallery display threshold.

    Delegates to core.settings_core.
    """
    return settings_core.get_gallery_display_threshold()


def get_edit_password() -> str | None:
    """
    Get the edit password if configured.

    Delegates to core.settings_core.
    """
    return settings_core.get_edit_password()
