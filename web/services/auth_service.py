"""
Auth Service - Web Layer Service for Authentication.

Handles authentication logic and redirect target validation.
"""

from core import settings_core


def authenticate(provided_password: str) -> bool:
    """
    Verify the provided password against the configuration.

    Args:
        provided_password: The password to check.

    Returns:
        True if password matches, False otherwise.
    """
    # Get raw password from settings (bypass security filter to match legacy behavior)
    stored_password = settings_core.get_setting("EDIT_PASSWORD", "")

    # Match legacy comparison logic: (stored_password or "")
    # Note: If stored_password is None, it becomes ""
    target = stored_password or ""

    return provided_password == target


def get_redirect_target(next_param: str | None, default: str = "/gallery") -> str:
    """
    Determine the redirect target URL.

    Args:
        next_param: The 'next' URL parameter or form field.
        default: Default URL if next_param is invalid/missing.

    Returns:
        The target URL.
    """
    if not next_param:
        return default

    # Basic safety check could go here, but matching legacy "trust it" behavior
    # Legacy just used .get("next", "/gallery")
    return next_param
