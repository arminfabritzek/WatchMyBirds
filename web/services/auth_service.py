"""
Auth Service - Web Layer Service for Authentication.

Handles authentication logic and redirect target validation.
"""

from urllib.parse import urlparse

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


_DEFAULT_PASSWORDS = {"watchmybirds", "SECRET_PASSWORD", "default_pass", ""}


def is_default_password() -> bool:
    """Return True if the configured password is a known insecure default."""
    stored = settings_core.get_setting("EDIT_PASSWORD", "") or ""
    return stored in _DEFAULT_PASSWORDS


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

    # Only allow relative paths (no scheme, no netloc) to prevent open redirect.
    parsed = urlparse(next_param)
    if parsed.scheme or parsed.netloc:
        return default

    return next_param
