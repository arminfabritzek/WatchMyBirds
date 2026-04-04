"""
Auth Service - Web Layer Service for Authentication.

Handles authentication logic and redirect target validation.
"""

from urllib.parse import urlparse

from core import settings_core

DEFAULT_PASSWORDS = {"watchmybirds", "SECRET_PASSWORD", "default_pass", ""}
MIN_PASSWORD_LENGTH = 8


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


def is_default_password() -> bool:
    """Return True if the configured password is a known insecure default."""
    stored = settings_core.get_setting("EDIT_PASSWORD", "") or ""
    return stored in DEFAULT_PASSWORDS


def should_require_password_setup() -> bool:
    """
    Return True when the appliance should force an initial password setup.

    Scope this to the Raspberry Pi appliance path so local dev and Docker
    workflows don't unexpectedly lose their lightweight default behavior.
    """
    return is_default_password() and settings_core.get_deploy_type() == "rpi"


def validate_new_password(
    password: str, password_confirm: str | None = None
) -> tuple[bool, str, str | None]:
    """Validate and normalize a newly chosen admin password."""
    cleaned = (password or "").strip()
    confirm_cleaned = (password_confirm or "").strip() if password_confirm is not None else None

    if len(cleaned) < MIN_PASSWORD_LENGTH:
        return False, "", f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."

    if cleaned in DEFAULT_PASSWORDS:
        return False, "", "Please choose a password that is not a known default."

    if confirm_cleaned is not None and cleaned != confirm_cleaned:
        return False, "", "Password confirmation does not match."

    return True, cleaned, None


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
