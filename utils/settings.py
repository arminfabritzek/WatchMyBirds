from pathlib import Path
from typing import Any

import yaml


def mask_rtsp_url(url: str | None) -> str | None:
    """
    Masks the password in an RTSP/HTTP URL.
    Format: scheme://user:password@host... -> scheme://user:*****@host...
    Handling passwords with '@' by finding the last '@' before the host.
    """
    if not url:
        return url

    val = str(url)
    scheme_end = val.find("://")
    if scheme_end == -1:
        return val  # Not a standard URL

    scheme_len = 3  # ://
    start_auth = scheme_end + scheme_len

    path_start = val.find("/", start_auth)
    if path_start == -1:
        authority = val[start_auth:]
        rest = ""
    else:
        authority = val[start_auth:path_start]
        rest = val[path_start:]

    last_at = authority.rfind("@")
    if last_at == -1:
        return val

    user_pass = authority[:last_at]
    host_port = authority[last_at + 1 :]

    first_colon = user_pass.find(":")
    if first_colon == -1:
        return val

    user = user_pass[:first_colon]
    new_authority = f"{user}:*****@{host_port}"

    return f"{val[:start_auth]}{new_authority}{rest}"


def unmask_rtsp_url(new_url: str | None, original_url: str | None) -> str | None:
    """
    Restores the original password if the new URL contains the placeholder '*****'.
    """
    if not new_url:
        return new_url

    if "*****" not in new_url:
        return new_url

    if not original_url:
        return new_url

    # We need to extract the original password using the exact same logic
    val = str(original_url)
    scheme_end = val.find("://")
    if scheme_end == -1:
        return new_url

    start_auth = scheme_end + 3
    path_start = val.find("/", start_auth)
    if path_start == -1:
        authority = val[start_auth:]
    else:
        authority = val[start_auth:path_start]

    last_at = authority.rfind("@")
    if last_at == -1:
        return new_url

    user_pass = authority[:last_at]
    first_colon = user_pass.find(":")
    if first_colon == -1:
        return new_url

    original_password = user_pass[first_colon + 1 :]

    # Now replace brackets in new_url
    return new_url.replace("*****", original_password, 1)


def get_settings_path(output_dir: str | None = None) -> Path:
    """Returns the path to the settings.yaml file."""
    if output_dir is None:
        from config import get_config

        output_dir = get_config()["OUTPUT_DIR"]
    return Path(output_dir) / "settings.yaml"


def load_settings_yaml(output_dir: str | None = None) -> dict[str, Any]:
    """Loads runtime settings from YAML; creates file if missing."""
    settings_path = get_settings_path(output_dir)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if not settings_path.exists():
        settings_path.write_text("{}", encoding="utf-8")
        return {}
    raw = settings_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def save_settings_yaml(
    settings_dict: dict[str, Any], output_dir: str | None = None
) -> None:
    """Saves runtime settings as YAML (only given keys)."""
    settings_path = get_settings_path(output_dir)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings_dict, handle, sort_keys=True)
    # Runtime settings can carry EDIT_PASSWORD and TELEGRAM_BOT_TOKEN, so the
    # file must not inherit a permissive umask.
    settings_path.chmod(0o600)
