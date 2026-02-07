from pathlib import Path
from typing import Any

import yaml


def get_settings_path(output_dir: str = None) -> Path:
    """Returns the path to the settings.yaml file."""
    if output_dir is None:
        from config import get_config

        output_dir = get_config()["OUTPUT_DIR"]
    return Path(output_dir) / "settings.yaml"


def load_settings_yaml(output_dir: str = None) -> dict[str, Any]:
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


def save_settings_yaml(settings_dict: dict[str, Any], output_dir: str = None) -> None:
    """Saves runtime settings as YAML (only given keys)."""
    settings_path = get_settings_path(output_dir)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings_dict, handle, sort_keys=True)
