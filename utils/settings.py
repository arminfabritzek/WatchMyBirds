import os
from pathlib import Path
from typing import Dict, Any

import yaml


def get_settings_path(output_dir: str = None) -> Path:
    """Gibt den Pfad zur settings.yaml-Datei zurück."""
    if output_dir is None:
        output_dir = os.getenv("OUTPUT_DIR", "/output")
    return Path(output_dir) / "settings.yaml"


def load_settings_yaml(output_dir: str = None) -> Dict[str, Any]:
    """Lädt Laufzeit-Settings aus YAML; erstellt Datei falls fehlend."""
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


def save_settings_yaml(settings_dict: Dict[str, Any], output_dir: str = None) -> None:
    """Speichert Laufzeit-Settings als YAML (nur gegebene Keys)."""
    settings_path = get_settings_path(output_dir)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings_dict, handle, sort_keys=True)
