# config.py
import os
from dotenv import load_dotenv

from utils.settings import load_settings_yaml, save_settings_yaml

# Load environment variables from .env file once.
load_dotenv()


_CONFIG = None

DEFAULTS = {
    "DEBUG_MODE": False,
    "OUTPUT_DIR": "/output",
    "INGEST_DIR": "/ingest",
    "VIDEO_SOURCE": "0",
    "LOCATION_DATA": {"latitude": 52.516, "longitude": 13.377},
    "DETECTOR_MODEL_CHOICE": "yolo",
    "CONFIDENCE_THRESHOLD_DETECTION": 0.55,
    "SAVE_THRESHOLD": 0.55,
    "DETECTION_INTERVAL_SECONDS": 2.0,
    "MODEL_BASE_PATH": "/models",
    "CLASSIFIER_CONFIDENCE_THRESHOLD": 0.55,
    "STREAM_FPS": 0.0,
    "STREAM_FPS_CAPTURE": 0.0,
    "STREAM_WIDTH_OUTPUT_RESIZE": 640,
    "DAY_AND_NIGHT_CAPTURE": True,
    "DAY_AND_NIGHT_CAPTURE_LOCATION": "Berlin",
    "CPU_LIMIT": 1,
    "TELEGRAM_COOLDOWN": 5.0,
    "EDIT_PASSWORD": None,
    "TELEGRAM_ENABLED": False,
    "GALLERY_DISPLAY_THRESHOLD": 0.5,
}

RUNTIME_KEYS = {
    "CONFIDENCE_THRESHOLD_DETECTION",
    "SAVE_THRESHOLD",
    "DETECTION_INTERVAL_SECONDS",
    "DAY_AND_NIGHT_CAPTURE",
    "DAY_AND_NIGHT_CAPTURE_LOCATION",
    "STREAM_FPS",
    "STREAM_FPS_CAPTURE",
    "CLASSIFIER_CONFIDENCE_THRESHOLD",
    "TELEGRAM_COOLDOWN",
    "EDIT_PASSWORD",
    "TELEGRAM_ENABLED",
    "GALLERY_DISPLAY_THRESHOLD",
}

BOOT_KEYS = set(DEFAULTS.keys()) - RUNTIME_KEYS


def _load_config():
    """Lädt Konfiguration aus Umgebungsvariablen und YAML."""
    config = dict(DEFAULTS)

    # Env overrides
    if os.getenv("DEBUG_MODE") is not None:
        config["DEBUG_MODE"] = os.getenv("DEBUG_MODE")
    if os.getenv("OUTPUT_DIR") is not None:
        config["OUTPUT_DIR"] = os.getenv("OUTPUT_DIR")
    if os.getenv("INGEST_DIR") is not None:
        config["INGEST_DIR"] = os.getenv("INGEST_DIR")
    if os.getenv("VIDEO_SOURCE") is not None:
        config["VIDEO_SOURCE"] = os.getenv("VIDEO_SOURCE")

    location_str = os.getenv("LOCATION_DATA")
    if location_str:
        config["LOCATION_DATA"] = location_str

    for key in (
        "DETECTOR_MODEL_CHOICE",
        "MODEL_BASE_PATH",
        "DAY_AND_NIGHT_CAPTURE_LOCATION",
        "EDIT_PASSWORD",
    ):
        if os.getenv(key) is not None:
            config[key] = os.getenv(key)

    for key in (
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "DETECTION_INTERVAL_SECONDS",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "TELEGRAM_COOLDOWN",
    ):
        if os.getenv(key) is not None:
            config[key] = os.getenv(key)

    if os.getenv("STREAM_WIDTH_OUTPUT_RESIZE") is not None:
        config["STREAM_WIDTH_OUTPUT_RESIZE"] = os.getenv(
            "STREAM_WIDTH_OUTPUT_RESIZE"
        )
    if os.getenv("DAY_AND_NIGHT_CAPTURE") is not None:
        config["DAY_AND_NIGHT_CAPTURE"] = os.getenv("DAY_AND_NIGHT_CAPTURE")
    if os.getenv("CPU_LIMIT") is not None:
        config["CPU_LIMIT"] = os.getenv("CPU_LIMIT")

    # YAML runtime overrides
    yaml_settings = load_settings_yaml(str(config["OUTPUT_DIR"]))
    
    if "MAX_FPS_DETECTION" in yaml_settings and "DETECTION_INTERVAL_SECONDS" not in yaml_settings:
        try:
            legacy_fps = float(yaml_settings["MAX_FPS_DETECTION"])
            if legacy_fps > 0:
                config["DETECTION_INTERVAL_SECONDS"] = 1.0 / legacy_fps
        except:
            pass

    for key, value in yaml_settings.items():
        if key in RUNTIME_KEYS:
            config[key] = value

    if os.getenv("MAX_FPS_DETECTION") and not os.getenv("DETECTION_INTERVAL_SECONDS"):
        try:
            legacy_fps = float(os.getenv("MAX_FPS_DETECTION"))
            if legacy_fps > 0:
                config["DETECTION_INTERVAL_SECONDS"] = 1.0 / legacy_fps
        except:
            pass

    _coerce_config_types(config)
    return config


def get_config():
    """Gibt die einmal geladene Konfiguration zurück."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def _coerce_config_types(config):
    """Validiert und erzwingt erwartete Typen für zentrale Keys."""
    # Booleans
    for key in ("DEBUG_MODE", "DAY_AND_NIGHT_CAPTURE", "TELEGRAM_ENABLED"):
        if key in config:
            config[key] = _coerce_bool(config.get(key))

    # LOCATION_DATA: parse "lat, lon" strings into dict
    location_val = config.get("LOCATION_DATA")
    if isinstance(location_val, str):
        try:
            lat_str, lon_str = location_val.split(",")
            config["LOCATION_DATA"] = {
                "latitude": float(lat_str),
                "longitude": float(lon_str),
            }
        except Exception:
            config["LOCATION_DATA"] = DEFAULTS["LOCATION_DATA"]

    # VIDEO_SOURCE: int for webcams, string otherwise (startup-only per locked decision).
    source = config.get("VIDEO_SOURCE", "0")
    try:
        if str(source).isdigit():
            config["VIDEO_SOURCE"] = int(source)
    except Exception:
        config["VIDEO_SOURCE"] = source

    # STREAM_FPS / STREAM_FPS_CAPTURE: allow 0 to disable throttling; otherwise positive float
    try:
        stream_fps = float(config.get("STREAM_FPS", 1))
        config["STREAM_FPS"] = stream_fps if stream_fps > 0 else 0.0
    except Exception:
        config["STREAM_FPS"] = 0.0
    try:
        stream_fps_capture = float(config.get("STREAM_FPS_CAPTURE", 0))
        config["STREAM_FPS_CAPTURE"] = (
            stream_fps_capture if stream_fps_capture > 0 else 0.0
        )
    except Exception:
        config["STREAM_FPS_CAPTURE"] = 0.0

    # CPU_LIMIT should be positive int; fallback to 1
    try:
        cpu_limit = int(float(config.get("CPU_LIMIT", 1)))
        config["CPU_LIMIT"] = cpu_limit if cpu_limit > 0 else 1
    except Exception:
        config["CPU_LIMIT"] = 1

    # Numeric values
    for key in (
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
    ):
        try:
            val = float(config.get(key, 0.55))
            config[key] = max(0.0, min(1.0, val))
        except Exception:
            config[key] = 0.55

    for key in ("DETECTION_INTERVAL_SECONDS", "TELEGRAM_COOLDOWN"):
        try:
            val = float(config.get(key, DEFAULTS.get(key, 1.0)))
            config[key] = val
        except Exception:
            config[key] = DEFAULTS.get(key, 1.0)

    # Derive MAX_FPS_DETECTION
    interval = config.get("DETECTION_INTERVAL_SECONDS", 2.0)
    if interval < 0.01: interval = 0.01 # Prevent division by zero
    config["MAX_FPS_DETECTION"] = 1.0 / interval

    try:
        config["STREAM_WIDTH_OUTPUT_RESIZE"] = int(
            float(config.get("STREAM_WIDTH_OUTPUT_RESIZE", 640))
        )
    except Exception:
        config["STREAM_WIDTH_OUTPUT_RESIZE"] = 640


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "on")
    return False


def get_settings_payload():
    """Liefert Settings inkl. Metadaten für UI/API."""
    cfg = get_config()
    yaml_settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
    env_overrides = {
        key
        for key in DEFAULTS
        if os.getenv(key) is not None
    }
    payload = {}
    for key, default in DEFAULTS.items():
        source = "default"
        if key in yaml_settings:
            source = "yaml"
        elif key in env_overrides:
            source = "env"
        payload[key] = {
            "value": cfg.get(key),
            "default": default,
            "source": source,
            "editable": key in RUNTIME_KEYS,
            "restart_required": key in BOOT_KEYS,
        }
    return payload


def validate_runtime_updates(updates):
    """Validiert Laufzeit-Updates und gibt (valid, errors) zurück."""
    valid = {}
    errors = {}
    for key, value in updates.items():
        if key not in RUNTIME_KEYS:
            continue
        ok, coerced = _validate_value(key, value)
        if ok:
            valid[key] = coerced
        else:
            errors[key] = "Invalid value"
    return valid, errors


def update_runtime_settings(updates):
    """Speichert Laufzeit-Settings und aktualisiert die laufende Konfiguration."""
    cfg = get_config()
    yaml_settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
    for key, value in updates.items():
        if key not in RUNTIME_KEYS:
            continue
        # Only remove from YAML if it matches default AND is not overridden by ENV.
        is_default = (value == DEFAULTS.get(key))
        has_env = (os.getenv(key) is not None)

        if is_default and not has_env:
            yaml_settings.pop(key, None)
        else:
            yaml_settings[key] = value
        cfg[key] = value
    _coerce_config_types(cfg)
    save_settings_yaml(yaml_settings, str(cfg["OUTPUT_DIR"]))


def _validate_value(key, value):
    if key in ("DAY_AND_NIGHT_CAPTURE", "TELEGRAM_ENABLED"):
        return True, _coerce_bool(value)
    if key in ("CONFIDENCE_THRESHOLD_DETECTION", "SAVE_THRESHOLD", "CLASSIFIER_CONFIDENCE_THRESHOLD", "GALLERY_DISPLAY_THRESHOLD"):
        try:
            val = float(value)
        except Exception:
            return False, None
        if 0.0 <= val <= 1.0:
            return True, val
        return False, None
    if key in ("STREAM_FPS", "STREAM_FPS_CAPTURE"):
        try:
            val = float(value)
        except Exception:
            return False, None
        if val >= 0.0:
            return True, val
        return False, None
    if key in ("DETECTION_INTERVAL_SECONDS", "TELEGRAM_COOLDOWN"):
        try:
            val = float(value)
        except Exception:
            return False, None
        if val >= 0.01: # Minimum interval of 10ms
            return True, val
        return False, None
    if key in ("DAY_AND_NIGHT_CAPTURE_LOCATION", "EDIT_PASSWORD"):
        if isinstance(value, str) and value.strip():
            return True, value.strip()
        return False, None
    return False, None


# Backward-compatible alias
def load_config():
    """Alias für Alt-Code; liefert die geteilte Konfiguration."""
    return get_config()


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_config())
