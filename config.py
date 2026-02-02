# config.py
import os

from dotenv import load_dotenv

from utils.settings import load_settings_yaml, save_settings_yaml

# Load environment variables from .env file once.
load_dotenv()


# Persistence for Secret Key
import secrets


def get_or_create_secret_key(config):
    """
    Retrieves or generates a persistent secret key.
    Tries to store it in the state directory (/var/lib/watchmybirds/secret.key).
    Falls back to volatile key if filesystem is read-only or permission denied.
    """
    from pathlib import Path

    # Determine safe storage path
    # Priority: 1. OUTPUT_DIR (State Dir), 2. Local File
    output_dir = config.get("OUTPUT_DIR", "./data/output")
    secret_file = Path(output_dir) / "secret.key"

    # 1. Try to load existing
    if secret_file.exists():
        try:
            with open(secret_file) as f:
                key = f.read().strip()
                if len(key) >= 32:
                    return key
        except Exception as e:
            print(f"Warning: Could not read secret key from {secret_file}: {e}")

    # 2. Generate new
    new_key = secrets.token_hex(32)

    # 3. Try to save
    try:
        # Ensure dir exists
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        with open(secret_file, "w") as f:
            f.write(new_key)
        # Fix permissions (600)
        os.chmod(secret_file, 0o600)
        print(f"Generated new persistent secret key at {secret_file}")
    except Exception as e:
        print(
            f"Warning: Could not save persistent secret key to {secret_file}: {e}. Using volatile key."
        )

    return new_key


_CONFIG = None

DEFAULTS = {
    "DEBUG_MODE": False,
    "OUTPUT_DIR": "./data/output",
    "INGEST_DIR": "./data/ingest",
    "VIDEO_SOURCE": "0",
    "LOCATION_DATA": {"latitude": 52.516, "longitude": 13.377},
    "DETECTOR_MODEL_CHOICE": "yolo",
    "CONFIDENCE_THRESHOLD_DETECTION": 0.65,
    "SAVE_THRESHOLD": 0.65,
    "DETECTION_INTERVAL_SECONDS": 2.0,
    "MODEL_BASE_PATH": "./data/models",
    "CLASSIFIER_CONFIDENCE_THRESHOLD": 0.55,
    "STREAM_FPS": 5.0,
    "STREAM_FPS_CAPTURE": 5.0,
    "STREAM_WIDTH_OUTPUT_RESIZE": 640,
    "DAY_AND_NIGHT_CAPTURE": True,
    "DAY_AND_NIGHT_CAPTURE_LOCATION": "Berlin",
    "CPU_LIMIT": 4,
    "TELEGRAM_COOLDOWN": 3600.0,
    "EDIT_PASSWORD": "watchmybirds",
    "TELEGRAM_ENABLED": False,
    "GALLERY_DISPLAY_THRESHOLD": 0.7,
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "EXIF_GPS_ENABLED": True,
    "MOTION_DETECTION_ENABLED": True,
    "MOTION_SENSITIVITY": 500,
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
    "VIDEO_SOURCE",
    # NEW
    "DEBUG_MODE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LOCATION_DATA",
    "EXIF_GPS_ENABLED",
    "MOTION_DETECTION_ENABLED",
    "MOTION_SENSITIVITY",
}

BOOT_KEYS = set(DEFAULTS.keys()) - RUNTIME_KEYS


def _load_config():
    """Loads configuration from environment variables and YAML."""
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
        config["STREAM_WIDTH_OUTPUT_RESIZE"] = os.getenv("STREAM_WIDTH_OUTPUT_RESIZE")
    if os.getenv("DAY_AND_NIGHT_CAPTURE") is not None:
        config["DAY_AND_NIGHT_CAPTURE"] = os.getenv("DAY_AND_NIGHT_CAPTURE")
    if os.getenv("CPU_LIMIT") is not None:
        config["CPU_LIMIT"] = os.getenv("CPU_LIMIT")

    # Telegram Credentials from ENV
    if os.getenv("TELEGRAM_BOT_TOKEN") is not None:
        config["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.getenv("TELEGRAM_CHAT_ID") is not None:
        config["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")

    # YAML runtime overrides
    yaml_settings = load_settings_yaml(str(config["OUTPUT_DIR"]))

    if (
        "MAX_FPS_DETECTION" in yaml_settings
        and "DETECTION_INTERVAL_SECONDS" not in yaml_settings
    ):
        try:
            legacy_fps = float(yaml_settings["MAX_FPS_DETECTION"])
            if legacy_fps > 0:
                config["DETECTION_INTERVAL_SECONDS"] = 1.0 / legacy_fps
        except Exception:
            pass

    for key, value in yaml_settings.items():
        if key in RUNTIME_KEYS:
            config[key] = value

    if os.getenv("MAX_FPS_DETECTION") and not os.getenv("DETECTION_INTERVAL_SECONDS"):
        try:
            legacy_fps = float(os.getenv("MAX_FPS_DETECTION"))
            if legacy_fps > 0:
                config["DETECTION_INTERVAL_SECONDS"] = 1.0 / legacy_fps
        except Exception:
            pass

    _coerce_config_types(config)
    # Load Persistent Secret Key (Late Binding to use populated config)
    # Allows sessions to survive restarts
    config["SECRET_KEY"] = get_or_create_secret_key(config)

    return config


def ensure_app_directories(config_dict=None):
    """
    Creates necessary directories based on configuration.
    Uses relative paths by default for portability.
    """
    if config_dict is None:
        config_dict = get_config()

    dirs_to_create = [
        config_dict["OUTPUT_DIR"],
        config_dict["INGEST_DIR"],
        config_dict["MODEL_BASE_PATH"],
        os.path.join(config_dict["OUTPUT_DIR"], "logs"),  # Ensure log dir exists
    ]

    for path in dirs_to_create:
        try:
            # path is relative or absolute, makedirs handles both.
            # Convert to absolute for safety/logging
            abs_path = os.path.abspath(path)
            os.makedirs(abs_path, exist_ok=True)
            # update config with absolute path to avoid ambiguity later
            # (optional, but safer if other modules do strictly path manipulations)
            # However, config object is global. Modifying it here is good.
        except Exception as e:
            # We cannot log easily here if logs dir creation fails,
            # so we print to stderr as last resort
            import sys

            print(f"CRITICAL: Failed to create directory {path}: {e}", file=sys.stderr)


def get_config():
    """Returns the loaded configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def _coerce_config_types(config):
    """Validates and enforces expected types for core keys."""
    # Booleans
    for key in (
        "DEBUG_MODE",
        "DAY_AND_NIGHT_CAPTURE",
        "TELEGRAM_ENABLED",
        "EXIF_GPS_ENABLED",
    ):
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

    # STREAM_FPS / STREAM_FPS_CAPTURE: Force safe defaults if 0.0 (legacy unthrottled)
    try:
        stream_fps = float(config.get("STREAM_FPS", 5.0))
        # 0.0 is legacy "unlimited" which kills the Pi. Force to 5.0.
        config["STREAM_FPS"] = stream_fps if stream_fps > 0.1 else 5.0
    except Exception:
        config["STREAM_FPS"] = 5.0

    try:
        stream_fps_capture = float(config.get("STREAM_FPS_CAPTURE", 5.0))
        # 0.0 is legacy "unlimited". Force to 5.0.
        config["STREAM_FPS_CAPTURE"] = (
            stream_fps_capture if stream_fps_capture > 0.1 else 5.0
        )
    except Exception:
        config["STREAM_FPS_CAPTURE"] = 5.0

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
    if interval < 0.01:
        interval = 0.01  # Prevent division by zero
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
    """Provides settings including metadata for UI/API."""
    cfg = get_config()
    yaml_settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
    env_overrides = {key for key in DEFAULTS if os.getenv(key) is not None}
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
    """Validates runtime updates and returns (valid, errors)."""
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
    """Saves runtime settings and updates the running configuration."""
    cfg = get_config()
    yaml_settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
    for key, value in updates.items():
        if key not in RUNTIME_KEYS:
            continue
        # Only remove from YAML if it matches default AND is not overridden by ENV.
        is_default = value == DEFAULTS.get(key)
        has_env = os.getenv(key) is not None

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
    if key in (
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "GALLERY_DISPLAY_THRESHOLD",
    ):
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
        if val >= 0.01:  # Minimum interval of 10ms
            return True, val
        return False, None
    if key == "EDIT_PASSWORD":
        if isinstance(value, str):
            return True, value.strip()
        return False, None
    if key == "DAY_AND_NIGHT_CAPTURE_LOCATION":
        if isinstance(value, str) and value.strip():
            return True, value.strip()
        return False, None
    if key == "VIDEO_SOURCE":
        # Integer string "0", "1" -> int
        # URL string "rtsp://..." -> str
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                return True, int(value)
            # Accept generic strings for RTSP/HTTP
            if value:
                return True, value
        elif isinstance(value, int):
            return True, value
        return False, None

    if key == "DEBUG_MODE":
        return True, _coerce_bool(value)

    if key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if isinstance(value, str):
            return True, value.strip()
        return True, ""  # Empty string fallback (disables feature)

    if key == "LOCATION_DATA":
        # Handle "lat, lon" string or dict
        if isinstance(value, str):
            try:
                parts = [float(x.strip()) for x in value.split(",")]
                if len(parts) == 2:
                    lat, lon = parts
                    # Basic Geo-Coordinate Validation
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return True, {"latitude": lat, "longitude": lon}
            except Exception:
                pass
        elif isinstance(value, dict) and "latitude" in value and "longitude" in value:
            try:
                lat = float(value["latitude"])
                lon = float(value["longitude"])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return True, {"latitude": lat, "longitude": lon}
            except Exception:
                pass
        return False, None

    if key == "EXIF_GPS_ENABLED":
        return True, _coerce_bool(value)

    if key == "MOTION_DETECTION_ENABLED":
        return True, _coerce_bool(value)

    if key == "MOTION_SENSITIVITY":
        try:
            val = int(float(value))
            return True, max(1, val)
        except Exception:
            return False, None

    return False, None


# Backward-compatible alias
def load_config():
    """Alias for legacy code; returns the shared configuration."""
    return get_config()


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_config())
