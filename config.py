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
    # Detection-confidence floor is now model-owned (read from the active
    # model_metadata.json). CONFIDENCE_THRESHOLD_DETECTION has been retired.
    "SAVE_THRESHOLD": 0.65,
    "SAVE_THRESHOLD_MODE": "auto",  # "auto" (derived from model) or "manual"
    "DETECTION_INTERVAL_SECONDS": 2.0,
    "MODEL_BASE_PATH": "./data/models",
    "BBOX_QUALITY_THRESHOLD": 0.40,
    "SPECIES_CONF_THRESHOLD": 0.70,
    "UNKNOWN_SCORE_THRESHOLD": 0.60,
    "STREAM_FPS": 5.0,
    "STREAM_FPS_CAPTURE": 5.0,
    "STREAM_WIDTH_OUTPUT_RESIZE": 640,
    "DAY_AND_NIGHT_CAPTURE": True,
    "DAY_AND_NIGHT_CAPTURE_LOCATION": "Berlin",
    "CPU_LIMIT": 0,
    "TELEGRAM_COOLDOWN": 3600.0,
    "EDIT_PASSWORD": "watchmybirds",
    "TELEGRAM_ENABLED": False,
    "GALLERY_DISPLAY_THRESHOLD": 0.1,
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "TELEGRAM_REPORT_TIME": "21:00",
    "TELEGRAM_MODE": "off",  # "off", "live", "daily", "interval"
    "TELEGRAM_REPORT_INTERVAL_HOURS": 1,
    "DEVICE_NAME": "",
    "EXIF_GPS_ENABLED": True,
    "INBOX_REQUIRE_EXIF_DATETIME": True,
    "INBOX_REQUIRE_EXIF_GPS": True,
    "MOTION_DETECTION_ENABLED": False,
    "MOTION_SENSITIVITY": 500,
    "CAMERA_URL": "",
    "ENABLE_NIGHTLY_DEEP_SCAN": False,
    "STREAM_SOURCE_MODE": "auto",  # "auto", "relay", "direct"
    "GO2RTC_STREAM_NAME": "camera",
    "GO2RTC_API_BASE": "http://127.0.0.1:1984",
    "GO2RTC_CONFIG_PATH": "./go2rtc.yaml",
    "SPECIES_COMMON_NAME_LOCALE": "DE",
}

RUNTIME_KEYS = {
    "SAVE_THRESHOLD",
    "SAVE_THRESHOLD_MODE",
    "DETECTION_INTERVAL_SECONDS",
    "DAY_AND_NIGHT_CAPTURE",
    "DAY_AND_NIGHT_CAPTURE_LOCATION",
    "STREAM_FPS",
    "STREAM_FPS_CAPTURE",
    "BBOX_QUALITY_THRESHOLD",
    "SPECIES_CONF_THRESHOLD",
    "UNKNOWN_SCORE_THRESHOLD",
    "TELEGRAM_COOLDOWN",
    "EDIT_PASSWORD",
    # TELEGRAM_ENABLED is derived from TELEGRAM_MODE (see _coerce_config_types)
    # and deliberately excluded from RUNTIME_KEYS so form POSTs can't clobber
    # the derived value. The notifier still reads it, and _coerce_config_types
    # keeps it in sync whenever settings change.
    "GALLERY_DISPLAY_THRESHOLD",
    "VIDEO_SOURCE",
    "CAMERA_URL",
    "STREAM_SOURCE_MODE",
    # NEW
    "DEBUG_MODE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "TELEGRAM_REPORT_TIME",
    "TELEGRAM_MODE",
    "TELEGRAM_REPORT_INTERVAL_HOURS",
    "DEVICE_NAME",
    "LOCATION_DATA",
    "EXIF_GPS_ENABLED",
    "INBOX_REQUIRE_EXIF_DATETIME",
    "INBOX_REQUIRE_EXIF_GPS",
    "MOTION_DETECTION_ENABLED",
    "MOTION_SENSITIVITY",
    "SPECIES_COMMON_NAME_LOCALE",
    # STREAM_WIDTH_OUTPUT_RESIZE is read once at mount time in
    # web_interface.py, so changing it here takes effect after the
    # next restart. It's still in RUNTIME_KEYS because the Settings
    # form exposes the field — without this entry, Apply would
    # silently drop the new value even though it was committed to
    # the form. See validator in _validate_value.
    "STREAM_WIDTH_OUTPUT_RESIZE",
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
    if os.getenv("CAMERA_URL") is not None:
        config["CAMERA_URL"] = os.getenv("CAMERA_URL")
    if os.getenv("STREAM_SOURCE_MODE") is not None:
        config["STREAM_SOURCE_MODE"] = os.getenv("STREAM_SOURCE_MODE")
    if os.getenv("GO2RTC_STREAM_NAME") is not None:
        config["GO2RTC_STREAM_NAME"] = os.getenv("GO2RTC_STREAM_NAME")
    if os.getenv("GO2RTC_API_BASE") is not None:
        config["GO2RTC_API_BASE"] = os.getenv("GO2RTC_API_BASE")
    if os.getenv("GO2RTC_CONFIG_PATH") is not None:
        config["GO2RTC_CONFIG_PATH"] = os.getenv("GO2RTC_CONFIG_PATH")

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

    if os.getenv("SPECIES_COMMON_NAME_LOCALE") is not None:
        config["SPECIES_COMMON_NAME_LOCALE"] = os.getenv("SPECIES_COMMON_NAME_LOCALE")

    for key in (
        "SAVE_THRESHOLD",
        "DETECTION_INTERVAL_SECONDS",
        "BBOX_QUALITY_THRESHOLD",
        "SPECIES_CONF_THRESHOLD",
        "UNKNOWN_SCORE_THRESHOLD",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "TELEGRAM_COOLDOWN",
        "GALLERY_DISPLAY_THRESHOLD",
        "MOTION_SENSITIVITY",
    ):
        if os.getenv(key) is not None:
            config[key] = os.getenv(key)

    if os.getenv("STREAM_WIDTH_OUTPUT_RESIZE") is not None:
        config["STREAM_WIDTH_OUTPUT_RESIZE"] = os.getenv("STREAM_WIDTH_OUTPUT_RESIZE")
    if os.getenv("DAY_AND_NIGHT_CAPTURE") is not None:
        config["DAY_AND_NIGHT_CAPTURE"] = os.getenv("DAY_AND_NIGHT_CAPTURE")
    if os.getenv("MOTION_DETECTION_ENABLED") is not None:
        config["MOTION_DETECTION_ENABLED"] = os.getenv("MOTION_DETECTION_ENABLED")
    if os.getenv("EXIF_GPS_ENABLED") is not None:
        config["EXIF_GPS_ENABLED"] = os.getenv("EXIF_GPS_ENABLED")
    if os.getenv("TELEGRAM_ENABLED") is not None:
        config["TELEGRAM_ENABLED"] = os.getenv("TELEGRAM_ENABLED")
    if os.getenv("ENABLE_NIGHTLY_DEEP_SCAN") is not None:
        config["ENABLE_NIGHTLY_DEEP_SCAN"] = os.getenv("ENABLE_NIGHTLY_DEEP_SCAN")
    if os.getenv("CPU_LIMIT") is not None:
        config["CPU_LIMIT"] = os.getenv("CPU_LIMIT")

    # Telegram Credentials from ENV
    if os.getenv("TELEGRAM_BOT_TOKEN") is not None:
        config["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.getenv("TELEGRAM_CHAT_ID") is not None:
        config["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
    if os.getenv("TELEGRAM_REPORT_TIME") is not None:
        config["TELEGRAM_REPORT_TIME"] = os.getenv("TELEGRAM_REPORT_TIME")
    if os.getenv("TELEGRAM_MODE") is not None:
        config["TELEGRAM_MODE"] = os.getenv("TELEGRAM_MODE")
    if os.getenv("TELEGRAM_REPORT_INTERVAL_HOURS") is not None:
        config["TELEGRAM_REPORT_INTERVAL_HOURS"] = os.getenv(
            "TELEGRAM_REPORT_INTERVAL_HOURS"
        )
    if os.getenv("DEVICE_NAME") is not None:
        config["DEVICE_NAME"] = os.getenv("DEVICE_NAME")

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

    # Legacy migration: users upgrading from pre-TELEGRAM_MODE builds have a
    # bare `TELEGRAM_ENABLED: true/false` in settings.yaml and no mode key.
    # Translate that to the closest equivalent so live alerts don't silently
    # stop after upgrade.
    if "TELEGRAM_MODE" not in yaml_settings:
        legacy_enabled = yaml_settings.get("TELEGRAM_ENABLED")
        if legacy_enabled is not None:
            config["TELEGRAM_MODE"] = "live" if _coerce_bool(legacy_enabled) else "off"

    if os.getenv("MAX_FPS_DETECTION") and not os.getenv("DETECTION_INTERVAL_SECONDS"):
        try:
            legacy_fps = float(os.getenv("MAX_FPS_DETECTION"))
            if legacy_fps > 0:
                config["DETECTION_INTERVAL_SECONDS"] = 1.0 / legacy_fps
        except Exception:
            pass

    # One-time migration: derive CAMERA_URL from legacy VIDEO_SOURCE when needed.
    _migrate_camera_url(config)
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


def probe_go2rtc(
    api_base: str = "http://127.0.0.1:1984",
    timeout_sec: float = 2.0,
) -> bool:
    """Return True if go2rtc API responds.

    Default timeout raised to 2.0s to accommodate Docker bridge-network
    DNS resolution which can take 200-500ms on first lookup.
    """
    import logging
    import urllib.request

    url = f"{api_base.rstrip('/')}/api/streams"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception as exc:
        logging.getLogger(__name__).debug("probe_go2rtc failed for %s: %s", url, exc)
        return False


def verify_go2rtc_stream_ready(
    api_base: str = "http://127.0.0.1:1984",
    stream_name: str = "camera",
    timeout_sec: float = 2.0,
) -> bool:
    """
    Return True when go2rtc has the stream configured.

    This intentionally checks configuration presence, not active producers.
    """
    import json
    import urllib.request

    url = f"{api_base.rstrip('/')}/api/streams"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
            return isinstance(data, dict) and stream_name in data
    except Exception:
        return False


def resolve_effective_sources(config: dict) -> dict:
    """
    Resolve effective runtime source for detection streaming.

    Returns keys: video_source, effective_mode, reason.
    """
    camera_url = config.get("CAMERA_URL", "")
    mode = config.get("STREAM_SOURCE_MODE", "auto")
    stream_name = config.get("GO2RTC_STREAM_NAME", "camera")
    api_base = config.get("GO2RTC_API_BASE", "http://127.0.0.1:1984")

    try:
        from urllib.parse import urlparse

        relay_host = urlparse(api_base).hostname or "127.0.0.1"
    except Exception:
        relay_host = "127.0.0.1"
    relay_url = f"rtsp://{relay_host}:8554/{stream_name}"

    if mode == "relay":
        return {
            "video_source": relay_url,
            "effective_mode": "relay",
            "reason": "mode=relay (forced)",
        }
    if mode == "direct":
        return {
            "video_source": camera_url,
            "effective_mode": "direct",
            "reason": "mode=direct (forced)",
        }

    # auto mode
    if (
        camera_url
        and probe_go2rtc(api_base)
        and verify_go2rtc_stream_ready(api_base, stream_name)
    ):
        return {
            "video_source": relay_url,
            "effective_mode": "relay",
            "reason": "mode=auto, go2rtc healthy + stream configured -> relay",
        }

    if not camera_url:
        reason = "mode=auto, CAMERA_URL empty -> direct (no source)"
    elif not probe_go2rtc(api_base):
        reason = "mode=auto, go2rtc unavailable -> direct"
    else:
        reason = "mode=auto, go2rtc stream not configured -> direct"

    return {
        "video_source": camera_url,
        "effective_mode": "direct",
        "reason": reason,
    }


def ensure_go2rtc_stream_synced(config: dict, *, with_retry: bool = False) -> None:
    """Proactively sync CAMERA_URL into go2rtc before resolving stream sources.

    This breaks the chicken-and-egg problem: ``resolve_effective_sources()``
    needs go2rtc to have the stream configured, but the old post-resolve sync
    only ran when the resolver had *already* chosen relay mode – which it
    never did on a fresh image because go2rtc had an empty source list.

    Safe to call at any point; silently returns when preconditions are not met
    (no camera URL, go2rtc unreachable).

    Drift-protection: this sync also runs when ``STREAM_SOURCE_MODE=direct``.
    Even in direct mode, the browser stream page may still rely on go2rtc, so
    keeping go2rtc's upstream aligned with CAMERA_URL prevents stale sources.

    Args:
        config: The application config dict (must already be loaded/coerced).
        with_retry: If True, use retry logic for the reload call (boot path).
    """
    import logging

    log = logging.getLogger(__name__)

    camera_url = config.get("CAMERA_URL", "")

    # Nothing to sync if there is no camera URL.
    if not camera_url:
        return

    api_base = config.get("GO2RTC_API_BASE", "http://127.0.0.1:1984")

    # Only sync if go2rtc is actually reachable.
    if not probe_go2rtc(api_base):
        log.debug("ensure_go2rtc_stream_synced: go2rtc unreachable, skipping")
        return

    try:
        from utils.go2rtc_config import (
            reload_go2rtc_stream,
            reload_go2rtc_stream_with_retry,
            sync_camera_stream_source,
        )

        go2rtc_path = config.get("GO2RTC_CONFIG_PATH", "./go2rtc.yaml")
        stream_name = config.get("GO2RTC_STREAM_NAME", "camera")

        sync_ok = sync_camera_stream_source(go2rtc_path, camera_url, stream_name)
        if not sync_ok:
            log.warning(
                "go2rtc pre-sync config write returned false (path=%s)",
                go2rtc_path,
            )

        # Push the source into the running go2rtc process.
        if with_retry:
            reload_go2rtc_stream_with_retry(
                api_base=api_base,
                stream_name=stream_name,
                camera_url=camera_url,
            )
        else:
            reload_go2rtc_stream(
                api_base=api_base,
                stream_name=stream_name,
                camera_url=camera_url,
            )
    except Exception as exc:
        log.warning("go2rtc pre-sync failed: %s", exc)


def _migrate_camera_url(config: dict) -> None:
    """Derive CAMERA_URL from legacy VIDEO_SOURCE when CAMERA_URL is empty."""
    camera_url = config.get("CAMERA_URL", "")
    if camera_url:
        return

    video_source = str(config.get("VIDEO_SOURCE", "0")).strip()
    if not video_source or video_source == "0":
        return

    # Legacy relay source; try to read real camera source from go2rtc config.
    if "127.0.0.1:8554" in video_source or "localhost:8554" in video_source:
        try:
            from utils.go2rtc_config import read_camera_stream_source

            go2rtc_path = config.get("GO2RTC_CONFIG_PATH", "./go2rtc.yaml")
            stream_name = config.get("GO2RTC_STREAM_NAME", "camera")
            real_url = read_camera_stream_source(go2rtc_path, stream_name)
            if real_url:
                config["CAMERA_URL"] = real_url
        except Exception:
            pass
        return

    config["CAMERA_URL"] = video_source


def _coerce_config_types(config):
    """Validates and enforces expected types for core keys."""
    # Booleans
    for key in (
        "DEBUG_MODE",
        "DAY_AND_NIGHT_CAPTURE",
        "TELEGRAM_ENABLED",
        "EXIF_GPS_ENABLED",
        "INBOX_REQUIRE_EXIF_DATETIME",
        "INBOX_REQUIRE_EXIF_GPS",
        "MOTION_DETECTION_ENABLED",
        "ENABLE_NIGHTLY_DEEP_SCAN",
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

    # CPU_LIMIT: 0 = disabled (no cpu pinning), positive int = limit cores
    try:
        cpu_limit = int(float(config.get("CPU_LIMIT", 0)))
        config["CPU_LIMIT"] = max(0, cpu_limit)
    except Exception:
        config["CPU_LIMIT"] = 0

    # Numeric values
    for key in (
        "SAVE_THRESHOLD",
        "BBOX_QUALITY_THRESHOLD",
        "SPECIES_CONF_THRESHOLD",
        "UNKNOWN_SCORE_THRESHOLD",
        "GALLERY_DISPLAY_THRESHOLD",
    ):
        try:
            val = float(config.get(key, DEFAULTS.get(key, 0.55)))
            config[key] = max(0.0, min(1.0, val))
        except Exception:
            config[key] = DEFAULTS.get(key, 0.55)

    # Integer values
    for key in ("MOTION_SENSITIVITY",):
        try:
            val = int(float(config.get(key, DEFAULTS.get(key, 500))))
            config[key] = max(1, val)
        except Exception:
            config[key] = DEFAULTS.get(key, 500)

    for key in ("DETECTION_INTERVAL_SECONDS", "TELEGRAM_COOLDOWN"):
        try:
            val = float(config.get(key, DEFAULTS.get(key, 1.0)))
            config[key] = val
        except Exception:
            config[key] = DEFAULTS.get(key, 1.0)

    # TELEGRAM_REPORT_TIME: strict HH:MM 24h format.
    report_time = config.get(
        "TELEGRAM_REPORT_TIME", DEFAULTS.get("TELEGRAM_REPORT_TIME", "21:00")
    )
    if not isinstance(report_time, str):
        report_time = str(report_time)
    report_time = report_time.strip()
    try:
        hh, mm = report_time.split(":")
        if not (len(hh) == 2 and len(mm) == 2 and hh.isdigit() and mm.isdigit()):
            raise ValueError("invalid format")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError("out of range")
        config["TELEGRAM_REPORT_TIME"] = f"{int(hh):02d}:{int(mm):02d}"
    except Exception:
        config["TELEGRAM_REPORT_TIME"] = DEFAULTS.get("TELEGRAM_REPORT_TIME", "21:00")

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

    # Stream resolver keys
    camera_url = config.get("CAMERA_URL", "")
    if camera_url is None:
        camera_url = ""
    elif not isinstance(camera_url, str):
        camera_url = str(camera_url)
    camera_url = camera_url.strip()
    if camera_url.lower() in ("none", "null"):
        camera_url = ""
    config["CAMERA_URL"] = camera_url

    mode = config.get("STREAM_SOURCE_MODE", "auto")
    if isinstance(mode, str) and mode.strip().lower() in ("auto", "relay", "direct"):
        config["STREAM_SOURCE_MODE"] = mode.strip().lower()
    else:
        config["STREAM_SOURCE_MODE"] = "auto"

    stream_name = config.get("GO2RTC_STREAM_NAME", "camera")
    if isinstance(stream_name, str) and stream_name.strip():
        config["GO2RTC_STREAM_NAME"] = stream_name.strip()
    else:
        config["GO2RTC_STREAM_NAME"] = "camera"

    api_base = config.get("GO2RTC_API_BASE", "http://127.0.0.1:1984")
    if isinstance(api_base, str) and api_base.strip():
        config["GO2RTC_API_BASE"] = api_base.strip().rstrip("/")
    else:
        config["GO2RTC_API_BASE"] = "http://127.0.0.1:1984"

    go2rtc_path = config.get("GO2RTC_CONFIG_PATH", "./go2rtc.yaml")
    if isinstance(go2rtc_path, str) and go2rtc_path.strip():
        config["GO2RTC_CONFIG_PATH"] = go2rtc_path.strip()
    else:
        config["GO2RTC_CONFIG_PATH"] = "./go2rtc.yaml"

    # SPECIES_COMMON_NAME_LOCALE: uppercase, only DE or NO
    locale_val = str(config.get("SPECIES_COMMON_NAME_LOCALE", "DE")).strip().upper()
    if locale_val not in ("DE", "NO"):
        locale_val = "DE"
    config["SPECIES_COMMON_NAME_LOCALE"] = locale_val

    # DEVICE_NAME: trimmed string, capped at 64 chars (shown in every message prefix)
    device_name = config.get("DEVICE_NAME", "")
    if device_name is None:
        device_name = ""
    elif not isinstance(device_name, str):
        device_name = str(device_name)
    config["DEVICE_NAME"] = device_name.strip()[:64]

    # TELEGRAM_MODE: restrict to the four known modes.
    mode_val = str(config.get("TELEGRAM_MODE", "off") or "off").strip().lower()
    if mode_val not in ("off", "live", "daily", "interval"):
        mode_val = "off"
    config["TELEGRAM_MODE"] = mode_val

    # TELEGRAM_ENABLED is derived from mode. Any non-"off" mode implies the
    # Telegram channel is active so downstream checks (e.g. the notifier) keep
    # working. "off" disables everything; individual features below gate on
    # TELEGRAM_MODE directly so the mode decides *what* gets sent.
    config["TELEGRAM_ENABLED"] = mode_val != "off"

    # TELEGRAM_REPORT_INTERVAL_HOURS: integer in [1, 24]. Used only when
    # TELEGRAM_MODE == "interval".
    try:
        interval_hours = int(float(config.get("TELEGRAM_REPORT_INTERVAL_HOURS", 1)))
    except Exception:
        interval_hours = 1
    config["TELEGRAM_REPORT_INTERVAL_HOURS"] = max(1, min(24, interval_hours))


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
        is_internal = key == "VIDEO_SOURCE"
        payload[key] = {
            "value": cfg.get(key),
            "default": default,
            "source": source,
            "editable": key in RUNTIME_KEYS and not is_internal,
            "restart_required": key in BOOT_KEYS,
            "internal": is_internal,
        }

    runtime_video = cfg.get("VIDEO_SOURCE", "")
    runtime_mode = (
        "relay"
        if isinstance(runtime_video, str) and ":8554/" in runtime_video
        else "direct"
    )
    payload["STREAM_SOURCE_RUNTIME_VIDEO"] = {
        "value": runtime_video,
        "default": "",
        "source": "runtime",
        "editable": False,
        "restart_required": False,
        "internal": True,
    }
    payload["STREAM_SOURCE_RUNTIME_MODE"] = {
        "value": runtime_mode,
        "default": "direct",
        "source": "runtime",
        "editable": False,
        "restart_required": False,
        "internal": True,
    }

    resolved = resolve_effective_sources(cfg)
    payload["STREAM_SOURCE_EFFECTIVE_MODE"] = {
        "value": resolved.get("effective_mode", "direct"),
        "default": "direct",
        "source": "derived",
        "editable": False,
        "restart_required": False,
        "internal": True,
    }
    payload["STREAM_SOURCE_REASON"] = {
        "value": resolved.get("reason", ""),
        "default": "",
        "source": "derived",
        "editable": False,
        "restart_required": False,
        "internal": True,
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
    if key in (
        "DAY_AND_NIGHT_CAPTURE",
        "TELEGRAM_ENABLED",
        "INBOX_REQUIRE_EXIF_DATETIME",
        "INBOX_REQUIRE_EXIF_GPS",
        "MOTION_DETECTION_ENABLED",
        "DEBUG_MODE",
        "EXIF_GPS_ENABLED",
    ):
        return True, _coerce_bool(value)
    if key == "SAVE_THRESHOLD_MODE":
        val = str(value).strip().lower() if value is not None else ""
        if val in ("auto", "manual"):
            return True, val
        return False, None
    if key in (
        "SAVE_THRESHOLD",
        "BBOX_QUALITY_THRESHOLD",
        "SPECIES_CONF_THRESHOLD",
        "UNKNOWN_SCORE_THRESHOLD",
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

    if key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if isinstance(value, str):
            return True, value.strip()
        return True, ""  # Empty string fallback (disables feature)

    if key == "DEVICE_NAME":
        if value is None:
            return True, ""
        if isinstance(value, str):
            return True, value.strip()[:64]
        return True, str(value).strip()[:64]

    if key == "TELEGRAM_MODE":
        if not isinstance(value, str):
            return False, None
        normalized = value.strip().lower()
        if normalized in ("off", "live", "daily", "interval"):
            return True, normalized
        return False, None

    if key == "TELEGRAM_REPORT_INTERVAL_HOURS":
        try:
            hours = int(float(value))
        except Exception:
            return False, None
        if 1 <= hours <= 24:
            return True, hours
        return False, None

    if key == "TELEGRAM_REPORT_TIME":
        if not isinstance(value, str):
            return False, None
        cleaned = value.strip()
        if not cleaned:
            return True, DEFAULTS.get("TELEGRAM_REPORT_TIME", "21:00")
        import re

        if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", cleaned):
            return True, cleaned
        return False, None

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

    if key == "MOTION_SENSITIVITY":
        try:
            val = int(float(value))
            return True, max(1, val)
        except Exception:
            return False, None

    if key == "CAMERA_URL":
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in ("none", "null"):
                cleaned = ""
            return True, cleaned
        if value is None:
            return True, ""
        return True, ""

    if key == "STREAM_SOURCE_MODE":
        if isinstance(value, str) and value.strip().lower() in (
            "auto",
            "relay",
            "direct",
        ):
            return True, value.strip().lower()
        return False, None

    if key == "SPECIES_COMMON_NAME_LOCALE":
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in ("DE", "NO"):
                return True, normalized
        return False, None

    if key == "STREAM_WIDTH_OUTPUT_RESIZE":
        # Accept int, numeric string, or empty string (= "full resolution",
        # coerced to the default 640 to keep the downstream consumer
        # happy — the feature advertised as "Empty = full" is not wired
        # up anywhere; it falls back to default).
        if value is None or (isinstance(value, str) and not value.strip()):
            return True, DEFAULTS.get("STREAM_WIDTH_OUTPUT_RESIZE", 640)
        try:
            width = int(float(value))
        except Exception:
            return False, None
        if 160 <= width <= 7680:  # reasonable bounds: QQVGA to 8K
            return True, width
        return False, None

    return False, None


# Backward-compatible alias
def load_config():
    """Alias for legacy code; returns the shared configuration."""
    return get_config()


# ---------------------------------------------------------------------------
# Save-threshold resolution (auto / manual)
# ---------------------------------------------------------------------------

# Constant offset above the model's detection floor in Auto mode.
# Locked to 0.10 so the rule does not quietly shift with each release.
SAVE_THRESHOLD_AUTO_OFFSET = 0.10


def effective_save_threshold(cfg: dict, detector_default_conf: float | None) -> float:
    """Return the save-threshold value the DetectionManager should apply.

    Mode resolution:
      - cfg["SAVE_THRESHOLD_MODE"] == "manual": honours cfg["SAVE_THRESHOLD"].
      - cfg["SAVE_THRESHOLD_MODE"] == "auto" (default): derived from the
        model's own detection floor as (conf_default + 0.10), clipped to
        [0, 1]. Falls back to cfg["SAVE_THRESHOLD"] when no detector
        instance is available yet (startup race before the first detector
        init).

    Unknown mode strings fall through to the "auto" path so a broken
    setting never blocks the pipeline.
    """
    mode = str(cfg.get("SAVE_THRESHOLD_MODE", "auto")).strip().lower()
    manual_val = float(cfg.get("SAVE_THRESHOLD", 0.65))

    if mode == "manual":
        return manual_val

    # Auto mode
    if detector_default_conf is None:
        # Detector not ready yet -> cannot derive; use last persisted value.
        return manual_val

    derived = float(detector_default_conf) + SAVE_THRESHOLD_AUTO_OFFSET
    if derived < 0.0:
        return 0.0
    if derived > 1.0:
        return 1.0
    return derived


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_config())
