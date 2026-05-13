"""
PTZ Core - camera PTZ use cases and configuration validation.
"""

import copy
import logging
import re
import threading
import time
from typing import Any

from camera.ptz_client import PtzClient
from config import get_config
from utils.camera_storage import DEFAULT_PTZ_CONFIG, get_camera_storage
from utils.log_safety import safe_log_value as _slv

logger = logging.getLogger(__name__)

VALID_PTZ_MODES = {"preset", "hybrid"}
_AUTO_CAMERA_CACHE_TTL_SEC = 2.0
_AUTO_CAMERA_CACHE_SENTINEL = object()
_auto_camera_cache_lock = threading.Lock()
_auto_camera_cache_ts = 0.0
_auto_camera_cache_value: dict[str, Any] | None | object = _AUTO_CAMERA_CACHE_SENTINEL


def _float_in_range(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _int_in_range(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def normalize_ptz_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize auto-PTZ config from API/storage into a stable shape."""
    defaults = DEFAULT_PTZ_CONFIG
    raw = raw_config or {}
    mode = str(raw.get("mode") or defaults["mode"]).strip().lower()
    if mode not in VALID_PTZ_MODES:
        mode = defaults["mode"]

    zones = _normalize_zones(raw.get("zones"))
    if not zones:
        zones = [zone.copy() for zone in defaults["zones"]]

    return {
        "enabled": bool(raw.get("enabled", defaults["enabled"])),
        "mode": mode,
        "profile_index": _int_in_range(
            raw.get("profile_index"), defaults["profile_index"], 0, 8
        ),
        "overview_preset": str(raw.get("overview_preset") or "").strip(),
        "acquire_frames": _int_in_range(
            raw.get("acquire_frames"), defaults["acquire_frames"], 1, 10
        ),
        "lost_timeout_sec": _float_in_range(
            raw.get("lost_timeout_sec"), defaults["lost_timeout_sec"], 1.0, 60.0
        ),
        "manual_view_sec": _float_in_range(
            raw.get("manual_view_sec"), defaults["manual_view_sec"], 3.0, 300.0
        ),
        "settle_max_sec": _float_in_range(
            raw.get("settle_max_sec"), defaults["settle_max_sec"], 1.0, 30.0
        ),
        "command_cooldown_ms": _int_in_range(
            raw.get("command_cooldown_ms"),
            defaults["command_cooldown_ms"],
            100,
            10000,
        ),
        "deadband": _float_in_range(
            raw.get("deadband"), defaults["deadband"], 0.02, 0.4
        ),
        "max_speed": _float_in_range(
            raw.get("max_speed"), defaults["max_speed"], 0.05, 1.0
        ),
        "move_duration_ms": _int_in_range(
            raw.get("move_duration_ms"), defaults["move_duration_ms"], 50, 2000
        ),
        "zones": zones,
        "overview_snapshot_path": str(raw.get("overview_snapshot_path") or "").strip(),
        "preset_metadata": _normalize_preset_metadata(raw.get("preset_metadata")),
    }


def _normalize_preset_metadata(raw_meta: Any) -> dict[str, dict[str, Any]]:
    """Pass through per-preset overlay metadata without losing entries.

    The map is keyed by ONVIF preset token; values are clamped on the
    write path (update_preset_metadata) so we just shape-check here.
    """
    if not isinstance(raw_meta, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for token, meta in raw_meta.items():
        if not isinstance(meta, dict):
            continue
        out[str(token)] = {str(k): v for k, v in meta.items()}
    return out


def _normalize_zones(raw_zones: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_zones, list):
        return []

    zones: list[dict[str, Any]] = []
    for idx, raw_zone in enumerate(raw_zones):
        if not isinstance(raw_zone, dict):
            continue
        x_min = _float_in_range(raw_zone.get("x_min"), 0.0, 0.0, 1.0)
        y_min = _float_in_range(raw_zone.get("y_min"), 0.0, 0.0, 1.0)
        x_max = _float_in_range(raw_zone.get("x_max"), 1.0, 0.0, 1.0)
        y_max = _float_in_range(raw_zone.get("y_max"), 1.0, 0.0, 1.0)
        if x_max <= x_min or y_max <= y_min:
            continue
        zones.append(
            {
                "name": str(raw_zone.get("name") or f"zone_{idx + 1}").strip(),
                "preset": str(raw_zone.get("preset") or "").strip(),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )
    return zones


def get_ptz_config(camera_id: int) -> dict[str, Any] | None:
    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=False)
    if not camera:
        return None
    return normalize_ptz_config(camera.get("ptz"))


def update_ptz_config(
    camera_id: int, raw_config: dict[str, Any]
) -> dict[str, Any] | None:
    storage = get_camera_storage()
    existing = storage.get_camera(camera_id, include_password=False)
    if not existing:
        return None
    prev_overview = (
        normalize_ptz_config(existing.get("ptz") or {}).get("overview_preset") or ""
    )
    config = normalize_ptz_config(raw_config)
    if not storage.update_ptz_config(camera_id, config):
        return None
    clear_auto_ptz_camera_cache()

    new_overview = (config.get("overview_preset") or "").strip()
    if new_overview and new_overview != prev_overview:
        try:
            client = _client_for_camera(camera_id)
            client.goto_preset(preset_token=new_overview)
            home_ok = client.set_home_position()
            logger.info(
                "PTZ overview preset bound camera_id=%s preset=%s set_home=%s",
                _slv(camera_id),
                _slv(new_overview),
                home_ok,
            )
        except Exception as exc:
            logger.warning(
                "PTZ overview-preset bind partial failure camera_id=%s: %s",
                _slv(camera_id),
                exc,
            )
    return config


def list_presets(camera_id: int) -> list[dict[str, str]]:
    client = _client_for_camera(camera_id)
    presets = client.list_presets()
    return [{"token": preset.token, "name": preset.name} for preset in presets]


_GENERIC_PRESET_NAME = re.compile(r"^Preset\d{1,4}$")


def list_presets_with_metadata(
    camera_id: int, show_all: bool = False
) -> list[dict[str, Any]]:
    """Combine ONVIF presets with per-preset metadata from cameras.yaml.

    Filters out generic 'PresetNNN' slots unless show_all is True or a
    preset has stored metadata.
    """
    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=False)
    if not camera:
        return []

    metadata_by_token = (camera.get("ptz") or {}).get("preset_metadata") or {}
    if not isinstance(metadata_by_token, dict):
        metadata_by_token = {}

    presets = list_presets(camera_id)
    result: list[dict[str, Any]] = []
    for preset in presets:
        token = preset["token"]
        name = preset["name"]
        meta = metadata_by_token.get(token) or {}
        is_generic = bool(_GENERIC_PRESET_NAME.match(name)) and not meta
        if is_generic and not show_all:
            continue
        result.append(
            {
                "token": token,
                "name": name,
                "metadata": {
                    "label": str(meta.get("label") or ""),
                    "center_x_pct": float(meta.get("center_x_pct") or 0.0),
                    "center_y_pct": float(meta.get("center_y_pct") or 0.0),
                    "box_w_pct": float(meta.get("box_w_pct") or 0.0),
                    "box_h_pct": float(meta.get("box_h_pct") or 0.0),
                }
                if meta
                else None,
            }
        )
    return result


def set_preset_at_current_position(
    camera_id: int,
    name: str,
    *,
    preset_token: str | None = None,
    center_x_pct: float | None = None,
    center_y_pct: float | None = None,
    box_w_pct: float | None = None,
    box_h_pct: float | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    """SetPreset at the current camera position, persist optional metadata."""
    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return None

    client = _client_for_camera(camera_id)
    logger.info(
        "PTZ SetPreset camera_id=%s name=%s token=%s",
        _slv(camera_id),
        _slv(name),
        _slv(preset_token or ""),
    )
    token = client.set_preset(name=name, preset_token=preset_token)

    metadata: dict[str, Any] = {}
    if label is not None:
        metadata["label"] = str(label)
    for key, value in (
        ("center_x_pct", center_x_pct),
        ("center_y_pct", center_y_pct),
        ("box_w_pct", box_w_pct),
        ("box_h_pct", box_h_pct),
    ):
        if value is not None:
            metadata[key] = max(0.0, min(1.0, float(value)))

    if metadata:
        storage.update_preset_metadata(camera_id, token, metadata)
        clear_auto_ptz_camera_cache()
    return {"token": token, "name": name, "metadata": metadata or None}


def capture_overview_snapshot(camera_id: int) -> dict[str, Any] | None:
    """Fly to overview, fetch ONVIF snapshot, persist as Mini-Map background.

    Returns dict with relative_path (under OUTPUT_DIR) on success, None when
    the camera is missing or no overview preset is configured. Raises on
    ONVIF or HTTP errors.
    """
    import time as _time

    import requests
    from requests.auth import HTTPBasicAuth, HTTPDigestAuth

    from config import get_config as _get_app_config
    from utils.path_manager import get_path_manager

    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=True)
    if not camera:
        return None

    ptz_config = normalize_ptz_config(camera.get("ptz"))
    overview = str(ptz_config.get("overview_preset") or "")
    if not overview:
        raise ValueError("Overview preset is not configured")

    client = _client_for_camera(camera_id)
    logger.info(
        "PTZ snapshot capture camera_id=%s overview=%s",
        _slv(camera_id),
        _slv(overview),
    )
    client.goto_preset(preset_token=overview)
    _time.sleep(2.5)  # camera settling time before snapshot

    snapshot_uri = client.get_snapshot_uri()

    username = str(camera.get("username") or "")
    password = str(camera.get("password") or "")
    auth_variants: list[Any] = []
    if username:
        auth_variants.append(HTTPDigestAuth(username, password))
        auth_variants.append(HTTPBasicAuth(username, password))
    else:
        auth_variants.append(None)

    response = None
    last_exc: Exception | None = None
    for auth in auth_variants:
        try:
            response = requests.get(snapshot_uri, auth=auth, timeout=10)
            if response.status_code == 200:
                break
        except Exception as exc:
            last_exc = exc
            response = None
    if response is None or response.status_code != 200:
        if last_exc:
            raise last_exc
        raise RuntimeError(
            "Snapshot HTTP fetch failed with status "
            f"{response.status_code if response else 'no-response'}"
        )

    app_cfg = _get_app_config()
    pm = get_path_manager(str(app_cfg.get("OUTPUT_DIR") or ""))
    abs_path = pm.get_ptz_snapshot_path(camera_id, "overview")
    abs_path.write_bytes(response.content)

    relative = abs_path.relative_to(pm.base_dir).as_posix()
    storage.update_overview_snapshot_path(camera_id, relative)
    clear_auto_ptz_camera_cache()
    return {"relative_path": relative, "bytes": len(response.content)}


def set_auto_enabled(camera_id: int, enabled: bool) -> dict[str, Any] | None:
    """Toggle the auto-PTZ enabled flag without touching other config fields."""
    storage = get_camera_storage()
    existing = storage.get_camera(camera_id, include_password=False)
    if not existing:
        return None
    config = normalize_ptz_config(existing.get("ptz"))
    config["enabled"] = bool(enabled)
    if not storage.update_ptz_config(camera_id, config):
        return None
    clear_auto_ptz_camera_cache()
    return config


def update_preset_metadata_only(
    camera_id: int,
    preset_token: str,
    *,
    center_x_pct: float | None = None,
    center_y_pct: float | None = None,
    box_w_pct: float | None = None,
    box_h_pct: float | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    """Update per-preset UI metadata without touching the camera position."""
    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return None
    metadata: dict[str, Any] = {}
    if label is not None:
        metadata["label"] = str(label)
    for key, value in (
        ("center_x_pct", center_x_pct),
        ("center_y_pct", center_y_pct),
        ("box_w_pct", box_w_pct),
        ("box_h_pct", box_h_pct),
    ):
        if value is not None:
            metadata[key] = max(0.0, min(1.0, float(value)))
    if not metadata:
        return {"token": preset_token, "metadata": None}
    storage.update_preset_metadata(camera_id, preset_token, metadata)
    clear_auto_ptz_camera_cache()
    return {"token": preset_token, "metadata": metadata}


def remove_preset(camera_id: int, preset_token: str) -> bool:
    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return False
    client = _client_for_camera(camera_id)
    logger.info(
        "PTZ RemovePreset camera_id=%s token=%s",
        _slv(camera_id),
        _slv(preset_token),
    )
    client.remove_preset(preset_token)
    storage.delete_preset_metadata(camera_id, preset_token)
    clear_auto_ptz_camera_cache()
    return True


def goto_preset(camera_id: int, preset_token: str, speed: float | None = None) -> None:
    logger.info(
        "PTZ goto preset camera_id=%s preset=%s",
        _slv(camera_id),
        _slv(preset_token),
    )
    client = _client_for_camera(camera_id)
    client.goto_preset(preset_token=preset_token, speed=speed)


def continuous_move(
    camera_id: int,
    *,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    duration_ms: int = 250,
) -> None:
    logger.info(
        "PTZ move camera_id=%s pan=%.3f tilt=%.3f zoom=%.3f duration=%sms",
        _slv(camera_id),
        pan,
        tilt,
        zoom,
        duration_ms,
    )
    client = _client_for_camera(camera_id)
    client.continuous_move(
        pan=pan,
        tilt=tilt,
        zoom=zoom,
        duration_ms=duration_ms,
    )


def stop(camera_id: int) -> None:
    logger.info("PTZ stop camera_id=%s", _slv(camera_id))
    client = _client_for_camera(camera_id)
    client.stop()


def find_auto_ptz_camera() -> dict[str, Any] | None:
    """Return the first enabled PTZ camera matching the active stream URL."""
    global _auto_camera_cache_ts, _auto_camera_cache_value

    now = time.monotonic()
    with _auto_camera_cache_lock:
        cache_age = now - _auto_camera_cache_ts
        if (
            _auto_camera_cache_value is not _AUTO_CAMERA_CACHE_SENTINEL
            and cache_age <= _AUTO_CAMERA_CACHE_TTL_SEC
        ):
            if isinstance(_auto_camera_cache_value, dict):
                return copy.deepcopy(_auto_camera_cache_value)
            return None

    camera = _find_auto_ptz_camera_uncached()
    with _auto_camera_cache_lock:
        _auto_camera_cache_ts = now
        _auto_camera_cache_value = copy.deepcopy(camera) if camera else None
    return copy.deepcopy(camera) if camera else None


def clear_auto_ptz_camera_cache() -> None:
    """Force the auto-PTZ camera lookup to re-read persisted camera config."""
    global _auto_camera_cache_ts, _auto_camera_cache_value

    with _auto_camera_cache_lock:
        _auto_camera_cache_ts = 0.0
        _auto_camera_cache_value = _AUTO_CAMERA_CACHE_SENTINEL


def _find_auto_ptz_camera_uncached() -> dict[str, Any] | None:
    storage = get_camera_storage()
    cameras = storage._load_cameras()
    cfg = get_config()
    source_candidates = [
        str(cfg.get("VIDEO_SOURCE") or ""),
        str(cfg.get("CAMERA_URL") or ""),
    ]

    fallback: dict[str, Any] | None = None
    for camera_id, camera in enumerate(cameras):
        ptz_config = normalize_ptz_config(camera.get("ptz"))
        if not ptz_config.get("enabled"):
            continue
        camera = camera.copy()
        camera.pop("password", None)
        camera["id"] = camera_id
        camera["ptz"] = ptz_config

        ip = str(camera.get("ip") or "")
        if ip and any(ip in source for source in source_candidates):
            return camera
        if fallback is None:
            fallback = camera

    if fallback:
        logger.debug(
            "Auto PTZ has enabled camera %s but active source did not include its IP",
            _slv(fallback.get("id")),
        )
    return None


def find_any_ptz_camera() -> dict[str, Any] | None:
    """Return the first PTZ-capable camera regardless of enabled flag.

    Used by the status endpoint so the operator can toggle auto-return on
    again from the UI even when the YAML currently says enabled=false.
    """
    storage = get_camera_storage()
    cameras = storage._load_cameras()
    cfg = get_config()
    source_candidates = [
        str(cfg.get("VIDEO_SOURCE") or ""),
        str(cfg.get("CAMERA_URL") or ""),
    ]

    fallback: dict[str, Any] | None = None
    for camera_id, camera in enumerate(cameras):
        if not camera.get("supports_onvif", True):
            continue
        ptz_config = normalize_ptz_config(camera.get("ptz"))
        cam = camera.copy()
        cam.pop("password", None)
        cam["id"] = camera_id
        cam["ptz"] = ptz_config

        ip = str(cam.get("ip") or "")
        if ip and any(ip in source for source in source_candidates):
            return cam
        if fallback is None:
            fallback = cam
    return fallback


def _client_for_camera(camera_id: int) -> PtzClient:
    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=True)
    if not camera:
        raise ValueError(f"Camera {camera_id} not found")

    return PtzClient(
        ip=str(camera.get("ip") or ""),
        port=int(camera.get("port", 80)),
        username=str(camera.get("username") or ""),
        password=str(camera.get("password") or ""),
        profile_index=int(
            normalize_ptz_config(camera.get("ptz")).get("profile_index", 0)
        ),
    )
