"""
PTZ Core - camera PTZ use cases and configuration validation.
"""

import copy
import logging
import re
import threading
import time
from typing import Any

import yaml

from camera.ptz_client import PtzClient
from config import get_config
from utils.camera_storage import DEFAULT_PTZ_CONFIG, get_camera_storage
from utils.log_safety import safe_log_value as _slv
from utils.path_manager import get_path_manager

logger = logging.getLogger(__name__)

VALID_PTZ_MODES = {"preset", "hybrid", "grid", "follow"}
_AUTO_CAMERA_CACHE_TTL_SEC = 2.0
_AUTO_CAMERA_CACHE_SENTINEL = object()
_auto_camera_cache_lock = threading.Lock()
_auto_camera_cache_ts = 0.0
_auto_camera_cache_value: dict[str, Any] | None | object = _AUTO_CAMERA_CACHE_SENTINEL

# Capability-probe cache keyed by camera_id. The 60s TTL applies only to
# the declared ONVIF probe (GetServiceCapabilities + GetNodes). Empirical
# results are read fresh from OUTPUT_DIR/ptz_capabilities/cam<id>.yaml
# each time, so updated probe results are visible without bouncing the app.
_CAPABILITIES_CACHE_TTL_SEC = 60.0
_capabilities_cache_lock = threading.Lock()
_capabilities_cache: dict[int, tuple[float, dict[str, Any]]] = {}

# Persistent PtzClient cache keyed by camera_id. Each PtzClient lazily
# performs an ONVIF handshake (create_media_service + create_ptz_service +
# GetProfiles) on first use and caches the profile token on the instance.
#
# Coherence: the embedded ip/port/credentials/profile_index can change when
# camera config is edited, so this cache is dropped wherever the auto-PTZ
# camera cache is dropped (clear_auto_ptz_camera_cache) and on credential
# mutations. Robustness: any ONVIF error during a command evicts the client
# so the next command rebuilds a fresh connection (self-healing).
#
# Serialization: a per-camera lock guards command execution so manual moves
# and the AutoPtzController worker thread cannot interleave ContinuousMove/
# Stop calls on the same non-thread-safe ONVIF connection.
_ptz_client_cache_lock = threading.Lock()
_ptz_client_cache: dict[int, tuple[PtzClient, dict[str, Any]]] = {}
_ptz_command_locks: dict[int, threading.Lock] = {}

# Hold-to-move backpressure. When the operator holds a joystick button, the
# frontend POSTs /ptz/move every ~180ms, but each move blocks for its
# duration (ContinuousMove + sleep + Stop) under the per-camera lock. Without
# backpressure the moves pile up in the Waitress thread queue and the
# release-Stop lands behind them, so the camera keeps moving for seconds
# after release. Two coupled mechanisms fix this WITHOUT touching the
# frontend heartbeat or the dead-man-switch sleep:
#
#   1. Coalescing: at most one move may WAIT for the lock per camera. A move
#      arriving while one runs and one already waits is dropped before it
#      queues — back-to-back holds carry no new information.
#   2. Stop generation: every stop bumps a per-camera counter. A move records
#      the counter when it arrives; if a stop happened before the move
#      acquires the lock, the move skips its ONVIF call. A release-Stop thus
#      invalidates every move queued before it.
_ptz_move_state_lock = threading.Lock()
_ptz_move_waiting: dict[int, int] = {}  # camera_id -> moves currently waiting
_ptz_stop_generation: dict[int, int] = {}  # camera_id -> monotonic stop counter


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
        "grid_shape": list(_normalize_grid_shape(raw.get("grid_shape"))),
        "grid_cells": _normalize_grid_cells(raw.get("grid_cells")),
        "grid_command_cooldown_ms": _int_in_range(
            raw.get("grid_command_cooldown_ms"),
            int(defaults.get("grid_command_cooldown_ms", 4000)),
            500,
            30000,
        ),
        "grid_hysteresis_margin": _float_in_range(
            raw.get("grid_hysteresis_margin"),
            float(defaults.get("grid_hysteresis_margin", 0.05)),
            0.0,
            0.3,
        ),
        "grid_acquire_frames": _int_in_range(
            raw.get("grid_acquire_frames"),
            int(defaults.get("grid_acquire_frames", 1)),
            1,
            10,
        ),
        "follow_zoom_target_pct": _float_in_range(
            raw.get("follow_zoom_target_pct"),
            float(defaults.get("follow_zoom_target_pct", 0.18)),
            0.02,
            0.80,
        ),
        "follow_zoom_deadband_pct": _float_in_range(
            raw.get("follow_zoom_deadband_pct"),
            float(defaults.get("follow_zoom_deadband_pct", 0.05)),
            0.0,
            0.5,
        ),
        "follow_zoom_speed": _float_in_range(
            raw.get("follow_zoom_speed"),
            float(defaults.get("follow_zoom_speed", 0.3)),
            0.05,
            1.0,
        ),
        "follow_zoom_max_burst_sec": _float_in_range(
            raw.get("follow_zoom_max_burst_sec"),
            float(defaults.get("follow_zoom_max_burst_sec", 0.0)),
            0.0,
            30.0,
        ),
        "manual_pan_tilt_burst": _int_in_range(
            raw.get("manual_pan_tilt_burst"),
            int(defaults.get("manual_pan_tilt_burst", 1)),
            1,
            6,
        ),
        "manual_zoom_burst": _int_in_range(
            raw.get("manual_zoom_burst"),
            int(defaults.get("manual_zoom_burst", 1)),
            1,
            6,
        ),
        "manual_move_duration_multiplier": _float_in_range(
            raw.get("manual_move_duration_multiplier"),
            float(defaults.get("manual_move_duration_multiplier", 1.0)),
            0.5,
            5.0,
        ),
        "zones": zones,
        "overview_snapshot_path": str(raw.get("overview_snapshot_path") or "").strip(),
        "preset_metadata": _normalize_preset_metadata(raw.get("preset_metadata")),
    }


def _normalize_grid_shape(raw: Any) -> tuple[int, int]:
    """Re-export-style wrapper so the public boundary stays in ptz_core."""
    from core.ptz_grid import normalize_grid_shape

    return normalize_grid_shape(raw)


def _normalize_grid_cells(raw_cells: Any) -> dict[str, str]:
    """Shape-check the grid_cells map. Keys are 'r{row}_c{col}' strings.

    Schema:
      {"r0_c0": "<onvif_preset_token>", "r0_c1": "...", ...}
    Values that aren't non-empty strings are dropped silently — bad
    persisted state should degrade to "cell unset" rather than crash.
    """
    if not isinstance(raw_cells, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw_cells.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        token = value.strip()
        if not token:
            continue
        out[key] = token
    return out


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

    ptz_cfg = camera.get("ptz") or {}
    metadata_by_token = ptz_cfg.get("preset_metadata") or {}
    if not isinstance(metadata_by_token, dict):
        metadata_by_token = {}

    # Cheap PTZ cams routinely report all 256 ONVIF slot stubs as
    # existing, regardless of whether the operator ever set them. The
    # honest way to identify "in-use" presets is to take everything
    # WMB has a record of: preset_metadata entries + overview_preset +
    # grid_cells values. Any slot outside that set with a generic
    # PresetNNN name is treated as a Cam-side stub and filtered.
    in_use_tokens: set[str] = set(metadata_by_token.keys())
    overview = str(ptz_cfg.get("overview_preset") or "").strip()
    if overview:
        in_use_tokens.add(overview)
    grid_cells = ptz_cfg.get("grid_cells") or {}
    if isinstance(grid_cells, dict):
        for tok in grid_cells.values():
            tok = str(tok or "").strip()
            if tok:
                in_use_tokens.add(tok)

    presets = list_presets(camera_id)
    result: list[dict[str, Any]] = []
    for preset in presets:
        token = preset["token"]
        name = preset["name"]
        meta = metadata_by_token.get(token) or {}
        is_generic_stub = (
            bool(_GENERIC_PRESET_NAME.match(name))
            and not meta
            and token not in in_use_tokens
        )
        if is_generic_stub and not show_all:
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


def _ensure_grid_mode(camera_id: int) -> bool:
    """Flip ptz.mode to "grid" if a wizard action implied that intent.

    Without this, an operator who completes the grid wizard (set_shape +
    link_cells) but never opens the modal to flip mode persists every
    grid_cell into cameras.yaml while the controller keeps running the
    preset-dispatch path. The symptom is "wizard works, auto-PTZ
    doesn't" — exactly the failure pattern that hid grid mode for days.

    Idempotent: returns True only when an actual write happened, so
    callers can surface a "mode auto-set" hint to the UI.
    """
    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=False)
    if not camera:
        return False
    existing_ptz = dict(camera.get("ptz") or {})
    if str(existing_ptz.get("mode") or "").lower() == "grid":
        return False
    existing_ptz["mode"] = "grid"
    if not storage.update_ptz_config(camera_id, existing_ptz):
        return False
    clear_auto_ptz_camera_cache()
    logger.info(
        "PTZ mode auto-set to grid by wizard action camera_id=%s",
        _slv(camera_id),
    )
    return True


def set_grid_shape(camera_id: int, rows: int, cols: int) -> dict[str, Any] | None:
    """Set the operator-chosen grid shape for grid-mode auto-tracking.

    Validates the shape against `ptz_grid.ALLOWED_GRID_SHAPES`. Does NOT
    clear existing grid_cells — the operator may want to expand from
    2×3 to 3×3 and reuse overlapping cells. Cleanup is a separate call.
    """
    from core.ptz_grid import ALLOWED_GRID_SHAPES

    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return None
    if (int(rows), int(cols)) not in ALLOWED_GRID_SHAPES:
        raise ValueError(
            f"Grid shape ({rows}, {cols}) not in allowed set {ALLOWED_GRID_SHAPES}"
        )
    if not storage.set_grid_shape(camera_id, int(rows), int(cols)):
        return None
    clear_auto_ptz_camera_cache()
    mode_auto_set = _ensure_grid_mode(camera_id)
    logger.info(
        "PTZ grid shape set camera_id=%s shape=%dx%d",
        _slv(camera_id),
        rows,
        cols,
    )
    return {
        "rows": int(rows),
        "cols": int(cols),
        "mode_auto_set": mode_auto_set,
    }


def set_grid_cell_at_current_position(
    camera_id: int, row: int, col: int
) -> dict[str, Any] | None:
    """Save the current camera position as a grid cell preset.

    Creates a new ONVIF preset named `grid_r{row}_c{col}` at the
    camera's current pan/tilt/zoom, then maps that preset's token to
    the cell key in `cameras.yaml > ptz.grid_cells`. Returns the
    cell's data on success, None if the camera doesn't exist.

    Re-calling for an existing cell overwrites: SetPreset on most
    ONVIF cameras updates the slot in place; the cell-key mapping
    just gets re-written to the (same or new) token.
    """
    from core.ptz_grid import cell_preset_name

    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=False)
    if not camera:
        return None

    cell_key = f"r{int(row)}_c{int(col)}"
    name = cell_preset_name(row, col)

    # Reuse existing token if this cell was already set, so SetPreset
    # updates the slot in place rather than creating a duplicate.
    existing_cells = (camera.get("ptz") or {}).get("grid_cells") or {}
    existing_token = str(existing_cells.get(cell_key) or "").strip() or None

    client = _client_for_camera(camera_id)
    logger.info(
        "PTZ grid SetPreset camera_id=%s cell=%s existing_token=%s",
        _slv(camera_id),
        _slv(cell_key),
        _slv(existing_token or ""),
    )
    token = client.set_preset(name=name, preset_token=existing_token)

    if not storage.set_grid_cell(camera_id, cell_key, token):
        return None
    clear_auto_ptz_camera_cache()
    mode_auto_set = _ensure_grid_mode(camera_id)
    return {
        "cell_key": cell_key,
        "name": name,
        "preset_token": token,
        "mode_auto_set": mode_auto_set,
    }


def link_grid_cell_to_existing_preset(
    camera_id: int, row: int, col: int, preset_token: str
) -> dict[str, Any] | None:
    """Map a grid cell to an existing ONVIF preset without moving the camera.

    Lets the operator reuse presets that already exist on the camera
    (the operator-placed 1–7 zones, the overview Preset005, the
    Re-Focus Preset008, etc.) as grid-cell targets. Multiple cells can
    point at the same token; routing in `_handle_detections_grid` just
    reads `grid_cells[key]` and fires a goto — duplicate targets are
    a feature, not a bug.

    Validates that the preset token actually exists on the camera so a
    typo doesn't silently create a dangling reference.
    """
    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return None

    token = str(preset_token or "").strip()
    if not token:
        raise ValueError("preset_token is required")

    # Cheap existence check via list_presets — most cameras return
    # their full preset table in one call. If the token isn't there,
    # refuse the link rather than write a dangling reference.
    client = _client_for_camera(camera_id)
    available = {p.token for p in client.list_presets()}
    if token not in available:
        raise ValueError(
            f"Preset token {token!r} not found on the camera "
            f"(available: {sorted(available)[:8]}…)"
        )

    cell_key = f"r{int(row)}_c{int(col)}"
    if not storage.set_grid_cell(camera_id, cell_key, token):
        return None
    clear_auto_ptz_camera_cache()
    mode_auto_set = _ensure_grid_mode(camera_id)
    logger.info(
        "PTZ grid cell linked camera_id=%s cell=%s preset=%s",
        _slv(camera_id),
        _slv(cell_key),
        _slv(token),
    )
    return {
        "cell_key": cell_key,
        "preset_token": token,
        "mode": "linked",
        "mode_auto_set": mode_auto_set,
    }


def clear_grid_cell(camera_id: int, row: int, col: int) -> bool:
    """Remove a grid cell mapping. Does NOT delete the ONVIF preset.

    The ONVIF preset slot stays so a re-add can reuse it. If the
    operator wants the slot freed on the camera too, they can call
    remove_preset() with the token they got back from set_grid_cell.
    """
    storage = get_camera_storage()
    if not storage.get_camera(camera_id, include_password=False):
        return False
    cell_key = f"r{int(row)}_c{int(col)}"
    if not storage.delete_grid_cell(camera_id, cell_key):
        return False
    clear_auto_ptz_camera_cache()
    logger.info(
        "PTZ grid cell cleared camera_id=%s cell=%s", _slv(camera_id), _slv(cell_key)
    )
    return True


def get_grid_state(camera_id: int) -> dict[str, Any] | None:
    """Return current grid config: shape, cells mapped, cells missing.

    Used by the setup wizard UI to render "which cells need a preset
    saved" without the frontend having to do its own math.
    """
    from core.ptz_grid import normalize_grid_shape, required_cell_count

    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=False)
    if not camera:
        return None
    ptz = normalize_ptz_config(camera.get("ptz"))
    shape = normalize_grid_shape(ptz.get("grid_shape"))
    cells: dict[str, str] = ptz.get("grid_cells") or {}
    rows, cols = shape
    expected_keys = [f"r{r}_c{c}" for r in range(rows) for c in range(cols)]
    missing = [k for k in expected_keys if k not in cells]
    return {
        "shape": list(shape),
        "rows": rows,
        "cols": cols,
        "cells": {k: cells[k] for k in expected_keys if k in cells},
        "missing": missing,
        "total_required": required_cell_count(shape),
        "total_set": len(cells),
        "mode_active": ptz.get("mode") == "grid",
    }


def goto_preset(camera_id: int, preset_token: str, speed: float | None = None) -> None:
    logger.info(
        "PTZ goto preset camera_id=%s preset=%s",
        _slv(camera_id),
        _slv(preset_token),
    )
    generation = _current_stop_generation(camera_id)
    _run_ptz_command(
        camera_id,
        lambda client: client.goto_preset(preset_token=preset_token, speed=speed),
        skip_if_superseded=generation,
    )


def continuous_move(
    camera_id: int,
    *,
    pan: float = 0.0,
    tilt: float = 0.0,
    zoom: float = 0.0,
    duration_ms: int = 250,
) -> None:
    # Coalesce: at most one move may wait for the lock. If a move is already
    # running and another already waiting, this one carries no new info while
    # the button is held — drop it before it queues so the backlog (and thus
    # the post-release run-on) cannot build.
    if not _reserve_move_slot(camera_id):
        logger.debug("PTZ move coalesced (slot busy) camera_id=%s", _slv(camera_id))
        return

    generation = _current_stop_generation(camera_id)
    logger.info(
        "PTZ move camera_id=%s pan=%.3f tilt=%.3f zoom=%.3f duration=%sms",
        _slv(camera_id),
        pan,
        tilt,
        zoom,
        duration_ms,
    )
    try:
        _run_ptz_command(
            camera_id,
            lambda client: client.continuous_move(
                pan=pan,
                tilt=tilt,
                zoom=zoom,
                duration_ms=duration_ms,
            ),
            skip_if_superseded=generation,
        )
    finally:
        _release_move_slot(camera_id)


def stop(camera_id: int) -> None:
    # Bump the stop generation FIRST so any move already waiting for the lock
    # sees that it has been superseded and skips its ONVIF call when it wakes.
    _bump_stop_generation(camera_id)
    logger.info("PTZ stop camera_id=%s", _slv(camera_id))
    _run_ptz_command(camera_id, lambda client: client.stop())


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

    # The PtzClient cache embeds ip/port/credentials/profile_index, which can
    # change in the same edit that invalidated the camera-config cache. Drop
    # cached clients so the next command rebuilds against current config.
    clear_ptz_client_cache()


def _connection_fingerprint(camera: dict[str, Any]) -> dict[str, Any]:
    """Connection-identifying fields. A change here means the cached client
    points at the wrong endpoint/credentials and must be rebuilt."""
    return {
        "ip": str(camera.get("ip") or ""),
        "port": int(camera.get("port", 80)),
        "username": str(camera.get("username") or ""),
        "password": str(camera.get("password") or ""),
        "profile_index": int(
            normalize_ptz_config(camera.get("ptz")).get("profile_index", 0)
        ),
    }


def _command_lock_for_camera(camera_id: int) -> threading.Lock:
    """Return the per-camera command lock, creating it on first use.

    Serializes ONVIF command execution so manual moves and the
    AutoPtzController worker cannot interleave ContinuousMove/Stop on the
    same non-thread-safe connection.
    """
    with _ptz_client_cache_lock:
        lock = _ptz_command_locks.get(camera_id)
        if lock is None:
            lock = threading.Lock()
            _ptz_command_locks[camera_id] = lock
        return lock


def _get_cached_client(camera_id: int) -> PtzClient:
    """Return a cached PtzClient for the camera, rebuilding if absent or if
    the persisted connection fingerprint changed since it was cached."""
    fingerprint = _connection_fingerprint(_camera_or_raise(camera_id))
    with _ptz_client_cache_lock:
        entry = _ptz_client_cache.get(camera_id)
        if entry is not None and entry[1] == fingerprint:
            return entry[0]
        client = PtzClient(
            ip=fingerprint["ip"],
            port=fingerprint["port"],
            username=fingerprint["username"],
            password=fingerprint["password"],
            profile_index=fingerprint["profile_index"],
        )
        _ptz_client_cache[camera_id] = (client, fingerprint)
        return client


def _current_stop_generation(camera_id: int) -> int:
    with _ptz_move_state_lock:
        return _ptz_stop_generation.get(camera_id, 0)


def _bump_stop_generation(camera_id: int) -> None:
    with _ptz_move_state_lock:
        _ptz_stop_generation[camera_id] = _ptz_stop_generation.get(camera_id, 0) + 1


# At most one move running plus one move waiting for the lock. A third
# concurrent move carries no new information while the button is held and is
# coalesced away. Keeping one waiting (rather than zero) keeps held-button
# motion fluid: the next move is already queued to fire the instant the
# running one finishes, so there is no per-step gap.
_MAX_MOVES_IN_FLIGHT = 2


def _reserve_move_slot(camera_id: int) -> bool:
    """Try to admit a move into the in-flight set (running + waiting).

    Returns True if this move may proceed to the lock, False if the camera
    already has one move running and one waiting (this one is coalesced
    away), so a held button cannot build an unbounded backlog.
    """
    with _ptz_move_state_lock:
        in_flight = _ptz_move_waiting.get(camera_id, 0)
        if in_flight >= _MAX_MOVES_IN_FLIGHT:
            return False
        _ptz_move_waiting[camera_id] = in_flight + 1
        return True


def _release_move_slot(camera_id: int) -> None:
    with _ptz_move_state_lock:
        waiting = _ptz_move_waiting.get(camera_id, 0)
        if waiting <= 1:
            _ptz_move_waiting.pop(camera_id, None)
        else:
            _ptz_move_waiting[camera_id] = waiting - 1


def _run_ptz_command(
    camera_id: int, action, *, skip_if_superseded: int | None = None
) -> None:
    """Run one PTZ command against the cached client under the per-camera lock.

    ``skip_if_superseded`` is the stop generation captured when the command
    was issued. If a stop bumped the generation while this command waited for
    the lock, the command is dropped without touching the camera — this is how
    a release-Stop cancels moves that were queued before it.

    On any error the cached client is evicted so the next command rebuilds a
    fresh ONVIF connection — credentials may have rotated, the socket may have
    dropped, or the camera may have rebooted.
    """
    lock = _command_lock_for_camera(camera_id)
    with lock:
        if (
            skip_if_superseded is not None
            and _current_stop_generation(camera_id) != skip_if_superseded
        ):
            logger.debug("PTZ command superseded by stop camera_id=%s", _slv(camera_id))
            return
        client = _get_cached_client(camera_id)
        try:
            action(client)
        except Exception:
            clear_ptz_client_cache(camera_id)
            raise


def clear_ptz_client_cache(camera_id: int | None = None) -> None:
    """Drop cached PtzClient instances.

    ``camera_id=None`` clears all entries (used on config reload and in tests).
    The per-camera command locks are intentionally retained: a lock may be
    held by an in-flight command, and Lock objects are cheap to keep.
    """
    with _ptz_client_cache_lock:
        if camera_id is None:
            _ptz_client_cache.clear()
        else:
            _ptz_client_cache.pop(int(camera_id), None)


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


def _camera_or_raise(camera_id: int) -> dict[str, Any]:
    storage = get_camera_storage()
    camera = storage.get_camera(camera_id, include_password=True)
    if not camera:
        raise ValueError(f"Camera {camera_id} not found")
    return camera


def _client_for_camera(camera_id: int) -> PtzClient:
    """Return the cached PtzClient for a camera (rebuilt on config change).

    Kept as the module-internal accessor for one-shot callers (e.g. applying
    a new overview preset). Routes through the same cache as the hot command
    path so a single connection is shared.
    """
    return _get_cached_client(camera_id)


def probe_capabilities(
    camera_id: int, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Probe declared PTZ capabilities for a saved camera (cached 60s).

    Returns a plain dict with the shape:

        {
            "camera_id": <int>,
            "probed_at": <unix-seconds-float>,
            "from_cache": <bool>,
            "ip": <str>,           # for the Settings UI subtitle
            "service_capabilities": {move_status, status_position, eflip, reverse} | None,
            "nodes": [...],
            "declared": {
                continuous_pan_tilt, continuous_zoom,
                relative_pan_tilt, relative_zoom,
                absolute_pan_tilt, absolute_zoom,
                home_position, move_status, status_position, max_presets
            },
            "error": <str | None>,  # populated if ONVIF subcall failed
        }

    Read-only — issues no PTZ move commands. Safe to call against any
    reachable cam. Empirical move tests live in the standalone
    ``scripts.ptz_probe`` tool; this function is the
    ``GetServiceCapabilities`` + ``GetNodes``-grade declared view only.

    Cache is per-process, 60s TTL. ``force_refresh=True`` bypasses
    the cache (used by the Settings UI's "Re-probe" button).
    """
    cam_id_int = int(camera_id)

    if not force_refresh:
        with _capabilities_cache_lock:
            entry = _capabilities_cache.get(cam_id_int)
            if entry is not None:
                ts, cached = entry
                if (time.monotonic() - ts) < _CAPABILITIES_CACHE_TTL_SEC:
                    result = copy.deepcopy(cached)
                    result["from_cache"] = True
                    return result

    # Look up the camera to extract IP for the UI subtitle and to fail
    # fast with a clear ValueError when the id is unknown.
    storage = get_camera_storage()
    camera = storage.get_camera(cam_id_int, include_password=False)
    if not camera:
        raise ValueError(f"Camera {cam_id_int} not found")

    # Reuse the existing client-construction path so tests can mock the
    # ONVIF surface the same way they do for goto/move/etc.
    client = _client_for_camera(cam_id_int)
    probed = client.get_capabilities()
    cam_ip = str(camera.get("ip") or "")
    result: dict[str, Any] = {
        "camera_id": cam_id_int,
        "probed_at": time.time(),
        "from_cache": False,
        "ip": cam_ip,
        **probed,
    }

    # Empirical augmentation: the standalone PTZ probe tool writes
    # operator-attended results to OUTPUT_DIR/ptz_capabilities/cam<id>.yaml.
    # The in-UI probe writes to the same path. Either way, the core just
    # reads: it never writes the empirical file. If the file is missing,
    # empirical is None and the UI shows yellow ?-pills.
    result["empirical"] = _load_empirical_from_disk(cam_id_int)

    with _capabilities_cache_lock:
        _capabilities_cache[cam_id_int] = (time.monotonic(), copy.deepcopy(result))

    return result


def clear_capabilities_cache(camera_id: int | None = None) -> None:
    """Drop cached capability-probe results.

    ``camera_id=None`` clears all entries (used in tests).
    """
    with _capabilities_cache_lock:
        if camera_id is None:
            _capabilities_cache.clear()
        else:
            _capabilities_cache.pop(int(camera_id), None)


def _load_empirical_from_disk(camera_id: int) -> dict[str, Any] | None:
    """Read the empirical-capabilities file written by the PTZ probe tool.

    Path: ``OUTPUT_DIR/ptz_capabilities/cam<id>.yaml``. The operator
    runs ``python -m scripts.ptz_probe --camera-id <id>`` (from the
    WMB repo, see ``scripts/ptz_probe/README.md``) to produce this
    file. The in-UI probe writes to the same path from inside WMB.

    Expected file shape (the probe tool's responsibility to produce):

        camera_id: 0
        probed_at: 2026-05-17T23:59:27Z
        connection:
          ip: 192.168.1.100
        empirical:
          continuous_works: true
          relative_works:   false
          absolute_works:   false
          movestatus_transitions: false
        recommended_strategy: continuous_pulse

    Returns the normalised empirical dict (boolean fields coerced,
    extra metadata preserved) or None when the file is missing, empty,
    or unreadable. None means "yellow ?-pills" for the UI.
    """
    try:
        pm = get_path_manager()
        # get_ptz_capabilities_path() mkdirs the parent — harmless when
        # the file doesn't yet exist; the existence check below handles
        # the empty-dir case.
        path = pm.get_ptz_capabilities_path(camera_id)
    except Exception as exc:  # noqa: BLE001 — PathManager init can fail in tests
        logger.debug(
            "PathManager unavailable for capability disk load (cam %s): %s",
            _slv(str(camera_id)),
            exc,
        )
        return None

    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.debug("Unreadable PTZ capability cache %s: %s", _slv(str(path)), exc)
        return None

    empirical = data.get("empirical") or {}
    if not empirical:
        return None

    def _coerce_tristate(raw: Any) -> bool | None:
        """YAML null → Python None ("not tested"). Bool → bool. Anything
        else (string, missing-via-default-None) → None for safety —
        better to render the pill as 'not tested' than fake-confirm a
        capability we can't read.
        """
        if raw is None:
            return None
        if isinstance(raw, bool):
            return raw
        return None

    result: dict[str, Any] = {
        "continuous_works": _coerce_tristate(empirical.get("continuous_works")),
        "relative_works": _coerce_tristate(empirical.get("relative_works")),
        "absolute_works": _coerce_tristate(empirical.get("absolute_works")),
        "movestatus_transitions": _coerce_tristate(
            empirical.get("movestatus_transitions")
        ),
        "recommended_strategy": str(data.get("recommended_strategy") or ""),
        "report_timestamp": str(data.get("probed_at") or ""),
        "report_path": str(path),
    }
    # Optional fields written by the in-UI probe wizard's near-focus
    # step. Whitelisted explicitly so the loader stays strict — never
    # forward unknown empirical keys without an explicit decision.
    if "follow_zoom_max_burst_sec" in empirical:
        try:
            result["follow_zoom_max_burst_sec"] = float(
                empirical["follow_zoom_max_burst_sec"]
            )
        except (TypeError, ValueError):
            # Malformed value in the YAML — silently drop and keep the
            # default; the rest of the empirical block is still useful.
            pass
    return result
