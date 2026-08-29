from unittest.mock import patch

from core.ptz_core import (
    clear_auto_ptz_camera_cache,
    find_auto_ptz_camera,
    list_presets_with_metadata,
    normalize_ptz_config,
)


def test_normalize_ptz_config_clamps_tracking_values():
    config = normalize_ptz_config(
        {
            "enabled": True,
            "lost_timeout_sec": -4,
            "command_cooldown_ms": 1,
            "deadband": 2,
            "max_speed": 4,
            "move_duration_ms": 9999,
            "mode": "hybrid",
            "grid_cells": {"r0_c0": "Preset001"},
            "zones": [{"preset": "Preset002"}],
        }
    )

    assert config["enabled"] is True
    assert config["lost_timeout_sec"] == 1.0
    assert config["command_cooldown_ms"] == 100
    assert config["deadband"] == 0.4
    assert config["max_speed"] == 1.0
    assert config["move_duration_ms"] == 2000
    assert "mode" not in config
    assert "grid_cells" not in config
    assert "zones" not in config


def test_normalize_ptz_config_manual_burst_defaults_and_clamp():
    """Manual joystick burst multipliers default to 1 and clamp to [1, 6].

    These fields are operator-facing knobs for cheap PTZ cams that
    ignore ContinuousMove velocity — fresh installs must see 1/1
    (legacy single-call behaviour) so nothing changes for users who
    don't need the asymmetry fix.
    """
    config = normalize_ptz_config({})
    assert config["manual_pan_tilt_burst"] == 1
    assert config["manual_zoom_burst"] == 1

    config = normalize_ptz_config({
        "manual_pan_tilt_burst": 3,
        "manual_zoom_burst": 2,
    })
    assert config["manual_pan_tilt_burst"] == 3
    assert config["manual_zoom_burst"] == 2

    config = normalize_ptz_config({
        "manual_pan_tilt_burst": 0,
        "manual_zoom_burst": 99,
    })
    assert config["manual_pan_tilt_burst"] == 1
    assert config["manual_zoom_burst"] == 6


def test_normalize_ptz_config_move_duration_multiplier_defaults_and_clamp():
    """Per-cam move-duration multiplier defaults to 1.0 and clamps to [0.5, 5.0].

    Independent from manual_pan_tilt_burst — operator may turn either or
    both knobs depending on which cam-firmware quirk they're working
    around (per-call cooldown vs. velocity-ignored-but-each-call-honoured).
    """
    config = normalize_ptz_config({})
    assert config["manual_move_duration_multiplier"] == 1.0

    config = normalize_ptz_config({"manual_move_duration_multiplier": 2.5})
    assert config["manual_move_duration_multiplier"] == 2.5

    config = normalize_ptz_config({"manual_move_duration_multiplier": 0.1})
    assert config["manual_move_duration_multiplier"] == 0.5

    config = normalize_ptz_config({"manual_move_duration_multiplier": 99.0})
    assert config["manual_move_duration_multiplier"] == 5.0


def test_normalize_ptz_config_follow_zoom_defaults_and_clamp():
    """Auto-follow zoom fields default sanely and clamp to safe ranges."""
    config = normalize_ptz_config({})
    assert config["follow_zoom_target_pct"] == 0.18
    assert config["follow_zoom_deadband_pct"] == 0.05
    assert config["follow_zoom_speed"] == 0.3

    # Out-of-range values clamp to bounds (zoom on cheap cams ignores
    # extremes anyway, but the controller should never enqueue a value
    # outside ONVIF's [-1, 1] range).
    config = normalize_ptz_config({
        "follow_zoom_target_pct": 5.0,
        "follow_zoom_deadband_pct": -1.0,
        "follow_zoom_speed": 99.0,
    })
    assert config["follow_zoom_target_pct"] == 0.80
    assert config["follow_zoom_deadband_pct"] == 0.0
    assert config["follow_zoom_speed"] == 1.0


def test_find_auto_ptz_camera_strips_password():
    """The cached auto-PTZ camera dict must never carry the raw password.

    The status route and the 2 s controller cache both surface the dict
    that find_auto_ptz_camera() returns; a password leak here would land
    in the API response and in process RAM.
    """

    class _FakeStorage:
        def _load_cameras(self):
            return [
                {
                    "ip": "192.168.1.50",
                    "username": "admin",
                    "password": "s3cret",
                    "ptz": {"enabled": True, "overview_preset": "home"},
                }
            ]

    clear_auto_ptz_camera_cache()
    try:
        with (
            patch(
                "core.ptz_core.get_camera_storage", return_value=_FakeStorage()
            ),
            patch(
                "core.ptz_core.get_config",
                return_value={"VIDEO_SOURCE": "rtsp://192.168.1.50/stream"},
            ),
        ):
            camera = find_auto_ptz_camera()
    finally:
        clear_auto_ptz_camera_cache()

    assert camera is not None
    assert "password" not in camera
    assert camera["ip"] == "192.168.1.50"
    assert camera["id"] == 0


# ---------------------------------------------------------------------------
# list_presets_with_metadata filter tests
# Cheap PTZ cams report all 256 ONVIF slot stubs as existing. The filter
# must show only operator-configured presets — anything in
# preset_metadata or overview_preset — and hide the
# generic stubs. This is what the empirical-probe wizard relies on so
# the operator sees a sane preset list instead of 256 fake entries.
# ---------------------------------------------------------------------------


def _stub_storage_for_presets(metadata=None, overview=""):
    """Build a CameraStorage mock that exposes a single camera with
    the given ptz config."""
    from unittest.mock import MagicMock

    cam = {
        "id": 0,
        "ip": "10.0.0.1",
        "port": 80,
        "username": "u",
        "password": "p",
        "ptz": {
            "enabled": True,
            "overview_preset": overview,
            "preset_metadata": metadata or {},
        },
    }
    storage = MagicMock()
    storage.get_camera.return_value = cam
    return storage


def _stub_256_generic_presets():
    """Mimic a cheap cam that reports 256 PresetNNN stubs."""
    return [
        {"token": f"Preset{i:03d}", "name": f"Preset{i:03d}"}
        for i in range(1, 257)
    ]


def test_list_presets_filters_generic_stubs_by_default():
    """Cam reports 256 stubs, operator configured 0 → show 0 (not 256)."""
    with (
        patch(
            "core.ptz_core.get_camera_storage",
            return_value=_stub_storage_for_presets(),
        ),
        patch(
            "core.ptz_core.list_presets",
            return_value=_stub_256_generic_presets(),
        ),
    ):
        result = list_presets_with_metadata(0, show_all=False)
    assert result == [], (
        "no operator-configured presets and all-generic ONVIF names "
        "must yield an empty list (filtering all 256 stubs)"
    )


def test_list_presets_includes_metadata_entries():
    """Generic stubs that DO have preset_metadata entries are kept."""
    metadata = {
        "Preset001": {"label": "1"},
        "Preset005": {"label": "Feeder"},
    }
    with (
        patch(
            "core.ptz_core.get_camera_storage",
            return_value=_stub_storage_for_presets(metadata=metadata),
        ),
        patch(
            "core.ptz_core.list_presets",
            return_value=_stub_256_generic_presets(),
        ),
    ):
        result = list_presets_with_metadata(0, show_all=False)
    tokens = {p["token"] for p in result}
    assert tokens == {"Preset001", "Preset005"}


def test_list_presets_includes_overview_preset_even_without_metadata():
    """overview_preset is operator intent — show it even when no
    preset_metadata entry exists for that token. Was a regression
    before this fix: cams whose operator set overview but never added
    metadata for it ended up with overview hidden from the wizard."""
    with (
        patch(
            "core.ptz_core.get_camera_storage",
            return_value=_stub_storage_for_presets(overview="Preset020"),
        ),
        patch(
            "core.ptz_core.list_presets",
            return_value=_stub_256_generic_presets(),
        ),
    ):
        result = list_presets_with_metadata(0, show_all=False)
    tokens = {p["token"] for p in result}
    assert "Preset020" in tokens


def test_list_presets_show_all_returns_every_stub():
    """show_all=True is the diagnostic mode — Settings power users may
    want to see every ONVIF-reported slot even if WMB hasn't tagged
    it. Filter must NOT trigger in that mode."""
    with (
        patch(
            "core.ptz_core.get_camera_storage",
            return_value=_stub_storage_for_presets(),
        ),
        patch(
            "core.ptz_core.list_presets",
            return_value=_stub_256_generic_presets(),
        ),
    ):
        result = list_presets_with_metadata(0, show_all=True)
    assert len(result) == 256


def test_list_presets_union_of_metadata_and_overview():
    """Metadata plus overview contribute to the in-use set without duplicates."""
    metadata = {"Preset001": {"label": "1"}, "Preset002": {"label": "2"}}
    with (
        patch(
            "core.ptz_core.get_camera_storage",
            return_value=_stub_storage_for_presets(
                metadata=metadata,
                overview="Preset020",
            ),
        ),
        patch(
            "core.ptz_core.list_presets",
            return_value=_stub_256_generic_presets(),
        ),
    ):
        result = list_presets_with_metadata(0, show_all=False)
    tokens = {p["token"] for p in result}
    assert tokens == {"Preset001", "Preset002", "Preset020"}
