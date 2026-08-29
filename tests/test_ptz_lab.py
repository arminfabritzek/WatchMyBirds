"""Offline tests for the read-only ONVIF PTZ laboratory helpers."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.ptz_lab.common import (
    READ_ONLY_OPERATIONS,
    analyse_status_samples,
    call_readonly,
    extract_space_ranges,
    sanitize,
)


def test_sanitize_redacts_credentials_and_sensitive_keys():
    payload = {
        "uri": (
            "rtsp://admin:secret@192.0.2.5:554/live/ch0?channel=1&auth_token=private"
        ),
        "password": "secret",
        "nested": {"username": "admin", "token": "Profile001"},
    }

    sanitized = sanitize(payload)

    assert sanitized["uri"] == (
        "rtsp://192.0.2.5:554/live/ch0?channel=1&auth_token=%3Credacted%3E"
    )
    assert sanitized["password"] == "<redacted>"
    assert sanitized["nested"]["username"] == "<redacted>"
    assert sanitized["nested"]["token"] == "Profile001"


def test_sanitize_converts_onvif_like_objects_recursively():
    payload = SimpleNamespace(
        URI="http://192.0.2.5/onvif/ptz_service",
        XRange=SimpleNamespace(Min=-1.0, Max=1.0),
    )

    sanitized = sanitize(payload)

    assert sanitized == {
        "URI": "http://192.0.2.5/onvif/ptz_service",
        "XRange": {"Max": 1.0, "Min": -1.0},
    }


def test_call_readonly_marks_not_supported_faults_without_raising():
    def unsupported():
        raise RuntimeError("ter:ActionNotSupported")

    result = call_readonly("GetConfigurationOptions", unsupported)

    assert result["status"] == "UNSUPPORTED"
    assert result["operation"] == "GetConfigurationOptions"
    assert "ActionNotSupported" in result["error"]


def test_readonly_allowlist_excludes_all_movement_operations():
    forbidden = {
        "AbsoluteMove",
        "ContinuousMove",
        "GotoHomePosition",
        "GotoPreset",
        "RelativeMove",
        "RemovePreset",
        "SetHomePosition",
        "SetPreset",
        "Stop",
    }

    assert READ_ONLY_OPERATIONS.isdisjoint(forbidden)


def test_extract_space_ranges_preserves_uri_and_axis_ranges():
    options = {
        "Spaces": {
            "RelativePanTiltTranslationSpace": [
                {
                    "URI": "http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationSpaceFov",
                    "XRange": {"Min": -1.0, "Max": 1.0},
                    "YRange": {"Min": -0.5, "Max": 0.5},
                }
            ],
            "ContinuousZoomVelocitySpace": {
                "URI": "http://www.onvif.org/ver10/tptz/ZoomSpaces/VelocityGenericSpace",
                "XRange": {"Min": -1.0, "Max": 1.0},
            },
        }
    }

    spaces = extract_space_ranges(options)

    by_kind = {space["kind"]: space for space in spaces}
    relative = by_kind["RelativePanTiltTranslationSpace"]
    assert relative["x"] == {"min": -1.0, "max": 1.0}
    assert relative["y"] == {"min": -0.5, "max": 0.5}
    assert relative["is_fov"] is True
    assert "ContinuousZoomVelocitySpace" in by_kind


def test_analyse_status_samples_distinguishes_observed_from_changing_position():
    samples = [
        {
            "status": "SUPPORTED",
            "value": {
                "Position": {
                    "PanTilt": {"x": 0.1, "y": -0.2},
                    "Zoom": {"x": 0.0},
                },
                "MoveStatus": {"PanTilt": "IDLE", "Zoom": "IDLE"},
            },
        },
        {
            "status": "SUPPORTED",
            "value": {
                "Position": {
                    "PanTilt": {"x": 0.1, "y": -0.2},
                    "Zoom": {"x": 0.0},
                },
                "MoveStatus": {"PanTilt": "IDLE", "Zoom": "IDLE"},
            },
        },
    ]

    summary = analyse_status_samples(samples)

    assert summary["successful_samples"] == 2
    assert summary["position_reported"] is True
    assert summary["position_changed"] is False
    assert summary["all_positions_zero"] is False
    assert summary["position_assessment"] == "OBSERVED_STATIC"
    assert summary["move_status_values"] == ["IDLE"]


def test_analyse_status_samples_flags_all_zero_position_as_inconclusive():
    samples = [
        {
            "status": "SUPPORTED",
            "value": {
                "Position": {
                    "PanTilt": {"x": 0.0, "y": 0.0},
                    "Zoom": {"x": 0.0},
                },
                "MoveStatus": {"PanTilt": "IDLE", "Zoom": "IDLE"},
            },
        }
    ]

    summary = analyse_status_samples(samples)

    assert summary["all_positions_zero"] is True
    assert summary["position_assessment"] == "INCONCLUSIVE_ZERO_STUB"
