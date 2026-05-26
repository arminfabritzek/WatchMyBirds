"""Tests for the Phase-2a PtzClient extensions.

PTZ probe client-extension tests.
Adds relative_move, absolute_move, emergency_stop, poll_move_status,
get_status (structured), get_device_info to the production PtzClient
so the empirical probe wizard + the standalone CLI tool share one
ONVIF surface instead of duplicating.

These tests exercise the new methods against a mocked ONVIF service —
no real cam is touched. Pattern follows test_ptz_capabilities_service.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from camera.ptz_client import DeviceInfo, PtzClient, StatusSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Attr:
    """Stand-in for zeep attribute-access objects."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _patched_client(ptz_svc: MagicMock) -> PtzClient:
    """Build a PtzClient with the ONVIF service short-circuited."""
    client = PtzClient(ip="0.0.0.0", port=80, username="u", password="p")
    client._ensure_services = lambda: (ptz_svc, "ProfileToken")  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# relative_move
# ---------------------------------------------------------------------------


def test_relative_move_builds_translation_request():
    ptz = MagicMock()
    # ptz.create_type returns a fresh object that the method populates.
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.relative_move(pan=0.05, tilt=-0.02, zoom=0.1, speed=0.5)

    ptz.create_type.assert_called_once_with("RelativeMove")
    assert request_obj.ProfileToken == "ProfileToken"
    assert request_obj.Translation == {
        "PanTilt": {"x": 0.05, "y": -0.02},
        "Zoom": {"x": 0.1},
    }
    assert request_obj.Speed == {
        "PanTilt": {"x": 0.5, "y": 0.5},
        "Zoom": {"x": 0.5},
    }
    ptz.RelativeMove.assert_called_once_with(request_obj)


def test_relative_move_clamps_out_of_range_inputs():
    """Inputs outside [-1, 1] get clamped — the ONVIF generic translation
    space won't accept e.g. pan=2.0 anyway."""
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.relative_move(pan=2.0, tilt=-3.0, zoom=0.5)

    assert request_obj.Translation == {
        "PanTilt": {"x": 1.0, "y": -1.0},
        "Zoom": {"x": 0.5},
    }


def test_relative_move_without_speed_omits_speed_field():
    """When speed=None, the Speed field is left as the mock's default
    (we don't write it). Lets cheap cams that error on unexpected
    Speed parameter still work."""
    ptz = MagicMock()
    request_obj = MagicMock()
    # Ensure Speed isn't pre-set by setting it to a sentinel; if our
    # method doesn't touch it, the sentinel stays.
    request_obj.Speed = "UNTOUCHED"
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.relative_move(pan=0.1)

    assert request_obj.Speed == "UNTOUCHED"


# ---------------------------------------------------------------------------
# absolute_move
# ---------------------------------------------------------------------------


def test_absolute_move_builds_position_request():
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.absolute_move(pan=0.3, tilt=0.0, zoom=0.0, speed=0.5)

    ptz.create_type.assert_called_once_with("AbsoluteMove")
    assert request_obj.Position == {
        "PanTilt": {"x": 0.3, "y": 0.0},
        "Zoom": {"x": 0.0},
    }
    ptz.AbsoluteMove.assert_called_once_with(request_obj)


def test_absolute_move_clamps_inputs():
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.absolute_move(pan=99.0, tilt=-50.0, zoom=2.5)

    assert request_obj.Position == {
        "PanTilt": {"x": 1.0, "y": -1.0},
        "Zoom": {"x": 1.0},
    }


# ---------------------------------------------------------------------------
# emergency_stop
# ---------------------------------------------------------------------------


def test_emergency_stop_calls_stop_with_both_axes():
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    client.emergency_stop()

    ptz.create_type.assert_called_once_with("Stop")
    assert request_obj.PanTilt is True
    assert request_obj.Zoom is True
    ptz.Stop.assert_called_once_with(request_obj)


def test_emergency_stop_swallows_exceptions_from_underlying_stop():
    """Emergency-stop is called from signal handlers + abort paths.
    A raise here would leave the cam spinning into an endstop."""
    ptz = MagicMock()
    ptz.Stop.side_effect = RuntimeError("network down")
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    client = _patched_client(ptz)

    # Must not raise.
    client.emergency_stop()


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


def test_get_status_decodes_full_response():
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    ptz.GetStatus.return_value = _Attr(
        Position=_Attr(
            PanTilt=_Attr(x=0.15, y=-0.05),
            Zoom=_Attr(x=0.4),
        ),
        MoveStatus=_Attr(PanTilt="MOVING", Zoom="IDLE"),
        UtcTime="2026-05-18T11:00:00Z",
    )
    client = _patched_client(ptz)

    sample = client.get_status()

    assert isinstance(sample, StatusSample)
    assert sample.pan == 0.15
    assert sample.tilt == -0.05
    assert sample.zoom == 0.4
    assert sample.move_status_pan_tilt == "MOVING"
    assert sample.move_status_zoom == "IDLE"
    assert sample.utc_time == "2026-05-18T11:00:00Z"
    assert sample.error is None


def test_get_status_handles_stub_cam_with_no_position():
    """The operator's cam returns no Position field (StatusPosition=False
    in service caps). get_status must return None for position fields,
    not raise."""
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    ptz.GetStatus.return_value = _Attr(
        Position=None,
        MoveStatus=_Attr(PanTilt="IDLE", Zoom="IDLE"),
        UtcTime="2026-05-18T11:00:00Z",
    )
    client = _patched_client(ptz)

    sample = client.get_status()

    assert sample.pan is None
    assert sample.tilt is None
    assert sample.zoom is None
    assert sample.move_status_pan_tilt == "IDLE"


def test_get_status_captures_error_in_sample_not_raised():
    """Malformed firmware can fault GetStatus. The probe needs partial
    data, not an exception — error is recorded in sample.error."""
    ptz = MagicMock()
    ptz.create_type.return_value = MagicMock()
    ptz.GetStatus.side_effect = RuntimeError("SOAP fault")
    client = _patched_client(ptz)

    sample = client.get_status()

    assert sample.error is not None
    assert "SOAP fault" in sample.error
    assert sample.pan is None


# ---------------------------------------------------------------------------
# poll_move_status
# ---------------------------------------------------------------------------


def test_poll_move_status_returns_sequence_of_samples():
    """poll_move_status drives get_status repeatedly. Returned list
    length is roughly duration / interval."""
    ptz = MagicMock()
    request_obj = MagicMock()
    ptz.create_type.return_value = request_obj
    ptz.GetStatus.return_value = _Attr(
        Position=None,
        MoveStatus=_Attr(PanTilt="IDLE", Zoom="IDLE"),
        UtcTime="t",
    )
    client = _patched_client(ptz)

    # 0.3 seconds at 0.05 cadence → ~6 samples (allow some slack for clock).
    samples = client.poll_move_status(duration_sec=0.3, interval_sec=0.05)

    assert len(samples) >= 3  # at least 3 even on the slowest CI runner
    assert all(isinstance(s, StatusSample) for s in samples)
    assert all(s.move_status_pan_tilt == "IDLE" for s in samples)


def test_poll_move_status_returns_empty_for_zero_duration():
    """Defensive: 0s window means no samples, no calls."""
    ptz = MagicMock()
    client = _patched_client(ptz)

    samples = client.poll_move_status(duration_sec=0.0)

    assert samples == []
    ptz.GetStatus.assert_not_called()


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------


def test_get_device_info_decodes_response():
    """Standard happy-path — GetDeviceInformation fills the dataclass."""
    ptz = MagicMock()
    client = PtzClient(ip="0.0.0.0", port=80, username="u", password="p")
    client._ensure_services = lambda: (ptz, "ProfileToken")  # type: ignore[method-assign]

    # _camera is set by _ensure_services in real life; here we stub it.
    fake_camera = MagicMock()
    fake_device_service = MagicMock()
    fake_device_service.GetDeviceInformation.return_value = _Attr(
        Manufacturer="IPCAM",
        Model="C6F0SoZ3N0PmL2",
        FirmwareVersion="V24.11.46.6.28-20240806",
        SerialNumber="9803CF674AC8",
        HardwareId="V24.11.46.6.28-20240806",
    )
    fake_camera.create_devicemgmt_service.return_value = fake_device_service
    client._camera = fake_camera

    info = client.get_device_info()

    assert isinstance(info, DeviceInfo)
    assert info.manufacturer == "IPCAM"
    assert info.model == "C6F0SoZ3N0PmL2"
    assert info.firmware_version == "V24.11.46.6.28-20240806"
    assert info.serial_number == "9803CF674AC8"


def test_get_device_info_returns_empty_on_failure():
    """If the cam doesn't support GetDeviceInformation, return an empty
    DeviceInfo rather than raising."""
    ptz = MagicMock()
    client = PtzClient(ip="0.0.0.0", port=80, username="u", password="p")
    client._ensure_services = lambda: (ptz, "ProfileToken")  # type: ignore[method-assign]

    fake_camera = MagicMock()
    fake_camera.create_devicemgmt_service.side_effect = RuntimeError(
        "devicemgmt not supported"
    )
    client._camera = fake_camera

    info = client.get_device_info()

    assert info == DeviceInfo("", "", "", "", "")
