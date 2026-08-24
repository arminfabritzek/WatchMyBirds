"""ONVIF Imaging focus control tests."""

from typing import Any
from unittest.mock import MagicMock, patch

from camera.ptz_client import PtzClient


class _Attr:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _focus_client(imaging: MagicMock) -> PtzClient:
    client = PtzClient(ip="0.0.0.0", port=80, username="u", password="p")
    client._ensure_imaging_service = lambda: (  # type: ignore[method-assign]
        imaging,
        "VideoSourceToken",
    )
    return client


def test_imaging_service_uses_profile_video_source_token():
    camera = MagicMock()
    profile = _Attr(
        token="ProfileToken",
        VideoSourceConfiguration=_Attr(SourceToken="VideoSourceToken"),
    )
    camera.create_media_service.return_value.GetProfiles.return_value = [profile]
    imaging = camera.create_imaging_service.return_value

    client = PtzClient(ip="192.0.2.5", port=80, username="u", password="p")
    with patch.object(client, "_create_camera", return_value=camera):
        service, source_token = client._ensure_imaging_service()

    assert service is imaging
    assert source_token == "VideoSourceToken"


def test_focus_capabilities_decode_move_and_autofocus_options():
    imaging = MagicMock()
    imaging.create_type.side_effect = lambda name: MagicMock()
    imaging.GetMoveOptions.return_value = _Attr(
        Continuous=_Attr(Speed=_Attr(Min=-1.0, Max=1.0)),
        Relative=None,
        Absolute=None,
    )
    imaging.GetOptions.return_value = _Attr(
        Focus=_Attr(AutoFocusModes=["AUTO", "MANUAL"])
    )

    capabilities = _focus_client(imaging).get_focus_capabilities()

    assert capabilities == {
        "continuous": True,
        "relative": False,
        "absolute": False,
        "autofocus": True,
        "speed_min": -1.0,
        "speed_max": 1.0,
    }


def test_continuous_focus_builds_move_and_stops(monkeypatch):
    imaging = MagicMock()
    move_request = MagicMock()
    stop_request = MagicMock()
    imaging.create_type.side_effect = [move_request, stop_request]
    monkeypatch.setattr("camera.ptz_client.time.sleep", MagicMock())

    _focus_client(imaging).continuous_focus(speed=3.0, duration_ms=700)

    assert move_request.VideoSourceToken == "VideoSourceToken"
    assert move_request.Focus == {"Continuous": {"Speed": 1.0}}
    imaging.Move.assert_called_once_with(move_request)
    assert stop_request.VideoSourceToken == "VideoSourceToken"
    imaging.Stop.assert_called_once_with(stop_request)


def test_relative_focus_builds_one_shot_move_without_stop():
    imaging = MagicMock()
    request = MagicMock()
    imaging.create_type.return_value = request

    _focus_client(imaging).relative_focus(distance=-0.08, speed=0.5)

    assert request.VideoSourceToken == "VideoSourceToken"
    assert request.Focus == {"Relative": {"Distance": -0.08, "Speed": 0.5}}
    imaging.Move.assert_called_once_with(request)
    imaging.Stop.assert_not_called()


def test_focus_stop_falls_back_to_zero_speed_move_when_unimplemented():
    imaging = MagicMock()
    stop_request = MagicMock()
    move_request = MagicMock()
    imaging.create_type.side_effect = [stop_request, move_request]
    imaging.Stop.side_effect = RuntimeError("Action Not Implemented")

    _focus_client(imaging).stop_focus()

    assert move_request.VideoSourceToken == "VideoSourceToken"
    assert move_request.Focus == {"Continuous": {"Speed": 0.0}}
    imaging.Move.assert_called_once_with(move_request)


def test_set_autofocus_uses_partial_imaging_settings():
    imaging = MagicMock()
    request = MagicMock()
    imaging.create_type.return_value = request

    _focus_client(imaging).set_autofocus(True)

    assert request.VideoSourceToken == "VideoSourceToken"
    assert request.ImagingSettings == {"Focus": {"AutoFocusMode": "AUTO"}}
    assert request.ForcePersistence is False
    imaging.SetImagingSettings.assert_called_once_with(request)
