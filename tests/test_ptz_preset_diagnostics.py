"""Preset diagnostics retain SOAP codes without dumping sensitive messages."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from onvif.exceptions import ONVIFError
from zeep.exceptions import Fault

from camera.ptz_client import PtzClient, _soap_fault_codes


def client_and_service() -> tuple[PtzClient, MagicMock]:
    service = MagicMock()
    service.create_type.side_effect = lambda _: SimpleNamespace()
    client = PtzClient("192.0.2.1", 80, "operator", "secret-password")
    client._ensure_services = lambda: (service, "profile-a")
    client._ptz_configuration_token = "ptz-a"
    return client, service


def test_wrapped_fault_retains_code_and_excludes_secret(caplog):
    client, service = client_and_service()

    def reject(request: object) -> None:
        try:
            raise Fault("secret-password", code="Sender", subcodes=["NoToken"])
        except Fault as exc:
            raise ONVIFError(exc)  # noqa: B904 - reproduce python-onvif wrapper

    service.GotoPreset.side_effect = reject
    with caplog.at_level(logging.INFO), pytest.raises(ONVIFError) as caught:
        client.goto_preset("Preset022")
    assert _soap_fault_codes(caught.value) == ("Sender", "NoToken")
    assert "soap_subcodes=NoToken" in caplog.text
    assert "profile=profile-a" in caplog.text
    assert "secret-password" not in caplog.text
    assert "event=accepted" not in caplog.text
    assert service.GotoPreset.call_count == 1
    service.GetStatus.assert_not_called()
    service.GetPresets.assert_not_called()


def test_success_is_acceptance_not_arrival_and_speed_is_omitted(caplog):
    client, service = client_and_service()
    with caplog.at_level(logging.INFO):
        client.stop()
        client.goto_preset("Preset022")
    request = service.GotoPreset.call_args.args[0]
    assert vars(request) == {"ProfileToken": "profile-a", "PresetToken": "Preset022"}
    assert "previous_command=Stop" in caplog.text
    assert "event=accepted" in caplog.text
    assert "arrival_verified=false" in caplog.text
    service.GetStatus.assert_not_called()


def test_diagnostics_neutralize_camera_supplied_linebreaks(caplog):
    client, service = client_and_service()
    with caplog.at_level(logging.INFO):
        client.goto_preset("preset\nforged")
    assert "preset\\nforged" in caplog.text
    assert "preset\nforged" not in caplog.text
    assert service.GotoPreset.call_args.args[0].PresetToken == "preset\nforged"


def test_transport_exception_is_preserved_without_message_dump(caplog):
    client, service = client_and_service()
    error = TimeoutError("http://operator:secret-password@camera/")
    service.GotoPreset.side_effect = error
    with caplog.at_level(logging.INFO), pytest.raises(TimeoutError) as caught:
        client.goto_preset("Preset022")
    assert caught.value is error
    assert "error_type=TimeoutError" in caplog.text
    assert "soap_code=unknown" in caplog.text
    assert "secret-password" not in caplog.text


def test_cyclic_exception_chain_terminates():
    error = RuntimeError("untrusted")
    error.__context__ = error
    assert _soap_fault_codes(error) == ("unknown", "")
