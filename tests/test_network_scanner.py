from unittest.mock import MagicMock, patch

import pytest
from zeep.cache import InMemoryCache

from camera import network_scanner as ns_module
from camera.network_scanner import NetworkScanner


def _build_camera_with_stream_uri(uri: str) -> MagicMock:
    cam = MagicMock()
    media = MagicMock()
    profile = MagicMock()
    profile.token = "profile_1"
    media.GetProfiles.return_value = [profile]
    uri_resp = MagicMock()
    uri_resp.Uri = uri
    media.GetStreamUri.return_value = uri_resp
    cam.create_media_service.return_value = media
    return cam


def test_create_onvif_camera_uses_resolved_wsdl(monkeypatch):
    scanner = NetworkScanner()
    monkeypatch.setattr(scanner, "_resolve_onvif_wsdl_dir", lambda: "/tmp/wsdl")

    with patch("camera.network_scanner.ONVIFCamera") as onvif_cls:
        scanner._create_onvif_camera("198.51.100.10", 80, "viewer", "example-password")
        onvif_cls.assert_called_once_with(
            "198.51.100.10",
            80,
            "viewer",
            "example-password",
            wsdl_dir="/tmp/wsdl",
            transport=ns_module._ZEEP_TRANSPORT,
        )


def test_create_onvif_camera_without_wsdl_uses_default(monkeypatch):
    scanner = NetworkScanner()
    monkeypatch.setattr(scanner, "_resolve_onvif_wsdl_dir", lambda: None)

    with patch("camera.network_scanner.ONVIFCamera") as onvif_cls:
        scanner._create_onvif_camera("198.51.100.10", 80, "viewer", "example-password")
        onvif_cls.assert_called_once_with(
            "198.51.100.10",
            80,
            "viewer",
            "example-password",
            transport=ns_module._ZEEP_TRANSPORT,
        )


def test_zeep_transport_uses_inmemory_cache():
    # zeep's default SqliteCache writes to /tmp/.../zeep, which crashes
    # in containers where the parent dir is root-owned (e.g. fontconfig
    # pre-creates /tmp/fontconfig at image build, PUID runtime user
    # can't os.makedirs underneath it). InMemoryCache must be used so
    # ONVIFCamera() construction never touches the filesystem cache.
    assert isinstance(ns_module._ZEEP_TRANSPORT.cache, InMemoryCache)


def test_get_stream_uri_falls_back_to_other_onvif_ports(monkeypatch):
    scanner = NetworkScanner()
    attempted_ports: list[int] = []

    def _fake_create(ip: str, port: int, user: str, password: str):
        attempted_ports.append(port)
        if port == 80:
            return _build_camera_with_stream_uri("rtsp://198.51.100.10:554/stream1")
        raise RuntimeError("connection failed")

    monkeypatch.setattr(scanner, "_create_onvif_camera", _fake_create)

    uri = scanner.get_stream_uri("198.51.100.10", 8080, "viewer", "example-password")

    assert attempted_ports[0] == 8080
    assert 80 in attempted_ports
    assert uri == "rtsp://viewer:example-password@198.51.100.10:554/stream1"


def test_get_stream_uri_raises_after_all_fallback_ports_fail(monkeypatch):
    scanner = NetworkScanner()
    monkeypatch.setattr(
        scanner,
        "_create_onvif_camera",
        lambda ip, port, user, password: (_ for _ in ()).throw(
            RuntimeError(f"fail-{port}")
        ),
    )

    with patch("camera.network_scanner.logger") as log_mock:
        with pytest.raises(RuntimeError, match="port fallback"):
            scanner.get_stream_uri("198.51.100.10", 8080, "viewer", "example-password")
        log_mock.error.assert_called()
