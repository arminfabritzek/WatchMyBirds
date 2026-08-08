"""Tests for XAddr host rewriting and ONVIF transport timeouts.

Some camera firmware advertises service XAddrs built from a stale
internally-configured address rather than the address the camera is
actually reachable on. python-onvif trusts those URLs verbatim for every
service except devicemgmt, so media/ptz calls are dispatched to an
unroutable host and block until the OS gives up on the SYN.

These tests pin both halves of the mitigation: rewrite the advertised
host to the address we connected to, and never dial without a timeout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from camera.ptz_client import _ZEEP_TRANSPORT, PtzClient, _rewrite_xaddrs_to_host


def _camera_with_xaddrs(xaddrs: dict[str, str]) -> MagicMock:
    camera = MagicMock()
    camera.xaddrs = dict(xaddrs)
    return camera


# ---------------------------------------------------------------------------
# _rewrite_xaddrs_to_host
# ---------------------------------------------------------------------------


def test_rewrite_replaces_foreign_host_keeping_path_and_port():
    camera = _camera_with_xaddrs(
        {"ns/media": "http://203.0.113.88:8080/onvif/media_service"}
    )

    rewritten = _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert rewritten == 1
    assert camera.xaddrs["ns/media"] == (
        "http://198.51.100.10:8080/onvif/media_service"
    )


def test_rewrite_covers_every_advertised_service():
    camera = _camera_with_xaddrs(
        {
            "ns/analytics": "http://203.0.113.88:8080/onvif/Analytics",
            "ns/events": "http://203.0.113.88:8080/onvif/event_service",
            "ns/imaging": "http://203.0.113.88:8080/onvif/image_service",
            "ns/media": "http://203.0.113.88:8080/onvif/media_service",
            "ns/ptz": "http://203.0.113.88:8080/onvif/ptz_service",
        }
    )

    rewritten = _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert rewritten == 5
    assert all("198.51.100.10" in url for url in camera.xaddrs.values())
    assert not any("203.0.113.88" in url for url in camera.xaddrs.values())


def test_rewrite_preserves_port_advertised_by_camera():
    """A camera may serve ONVIF on a port other than the one we connected to."""
    camera = _camera_with_xaddrs(
        {"ns/media": "http://203.0.113.88:8899/onvif/media_service"}
    )

    _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert camera.xaddrs["ns/media"] == (
        "http://198.51.100.10:8899/onvif/media_service"
    )


def test_rewrite_is_noop_when_host_already_correct():
    camera = _camera_with_xaddrs(
        {"ns/media": "http://198.51.100.10:8080/onvif/media_service"}
    )

    rewritten = _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert rewritten == 0
    assert camera.xaddrs["ns/media"] == (
        "http://198.51.100.10:8080/onvif/media_service"
    )


def test_rewrite_tolerates_missing_or_empty_xaddrs():
    assert _rewrite_xaddrs_to_host(_camera_with_xaddrs({}), "10.0.0.1", 80) == 0

    camera = MagicMock()
    camera.xaddrs = None
    assert _rewrite_xaddrs_to_host(camera, "10.0.0.1", 80) == 0


def test_rewrite_skips_unparsable_entries_without_raising():
    camera = _camera_with_xaddrs(
        {
            "ns/broken": "",
            "ns/media": "http://203.0.113.88:8080/onvif/media_service",
        }
    )

    rewritten = _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert rewritten == 1
    assert camera.xaddrs["ns/broken"] == ""


def test_rewrite_uses_fallback_port_when_url_omits_one():
    camera = _camera_with_xaddrs(
        {"ns/media": "http://203.0.113.88/onvif/media_service"}
    )

    _rewrite_xaddrs_to_host(camera, "198.51.100.10", 8080)

    assert camera.xaddrs["ns/media"] == (
        "http://198.51.100.10:8080/onvif/media_service"
    )


# ---------------------------------------------------------------------------
# Transport timeout
# ---------------------------------------------------------------------------


def test_shared_transport_declares_finite_timeouts():
    """A None timeout pins the calling thread for the full SYN retry budget."""
    assert _ZEEP_TRANSPORT.operation_timeout is not None
    assert _ZEEP_TRANSPORT.operation_timeout > 0
    assert _ZEEP_TRANSPORT.load_timeout is not None
    assert _ZEEP_TRANSPORT.load_timeout > 0


# ---------------------------------------------------------------------------
# Wiring into _ensure_services
# ---------------------------------------------------------------------------


def test_ensure_services_rewrites_xaddrs_before_creating_media_service():
    """The rewrite must land before any non-devicemgmt service is built."""
    camera = _camera_with_xaddrs(
        {"ns/media": "http://203.0.113.88:8080/onvif/media_service"}
    )
    seen: dict[str, str] = {}

    def _capture_media_service():
        seen.update(camera.xaddrs)
        return MagicMock(GetProfiles=lambda: [MagicMock(token="ProfileToken")])

    camera.create_media_service.side_effect = _capture_media_service

    client = PtzClient(ip="198.51.100.10", port=8080, username="u", password="p")
    with patch.object(client, "_create_camera", return_value=camera):
        client._ensure_services()

    assert seen["ns/media"] == "http://198.51.100.10:8080/onvif/media_service"


def test_ensure_services_returns_profile_token_after_rewrite():
    camera = _camera_with_xaddrs(
        {"ns/ptz": "http://203.0.113.88:8080/onvif/ptz_service"}
    )
    profile = MagicMock()
    profile.token = "MainStreamProfileToken"
    camera.create_media_service.return_value = MagicMock(GetProfiles=lambda: [profile])
    ptz_service = MagicMock()
    camera.create_ptz_service.return_value = ptz_service

    client = PtzClient(ip="198.51.100.10", port=8080, username="u", password="p")
    with patch.object(client, "_create_camera", return_value=camera):
        ptz, token = client._ensure_services()

    assert ptz is ptz_service
    assert token == "MainStreamProfileToken"
    assert camera.xaddrs["ns/ptz"] == "http://198.51.100.10:8080/onvif/ptz_service"
