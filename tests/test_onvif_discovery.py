"""
Unit tests for ONVIF Discovery module.
Uses mocks to test without actual network/camera access.
"""

from unittest.mock import MagicMock, patch

from camera.onvif_discovery import ONVIFDiscovery


class TestONVIFDiscovery:
    """Tests for ONVIFDiscovery class."""

    def test_init_creates_empty_discovered_list(self):
        """Discovery initializes with empty camera list."""
        discovery = ONVIFDiscovery()
        assert discovery.last_discovered == []

    @patch("camera.onvif_discovery.ThreadedWSDiscovery")
    def test_discover_cameras_empty_network(self, mock_wsd_class):
        """Discovery returns empty list when no cameras found."""
        mock_wsd = MagicMock()
        mock_wsd.searchServices.return_value = []
        mock_wsd_class.return_value = mock_wsd

        discovery = ONVIFDiscovery()
        result = discovery.discover_cameras(timeout=1)

        assert result == []
        mock_wsd.start.assert_called_once()
        mock_wsd.stop.assert_called_once()

    @patch("camera.onvif_discovery.ThreadedWSDiscovery")
    def test_discover_cameras_finds_camera(self, mock_wsd_class):
        """Discovery correctly parses found camera."""
        # Create mock service
        mock_service = MagicMock()
        mock_service.getXAddrs.return_value = [
            "http://192.168.1.100:80/onvif/device_service"
        ]
        mock_service.getScopes.return_value = [
            "onvif://www.onvif.org/name/TestCam",
            "onvif://www.onvif.org/hardware/Hikvision",
        ]

        mock_wsd = MagicMock()
        mock_wsd.searchServices.return_value = [mock_service]
        mock_wsd_class.return_value = mock_wsd

        discovery = ONVIFDiscovery()
        result = discovery.discover_cameras(timeout=1)

        assert len(result) == 1
        assert result[0]["ip"] == "192.168.1.100"
        assert result[0]["port"] == 80
        assert result[0]["name"] == "TestCam"
        assert result[0]["manufacturer"] == "Hikvision"

    @patch("camera.onvif_discovery.ThreadedWSDiscovery")
    def test_discover_cameras_handles_discovery_error(self, mock_wsd_class):
        """Discovery handles errors gracefully."""
        mock_wsd = MagicMock()
        mock_wsd.start.side_effect = Exception("Network error")
        mock_wsd_class.return_value = mock_wsd

        discovery = ONVIFDiscovery()
        result = discovery.discover_cameras(timeout=1)

        assert result == []

    def test_parse_service_returns_none_for_empty_xaddrs(self):
        """_parse_service returns None when no addresses."""
        mock_service = MagicMock()
        mock_service.getXAddrs.return_value = []

        discovery = ONVIFDiscovery()
        result = discovery._parse_service(mock_service)

        assert result is None

    def test_extract_scope_value_finds_name(self):
        """_extract_scope_value extracts name from scope URI."""
        discovery = ONVIFDiscovery()
        scopes = ["onvif://www.onvif.org/name/MyCameraName"]

        result = discovery._extract_scope_value(scopes, "name")

        assert result == "MyCameraName"

    def test_extract_scope_value_returns_empty_when_not_found(self):
        """_extract_scope_value returns empty string when key not found."""
        discovery = ONVIFDiscovery()
        scopes = ["onvif://www.onvif.org/type/video_encoder"]

        result = discovery._extract_scope_value(scopes, "name")

        assert result == ""

    def test_last_discovered_returns_copy(self):
        """last_discovered returns a copy, not the internal list."""
        discovery = ONVIFDiscovery()
        discovery._discovered_cameras = [{"ip": "1.2.3.4"}]

        result = discovery.last_discovered
        result.append({"ip": "5.6.7.8"})

        assert len(discovery._discovered_cameras) == 1

    @patch("camera.onvif_discovery.ONVIFCamera")
    def test_get_camera_details_returns_info(self, mock_camera_class):
        """get_camera_details returns device info dict."""
        mock_camera = MagicMock()
        mock_device_info = MagicMock()
        mock_device_info.Manufacturer = "Hikvision"
        mock_device_info.Model = "DS-2CD2343G0"
        mock_device_info.FirmwareVersion = "5.6.0"
        mock_device_info.SerialNumber = "ABC123"

        mock_capabilities = MagicMock()
        mock_capabilities.PTZ = MagicMock()
        mock_capabilities.Media = MagicMock()

        mock_camera.devicemgmt.GetDeviceInformation.return_value = mock_device_info
        mock_camera.devicemgmt.GetCapabilities.return_value = mock_capabilities
        mock_camera_class.return_value = mock_camera

        discovery = ONVIFDiscovery()
        result = discovery.get_camera_details("192.168.1.100", 80)

        assert result["ip"] == "192.168.1.100"
        assert result["manufacturer"] == "Hikvision"
        assert result["model"] == "DS-2CD2343G0"
        assert result["has_ptz"] is True
        assert result["has_media"] is True

    @patch("camera.onvif_discovery.ONVIFCamera")
    def test_get_camera_details_handles_error(self, mock_camera_class):
        """get_camera_details returns None on connection error."""
        from onvif import ONVIFError

        mock_camera_class.side_effect = ONVIFError("Connection refused")

        discovery = ONVIFDiscovery()
        result = discovery.get_camera_details("192.168.1.100", 80)

        assert result is None

    @patch("camera.onvif_discovery.ONVIFCamera")
    def test_get_stream_uri_returns_rtsp_url(self, mock_camera_class):
        """get_stream_uri returns formatted RTSP URI."""
        mock_camera = MagicMock()
        mock_media = MagicMock()

        mock_profile = MagicMock()
        mock_profile.token = "profile_1"
        mock_media.GetProfiles.return_value = [mock_profile]

        mock_uri_response = MagicMock()
        mock_uri_response.Uri = "rtsp://192.168.1.100:554/stream1"
        mock_media.GetStreamUri.return_value = mock_uri_response

        mock_camera.create_media_service.return_value = mock_media
        mock_camera_class.return_value = mock_camera

        discovery = ONVIFDiscovery()
        result = discovery.get_stream_uri("192.168.1.100", 80)

        assert result == "rtsp://192.168.1.100:554/stream1"

    @patch("camera.onvif_discovery.ONVIFCamera")
    def test_get_stream_uri_injects_credentials(self, mock_camera_class):
        """get_stream_uri includes credentials in URI when provided."""
        mock_camera = MagicMock()
        mock_media = MagicMock()

        mock_profile = MagicMock()
        mock_profile.token = "profile_1"
        mock_media.GetProfiles.return_value = [mock_profile]

        mock_uri_response = MagicMock()
        mock_uri_response.Uri = "rtsp://192.168.1.100:554/stream1"
        mock_media.GetStreamUri.return_value = mock_uri_response

        mock_camera.create_media_service.return_value = mock_media
        mock_camera_class.return_value = mock_camera

        discovery = ONVIFDiscovery()
        result = discovery.get_stream_uri(
            "192.168.1.100", 80, username="admin", password="secret123"
        )

        assert "admin:secret123@" in result
        assert "192.168.1.100" in result
