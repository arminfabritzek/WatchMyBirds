"""
ONVIF Camera Discovery Module.
Discovers ONVIF cameras on the local network using WS-Discovery.

Usage:
    from camera.onvif_discovery import ONVIFDiscovery

    discovery = ONVIFDiscovery()
    cameras = discovery.discover_cameras()
    for cam in cameras:
        print(f"{cam['ip']}:{cam['port']} - {cam.get('name', 'Unknown')}")
"""

import logging
import threading
from urllib.parse import urlparse, urlunparse

from onvif import ONVIFCamera, ONVIFError
from wsdiscovery.discovery import ThreadedWSDiscovery

logger = logging.getLogger(__name__)


class ONVIFDiscovery:
    """Discovers and manages ONVIF cameras on the network."""

    DISCOVERY_TIMEOUT = 5  # seconds

    def __init__(self):
        self._discovered_cameras: list[dict] = []
        self._discovery_lock = threading.Lock()

    def discover_cameras(self, timeout: int | None = None) -> list[dict]:
        """
        Scans the network for ONVIF cameras using WS-Discovery.

        Args:
            timeout: Discovery timeout in seconds (default: 5)

        Returns:
            List of camera info dicts with keys:
            - ip: str
            - port: int
            - name: str (if available)
            - manufacturer: str (if available)
            - xaddr: str (ONVIF device address)
            - scopes: List[str]
        """
        timeout = timeout or self.DISCOVERY_TIMEOUT
        discovered = []

        wsd = None
        try:
            wsd = ThreadedWSDiscovery()
            wsd.start()

            # Search for ONVIF video encoders
            services = wsd.searchServices(
                scopes=["onvif://www.onvif.org/type/video_encoder"],
                timeout=timeout,
            )

            for service in services:
                camera_info = self._parse_service(service)
                if camera_info:
                    discovered.append(camera_info)

        except Exception as e:
            logger.error(f"ONVIF Discovery failed: {e}")
        finally:
            if wsd:
                try:
                    wsd.stop()
                except Exception:
                    pass

        with self._discovery_lock:
            self._discovered_cameras = discovered

        logger.info(f"Discovered {len(discovered)} ONVIF cameras")
        return discovered

    def _parse_service(self, service) -> dict | None:
        """Parses WS-Discovery service into camera info dict."""
        try:
            xaddrs = service.getXAddrs()
            if not xaddrs:
                return None

            # Parse first address (usually http://ip:port/onvif/device_service)
            addr = xaddrs[0]
            parsed = urlparse(addr)

            return {
                "ip": parsed.hostname,
                "port": parsed.port or 80,
                "xaddr": addr,
                "scopes": [str(s) for s in service.getScopes()],
                "name": self._extract_scope_value(service.getScopes(), "name"),
                "manufacturer": self._extract_scope_value(
                    service.getScopes(), "hardware"
                ),
            }
        except Exception as e:
            logger.debug(f"Failed to parse service: {e}")
            return None

    def _extract_scope_value(self, scopes, key: str) -> str:
        """Extracts value from ONVIF scope URIs."""
        for scope in scopes:
            scope_str = str(scope)
            if f"/{key}/" in scope_str.lower():
                return scope_str.split("/")[-1]
        return ""

    def get_camera_details(
        self,
        ip: str,
        port: int = 80,
        username: str = "",
        password: str = "",
    ) -> dict | None:
        """
        Connects to an ONVIF camera and retrieves detailed information.

        Args:
            ip: Camera IP address
            port: ONVIF port (usually 80 or 8080)
            username: ONVIF username (optional for discovery)
            password: ONVIF password (optional for discovery)

        Returns:
            Dict with device info, or None if failed.
        """
        try:
            camera = ONVIFCamera(ip, port, username, password)

            # Get device information
            device_mgmt = camera.devicemgmt
            device_info = device_mgmt.GetDeviceInformation()

            # Get capabilities
            capabilities = device_mgmt.GetCapabilities({"Category": "All"})

            result = {
                "ip": ip,
                "port": port,
                "manufacturer": device_info.Manufacturer,
                "model": device_info.Model,
                "firmware": device_info.FirmwareVersion,
                "serial": device_info.SerialNumber,
                "has_ptz": hasattr(capabilities, "PTZ")
                and capabilities.PTZ is not None,
                "has_media": hasattr(capabilities, "Media")
                and capabilities.Media is not None,
            }

            return result

        except ONVIFError as e:
            logger.error(f"ONVIF connection to {ip}:{port} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error connecting to {ip}:{port}: {e}")
            return None

    def get_stream_uri(
        self,
        ip: str,
        port: int = 80,
        username: str = "",
        password: str = "",
        profile_index: int = 0,
        include_credentials: bool = True,
    ) -> str | None:
        """
        Retrieves the RTSP stream URI for the camera.

        Args:
            ip: Camera IP address
            port: ONVIF port (usually 80)
            username: ONVIF username
            password: ONVIF password
            profile_index: 0 = main stream, 1 = sub stream
            include_credentials: Whether to include credentials in URI

        Returns:
            RTSP URI string or None if failed.
        """
        try:
            camera = ONVIFCamera(ip, port, username, password)
            media_service = camera.create_media_service()

            # Get available profiles
            profiles = media_service.GetProfiles()
            if not profiles:
                logger.warning(f"No media profiles found on {ip}")
                return None

            # Select profile (main or sub stream)
            profile = profiles[min(profile_index, len(profiles) - 1)]

            # Request stream URI
            stream_setup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
            uri_response = media_service.GetStreamUri(
                {"StreamSetup": stream_setup, "ProfileToken": profile.token}
            )

            rtsp_uri = uri_response.Uri

            # Inject credentials into URI if requested
            if include_credentials and username and password:
                parsed = urlparse(rtsp_uri)
                rtsp_uri = urlunparse(
                    (
                        parsed.scheme,
                        f"{username}:{password}@{parsed.hostname}:{parsed.port or 554}",
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )

            # Log without revealing full credentials
            safe_log = rtsp_uri.replace(password, "***") if password else rtsp_uri
            logger.info(f"Retrieved stream URI for {ip}: {safe_log[:60]}...")
            return rtsp_uri

        except Exception as e:
            logger.error(f"Failed to get stream URI from {ip}: {e}")
            return None

    @property
    def last_discovered(self) -> list[dict]:
        """Returns last discovered cameras."""
        with self._discovery_lock:
            return list(self._discovered_cameras)
