import ipaddress
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import ifaddr
from onvif import ONVIFCamera
from wsdiscovery.discovery import ThreadedWSDiscovery

logger = logging.getLogger(__name__)


class NetworkScanner:
    """
    Robust network scanner for finding ONVIF cameras.
    Combines WS-Discovery (Multicast) with active Subnet/Port scanning.
    """

    COMMON_PORTS = [80, 8080, 8899, 554, 10080, 8000]
    SCAN_TIMEOUT = 0.5  # Timeout for socket connection
    WSD_TIMEOUT = 3  # Timeout for WS-Discovery

    def __init__(self):
        self._found_devices: dict[str, dict] = {}  # Key: "ip:port"
        self._lock = threading.Lock()

    def scan(self, fast: bool = False) -> list[dict]:
        """
        Perform a network scan.
        Args:
            fast: If True, skips the aggressive subnet scan and only does WS-Discovery.
        """
        self._found_devices = {}

        # 1. Start WS-Discovery in background
        wsd_thread = threading.Thread(target=self._scan_ws_discovery)
        wsd_thread.start()

        # 2. Start Subnet Scan (if not fast mode)
        msg_scanner = "Active Subnet Scan..."
        if not fast:
            self._scan_subnet()
        else:
            msg_scanner = "Skipping Subnet Scan (Fast Mode)"
            logger.info(msg_scanner)

        # Wait for WSD
        wsd_thread.join()

        # Convert devices to list
        results = list(self._found_devices.values())
        logger.info(f"Scan complete. Found {len(results)} devices.")
        return results

    def _scan_ws_discovery(self):
        """Standard ONVIF WS-Discovery."""
        try:
            logger.info("Starting WS-Discovery...")
            wsd = ThreadedWSDiscovery()
            wsd.start()

            # Scope for VideoTransmitter or specific ONVIF types
            # Note: Some older cameras might not advertise types strictly, but we'll try.
            # Using empty scopes finds everything, then we filter.
            services = wsd.searchServices(timeout=self.WSD_TIMEOUT)

            for service in services:
                # Parse xAddrs
                xaddrs = service.getXAddrs()
                if not xaddrs:
                    continue

                for addr in xaddrs:
                    # addr is like http://192.168.1.100:80/onvif/device_service
                    try:
                        parsed = urlparse(addr)
                        ip = parsed.hostname
                        port = parsed.port or 80

                        # Gather metadata from scopes
                        scopes = service.getScopes()
                        name = self._extract_scope(scopes, "name")
                        hardware = self._extract_scope(scopes, "hardware")

                        self._add_device(ip, port, name, hardware, "WS-Discovery")
                    except Exception as e:
                        logger.debug(f"Error parsing WSD service result: {e}")

            wsd.stop()
            logger.info("WS-Discovery finished.")
        except Exception as e:
            logger.error(f"WS-Discovery failed: {e}")

    def _scan_subnet(self):
        """Active scan of local subnet common ports."""
        logger.info("Starting Active Subnet Scan...")
        local_nets = self._get_local_networks()

        # Get own IPs to skip
        own_ips = set()
        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str):
                    own_ips.add(ip.ip)

        tasks = []
        # Reduced workers to prevent finding self or starving other threads (like Video Feed)
        with ThreadPoolExecutor(max_workers=32) as executor:
            for net in local_nets:
                logger.info(f"Scanning subnet: {net}")
                try:
                    network = ipaddress.IPv4Network(net, strict=False)
                except ValueError:
                    continue

                for ip in network:
                    if ip.is_loopback or ip.is_multicast or ip.is_reserved:
                        continue

                    ip_str = str(ip)
                    if ip_str in own_ips:
                        continue

                    # Check common ports
                    for port in self.COMMON_PORTS:
                        tasks.append(executor.submit(self._probe_port, ip_str, port))

            # Wait for all
            for _future in as_completed(tasks):
                pass
        logger.info("Subnet Scan finished.")

    def _probe_port(self, ip: str, port: int):
        """Check if port is open and maybe ONVIF."""
        key = f"{ip}:{port}"
        # Skip if already found via WSD
        if key in self._found_devices:
            return

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.SCAN_TIMEOUT)
                result = s.connect_ex((ip, port))
                if result == 0:
                    # Port is Open. STRICT verification required.
                    if self._verify_onvif_http(ip, port):
                        self._add_device(
                            ip,
                            port,
                            f"Unknown Device ({ip})",
                            "Generic ONVIF",
                            "Port Scan",
                        )
        except Exception:
            pass

    def _verify_onvif_http(self, ip: str, port: int) -> bool:
        """
        Stricter verification: Sends a POST to /onvif/device_service.
        ONVIF is SOAP-based (POST).

        We expect:
        - 400 Bad Request (Valid! Sent empty/garbage body, service complained)
        - 401 Unauthorized (Valid! Service exists/secured)
        - 500 Internal Error (Valid! SOAP Fault)
        - 200 OK (Valid)
        - 405 Method Not Allowed (Valid)

        We REJECT:
        - 404 Not Found (Standard web server without ONVIF)
        - Connection Refused/Timeout
        """
        import http.client

        try:
            conn = http.client.HTTPConnection(ip, port, timeout=2.0)
            # Try POST to standard ONVIF path (SOAP endpoint)
            # Sending empty body should trigger 400 or 500 if service exists.
            conn.request(
                "POST",
                "/onvif/device_service",
                body="",
                headers={"Content-Type": "application/soap+xml"},
            )
            resp = conn.getresponse()
            conn.close()

            # 404 means endpoint not found -> Not an ONVIF camera
            if resp.status == 404:
                return False

            # Acceptable ONVIF-like responses
            if resp.status in [200, 400, 401, 403, 405, 500]:
                return True

            return False
        except Exception:
            return False

    def _add_device(self, ip, port, name, hw, source):
        key = f"{ip}:{port}"
        with self._lock:
            if key not in self._found_devices:
                self._found_devices[key] = {
                    "ip": ip,
                    "port": port,
                    "name": name or f"Camera {ip}",
                    "manufacturer": hw or "Unknown",
                    "source": source,
                }
            else:
                # Update info if better
                if name and "Unknown" in self._found_devices[key]["name"]:
                    self._found_devices[key]["name"] = name
                if hw and "Unknown" in self._found_devices[key]["manufacturer"]:
                    self._found_devices[key]["manufacturer"] = hw

    def _get_local_networks(self) -> list[str]:
        """Returns list of local subnets (e.g. ['192.168.1.0/24'])"""
        nets = []
        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and isinstance(ip.network_prefix, int):
                    # IPv4
                    if ip.ip == "127.0.0.1":
                        continue

                    # Calculate network
                    try:
                        # naive /24 assumption if prefix missing, but ifaddr gives prefix
                        # Using ipaddress module
                        iface = ipaddress.IPv4Interface(f"{ip.ip}/{ip.network_prefix}")
                        nets.append(str(iface.network))
                    except Exception:
                        pass
        return list(set(nets))

    def _extract_scope(self, scopes, key):
        for scope in scopes:
            s = str(scope)
            if f"/{key}/" in s.lower():
                return s.split("/")[-1].replace("_", " ")
        return None

    # --- Helper methods for direct connection ---

    def get_device_info(self, ip, port, user, password):
        """Directly query a specific camera."""
        try:
            cam = ONVIFCamera(ip, port, user, password)
            info = cam.devicemgmt.GetDeviceInformation()
            return {
                "manufacturer": info.Manufacturer,
                "model": info.Model,
                "firmware": info.FirmwareVersion,
                "serial": info.SerialNumber,
            }
        except Exception as e:
            logger.error(f"GetInfo failed: {e}")
            raise

    def get_stream_uri(self, ip, port, user, password, profile_index=0):
        """Get RTSP URI."""
        try:
            c = ONVIFCamera(ip, port, user, password)
            media = c.create_media_service()
            profiles = media.GetProfiles()
            if not profiles:
                raise Exception("No profiles found")
            token = profiles[profile_index].token

            uri_resp = media.GetStreamUri(
                {
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    },
                    "ProfileToken": token,
                }
            )
            uri = uri_resp.Uri

            # Inject creds
            if user and password:
                p = urlparse(uri)
                # Reconstruct with auth
                netloc = f"{user}:{password}@{p.hostname}"
                if p.port:
                    netloc += f":{p.port}"
                uri = p._replace(netloc=netloc).geturl()
            return uri

        except Exception as e:
            logger.error(f"GetStream failed: {e}")
            raise
