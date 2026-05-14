import ipaddress
import logging
import os
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import ifaddr
from onvif import ONVIFCamera
from wsdiscovery.discovery import ThreadedWSDiscovery
from zeep.cache import InMemoryCache
from zeep.transports import Transport

from utils.log_safety import safe_log_value as _slv

logger = logging.getLogger(__name__)

# zeep's default Transport builds a SqliteCache that calls
# os.makedirs('$XDG_CACHE_HOME/zeep' or '/tmp/.../zeep'). On containers
# where /tmp/<parent> exists but is owned by root (e.g. fontconfig
# pre-created during image build), the runtime user can't write there
# and every ONVIFCamera() raises PermissionError. WSDLs are local files
# under wsdl_dir, so in-memory caching is sufficient and side-effect-free.
_ZEEP_TRANSPORT = Transport(cache=InMemoryCache())

# Interface-name patterns we never scan: Docker bridges, container veths,
# and common VPN tunnel devices. The scanner is only useful on real LAN
# interfaces where ONVIF cameras might actually live.
_SKIP_IFACE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^docker\d*$"),
    re.compile(r"^br-[0-9a-f]+$"),
    re.compile(r"^veth"),
    re.compile(r"^tun\d*$"),
    re.compile(r"^tap\d*$"),
    re.compile(r"^wg\d*$"),
    re.compile(r"^tailscale"),
    re.compile(r"^zt"),
)

# Largest network we will expand into a host-by-host scan. A /22 is 1022
# hosts; anything larger is an SMB/enterprise range, a Docker bridge, or
# a misconfigured interface — never a home camera LAN. Without this cap,
# a single /16 produces ~65k * 6 ports = ~400k ThreadPool tasks resident
# in RAM, which OOMs small NAS hosts.
_MAX_SCAN_PREFIX = 22


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
        self._scan_lock = threading.Lock()
        self._last_result: list[dict] = []

    def _candidate_onvif_ports(self, preferred_port: int | None) -> list[int]:
        """
        Returns ordered ONVIF ports to try.
        Preferred port first, then known common ONVIF ports without duplicates.
        """
        ordered: list[int] = []
        if isinstance(preferred_port, int) and preferred_port > 0:
            ordered.append(preferred_port)

        for port in self.COMMON_PORTS:
            if port not in ordered:
                ordered.append(port)

        return ordered

    def _resolve_onvif_wsdl_dir(self) -> str | None:
        """
        Resolve WSDL directory for onvif-zeep.
        This keeps ONVIF working even if the package wheel misses bundled WSDL files.
        """
        candidates: list[Path] = []

        env_wsdl = os.getenv("ONVIF_WSDL_DIR", "").strip()
        if env_wsdl:
            candidates.append(Path(env_wsdl))

        # Default onvif-zeep layout: site-packages/wsdl
        try:
            import onvif as onvif_module

            candidates.append(
                Path(onvif_module.__file__).resolve().parent.parent / "wsdl"
            )
        except (ImportError, AttributeError, OSError):
            # onvif-zeep absent or installed without packaged WSDL dir.
            pass

        # Repository-bundled fallback for appliance/dev images.
        project_root = Path(__file__).resolve().parents[1]
        candidates.append(project_root / "assets" / "onvif_wsdl")

        for candidate in candidates:
            try:
                if (candidate / "devicemgmt.wsdl").exists():
                    return str(candidate)
            except Exception:
                continue
        return None

    def _create_onvif_camera(
        self, ip: str, port: int, user: str, password: str
    ) -> ONVIFCamera:
        """Create ONVIF camera client with explicit WSDL path when available."""
        wsdl_dir = self._resolve_onvif_wsdl_dir()
        if wsdl_dir:
            return ONVIFCamera(
                ip,
                port,
                user,
                password,
                wsdl_dir=wsdl_dir,
                transport=_ZEEP_TRANSPORT,
            )
        return ONVIFCamera(ip, port, user, password, transport=_ZEEP_TRANSPORT)

    def scan(self, fast: bool = False) -> list[dict]:
        """
        Perform a network scan.
        Args:
            fast: If True, skips the aggressive subnet scan and only does WS-Discovery.
        """
        # Non-blocking reentrancy guard: a second UI-triggered scan while
        # the first is still running would double the ThreadPool pressure
        # and (on host-network NAS deploys) OOM the container.
        if not self._scan_lock.acquire(blocking=False):
            logger.info(
                "Scan already in progress; returning cached result (%d device(s))",
                len(self._last_result),
            )
            return list(self._last_result)

        try:
            self._found_devices = {}

            # 1. Start WS-Discovery in background
            wsd_thread = threading.Thread(target=self._scan_ws_discovery)
            wsd_thread.start()

            # 2. Start Subnet Scan (if not fast mode)
            if not fast:
                self._scan_subnet()
            else:
                logger.info("Skipping Subnet Scan (Fast Mode)")

            # Wait for WSD
            wsd_thread.join()

            # Convert devices to list
            results = list(self._found_devices.values())
            self._last_result = results
            logger.info(f"Scan complete. Found {len(results)} devices.")
            return results
        finally:
            self._scan_lock.release()

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
        except (OSError, TimeoutError):
            # Connection refused/timed out; not an ONVIF host.
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
        """Returns list of local subnets to scan (e.g. ['192.168.1.0/24']).

        Filters out Docker bridges, container veths, link-local, and
        common VPN tunnel interfaces — none of which host ONVIF cameras.
        Caps at /22 to keep a single scan from expanding into hundreds of
        thousands of probe tasks. See the module-level docstring on
        ``_SKIP_IFACE_PATTERNS`` and ``_MAX_SCAN_PREFIX`` for the rationale.
        """
        nets: set[str] = set()
        for adapter in ifaddr.get_adapters():
            iface_name = getattr(adapter, "nice_name", None) or getattr(
                adapter, "name", ""
            )
            if self._should_skip_interface(iface_name):
                logger.info(
                    "skipping interface %s: docker/vpn/container interface",
                    iface_name,
                )
                continue

            for ip in adapter.ips:
                if not (isinstance(ip.ip, str) and isinstance(ip.network_prefix, int)):
                    continue
                if ip.ip == "127.0.0.1":
                    continue

                try:
                    iface = ipaddress.IPv4Interface(f"{ip.ip}/{ip.network_prefix}")
                except (ValueError, TypeError):
                    continue

                network = iface.network
                if network.is_link_local:
                    logger.info(
                        "skipping subnet %s on %s: link-local",
                        network,
                        iface_name,
                    )
                    continue
                if network.prefixlen < _MAX_SCAN_PREFIX:
                    logger.info(
                        "skipping subnet %s on %s: too large (prefix /%d < /%d)",
                        network,
                        iface_name,
                        network.prefixlen,
                        _MAX_SCAN_PREFIX,
                    )
                    continue

                nets.add(str(network))
        return list(nets)

    @staticmethod
    def _should_skip_interface(name: str) -> bool:
        if not name:
            return False
        return any(pattern.match(name) for pattern in _SKIP_IFACE_PATTERNS)

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
            cam = self._create_onvif_camera(ip, port, user, password)
            info = cam.devicemgmt.GetDeviceInformation()
            has_ptz = self._probe_ptz_capability(cam)
            return {
                "manufacturer": info.Manufacturer,
                "model": info.Model,
                "firmware": info.FirmwareVersion,
                "serial": info.SerialNumber,
                "has_ptz": has_ptz,
            }
        except Exception as e:
            logger.error(f"GetInfo failed: {e}")
            raise

    @staticmethod
    def _probe_ptz_capability(cam) -> bool:
        """Best-effort PTZ-capability check via ONVIF GetCapabilities."""
        try:
            caps = cam.devicemgmt.GetCapabilities({"Category": "All"})
            ptz_caps = getattr(caps, "PTZ", None)
            xaddr = getattr(ptz_caps, "XAddr", None) if ptz_caps else None
            return bool(xaddr)
        except Exception as e:
            logger.debug("PTZ capability probe failed: %s", e)
            return False

    def get_stream_uri(self, ip, port, user, password, profile_index=0):
        """Get RTSP URI."""
        last_error = None
        tried_ports = self._candidate_onvif_ports(port)

        for try_port in tried_ports:
            try:
                c = self._create_onvif_camera(ip, try_port, user, password)
                media = c.create_media_service()
                profiles = media.GetProfiles()
                if not profiles:
                    raise RuntimeError("No profiles found")
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

                if try_port != port:
                    logger.info(
                        "GetStream succeeded via ONVIF fallback port %s (requested %s)",
                        try_port,
                        port,
                    )
                return uri

            except Exception as e:
                last_error = e
                logger.debug(
                    "GetStream failed for %s:%s: %s", _slv(ip), _slv(try_port), e
                )

        logger.error(
            "GetStream failed for %s on ONVIF ports %s: %s",
            _slv(ip),
            tried_ports,
            last_error,
        )
        raise RuntimeError(f"GetStream failed after port fallback: {last_error}")
