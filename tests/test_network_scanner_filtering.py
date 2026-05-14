"""Tests for NetworkScanner interface filtering, network-size cap, and reentrancy lock.

Background: on hosts where the WatchMyBirds container runs with
``network_mode: host`` (typical NAS deployment), the scanner inherits the
host's full routing table. Without filtering, every Docker bridge and VPN
tunnel becomes a scan target, and ``/16`` ranges expand to ~65k hosts ×
6 ports = ~400k ThreadPool tasks resident in RAM. See plan
2026-05-15_FIX_network-scanner-docker-bridge-filter for the live-log
evidence (NAS RAM 90.9 % -> 99.6 % during one scan).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from camera.network_scanner import NetworkScanner


def _fake_adapter(nice_name: str, ip: str, prefix: int) -> SimpleNamespace:
    """Build a minimal ifaddr-Adapter-compatible stub."""
    return SimpleNamespace(
        name=nice_name,
        nice_name=nice_name,
        index=0,
        ips=[
            SimpleNamespace(
                ip=ip,
                network_prefix=prefix,
                is_IPv4=True,
                is_IPv6=False,
                nice_name=nice_name,
            )
        ],
    )


@pytest.fixture
def patch_adapters(monkeypatch):
    """Patch ifaddr.get_adapters with a caller-provided list of fake adapters."""

    def _apply(adapters: list) -> None:
        monkeypatch.setattr(
            "camera.network_scanner.ifaddr.get_adapters", lambda: adapters
        )

    return _apply


# ---------------------------------------------------------------------------
# Interface-name filter (fix 1)
# ---------------------------------------------------------------------------


def test_local_networks_keep_real_lan(patch_adapters):
    patch_adapters([_fake_adapter("eth0", "192.168.178.5", 24)])
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert "192.168.178.0/24" in nets


def test_local_networks_skip_docker0(patch_adapters):
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("docker0", "172.17.0.1", 16),
        ]
    )
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert "192.168.178.0/24" in nets
    assert not any(n.startswith("172.17.") for n in nets)


def test_local_networks_skip_compose_bridge(patch_adapters):
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("br-1a2b3c4d5e6f", "172.22.0.1", 16),
        ]
    )
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert not any(n.startswith("172.22.") for n in nets)


def test_local_networks_skip_veth(patch_adapters):
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("veth0815", "172.18.0.2", 16),
        ]
    )
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert not any(n.startswith("172.18.") for n in nets)


def test_local_networks_skip_vpn_interfaces(patch_adapters):
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("tun0", "10.10.0.5", 24),
            _fake_adapter("wg0", "10.20.0.5", 24),
            _fake_adapter("tailscale0", "100.64.0.5", 32),
        ]
    )
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert "192.168.178.0/24" in nets
    assert not any(n.startswith("10.10.") for n in nets)
    assert not any(n.startswith("10.20.") for n in nets)
    assert not any(n.startswith("100.64.") for n in nets)


def test_local_networks_skip_link_local(patch_adapters):
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("eth0:avahi", "169.254.5.5", 16),
        ]
    )
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert not any(n.startswith("169.254.") for n in nets)


# ---------------------------------------------------------------------------
# Network-size cap (fix 2) — the load-bearing safety net
# ---------------------------------------------------------------------------


def test_local_networks_skip_oversized_even_on_eth0(patch_adapters):
    """Even if an interface name looks legitimate, a /16 is always skipped.

    This is the fix that protects against unknown bridge names (Synology
    DSM sometimes labels bridges differently than upstream Docker) and
    against any future filter-evasion mode we have not enumerated.
    """
    patch_adapters([_fake_adapter("eth0", "10.0.0.5", 16)])
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert nets == [], f"Expected empty (oversized), got {nets}"


def test_local_networks_keep_slash_22(patch_adapters):
    """A /22 (1022 hosts) is the largest legitimate LAN size we accept."""
    patch_adapters([_fake_adapter("eth0", "192.168.4.5", 22)])
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert "192.168.4.0/22" in nets


def test_local_networks_skip_slash_20(patch_adapters):
    patch_adapters([_fake_adapter("eth0", "192.168.112.5", 20)])
    scanner = NetworkScanner()
    nets = scanner._get_local_networks()
    assert nets == []


def test_local_networks_log_skip_reasons(patch_adapters, caplog):
    """Each skip must log a reason; humans need this in NAS logs to verify the fix.

    Docker-interface skips and size-cap skips both go to logger.info and
    name the subnet plus the reason. The exact wording is not asserted
    here (one assertion per behavior is enough), only that *some* info
    record mentions each skipped subnet.
    """
    patch_adapters(
        [
            _fake_adapter("eth0", "192.168.178.5", 24),
            _fake_adapter("docker0", "172.17.0.1", 16),
            _fake_adapter("eth1", "10.5.0.1", 16),
        ]
    )
    scanner = NetworkScanner()
    with caplog.at_level("INFO", logger="camera.network_scanner"):
        scanner._get_local_networks()
    combined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "172.17" in combined or "docker0" in combined
    assert "10.5" in combined or "/16" in combined


# ---------------------------------------------------------------------------
# Reentrancy lock (fix 3)
# ---------------------------------------------------------------------------


def test_scan_returns_cached_result_when_already_running(patch_adapters):
    """A second concurrent scan() call must not spawn a parallel scan.

    The contract: if a scan is in progress, the second caller gets the
    *previous* result list back immediately (empty list on first ever
    call) and the scan loop runs only once.
    """
    patch_adapters([_fake_adapter("eth0", "192.168.178.5", 24)])
    scanner = NetworkScanner()

    start_barrier = threading.Event()
    release_barrier = threading.Event()
    call_count = {"n": 0}

    def slow_subnet_scan(self):
        call_count["n"] += 1
        start_barrier.set()
        release_barrier.wait(timeout=5)

    with (
        patch.object(NetworkScanner, "_scan_subnet", slow_subnet_scan),
        patch.object(NetworkScanner, "_scan_ws_discovery", lambda self: None),
    ):
        t1 = threading.Thread(target=scanner.scan)
        t1.start()
        assert start_barrier.wait(timeout=2), "first scan never started"

        # Second call lands while first is still running.
        result_second = scanner.scan()
        assert result_second == [], "Reentrant scan should return cached result"

        release_barrier.set()
        t1.join(timeout=5)

    assert call_count["n"] == 1, (
        f"Expected only one _scan_subnet invocation, got {call_count['n']}"
    )
