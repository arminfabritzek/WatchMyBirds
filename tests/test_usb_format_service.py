"""Tests for web/services/usb_format_service.py.

Covers:
  - Strict device-path validation (only /dev/sd[a-z], no partitions)
  - Confirmation token requirement
  - Defense-in-depth re-validation against the discovery list
  - Polkit-failure recognition in the error message

The real format script is bash-only and runs as root via systemd; we
do not exercise it from Python tests. We assert the Python layer's
own guards instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.services import usb_format_service  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def stub_supported(monkeypatch):
    """Pretend we're on a Pi where the script + systemctl are available."""
    monkeypatch.setattr(usb_format_service, "is_format_supported", lambda: True)


@pytest.fixture
def stub_devices(monkeypatch):
    """Discovery returns a single fake stick at /dev/sda."""
    monkeypatch.setattr(
        usb_format_service,
        "list_usb_block_devices",
        lambda: [
            {
                "device": "/dev/sda",
                "size_bytes": 120 * 1024 * 1024 * 1024,
                "model": "FakeStick",
                "vendor": "Test",
                "current_label": "WMB-BACKUP",
                "current_fstype": "ext4",
                "is_already_wmb_backup": True,
            },
        ],
    )


# ----------------------------------------------------------------------
# Validation guards
# ----------------------------------------------------------------------


class TestTriggerValidation:
    def test_refuses_when_not_supported(self, monkeypatch):
        monkeypatch.setattr(usb_format_service, "is_format_supported", lambda: False)
        ok, msg = usb_format_service.trigger_format("/dev/sda", "FORMAT")
        assert ok is False
        assert "not available" in msg.lower()

    @pytest.mark.parametrize(
        "bad",
        [
            "/dev/sda1",         # partition, not whole disk
            "/dev/sdaa",         # too long
            "/dev/sd",           # too short
            "/dev/mmcblk0",      # SD card!
            "/dev/nvme0n1",      # internal NVMe!
            "/dev/loop0",
            "../etc/passwd",     # traversal nonsense
            "",
            "sda",               # no /dev/ prefix
            "/dev/SDA",          # case
        ],
    )
    def test_refuses_invalid_device_paths(self, stub_supported, bad):
        ok, msg = usb_format_service.trigger_format(bad, "FORMAT")
        assert ok is False, f"Should have refused {bad!r}, got success"

    def test_refuses_missing_confirm(self, stub_supported, stub_devices):
        ok, msg = usb_format_service.trigger_format("/dev/sda", "")
        assert ok is False
        assert "confirm" in msg.lower()

    def test_refuses_wrong_confirm(self, stub_supported, stub_devices):
        ok, msg = usb_format_service.trigger_format("/dev/sda", "format")  # lowercase
        assert ok is False
        assert "confirm" in msg.lower()

    def test_refuses_target_not_in_discovery_list(self, stub_supported, monkeypatch):
        # Discovery returns no devices -- crafted POST shouldn't bypass.
        monkeypatch.setattr(
            usb_format_service, "list_usb_block_devices", lambda: []
        )
        ok, msg = usb_format_service.trigger_format("/dev/sda", "FORMAT")
        assert ok is False
        assert "not a recognised" in msg.lower() or "not recognised" in msg.lower()


# ----------------------------------------------------------------------
# Polkit failure recognition
# ----------------------------------------------------------------------


class TestPolkitErrorMessage:
    def test_polkit_failure_surfaces_clean_message(
        self, stub_supported, stub_devices, monkeypatch
    ):
        import subprocess as sp

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # First two calls: set-environment (succeed)
            if cmd[:2] == ["systemctl", "set-environment"]:
                return sp.CompletedProcess(cmd, 0, "", "")
            # Third call: start (fail with polkit error)
            if "start" in cmd:
                raise sp.CalledProcessError(
                    1, cmd,
                    stderr="Failed to start wmb-format-backup.service: "
                           "Interactive authentication required by polkit.",
                )
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(sp, "run", fake_run)
        ok, msg = usb_format_service.trigger_format("/dev/sda", "FORMAT")
        assert ok is False
        assert "polkit" in msg.lower()


# ----------------------------------------------------------------------
# Status reading
# ----------------------------------------------------------------------


class TestStatusReading:
    def test_idle_when_no_status_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            usb_format_service, "STATUS_FILE", tmp_path / "no-such-file.json"
        )
        result = usb_format_service.get_format_status()
        assert result["state"] == "idle"

    def test_returns_file_contents(self, monkeypatch, tmp_path):
        sf = tmp_path / "status.json"
        sf.write_text(
            '{"state":"formatting","message":"halfway","ts":"2026-05-01T00:00:00Z","target":"/dev/sda"}'
        )
        monkeypatch.setattr(usb_format_service, "STATUS_FILE", sf)
        result = usb_format_service.get_format_status()
        assert result["state"] == "formatting"
        assert result["target"] == "/dev/sda"

    def test_handles_corrupt_json(self, monkeypatch, tmp_path):
        sf = tmp_path / "status.json"
        sf.write_text("{not json")
        monkeypatch.setattr(usb_format_service, "STATUS_FILE", sf)
        result = usb_format_service.get_format_status()
        assert result["state"] == "error"

    def test_clear_removes_file(self, monkeypatch, tmp_path):
        sf = tmp_path / "status.json"
        sf.write_text('{"state":"success"}')
        monkeypatch.setattr(usb_format_service, "STATUS_FILE", sf)
        assert usb_format_service.clear_format_status() is True
        assert not sf.exists()

    def test_clear_idempotent_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            usb_format_service, "STATUS_FILE", tmp_path / "absent.json"
        )
        # Removing absent file should still return True (Path.unlink missing_ok).
        assert usb_format_service.clear_format_status() is True
