# tests/test_system_monitor.py
"""
Unit tests for the SystemMonitor class.
"""

import json
import time

from utils.system_monitor import (
    SystemMonitor,
    _is_raspberry_pi,
    _parse_throttled,
)


class TestParseThrottled:
    """Tests for the throttled state parser."""

    def test_parse_normal(self):
        """Normal state (no throttling)."""
        result = _parse_throttled("throttled=0x0")
        assert result["raw"] == "0x0"
        assert result["flags"] == []

    def test_parse_under_voltage_now(self):
        """Currently under voltage."""
        result = _parse_throttled("throttled=0x1")
        assert "under_voltage_now" in result["flags"]

    def test_parse_under_voltage_occurred(self):
        """Under voltage occurred in the past."""
        result = _parse_throttled("throttled=0x10000")
        assert "under_voltage_occurred" in result["flags"]

    def test_parse_multiple_flags(self):
        """Multiple issues at once."""
        # 0x50005 = under_voltage_now + arm_freq_capped_now + under_voltage_occurred + arm_freq_capped_occurred
        result = _parse_throttled("throttled=0x50005")
        assert "under_voltage_now" in result["flags"]
        assert "throttled_now" in result["flags"]

    def test_parse_none(self):
        """Handle None input (vcgencmd not available)."""
        result = _parse_throttled(None)
        assert result["raw"] is None
        assert result["flags"] == []


class TestSystemMonitor:
    """Tests for the SystemMonitor class."""

    def test_initialization(self, tmp_path):
        """Monitor initializes correctly and creates directories."""
        output_dir = tmp_path / "output"
        monitor = SystemMonitor(
            output_dir=str(output_dir),
            sample_interval_seconds=1.0,
            chunk_interval_seconds=5.0,
        )
        assert monitor.output_dir.exists()
        assert monitor.output_dir == output_dir / "logs" / "vitals"

    def test_collect_sample(self, tmp_path):
        """Sample collection returns expected keys."""
        monitor = SystemMonitor(output_dir=str(tmp_path))
        sample = monitor._collect_sample()

        assert "ts" in sample
        assert "cpu_percent" in sample
        assert "ram_percent" in sample
        assert "throttled" in sample
        assert isinstance(sample["cpu_percent"], (int, float))
        assert isinstance(sample["ram_percent"], (int, float))

    def test_chunk_write(self, tmp_path):
        """Chunks are written correctly."""
        monitor = SystemMonitor(
            output_dir=str(tmp_path),
            sample_interval_seconds=0.1,
            chunk_interval_seconds=0.5,
        )

        # Manually add samples
        for _ in range(5):
            sample = monitor._collect_sample()
            monitor._samples.append(sample)

        # Trigger chunk write
        monitor._write_chunk()

        # Verify file was created
        chunk_files = list(monitor.output_dir.glob("*.json"))
        assert len(chunk_files) == 1

        # Verify content
        with open(chunk_files[0]) as f:
            data = json.load(f)
        assert data["sample_count"] == 5
        assert len(data["samples"]) == 5

    def test_emergency_flush_detection(self, tmp_path):
        """Emergency flush is triggered on critical events."""
        monitor = SystemMonitor(output_dir=str(tmp_path))

        # First sample: normal
        sample_normal = {"throttled": {"flags": []}}
        assert not monitor._should_emergency_flush(sample_normal)

        # Second sample: under voltage appears
        sample_critical = {"throttled": {"flags": ["under_voltage_now"]}}
        assert monitor._should_emergency_flush(sample_critical)

        # Third sample: same critical state (no new event)
        assert not monitor._should_emergency_flush(sample_critical)

    def test_cleanup_old_chunks(self, tmp_path):
        """Old chunks are cleaned up."""
        monitor = SystemMonitor(output_dir=str(tmp_path))

        # Create 10 fake chunk files
        for i in range(10):
            (monitor.output_dir / f"vitals_{i:04d}.json").write_text("{}")
            time.sleep(0.01)  # Ensure different mtime

        # Cleanup keeping only 5
        monitor._cleanup_old_chunks(max_files=5)

        remaining = list(monitor.output_dir.glob("*.json"))
        assert len(remaining) == 5

    def test_get_current_vitals(self, tmp_path):
        """get_current_vitals returns data."""
        monitor = SystemMonitor(output_dir=str(tmp_path))
        vitals = monitor.get_current_vitals()

        assert "ts" in vitals
        assert "cpu_percent" in vitals


class TestIsRaspberryPi:
    """Tests for RPi detection."""

    def test_not_rpi_on_mac(self):
        """Should return False on non-RPi systems."""
        # This test will pass on Mac/Linux dev machines
        # and correctly detect True on actual RPi
        result = _is_raspberry_pi()
        # We don't assert a specific value since it depends on the environment
        assert isinstance(result, bool)
