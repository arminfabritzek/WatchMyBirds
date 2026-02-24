# tests/test_system_monitor.py
"""
Unit tests for the SystemMonitor class.
"""

import csv
import time

import utils.system_monitor as system_monitor_mod
from utils.system_monitor import (
    CSV_HEADERS,
    SystemMonitor,
    is_raspberry_pi,
    parse_throttled,
)


class TestParseThrottled:
    """Tests for the throttled state parser."""

    def test_parse_normal(self):
        """Normal state (no throttling)."""
        result = parse_throttled("throttled=0x0")
        assert result["raw"] == "0x0"
        assert result["flags"] == []

    def test_parse_under_voltage_now(self):
        """Currently under voltage."""
        result = parse_throttled("throttled=0x1")
        assert "under_voltage_now" in result["flags"]

    def test_parse_under_voltage_occurred(self):
        """Under voltage occurred in the past."""
        result = parse_throttled("throttled=0x10000")
        assert "under_voltage_occurred" in result["flags"]

    def test_parse_multiple_flags(self):
        """Multiple issues at once."""
        # 0x50005 = under_voltage_now + arm_freq_capped_now + under_voltage_occurred + arm_freq_capped_occurred
        result = parse_throttled("throttled=0x50005")
        assert "under_voltage_now" in result["flags"]
        assert "throttled_now" in result["flags"]

    def test_parse_none(self):
        """Handle None input (vcgencmd not available)."""
        result = parse_throttled(None)
        assert result["raw"] is None
        assert result["flags"] == []

    def test_parse_raw_hex_without_prefix(self):
        """Parser accepts plain hex values without 'throttled=' prefix."""
        result = parse_throttled("0x0")
        assert result["raw"] == "0x0"
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
        assert monitor.output_dir == output_dir / "logs"
        assert monitor.csv_path == output_dir / "logs" / "vital_signs.csv"
        assert monitor.csv_path.exists()
        with open(monitor.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == CSV_HEADERS

    def test_initialization_rotates_legacy_header(self, tmp_path):
        """Legacy vital_signs schema is rotated before writing the new header."""
        output_dir = tmp_path / "output"
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True)
        legacy_csv = log_dir / "vital_signs.csv"
        with open(legacy_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_temp_c"])
            writer.writerow(["2026-02-09T00:00:00", "50.0"])

        monitor = SystemMonitor(output_dir=str(output_dir))
        with open(monitor.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == CSV_HEADERS
        legacy_rotations = list(log_dir.glob("vital_signs_legacy_*.csv"))
        assert len(legacy_rotations) == 1

    def test_collect_sample(self, tmp_path):
        """Sample collection returns expected keys."""
        monitor = SystemMonitor(output_dir=str(tmp_path))
        sample = monitor._collect_sample()

        assert "ts" in sample
        assert "cpu_percent" in sample
        assert "ram_percent" in sample
        assert "throttled_hex" in sample
        assert "throttled" in sample
        assert "uptime_seconds" in sample
        assert "children_count" in sample
        assert "children_rss_mb" in sample
        assert "ffmpeg_count" in sample
        assert "ffmpeg_rss_mb" in sample
        assert "app_total_rss_mb" in sample
        assert isinstance(sample["cpu_percent"], (int, float))
        assert isinstance(sample["ram_percent"], (int, float))
        assert isinstance(sample["uptime_seconds"], int)
        assert isinstance(sample["children_count"], int)
        assert isinstance(sample["ffmpeg_count"], int)

    def test_collect_sample_boot_time_fallback(self, tmp_path, monkeypatch):
        """Sample collection survives psutil.boot_time failures."""
        monitor = SystemMonitor(output_dir=str(tmp_path))

        def fail_boot_time():
            raise PermissionError("sysctl denied")

        monkeypatch.setattr(system_monitor_mod.psutil, "boot_time", fail_boot_time)
        sample = monitor._collect_sample()
        assert isinstance(sample["uptime_seconds"], int)
        assert sample["uptime_seconds"] >= 0

    def test_log_sample_diagnostics_emits_structured_event(self, tmp_path, monkeypatch):
        """Diagnostics logger emits system_vitals entries with deltas/HWM."""
        monitor = SystemMonitor(output_dir=str(tmp_path))
        messages: list[str] = []

        monkeypatch.setattr(
            system_monitor_mod.logger, "debug", lambda msg: messages.append(str(msg))
        )

        sample1 = {
            "cpu_temp_c": 50.0,
            "cpu_percent": 10.0,
            "ram_percent": 20.0,
            "disk_percent": 30.0,
            "throttled_hex": "0x0",
            "throttled": {"flags": []},
            "uptime_seconds": 100,
            "fd_count": 10,
            "thread_count": 5,
            "process_rss_mb": 100.0,
            "children_count": 2,
            "children_rss_mb": 40.0,
            "ffmpeg_count": 1,
            "ffmpeg_rss_mb": 30.0,
            "app_total_rss_mb": 140.0,
        }
        sample2 = {
            **sample1,
            "fd_count": 12,
            "ffmpeg_rss_mb": 33.5,
            "app_total_rss_mb": 145.5,
        }

        monitor._log_sample_diagnostics(sample1)
        monitor._log_sample_diagnostics(sample2)

        assert any("event=system_vitals" in msg for msg in messages)
        assert "fd_high_watermark=12" in messages[-1]
        assert "fd_delta=2" in messages[-1]
        assert "app_total_rss_delta_mb=5.5" in messages[-1]
        assert "ffmpeg_rss_delta_mb=3.5" in messages[-1]

    def test_monitor_writes_csv_rows(self, tmp_path):
        """Background monitor writes rows to the CSV file."""
        monitor = SystemMonitor(
            output_dir=str(tmp_path),
            sample_interval_seconds=0.1,
        )
        monitor.start()
        time.sleep(0.35)
        monitor.stop()
        with open(monitor.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # Header + at least one sample row.
        assert len(rows) >= 2

    def test_get_current_vitals(self, tmp_path):
        """get_current_vitals always returns API-compatible data."""
        monitor = SystemMonitor(output_dir=str(tmp_path))
        vitals = monitor.get_current_vitals()

        assert "ts" in vitals
        assert "cpu_percent" in vitals
        assert "ram_percent" in vitals
        assert "throttled" in vitals

    def test_get_current_vitals_after_start_contains_cached_values(self, tmp_path):
        """get_current_vitals returns cached monitor values after start."""
        monitor = SystemMonitor(output_dir=str(tmp_path), sample_interval_seconds=0.1)
        monitor.start()
        time.sleep(0.25)
        vitals = monitor.get_current_vitals()
        monitor.stop()

        assert "throttled" in vitals
        assert "throttled_hex" in vitals
        assert "fd_count" in vitals


class TestIsRaspberryPi:
    """Tests for RPi detection."""

    def test_not_rpi_on_mac(self):
        """Should return False on non-RPi systems."""
        # This test will pass on Mac/Linux dev machines
        # and correctly detect True on actual RPi
        result = is_raspberry_pi()
        # We don't assert a specific value since it depends on the environment
        assert isinstance(result, bool)
