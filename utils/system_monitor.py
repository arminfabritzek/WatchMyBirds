# utils/system_monitor.py
"""
System Vitals Monitor for Raspberry Pi crash diagnosis.

Collects hardware metrics (voltage, temperature, throttling) and writes them
in "chunks" to minimize SD card wear while maintaining a 24-hour history.
"""

import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil

from logging_config import get_logger


logger = get_logger(__name__)

# Throttled state bit definitions (from vcgencmd get_throttled)
THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def _is_raspberry_pi() -> bool:
    """Check if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry pi" in f.read().lower()
    except FileNotFoundError:
        return False


def _run_vcgencmd(cmd: str) -> str | None:
    """Run a vcgencmd command and return the output."""
    try:
        result = subprocess.run(
            ["vcgencmd", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _parse_throttled(value_str: str | None) -> dict:
    """Parse vcgencmd get_throttled output into human-readable flags."""
    if not value_str:
        return {"raw": None, "flags": []}
    try:
        # Output format: "throttled=0x0"
        hex_val = value_str.split("=")[1]
        value = int(hex_val, 16)
        flags = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
        return {"raw": hex_val, "flags": flags}
    except (IndexError, ValueError):
        return {"raw": value_str, "flags": []}


def _get_cpu_temp() -> float | None:
    """Get CPU temperature in Celsius."""
    # Try vcgencmd first (RPi specific)
    temp_str = _run_vcgencmd("measure_temp")
    if temp_str:
        try:
            # Output format: "temp=42.0'C"
            return float(temp_str.split("=")[1].replace("'C", ""))
        except (IndexError, ValueError):
            pass
    # Fallback to psutil (works on most Linux)
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            return temps["cpu_thermal"][0].current
        if "coretemp" in temps:
            return temps["coretemp"][0].current
    except Exception:
        pass
    return None


def _get_core_voltage() -> str | None:
    """Get core voltage (RPi specific)."""
    volt_str = _run_vcgencmd("measure_volts core")
    if volt_str:
        try:
            # Output format: "volt=1.2000V"
            return volt_str.split("=")[1]
        except IndexError:
            return volt_str
    return None


class SystemMonitor:
    """
    Monitors system vitals and writes chunked logs to disk.

    - Collects samples every `sample_interval_seconds` (default: 30s)
    - Writes chunks to disk every `chunk_interval_seconds` (default: 15 min)
    - Emergency flush on critical events (under-voltage, over-temp)
    """

    def __init__(
        self,
        output_dir: str,
        sample_interval_seconds: float = 30.0,
        chunk_interval_seconds: float = 900.0,  # 15 minutes
        max_samples_in_memory: int = 200,  # ~1.6 hours at 30s interval
    ):
        self.output_dir = Path(output_dir) / "logs" / "vitals"
        self.sample_interval = sample_interval_seconds
        self.chunk_interval = chunk_interval_seconds
        self.max_samples = max_samples_in_memory

        self._samples: deque = deque(maxlen=max_samples_in_memory)
        self._last_chunk_write = time.time()
        self._last_throttle_flags: set = set()
        self._running = False
        self._thread: threading.Thread | None = None
        self._is_rpi = _is_raspberry_pi()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_sample(self) -> dict:
        """Collect a single system vitals sample."""
        now = datetime.now()
        throttled = _parse_throttled(_run_vcgencmd("get_throttled"))

        sample = {
            "ts": now.isoformat(),
            "cpu_temp_c": _get_cpu_temp(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
            "throttled": throttled,
        }

        # Add RPi-specific metrics
        if self._is_rpi:
            sample["core_voltage"] = _get_core_voltage()

        return sample

    def _should_emergency_flush(self, sample: dict) -> bool:
        """Check if we should immediately flush due to critical events."""
        current_flags = set(sample.get("throttled", {}).get("flags", []))
        critical_flags = {"under_voltage_now", "throttled_now", "soft_temp_limit_now"}
        new_critical = current_flags & critical_flags - self._last_throttle_flags
        self._last_throttle_flags = current_flags
        return bool(new_critical)

    def _write_chunk(self, emergency: bool = False):
        """Write buffered samples to disk."""
        if not self._samples:
            return

        now = datetime.now()
        prefix = "EMERGENCY_" if emergency else ""
        filename = f"{prefix}vitals_{now.strftime('%Y%m%d_%H%M')}.json"
        filepath = self.output_dir / filename

        try:
            chunk_data = {
                "written_at": now.isoformat(),
                "sample_count": len(self._samples),
                "emergency_flush": emergency,
                "samples": list(self._samples),
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2)

            logger.info(
                f"Wrote vitals chunk: {filename} ({len(self._samples)} samples)"
            )
            self._samples.clear()
            self._last_chunk_write = time.time()

            # Cleanup old chunks (keep last 24 hours = ~96 files at 15min interval)
            self._cleanup_old_chunks(max_files=100)
        except Exception as e:
            logger.error(f"Failed to write vitals chunk: {e}")

    def _cleanup_old_chunks(self, max_files: int = 100):
        """Remove old chunk files, keeping only the most recent ones."""
        try:
            files = sorted(self.output_dir.glob("*.json"), key=os.path.getmtime)
            if len(files) > max_files:
                for old_file in files[:-max_files]:
                    old_file.unlink()
                    logger.debug(f"Removed old vitals chunk: {old_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old vitals chunks: {e}")

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info(
            f"SystemMonitor started (interval={self.sample_interval}s, chunk={self.chunk_interval}s)"
        )

        while self._running:
            try:
                sample = self._collect_sample()
                self._samples.append(sample)

                # Check for emergency flush
                if self._should_emergency_flush(sample):
                    logger.warning(
                        f"Critical throttle event detected: {sample['throttled']['flags']}"
                    )
                    self._write_chunk(emergency=True)
                # Regular chunk interval
                elif time.time() - self._last_chunk_write >= self.chunk_interval:
                    self._write_chunk()

            except Exception as e:
                logger.error(f"Error in vitals collection: {e}")

            # Sleep in small increments to allow quick shutdown
            sleep_end = time.time() + self.sample_interval
            while self._running and time.time() < sleep_end:
                time.sleep(1.0)

    def start(self):
        """Start the monitoring thread."""
        if self._running:
            logger.warning("SystemMonitor already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread and flush remaining samples."""
        if not self._running:
            return

        logger.info("Stopping SystemMonitor...")
        self._running = False

        if self._thread:
            self._thread.join(timeout=5.0)

        # Final flush on shutdown
        if self._samples:
            self._write_chunk()
        logger.info("SystemMonitor stopped")

    def get_current_vitals(self) -> dict:
        """Get the most recent vitals sample (for API/UI use)."""
        if self._samples:
            return self._samples[-1]
        return self._collect_sample()
