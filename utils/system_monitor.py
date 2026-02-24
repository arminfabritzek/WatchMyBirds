# utils/system_monitor.py
"""
System vitals monitor for Raspberry Pi crash diagnosis.

Collects hardware metrics (voltage, temperature, throttling, FDs) and writes them
to a CSV file with forced fsync to survive hard crashes.
"""

import csv
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from logging_config import get_logger

logger = get_logger(__name__)

CSV_HEADERS = [
    "timestamp",
    "cpu_temp_c",
    "cpu_percent",
    "ram_percent",
    "disk_percent",
    "fd_count",
    "thread_count",
    "throttled_hex",
    "uptime_seconds",
    "process_rss_mb",
    "children_count",
    "children_rss_mb",
    "ffmpeg_count",
    "ffmpeg_rss_mb",
    "app_total_rss_mb",
]

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


def is_raspberry_pi() -> bool:
    """Check if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            return "raspberry pi" in f.read().lower()
    except FileNotFoundError:
        return False


def run_vcgencmd(cmd: str) -> str | None:
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


def parse_throttled(value_str: str | None) -> dict[str, Any]:
    """Parse vcgencmd get_throttled output into human-readable flags."""
    if not value_str:
        return {"raw": None, "flags": []}
    try:
        # Accept both "throttled=0x0" and "0x0"
        hex_val = value_str.split("=", 1)[1] if "=" in value_str else value_str
        value = int(hex_val, 16)
        flags = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
        return {"raw": hex_val, "flags": flags}
    except (IndexError, ValueError):
        return {"raw": value_str, "flags": []}


def get_cpu_temp() -> float | None:
    """Get CPU temperature in Celsius."""
    # Try vcgencmd first (RPi specific)
    temp_str = run_vcgencmd("measure_temp")
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


def get_core_voltage() -> str | None:
    """Get core voltage (RPi specific)."""
    volt_str = run_vcgencmd("measure_volts core")
    if volt_str:
        try:
            # Output format: "volt=1.2000V"
            return volt_str.split("=")[1]
        except IndexError:
            return volt_str
    return None


class SystemMonitor:
    """
    Robust system vitals monitor for crash diagnosis.

    - Collects hardware metrics (CPU, RAM, Temp, Disk, FDs).
    - Writes CSV logs with explicit fsync to survive hard power cuts.
    - Provides real-time stats for the UI.
    """

    def __init__(
        self,
        output_dir: str,
        sample_interval_seconds: float = 60.0,
        chunk_interval_seconds: float = 0,  # Deprecated, kept for compatibility
        max_samples_in_memory: int = 0,  # Deprecated, kept for compatibility
    ):
        _ = (chunk_interval_seconds, max_samples_in_memory)
        self.output_dir = Path(output_dir) / "logs"
        self.csv_path = self.output_dir / "vital_signs.csv"
        self.interval = (
            float(sample_interval_seconds) if sample_interval_seconds > 0 else 60.0
        )

        self._running = False
        self._thread: threading.Thread | None = None
        self._is_rpi = is_raspberry_pi()
        self._process = psutil.Process()
        self._last_sample: dict[str, Any] = {}
        self._prev_sample: dict[str, Any] | None = None
        self._fd_high_watermark = 0
        self._app_rss_high_watermark_mb = 0.0
        self._ffmpeg_rss_high_watermark_mb = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Create CSV with header if missing/empty, rotate if header schema changed."""
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            try:
                with open(self.csv_path, newline="", encoding="utf-8") as f:
                    first_row = next(csv.reader(f), [])
                if first_row == CSV_HEADERS:
                    return

                legacy_path = self.output_dir / (
                    f"vital_signs_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                self.csv_path.rename(legacy_path)
                logger.warning(
                    f"Existing vital_signs.csv header mismatch. Rotated legacy file to {legacy_path.name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to validate/rotate existing vital_signs.csv: {e}"
                )

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Failed to init CSV header: {e}")

    def _get_uptime_seconds(self) -> int:
        """Get uptime with fallbacks. Never raises."""
        try:
            return int(max(0.0, time.time() - psutil.boot_time()))
        except Exception:
            # Linux fallback independent of psutil permission model.
            try:
                with open("/proc/uptime", encoding="utf-8") as f:
                    return int(float(f.read().split()[0]))
            except Exception:
                return 0

    def _collect_sample(self) -> dict[str, Any]:
        """Collect a single system vitals sample."""
        now = datetime.now()

        throttled_hex = "NA"
        if self._is_rpi:
            throttled_raw = run_vcgencmd("get_throttled")
            if throttled_raw:
                throttled_hex = (
                    throttled_raw.split("=", 1)[1]
                    if "=" in throttled_raw
                    else throttled_raw
                )

        try:
            disk_usage = psutil.disk_usage(str(self.output_dir)).percent
        except Exception:
            disk_usage = 0.0

        fd_count = -1
        thread_count = -1
        process_rss_mb = 0.0
        try:
            with self._process.oneshot():
                try:
                    fd_count = self._process.num_fds()
                except Exception:
                    fd_count = -1
                thread_count = self._process.num_threads()
                process_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        children_count = 0
        children_rss_mb = 0.0
        ffmpeg_count = 0
        ffmpeg_rss_mb = 0.0
        try:
            for child in self._process.children(recursive=True):
                children_count += 1
                child_rss_mb = 0.0
                try:
                    child_rss_mb = child.memory_info().rss / (1024 * 1024)
                    children_rss_mb += child_rss_mb
                except Exception:
                    pass

                # Classify ffmpeg children to separate camera/stream process pressure.
                try:
                    name = child.name().lower()
                    cmd = " ".join(child.cmdline()).lower()
                    if "ffmpeg" in name or "ffmpeg" in cmd:
                        ffmpeg_count += 1
                        ffmpeg_rss_mb += child_rss_mb
                except Exception:
                    pass
        except Exception:
            pass

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_percent = 0.0
        try:
            ram_percent = psutil.virtual_memory().percent
        except Exception:
            ram_percent = 0.0

        sample: dict[str, Any] = {
            "ts": now.isoformat(),
            "cpu_temp_c": get_cpu_temp(),
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "disk_percent": disk_usage,
            "fd_count": fd_count,
            "thread_count": thread_count,
            "throttled_hex": throttled_hex,
            "uptime_seconds": self._get_uptime_seconds(),
            "process_rss_mb": round(process_rss_mb, 1),
            "children_count": children_count,
            "children_rss_mb": round(children_rss_mb, 1),
            "ffmpeg_count": ffmpeg_count,
            "ffmpeg_rss_mb": round(ffmpeg_rss_mb, 1),
            "app_total_rss_mb": round(process_rss_mb + children_rss_mb, 1),
        }

        if self._is_rpi:
            sample["core_voltage"] = get_core_voltage()

        # Keep backward-compatible field for API/UI callers.
        sample["throttled"] = parse_throttled(f"throttled={throttled_hex}")
        return sample

    def _append_csv_row(self, sample: dict[str, Any]) -> None:
        """Append one sample row and fsync it."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    sample["ts"],
                    sample["cpu_temp_c"],
                    sample["cpu_percent"],
                    sample["ram_percent"],
                    sample["disk_percent"],
                    sample["fd_count"],
                    sample["thread_count"],
                    sample["throttled_hex"],
                    sample["uptime_seconds"],
                    sample.get("process_rss_mb", 0),
                    sample.get("children_count", 0),
                    sample.get("children_rss_mb", 0),
                    sample.get("ffmpeg_count", 0),
                    sample.get("ffmpeg_rss_mb", 0),
                    sample.get("app_total_rss_mb", 0),
                ]
            )
            f.flush()
            os.fsync(f.fileno())

    def _dump_fds(self, count: int) -> None:
        """Dump open FDs to a file to debug leaks."""
        if not os.path.isdir("/proc/self/fd"):
            return
        dump_file = self.output_dir / "fd_leak_dump.txt"
        try:
            fds = os.listdir("/proc/self/fd")
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total FDs: {count}\n")
                f.write("-" * 40 + "\n")
                for fd in fds:
                    try:
                        path = os.readlink(f"/proc/self/fd/{fd}")
                        f.write(f"{fd} -> {path}\n")
                    except Exception:
                        f.write(f"{fd} -> (unknown)\n")
            logger.warning(f"Dumped {count} FDs to {dump_file}")
        except Exception as e:
            logger.error(f"Failed to dump FDs: {e}")

    @staticmethod
    def _format_delta(value: float | int | None) -> str:
        """Format delta values for structured logs."""
        if value is None:
            return "na"
        if isinstance(value, int):
            return str(value)
        return f"{value:.1f}"

    def _log_sample_diagnostics(self, sample: dict[str, Any]) -> None:
        """Emit a compact diagnostics summary into app.log for UI visibility."""
        fd_count = int(sample.get("fd_count", -1))
        app_total_rss_mb = float(sample.get("app_total_rss_mb", 0.0))
        ffmpeg_rss_mb = float(sample.get("ffmpeg_rss_mb", 0.0))

        if fd_count >= 0:
            self._fd_high_watermark = max(self._fd_high_watermark, fd_count)
        self._app_rss_high_watermark_mb = max(
            self._app_rss_high_watermark_mb, app_total_rss_mb
        )
        self._ffmpeg_rss_high_watermark_mb = max(
            self._ffmpeg_rss_high_watermark_mb, ffmpeg_rss_mb
        )

        fd_delta: int | None = None
        app_total_rss_delta_mb: float | None = None
        ffmpeg_rss_delta_mb: float | None = None
        if self._prev_sample:
            prev_fd = self._prev_sample.get("fd_count")
            if isinstance(prev_fd, int) and prev_fd >= 0 and fd_count >= 0:
                fd_delta = fd_count - prev_fd

            prev_total_rss = self._prev_sample.get("app_total_rss_mb")
            if isinstance(prev_total_rss, (int, float)):
                app_total_rss_delta_mb = app_total_rss_mb - float(prev_total_rss)

            prev_ffmpeg_rss = self._prev_sample.get("ffmpeg_rss_mb")
            if isinstance(prev_ffmpeg_rss, (int, float)):
                ffmpeg_rss_delta_mb = ffmpeg_rss_mb - float(prev_ffmpeg_rss)

        throttle_flags = sample.get("throttled", {}).get("flags", [])
        throttle_flags_str = ",".join(throttle_flags) if throttle_flags else "none"

        logger.debug(
            "event=system_vitals "
            f"cpu_temp_c={sample.get('cpu_temp_c')} "
            f"cpu_percent={sample.get('cpu_percent')} "
            f"ram_percent={sample.get('ram_percent')} "
            f"disk_percent={sample.get('disk_percent')} "
            f"throttled_hex={sample.get('throttled_hex')} "
            f"throttle_flags={throttle_flags_str} "
            f"uptime_seconds={sample.get('uptime_seconds')} "
            f"fd_count={fd_count} "
            f"fd_high_watermark={self._fd_high_watermark} "
            f"fd_delta={self._format_delta(fd_delta)} "
            f"thread_count={sample.get('thread_count')} "
            f"process_rss_mb={sample.get('process_rss_mb')} "
            f"children_count={sample.get('children_count')} "
            f"children_rss_mb={sample.get('children_rss_mb')} "
            f"ffmpeg_count={sample.get('ffmpeg_count')} "
            f"ffmpeg_rss_mb={sample.get('ffmpeg_rss_mb')} "
            f"ffmpeg_rss_high_watermark_mb={self._ffmpeg_rss_high_watermark_mb:.1f} "
            f"ffmpeg_rss_delta_mb={self._format_delta(ffmpeg_rss_delta_mb)} "
            f"app_total_rss_mb={sample.get('app_total_rss_mb')} "
            f"app_total_rss_high_watermark_mb={self._app_rss_high_watermark_mb:.1f} "
            f"app_total_rss_delta_mb={self._format_delta(app_total_rss_delta_mb)}"
        )
        self._prev_sample = sample

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.info(
            f"SystemMonitor started (interval={self.interval}s, target={self.csv_path})"
        )
        while self._running:
            try:
                sample = self._collect_sample()
                self._last_sample = sample
                self._append_csv_row(sample)
                self._log_sample_diagnostics(sample)

                if sample["ram_percent"] > 90:
                    logger.warning(
                        f"HIGH MEMORY USAGE WARNING: {sample['ram_percent']}%"
                    )

                if sample["fd_count"] > 600:
                    logger.warning(
                        f"HIGH FD COUNT LEAK WARNING: {sample['fd_count']} open files!"
                    )
                    self._dump_fds(sample["fd_count"])
            except Exception as e:
                logger.error(f"Error in vitals collection: {e}")

            sleep_end = time.time() + self.interval
            while self._running and time.time() < sleep_end:
                time.sleep(1.0)

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            logger.warning("SystemMonitor already running")
            return
        self._ensure_csv_header()
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self._running:
            return
        logger.info("Stopping SystemMonitor...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SystemMonitor stopped")

    def get_current_vitals(self) -> dict[str, Any]:
        """Get the most recent vitals sample for API/UI use."""
        if self._last_sample:
            return self._last_sample.copy()
        return self._collect_sample()
