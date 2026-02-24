# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Web Interface
# main.py
# ------------------------------------------------------------------------------
import atexit
import json
import os
import signal
import socket
import threading
from datetime import datetime

from config import (
    ensure_app_directories,
    ensure_go2rtc_stream_synced,
    get_config,
    resolve_effective_sources,
)
from logging_config import get_logger
from utils.cpu_limiter import restrict_to_cpus
from utils.system_monitor import SystemMonitor
from utils.telegram_notifier import send_telegram_message
from web.web_interface import create_web_interface

logger = get_logger(__name__)


def _detect_runtime_environment():
    """Detect whether app runs on host or inside a container runtime."""
    if os.path.exists("/.dockerenv"):
        return "docker"
    cgroup_paths = ("/proc/1/cgroup", "/proc/self/cgroup")
    for path in cgroup_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                content = handle.read().lower()
                if "docker" in content or "containerd" in content:
                    return "container"
        except Exception:
            continue
    return "host"


def _log_restart_marker():
    """
    Emit a highly visible startup marker so restarts are obvious in app.log.
    """
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    pid = os.getpid()
    ppid = os.getppid()
    runtime = _detect_runtime_environment()
    hostname = socket.gethostname()
    marker = "#" * 92
    logger.info(marker)
    logger.info(
        "APP_RESTART_MARKER started_at=%s pid=%s ppid=%s runtime=%s host=%s",
        timestamp,
        pid,
        ppid,
        runtime,
        hostname,
    )
    logger.info(marker)


def _create_runtime():
    """
    Build runtime components only for real app execution.
    This function is intentionally not run on module import so multiprocessing
    worker imports (e.g. __mp_main__) do not start duplicate app instances.
    """
    config = get_config()
    ensure_app_directories(config)
    _log_restart_marker()
    from detectors.detection_manager import DetectionManager

    # Apply CPU restriction before starting any threads for slow systems
    restrict_to_cpus()

    debug_mode = config["DEBUG_MODE"]
    output_dir = config["OUTPUT_DIR"]

    # --- Sync go2rtc before resolving stream sources ---
    # Must run BEFORE resolve_effective_sources() to break the chicken-and-egg
    # problem: the resolver needs go2rtc to have the stream configured, but the
    # old post-resolve sync only ran when mode was already 'relay'.
    ensure_go2rtc_stream_synced(config, with_retry=True)

    # --- Resolve effective stream sources (§5) ---
    resolved = resolve_effective_sources(config)
    config["VIDEO_SOURCE"] = resolved["video_source"]
    logger.info(
        "STREAM_SOURCE stream_mode=%s video_source=%s reason=%s",
        resolved["effective_mode"],
        resolved["video_source"][:40] + "..."
        if len(str(resolved["video_source"])) > 40
        else resolved["video_source"],
        resolved["reason"],
    )

    logger.info(f"Debug mode is {'enabled' if debug_mode else 'disabled'}.")
    logger.debug(f"Configuration: {json.dumps(config, indent=2)}")

    # Security Audit Warning
    if config.get("EDIT_PASSWORD") == "watchmybirds":
        logger.warning(
            "SECURITY WARNING: Using default password 'watchmybirds'. "
            "If EDIT_PASSWORD is unchanged, the UI is protected but not personalized."
        )

    if debug_mode:
        send_telegram_message(
            text="🐦 Birdwatching has started in DEBUG mode!",
            photo_path="assets/debug.jpg",
        )

    # go2rtc sync already handled by ensure_go2rtc_stream_synced() above.

    detection_manager = DetectionManager()
    threading.Thread(target=detection_manager.start, daemon=True).start()
    atexit.register(detection_manager.stop)

    monitor_interval_raw = os.environ.get("SYSTEM_MONITOR_INTERVAL_SECONDS", "15")
    try:
        monitor_interval_seconds = float(monitor_interval_raw)
        if monitor_interval_seconds <= 0:
            raise ValueError("interval must be positive")
    except ValueError:
        logger.warning(
            "Invalid SYSTEM_MONITOR_INTERVAL_SECONDS=%r. Falling back to 15s.",
            monitor_interval_raw,
        )
        monitor_interval_seconds = 15.0

    logger.info(
        "SystemMonitor configured with interval=%.1fs",
        monitor_interval_seconds,
    )
    system_monitor = SystemMonitor(
        output_dir=output_dir,
        sample_interval_seconds=monitor_interval_seconds,
    )
    system_monitor.start()
    atexit.register(system_monitor.stop)

    # Start Weather background service (polls Open-Meteo every 30 min)
    try:
        from web.services.weather_service import start_weather_loop

        start_weather_loop(interval=1800)
    except Exception as e:
        logger.warning(f"Weather service failed to start: {e}")

    # Start Analysis Queue Worker (Deep Review)
    try:
        from core.analysis_queue import analysis_queue
        from web.services.analysis_service import (
            process_deep_analysis_job,
            start_nightly_analysis_sweep,
        )

        # Inject DetectionManager for deep-scan gate control
        analysis_queue.set_detection_manager(detection_manager)

        # Use lambda to inject detection_manager dependency
        analysis_queue.start(
            lambda job: process_deep_analysis_job(detection_manager, job)
        )

        # Start nightly sweep
        start_nightly_analysis_sweep()

        atexit.register(analysis_queue.stop)
    except Exception as e:
        logger.warning(f"Analysis Queue failed to start: {e}")

    # Start Daily Report Scheduler (sends Telegram evening report)
    try:
        from web.services.report_scheduler import start_report_scheduler

        start_report_scheduler(detection_manager=detection_manager)
    except Exception as e:
        logger.warning(f"Daily report scheduler failed to start: {e}")

    app = create_web_interface(detection_manager, system_monitor=system_monitor)
    return app, detection_manager


def main():
    app, detection_manager = _create_runtime()

    # Handle SIGTERM (Docker stop) gracefully
    def handle_sigterm(*args):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, handle_sigterm)

    # Use Waitress instead of Werkzeug dev server.
    # Werkzeug delays accepting connections for ~8s after socket.bind(),
    # causing the UI to be unreachable immediately after startup.
    # Waitress responds instantly, providing consistent dev/prod behavior.
    from waitress import serve

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8050))
    logger.info(f"Starting Waitress server on http://{host}:{port}")
    try:
        # threads=8 because /video_feed holds connections open indefinitely (streaming),
        # which can exhaust the default 4 threads and block normal page requests.
        # max_request_body_size=10GB to allow large backup uploads (default is 1GB)
        serve(
            app,
            host=host,
            port=port,
            threads=8,
            max_request_body_size=10 * 1024 * 1024 * 1024,  # 10 GB
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down detection manager...")
        detection_manager.stop()


if __name__ == "__main__":
    main()
