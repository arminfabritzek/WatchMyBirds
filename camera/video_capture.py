# ------------------------------------------------------------------------------
# Video Capture Class for Input Streams
# camera/video_capture.py
# ------------------------------------------------------------------------------
# video_capture.py
from config import get_config

config = get_config()

import atexit
import itertools
import json
import os
import queue
import select
import signal
import subprocess
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)


class VideoCapture:
    """
    Handles video input streams from various sources (RTSP, WebRTC, Webcam, FFmpeg).
    """

    RTSP = "rtsp"
    HTTP = "http"
    WEBCAM = "webcam"
    BACKEND_OPENCV = "opencv"
    BACKEND_FFMPEG = "ffmpeg"
    _instances = weakref.WeakSet()
    _shutdown_hooks_registered = False
    _ffmpeg_version_cache = None

    def __init__(self, source, debug=False, auto_start=False):
        """
        Initializes the VideoCapture object and detects the stream type.
        """
        self.source = source
        self.debug = debug
        self.stop_flag = False
        self.stop_event = Event()
        self.cap = None
        self.ffmpeg_process = None
        self.ffmpeg_pid = None
        self.stream_width = None
        self.stream_height = None
        self.q = queue.Queue(maxsize=1)
        self.reader_thread = None
        self.last_frame_time = time.time()
        self.health_check_thread = None
        self.retry_count = 0
        self.stream_type = self._detect_stream_type()
        self.reinit_lock = threading.Lock()
        self.consecutive_none_frames = 0
        self.max_none_frames_before_reconnect = 50

        self.failed_reads = 0
        self.last_codec_switch_time = 0
        self.codec_switch_cooldown = 10  # seconds between codec switch attempts
        self.backend_switch_cooldown = 15  # seconds between backend switches
        self.stream_settings_loaded = False
        self._cache_reuse_failed_once = False
        self.runtime_resolution_lock = threading.Lock()
        self._ffmpeg_last_availability = True
        self._last_codec_switch_skip_logged = False
        self._last_frame_drop_logged = False
        self._last_frame_drop_warn_mono = 0.0
        self._last_read_error_logged = False
        self._health_check_error_state = False
        self._last_logged_ffmpeg_line = None
        self._preexec_skip_logged = False
        self._startup_initial_frame_wait_sec = 5.0
        self._recovery_initial_frame_wait_sec = 12.0

        # -- Recovery Dispatcher Infrastructure (Steps 0-6) --
        self._recovery_gate = threading.Lock()          # single-flight gate
        self._recovery_gen = itertools.count(1)         # attempt id generator
        self._recovery_cooldown_until = 0.0             # earliest next recovery ts
        self._recovery_cooldown_sec = 5.0               # min gap between recoveries
        # Circuit breaker (Step 6)
        self._breaker_timestamps = []                   # recent recovery attempt times
        self._breaker_threshold = 5                     # max attempts in window
        self._breaker_window_sec = 60.0                 # window size
        self._breaker_offline_until = 0.0               # OFFLINE cooldown end
        self._breaker_offline_sec = 60.0                # OFFLINE duration
        # Process registry (Step 4)
        self._child_registry = {}                       # pid -> {handle, start_ts, kind, gen}
        self._child_hard_cap = 4                        # max concurrent children
        # Timer coalescing (Step 5)
        self._pending_reinit_timer = None               # at most one pending timer
        self._pending_reinit_lock = threading.Lock()

        logger.debug(
            f"Initialized VideoCapture with source: {self.source}, stream_type: {self.stream_type}"
        )
        self._register_instance_for_shutdown()
        self._prime_stream_settings_from_cache()
        if auto_start:
            self.start()

    def start(self):
        """Starts the video stream and background threads."""
        self.stop_flag = False
        if self.is_running():
            logger.debug("VideoCapture already running.")
            return
        if self.stream_type == self.RTSP:
            if not self.stream_settings_loaded:
                self._get_stream_resolution_ffprobe()
                logger.debug(f"Initial resolution: {self.resolution}")
            else:
                logger.debug(
                    f"Using cached stream settings without re-probe: resolution={self.resolution}"
                )
        try:
            self._setup_capture()
        except Exception as start_error:
            if (
                self.stream_type == self.RTSP
                and self.stream_settings_loaded
                and not self._cache_reuse_failed_once
            ):
                logger.warning(
                    "Cached stream settings failed to start. "
                    "Invalidating cache and retrying with probing."
                )
                self._cache_reuse_failed_once = True
                self._invalidate_cache_entry()
                self.stream_settings_loaded = False
                self.stream_width = None
                self.stream_height = None
                self._get_stream_resolution_ffprobe()
                self._setup_capture()
            else:
                raise start_error
        self._start_reader_thread()
        self._start_health_check_thread()

    def is_running(self):
        """Checks if the reader thread is active."""
        return self.reader_thread is not None and self.reader_thread.is_alive()

    def _detect_stream_type(self):
        """Determines the stream type based on the source."""
        logger.debug("Detect stream type...")
        if isinstance(self.source, str):
            if self.source.startswith("rtsp://"):
                logger.debug("Stream type detected as RTSP.")
                return self.RTSP
            elif self.source.startswith(("http://", "https://")):
                logger.debug("Stream type detected as HTTP.")
                return self.HTTP
        elif isinstance(self.source, int):
            logger.debug("Stream type detected as Webcam.")
            return self.WEBCAM
        error_msg = f"Unable to determine stream type for source: {self.source}"
        logger.error(error_msg)
        raise ValueError(f"Unable to determine stream type for source: {self.source}")

    def _setup_capture(
        self, require_initial_frame=False, initial_frame_wait_sec=None
    ):
        logger.debug("Setting up video capture...")
        if self.stream_type == self.RTSP:
            try:
                self._setup_ffmpeg()
                self.backend = self.BACKEND_FFMPEG
                logger.debug("Using FFmpeg backend successfully.")
            except Exception as e:
                logger.warning(f"FFmpeg capture failed: {e}. Falling back to OpenCV.")
                self._setup_opencv_rtsp()
                self.backend = self.BACKEND_OPENCV
        elif self.stream_type == self.HTTP:
            self._setup_http()
            self.backend = self.BACKEND_OPENCV
        elif self.stream_type == self.WEBCAM:
            self._setup_webcam()
            self.backend = self.BACKEND_OPENCV
        else:
            raise ValueError(f"Unsupported stream type: {self.stream_type}")
        logger.debug("Video capture setup completed.")

        if initial_frame_wait_sec is None:
            initial_frame_wait_sec = self._startup_initial_frame_wait_sec
        try:
            initial_frame_wait_sec = max(0.5, float(initial_frame_wait_sec))
        except (TypeError, ValueError):
            initial_frame_wait_sec = self._startup_initial_frame_wait_sec
        warmup_attempts = max(1, int(initial_frame_wait_sec / 0.5))

        test_frame = None
        try:
            if self.backend == self.BACKEND_FFMPEG:
                logger.debug(
                    "Waiting up to "
                    f"{initial_frame_wait_sec:.1f}s for FFmpeg to produce the first frame..."
                )
                for _ in range(warmup_attempts):
                    test_frame = self._read_ffmpeg_frame(read_timeout_sec=0.5)
                    if test_frame is not None:
                        logger.debug("Received initial frame from FFmpeg.")
                        break
                    time.sleep(0.5)
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                test_frame = frame if ret else None
        except Exception as e:
            logger.warning(f"Error testing frame availability: {e}")

        if test_frame is None:
            if require_initial_frame:
                raise RuntimeError(
                    "Initial test frame missing during strict capture setup "
                    f"(backend={self.backend}, wait_sec={initial_frame_wait_sec:.1f})."
                )
            if self.stream_type == self.RTSP and self.backend == self.BACKEND_FFMPEG:
                # Keep FFmpeg-first startup path: some cameras need extra warm-up
                # before the first decodable frame appears.
                logger.warning(
                    "Initial test frame missing from FFmpeg. "
                    "Keeping FFmpeg backend and relying on startup grace/recovery."
                )
                return
            now = time.time()
            if now - self.last_codec_switch_time < self.backend_switch_cooldown:
                logger.debug(
                    "Backend switch cooldown active, skipping immediate switch."
                )
                return
            logger.warning(
                "Initial test frame still missing. Proceeding anyway to allow stream startup on slow devices (e.g., NAS)."
            )
            if self.backend == self.BACKEND_FFMPEG:
                self._terminate_ffmpeg_process(
                    reason="switching backend after missing initial frame"
                )
                if self.cap:
                    self.cap.release()
                self._setup_opencv_rtsp()
                self.backend = self.BACKEND_OPENCV
            else:
                if self.cap:
                    self.cap.release()
                self._setup_ffmpeg()
                self.backend = self.BACKEND_FFMPEG
            self.last_codec_switch_time = now

    def _setup_opencv_rtsp(self):
        """Try to initialize RTSP stream with OpenCV."""
        logger.debug("Trying OpenCV backend for RTSP stream...")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("OpenCV could not open RTSP stream")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(
            f"OpenCV RTSP setup successful: {self.stream_width}x{self.stream_height}"
        )
        self._persist_stream_settings()
        self.backend = self.BACKEND_OPENCV

    def _setup_ffmpeg(self):
        self._terminate_ffmpeg_process(reason="preparing new FFmpeg start")
        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "50000000",
            "-analyzeduration",
            "10000000",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.source,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "pipe:",
        ]

        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        for attempt in range(5):  # Retry up to 5 times
            try:
                logger.debug(f"Starting FFmpeg process (attempt {attempt + 1})...")
                popen_kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "bufsize": 10**8,
                    "start_new_session": True,
                }
                if self._can_use_ffmpeg_preexec_fn():
                    preexec_fn = self._make_ffmpeg_preexec_fn()
                    if preexec_fn is not None:
                        popen_kwargs["preexec_fn"] = preexec_fn
                        popen_kwargs["start_new_session"] = False

                self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, **popen_kwargs)
                self.ffmpeg_pid = self.ffmpeg_process.pid
                self._register_child(self.ffmpeg_process, kind="ffmpeg")
                logger.debug(f"FFmpeg process started with pid {self.ffmpeg_pid}.")
                logger.debug("FFmpeg process started successfully.")

                # Start logging FFmpeg errors if in debug mode
                if self.debug:
                    threading.Thread(
                        target=self._log_ffmpeg_errors, daemon=True
                    ).start()

                # Verify FFmpeg process is running
                if self.ffmpeg_process.poll() is not None:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    error_msg = f"FFmpeg process terminated prematurely. STDERR: {stderr_output}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                return  # Exit loop on success
            except Exception as e:
                logger.error(f"Failed to start FFmpeg process: {e}. Retrying...")
                self._terminate_ffmpeg_process(reason="failed FFmpeg start")
                time.sleep(2**attempt)  # Exponential backoff
        error_msg = "Failed to start FFmpeg after multiple attempts."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _terminate_ffmpeg_process(self, reason=""):
        """Beendet laufenden FFmpeg-Prozess inklusive Prozessgruppe."""
        process = self.ffmpeg_process
        if not process:
            return

        pid = self.ffmpeg_pid or process.pid
        reason_suffix = f" ({reason})" if reason else ""
        logger.debug(f"Stopping FFmpeg process{reason_suffix} (pid {pid}).")

        try:
            if process.poll() is None:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except Exception as term_error:
                        logger.debug(
                            f"SIGTERM process group failed: {term_error}; "
                            "falling back to terminate()."
                        )
                        process.terminate()
                else:
                    process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "FFmpeg did not terminate after SIGTERM, sending SIGKILL."
                    )
                    if hasattr(os, "killpg"):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except Exception as kill_error:
                            logger.error(
                                f"Failed to kill FFmpeg process group: {kill_error}"
                            )
                    else:
                        process.kill()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.error(
                            "FFmpeg process did not exit after SIGKILL attempt."
                        )
        finally:
            for stream in (process.stdout, process.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            self.ffmpeg_process = None
            self.ffmpeg_pid = None

    def _stream_settings_path(self):
        """Returns the path for persisted stream settings."""
        try:
            output_dir = Path(config["OUTPUT_DIR"])
        except Exception:
            # Should practically never happen with new strict config
            output_dir = Path("data/output")
        return output_dir / "stream_settings.json"

    def _get_ffmpeg_version(self):
        """Liest die FFmpeg-Version zur Cache-Validierung."""
        if VideoCapture._ffmpeg_version_cache is not None:
            return VideoCapture._ffmpeg_version_cache
        try:
            output = subprocess.check_output(
                ["ffmpeg", "-version"], stderr=subprocess.STDOUT
            )
            first_line = output.decode().splitlines()[0]
            VideoCapture._ffmpeg_version_cache = first_line.strip()
        except Exception as version_error:
            logger.debug(f"Could not determine FFmpeg version: {version_error}")
            VideoCapture._ffmpeg_version_cache = None
        return VideoCapture._ffmpeg_version_cache

    def _load_stream_settings_from_cache(self):
        """Reads cached stream settings for the current source."""
        cache_path = self._stream_settings_path()
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
            return data.get(str(self.source))
        except Exception as cache_error:
            logger.debug(f"Could not read stream settings cache: {cache_error}")
            return None

    def _prime_stream_settings_from_cache(self):
        """Loads cached settings into memory to skip probing."""
        cached = self._load_stream_settings_from_cache()
        if not cached:
            return
        try:
            if not self._cache_metadata_matches(cached):
                logger.debug("Stream settings cache ignored due to metadata mismatch.")
                return
            self.stream_width = int(cached["width"])
            self.stream_height = int(cached["height"])
            self.stream_settings_loaded = True
            logger.debug(
                f"Cache loaded for source {self.source}: "
                f"{self.stream_width}x{self.stream_height}"
            )
        except Exception as cache_error:
            logger.debug(f"Invalid cached stream settings: {cache_error}")

    def _cache_metadata_matches(self, cached):
        """Validates cached metadata (stream URL, type, FFmpeg version)."""
        try:
            if str(cached.get("stream_url")) != str(self.source):
                logger.debug("Stream settings cache ignored (URL changed).")
                return False
            if cached.get("stream_type") != self.stream_type:
                logger.debug("Stream settings cache ignored (stream type changed).")
                return False
            cached_version = cached.get("ffmpeg_version")
            current_version = self._get_ffmpeg_version()
            if cached_version and current_version and cached_version != current_version:
                logger.debug("Stream settings cache ignored (FFmpeg version changed).")
                return False
            return True
        except Exception as metadata_error:
            logger.debug(f"Cache metadata validation failed: {metadata_error}")
            return False

    def _validate_cached_stream_settings(self):
        """
        Re-probes resolution to ensure cached settings are still correct.
        Returns True if validation succeeded (even if resolution unchanged),
        False if validation failed.
        """
        try:
            prev_width, prev_height = self.stream_width, self.stream_height
            self._get_stream_resolution_ffprobe()
            if self.stream_width != prev_width or self.stream_height != prev_height:
                logger.debug(
                    f"Stream settings cache updated after resolution change: "
                    f"{prev_width}x{prev_height} -> {self.stream_width}x{self.stream_height}"
                )
            else:
                logger.debug("Stream settings cache validated; resolution unchanged.")
            return True
        except Exception as validation_error:
            logger.warning(
                f"Cached stream settings validation failed: {validation_error}"
            )
            return False

    def _persist_stream_settings(self):
        """Persists discovered stream settings to speed up future startups."""
        if (
            not hasattr(self, "stream_width")
            or not hasattr(self, "stream_height")
            or self.stream_width is None
            or self.stream_height is None
        ):
            return
        if self.stop_event.is_set() or self.stop_flag:
            logger.debug("Skipping stream settings persist during shutdown.")
            return
        cache_path = self._stream_settings_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                except Exception:
                    data = {}
            data[str(self.source)] = {
                "stream_type": self.stream_type,
                "stream_url": str(self.source),
                "width": int(self.stream_width),
                "height": int(self.stream_height),
                "ffmpeg_version": self._get_ffmpeg_version(),
                "cached_at": time.time(),
            }
            self._write_cache_atomically(cache_path, data)
            logger.debug(
                f"Cache overwritten for source {self.source}: "
                f"{self.stream_width}x{self.stream_height} -> {cache_path}"
            )
        except Exception as cache_error:
            logger.debug(f"Failed to persist stream settings: {cache_error}")

    def _write_cache_atomically(self, cache_path, data):
        """Writes cache atomically using a temp file and fsync."""
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            tmp_path.replace(cache_path)
        except Exception:
            # Best-effort cleanup; ignore failure.
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _invalidate_cache_entry(self):
        """Removes cached settings for the current source."""
        cache_path = self._stream_settings_path()
        if not cache_path.exists():
            return
        try:
            data = json.loads(cache_path.read_text())
            if str(self.source) in data:
                data.pop(str(self.source), None)
                self._write_cache_atomically(cache_path, data)
                logger.debug("Stream settings cache entry removed for current source.")
        except Exception as cache_error:
            logger.debug(f"Failed to invalidate cache entry: {cache_error}")

    def _log_ffmpeg_errors(self):
        """
        Continuously logs FFmpeg's stderr for debugging purposes, only if debug mode is enabled.
        """
        logger.debug("Starting FFmpeg stderr logging.")
        process = self.ffmpeg_process
        if not process or not process.stderr:
            logger.debug("FFmpeg stderr stream not available for logging.")
            return
        try:
            for line in iter(process.stderr.readline, b""):
                if line:
                    line_str = line.decode("utf-8", errors="replace").strip()
                    # Filter out progress noise and version banner
                    prefixes_to_ignore = [
                        "frame=",
                        "size=",
                        "ffmpeg version",
                        "built with",
                        "configuration:",
                        "libavutil",
                        "libavcodec",
                        "libavformat",
                        "libavdevice",
                        "libavfilter",
                        "libswscale",
                        "libswresample",
                        "libpostproc",
                        "  libavutil",
                        "  libavcodec",
                        "  libavformat",
                        "  libavdevice",
                        "  libavfilter",
                        "  libswscale",
                        "  libswresample",
                        "  libpostproc",
                        "  built with",
                        "  configuration:",
                    ]
                    if not any(line_str.startswith(p) for p in prefixes_to_ignore):
                        if line_str != self._last_logged_ffmpeg_line:
                            logger.debug(f"FFmpeg STDERR: {line_str}")
                            self._last_logged_ffmpeg_line = line_str
                if self.stop_event.is_set():
                    logger.debug("Stop event set. Ending FFmpeg stderr logging.")
                    break
        except Exception as e:
            logger.error(f"Exception while logging FFmpeg stderr: {e}")
        logger.debug("FFmpeg stderr logging thread terminated.")

    def _get_stream_resolution_ffprobe(self):
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            self.source,
        ]

        logger.debug(f"Running FFprobe command: {' '.join(ffprobe_cmd)}")

        try:
            output = (
                subprocess.check_output(
                    ffprobe_cmd, stderr=subprocess.STDOUT, timeout=60
                )
                .decode()
                .strip()
            )
            logger.debug(f"FFprobe output: {output}")
            width, height = map(int, output.split("\n"))
            self.stream_width = width
            self.stream_height = height
            logger.debug(f"Detected stream resolution via FFprobe: {width}x{height}")
            self._persist_stream_settings()
            return
        except Exception as ffprobe_error:
            logger.debug(f"FFprobe failed for resolution detection: {ffprobe_error}")

        # Fallback: use ffmpeg to parse stderr for resolution
        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.source,
            "-t",
            "1",
            "-f",
            "null",
            "-",
        ]

        logger.debug(
            f"Running FFmpeg command for resolution fallback: {' '.join(ffmpeg_cmd)}"
        )

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
            stderr_output = result.stderr.decode()
            logger.debug(f"FFmpeg stderr output:\n{stderr_output}")

            # Parse resolution from stderr
            import re

            match = re.search(r"(\d{2,5})x(\d{2,5})", stderr_output)
            if match:
                self.stream_width = int(match.group(1))
                self.stream_height = int(match.group(2))
                logger.debug(
                    f"FFmpeg fallback successfully detected resolution: {self.stream_width}x{self.stream_height}"
                )
                logger.debug(
                    f"Detected stream resolution via FFmpeg fallback: {self.stream_width}x{self.stream_height}"
                )
                self._persist_stream_settings()
            else:
                raise RuntimeError("FFmpeg did not return a parsable resolution.")
        except Exception as ffmpeg_error:
            logger.debug(
                f"FFmpeg fallback failed for resolution detection: {ffmpeg_error}"
            )
            raise RuntimeError(
                "Unable to determine stream resolution using probe or fallback."
            ) from ffmpeg_error

    def _setup_http(self):
        """
        Sets up OpenCV's VideoCapture for HTTP streams.
        """
        logger.debug(f"Setting up HTTP stream with source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open HTTP stream: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(
            f"Detected HTTP stream resolution: {self.stream_width}x{self.stream_height}"
        )
        self._persist_stream_settings()

    def _setup_webcam(self):
        """
        Sets up OpenCV's VideoCapture for webcam streams.
        """
        logger.debug(f"Setting up Webcam with source index: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open webcam with index: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(
            f"Detected Webcam resolution: {self.stream_width}x{self.stream_height}"
        )
        self._persist_stream_settings()

    def _start_reader_thread(self):
        """
        Starts the reader thread if it's not already running.
        Ensures only one thread is active at a time to avoid conflicts.
        """
        if self.reader_thread and self.reader_thread.is_alive():
            if self.stop_event.is_set() or self.stop_flag:
                logger.warning(
                    "Reader thread still alive while shutdown/restart is in progress; "
                    "waiting briefly before reusing it."
                )
                self.reader_thread.join(timeout=1.5)
            if self.reader_thread and self.reader_thread.is_alive():
                logger.warning(
                    "Reader thread is still alive; skipping new reader startup for now."
                )
                return

        else:
            # Add a short delay to allow FFmpeg to initialize properly
            logger.debug("Starting reader thread...")
            time.sleep(2)
            self.reader_thread = threading.Thread(
                target=self._reader, daemon=True, name="ReaderThread"
            )
            self.reader_thread.start()
            logger.debug("Reader thread started successfully.")

    def _start_health_check_thread(self):
        """
        Starts a health check thread to monitor FFmpeg subprocess.
        """
        if self.stream_type != self.RTSP:
            logger.debug(
                "Health check thread is only applicable for RTSP streams. Skipping."
            )
            return

        if self.health_check_thread and self.health_check_thread.is_alive():
            logger.debug(
                "Health check thread already running; skipping reinitialization."
            )
        else:
            logger.debug("Starting health check thread...")
            self.health_check_thread = threading.Thread(
                target=self._health_check, daemon=True, name="HealthCheckThread"
            )
            self.health_check_thread.start()
            logger.debug("Health check thread started successfully.")

    def _health_check(self):
        """
        Periodically checks if the stream (FFmpeg for RTSP, or OpenCV for HTTP) is alive.
        If not, attempts to reinitialize.
        """
        logger.debug("Health check thread is running.")
        try:
            while not self.stop_event.is_set():
                if self.stream_type == self.RTSP:
                    if self.ffmpeg_process:
                        retcode = self.ffmpeg_process.poll()
                        if retcode is not None:
                            if not self._health_check_error_state:
                                # Step 4 (stderr ownership): Do NOT read stderr here.
                                # Only _log_ffmpeg_errors is the designated consumer.
                                logger.error(
                                    f"FFmpeg subprocess terminated with return code {retcode}."
                                )
                                self._health_check_error_state = True
                            self.request_recovery(
                                trigger="health_check",
                                reason="RTSP FFmpeg subprocess terminated unexpectedly."
                            )
                        else:
                            if time.time() - self.last_frame_time > 10:
                                if not self._health_check_error_state:
                                    logger.warning(
                                        "No frame received for over 10 seconds; triggering reinitialization."
                                    )
                                    self._health_check_error_state = True
                                self.request_recovery(trigger="health_check", reason="RTSP stream stale.")
                            else:
                                if self._health_check_error_state:
                                    logger.debug(
                                        "RTSP stream recovered (frames flowing)."
                                    )
                                    self._health_check_error_state = False
                elif self.stream_type == self.HTTP:
                    if not self.cap or not self.cap.isOpened():
                        if not self._health_check_error_state:
                            logger.error(
                                "HTTP stream is not opened. Triggering reinitialization."
                            )
                            self._health_check_error_state = True
                        self.request_recovery(trigger="health_check", reason="HTTP stream not opened.")
                    else:
                        if self._health_check_error_state:
                            logger.debug("HTTP stream recovered.")
                            self._health_check_error_state = False
                time.sleep(5)
        except Exception as e:
            logger.error(f"Health check encountered error: {e}")
        logger.debug("Health check thread is stopping.")

    def _reader(self):
        """
        Continuously reads frames from the video source and places only the latest frame in the queue.
        Terminates cleanly if stop_event is set.
        """
        logger.debug("Reader thread is running.")
        try:
            read_counter = 0
            last_log_time = time.time()
            skipped_counter = 0

            while not self.stop_event.is_set():
                start_read = time.time()

                # ---------------------------------------------------------
                # LAG FIX + CPU OPTIMIZATION
                # ---------------------------------------------------------
                # 1. Determine if we NEED this frame (for throttling)
                try:
                    capture_fps = float(config.get("STREAM_FPS_CAPTURE", 0))
                except Exception:
                    capture_fps = 0.0

                should_decode = True
                if capture_fps > 0:
                    min_interval = 1.0 / capture_fps
                    if not hasattr(self, "last_enqueued_time"):
                        self.last_enqueued_time = 0
                    if (time.time() - self.last_enqueued_time) < min_interval:
                        should_decode = False

                # 2. Read Frame (with optimization: skip decode if possible)
                try:
                    frame = self._read_frame(decode=should_decode)
                    duration = time.time() - start_read
                    read_counter += 1

                    if not should_decode:
                        skipped_counter += 1

                    # Periodically log diagnostic information
                    if time.time() - last_log_time >= 60:
                        logger.debug(
                            f"Diagnostics: {read_counter} reads in last 60s "
                            f"(Skipped Decode: {skipped_counter}), last read duration={duration:.4f}s"
                        )
                        read_counter = 0
                        skipped_counter = 0
                        last_log_time = time.time()

                    if self.stop_event.is_set():
                        break

                    # Handle Frame Result
                    if frame is not None:
                        # "SKIPPED" indicates successful grab but no decode
                        if isinstance(frame, str) and frame == "SKIPPED":
                            self.consecutive_none_frames = 0
                            continue

                        if self._last_frame_drop_logged:
                            logger.debug("Stream stabilized (receiving frames).")
                            self._last_frame_drop_logged = False

                        if self._last_read_error_logged:
                            logger.debug(
                                "Frame read recovered (no longer throwing exceptions)."
                            )
                            self._last_read_error_logged = False

                        self.consecutive_none_frames = 0

                        # We have a real decoded frame, enqueue it
                        self.last_enqueued_time = time.time()

                        # Apply Clock Overlay for visual reference and sync check (P1)
                        try:
                            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Bottom-left position, green text with black outline for visibility
                            cv2.putText(
                                frame,
                                ts_str,
                                (20, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 0),
                                4,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                frame,
                                ts_str,
                                (20, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                        except Exception:
                            pass

                        try:
                            self.q.put(frame, block=False)
                        except queue.Full:
                            try:
                                self.q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.q.put(frame, block=False)
                            except queue.Full:
                                pass
                    else:
                        # Frame error (None)
                        self.consecutive_none_frames += 1
                        # Time-based log guard: warn at most once per 60s
                        # (immune to counter resets and object recreation)
                        if self.consecutive_none_frames >= 50:
                            now_mono = time.monotonic()
                            if now_mono - self._last_frame_drop_warn_mono >= 60.0:
                                logger.warning(
                                    "Significant frame drop detected "
                                    f"({self.consecutive_none_frames} consecutive None frames). "
                                    "Stream may be unstable."
                                )
                                self._last_frame_drop_warn_mono = now_mono
                                self._last_frame_drop_logged = True

                        if (
                            self.stream_type == self.RTSP
                            and self.consecutive_none_frames
                            >= self.max_none_frames_before_reconnect
                        ):
                            now = time.time()
                            if (
                                now - self.last_codec_switch_time
                                > self.codec_switch_cooldown
                            ):
                                logger.debug(
                                    "Exceeded max consecutive None frames. Attempting codec switch recovery."
                                )
                                self.request_recovery(trigger="reader_codec_switch", reason="Exceeded max consecutive None frames")
                                self.last_codec_switch_time = now
                                self._last_codec_switch_skip_logged = False
                            else:
                                if not self._last_codec_switch_skip_logged:
                                    logger.debug(
                                        "Skipping codec switch, still in cooldown period."
                                    )
                                    self._last_codec_switch_skip_logged = True
                            self.consecutive_none_frames = 0
                        elif (
                            self.stream_type != self.RTSP
                            and self.consecutive_none_frames
                            >= self.max_none_frames_before_reconnect
                        ):
                            self.request_recovery(
                                trigger="reader_none_frames",
                                reason="Multiple None frames received."
                            )
                            self.consecutive_none_frames = 0

                except Exception as e:
                    if not self._last_read_error_logged:
                        logger.error(
                            f"Error reading frame: {e} (Entering wait state for reconnection)."
                        )
                        self._last_read_error_logged = True
                    if self.stop_event.is_set():
                        break
                    self.request_recovery(trigger="reader_exception", reason=str(e))
        finally:
            logger.debug("Reader thread has exited.")

    def _read_frame(self, decode=True):
        if self.stream_type == self.RTSP and self.backend == self.BACKEND_FFMPEG:
            # FFmpeg always decodes; we must read to drain pipe regardless of 'decode' flag.
            return self._read_ffmpeg_frame(read_timeout_sec=1.0)
        elif self.cap and self.cap.isOpened():
            if not decode:
                # OPTIMIZATION: Just grab the frame (drain buffer) without decoding (save CPU)
                ret = self.cap.grab()
                if ret:
                    self.failed_reads = 0
                    return "SKIPPED"
                else:
                    self.failed_reads += 1
                    # Treat failed grab same as failed read
                    if self.failed_reads >= 3:
                        logger.error(
                            f"OpenCV failed to grab {self.failed_reads} consecutive frames."
                        )
                    return None
            else:
                # Full read (grab + retrieve)
                ret, frame = self.cap.read()
                if ret:
                    self.failed_reads = 0
                    return frame
                else:
                    self.failed_reads += 1
                    if self.failed_reads >= 3:
                        logger.error(
                            f"OpenCV failed to read {self.failed_reads} consecutive frames. Switching to FFmpeg backend."
                        )
                        self._switch_to_ffmpeg()
                    return None
        else:
            logger.error("HTTP/RTSP capture not opened.")
            if self.stream_type == self.RTSP:
                logger.debug("Requesting recovery from _read_frame.")
                self.request_recovery(trigger="read_frame_codec", reason="Capture not opened (RTSP)")
            else:
                self.request_recovery(trigger="read_frame", reason="Capture not opened.")
            return None

    def _switch_to_ffmpeg(self):
        """Releases OpenCV backend and initializes FFmpeg."""
        now = time.time()
        if now - self.last_codec_switch_time < self.backend_switch_cooldown:
            logger.debug("Backend switch cooldown active, skipping FFmpeg switch.")
            return
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.backend = self.BACKEND_FFMPEG
            self._setup_ffmpeg()
            logger.debug("Successfully switched to FFmpeg backend.")
            self.last_codec_switch_time = now
        except Exception as e:
            logger.error(f"Failed to switch to FFmpeg: {e}")

    def _handle_codec_switch(self):
        """
        Handles RTSP codec switches gracefully without triggering full reinitialization loops.
        Step 3: No ffprobe in hot recovery path - use cached resolution.
        """
        if self.stop_event.is_set() or self.stop_flag:
            logger.debug("Skipping codec switch handling during shutdown.")
            return
        logger.debug("Detected potential codec switch. Attempting fast reconnection...")
        try:
            # Release resources
            if self.cap:
                self.cap.release()
            self._terminate_ffmpeg_process(reason="codec switch")
            self.cap = None

            # Short pause to let codec change settle
            time.sleep(1)

            # Step 3: Use cached/known resolution - NO ffprobe here
            if self.stream_width and self.stream_height:
                logger.debug(
                    f"Using cached resolution for codec switch recovery: "
                    f"{self.stream_width}x{self.stream_height}"
                )
            else:
                logger.warning(
                    "No cached resolution available during codec switch. "
                    "Failing fast and backing off."
                )
                return

            # Reconnect quickly
            self._setup_capture()
            self._start_reader_thread()
            logger.debug("Fast reconnection after codec switch successful.")
        except Exception as e:
            logger.error(f"Codec switch handling failed: {e}")
            self.request_recovery(trigger="codec_switch_fallback", reason=str(e))

    def _handle_runtime_resolution_change(self, new_width, new_height, reason=""):
        """
        Reconnects FFmpeg when runtime resolution changes to keep buffer aligned.
        Step 3: No ffprobe in hot recovery path.
        """
        if self.stop_event.is_set() or self.stop_flag:
            logger.debug("Skipping runtime resolution change handling during shutdown.")
            return
        if not self.runtime_resolution_lock.acquire(blocking=False):
            logger.debug(
                "Skipping runtime resolution change handling (already running)."
            )
            return
        try:
            # Step 3: If no explicit dimensions, fail fast - NO ffprobe here
            if new_width is None or new_height is None:
                logger.warning(
                    "Runtime resolution change detected but no dimensions provided. "
                    "Failing fast (no probe in hot path). Requesting full recovery."
                )
                self.request_recovery(
                    trigger="resolution_change_no_dims",
                    reason="Resolution changed but no cached dimensions available"
                )
                return

            reason_suffix = f" ({reason})" if reason else ""
            logger.debug(
                f"Runtime resolution change detected{reason_suffix}: "
                f"{self.stream_width}x{self.stream_height} -> {new_width}x{new_height}"
            )
            self.stream_width = new_width
            self.stream_height = new_height
            self._persist_stream_settings()

            if self.stream_type == self.RTSP and self.backend == self.BACKEND_FFMPEG:
                self._terminate_ffmpeg_process(reason="runtime resolution change")
                try:
                    self._setup_ffmpeg()
                    logger.debug("FFmpeg restarted after runtime resolution change.")
                except Exception as restart_error:
                    logger.error(
                        f"Failed to restart FFmpeg after runtime resolution change: {restart_error}"
                    )
                    self.request_recovery(
                        trigger="resolution_restart_failed",
                        reason=str(restart_error)
                    )
        finally:
            self.runtime_resolution_lock.release()

    def _read_ffmpeg_frame(self, read_timeout_sec=None):
        frame_size = (
            self.stream_width * self.stream_height * 3
        )  # For bgr24 (3 bytes per pixel)
        process = self.ffmpeg_process
        if not process or process.poll() is not None:
            if self._ffmpeg_last_availability:
                logger.warning(
                    "FFmpeg process unavailable (Entering wait state for stream reconnection)."
                )
                self._ffmpeg_last_availability = False
            return None

        if not self._ffmpeg_last_availability:
            logger.debug("FFmpeg process recovered (receiving data).")
            self._ffmpeg_last_availability = True
        if self.stop_event.is_set() or self.stop_flag:
            return None

        if read_timeout_sec is not None:
            try:
                if not process.stdout:
                    return None
                ready, _, _ = select.select(
                    [process.stdout.fileno()], [], [], read_timeout_sec
                )
                if not ready:
                    return None
            except Exception as select_error:
                logger.debug(
                    f"FFmpeg read readiness check failed; continuing with blocking read: {select_error}"
                )
        try:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                actual_size = len(raw_frame)
                # New block: handle empty frame
                if actual_size == 0:
                    now = time.time()
                    if now - self.last_codec_switch_time > self.codec_switch_cooldown:
                        logger.warning("Empty frame detected - possible codec switch.")
                        self.request_recovery(trigger="ffmpeg_empty_frame", reason="Empty frame - possible codec switch")
                        self.last_codec_switch_time = now
                        self._last_codec_switch_skip_logged = False
                    else:
                        if not self._last_codec_switch_skip_logged:
                            logger.debug(
                                "Skipping codec switch on empty frame, still in cooldown period."
                            )
                            self._last_codec_switch_skip_logged = True
                    return None
                logger.warning(
                    f"FFmpeg produced incomplete frame: expected {frame_size} bytes, got {actual_size} bytes."
                )
                self.request_recovery(
                    trigger="ffmpeg_resolution_mismatch",
                    reason="FFmpeg output size changed",
                )
                return None
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (self.stream_height, self.stream_width, 3)
            )
            self.last_frame_time = time.time()
            return frame
        except Exception as e:
            logger.error(f"Error reading frame from FFmpeg: {e}")
            return None

    # ------------------------------------------------------------------
    # Child Process Registry (Step 4)
    # ------------------------------------------------------------------
    def _register_child(self, handle, kind="ffmpeg"):
        """Track a spawned child; enforce hard cap before registering."""
        self._cleanup_stale_children()
        if len(self._child_registry) >= self._child_hard_cap:
            logger.warning(
                f"event=child_hard_cap_hit count={len(self._child_registry)} "
                f"cap={self._child_hard_cap} kind={kind}"
            )
            # Kill oldest child to make room
            oldest_pid = min(self._child_registry, key=lambda p: self._child_registry[p]["start_ts"])
            self._force_kill_child(oldest_pid, reason="hard cap exceeded")
        pid = handle.pid
        self._child_registry[pid] = {
            "handle": handle,
            "start_ts": time.time(),
            "kind": kind,
        }
        logger.debug(f"event=child_registered pid={pid} kind={kind} total={len(self._child_registry)}")

    def _cleanup_stale_children(self):
        """Remove finished processes from registry."""
        stale = [pid for pid, info in self._child_registry.items() if info["handle"].poll() is not None]
        for pid in stale:
            del self._child_registry[pid]

    def _force_kill_child(self, pid, reason=""):
        """Force-terminate a tracked child."""
        info = self._child_registry.pop(pid, None)
        if not info:
            return
        handle = info["handle"]
        try:
            if handle.poll() is None:
                handle.kill()
                handle.wait(timeout=3)
        except Exception as e:
            logger.debug(f"event=child_kill_failed pid={pid} error={e}")
        logger.debug(f"event=child_killed pid={pid} reason={reason}")

    def _cleanup_all_children(self):
        """Terminate all tracked children (shutdown hook)."""
        for pid in list(self._child_registry):
            self._force_kill_child(pid, reason="full cleanup")

    # ------------------------------------------------------------------
    # Recovery Dispatcher (Steps 0-3, 5-6)
    # ------------------------------------------------------------------
    def request_recovery(self, trigger, reason=""):
        """
        Central recovery entry point.  All hot-path recovery triggers
        MUST route through here instead of calling recovery actions directly.
        Enforces: single-flight, cooldown, circuit breaker, timer coalescing.
        """
        attempt_id = next(self._recovery_gen)
        now = time.time()

        # --- Circuit breaker check (Step 6) ---
        if now < self._breaker_offline_until:
            remaining = self._breaker_offline_until - now
            logger.debug(
                f"event=recovery_blocked_breaker attempt_id={attempt_id} "
                f"trigger={trigger} offline_remaining={remaining:.1f}s"
            )
            return
        # Prune old timestamps and check threshold
        self._breaker_timestamps = [t for t in self._breaker_timestamps if now - t < self._breaker_window_sec]
        if len(self._breaker_timestamps) >= self._breaker_threshold:
            self._breaker_offline_until = now + self._breaker_offline_sec
            logger.warning(
                f"event=circuit_breaker_tripped attempt_id={attempt_id} "
                f"trigger={trigger} attempts_in_window={len(self._breaker_timestamps)} "
                f"offline_sec={self._breaker_offline_sec}"
            )
            return

        # --- Cooldown check (Step 2) ---
        if now < self._recovery_cooldown_until:
            remaining = self._recovery_cooldown_until - now
            logger.debug(
                f"event=recovery_cooldown attempt_id={attempt_id} "
                f"trigger={trigger} cooldown_remaining={remaining:.1f}s"
            )
            return

        # --- Single-flight gate (Step 1) ---
        if not self._recovery_gate.acquire(blocking=False):
            logger.debug(
                f"event=recovery_coalesced attempt_id={attempt_id} trigger={trigger} "
                f"reason=gate_busy"
            )
            return

        # Set cooldown BEFORE execution (Step 2)
        self._recovery_cooldown_until = now + self._recovery_cooldown_sec
        self._breaker_timestamps.append(now)

        try:
            logger.info(
                f"event=recovery_start attempt_id={attempt_id} trigger={trigger} "
                f"reason={reason} breaker_count={len(self._breaker_timestamps)}"
            )
            self._reinitialize_camera(reason=f"[{trigger}] {reason}")
            logger.info(
                f"event=recovery_end attempt_id={attempt_id} trigger={trigger} result=ok"
            )
            # Reset breaker on SUCCESS (stable window)
            self._breaker_timestamps.clear()
            self._breaker_offline_until = 0.0
        except Exception as e:
            logger.warning(
                f"event=recovery_end attempt_id={attempt_id} trigger={trigger} "
                f"result=error error={e}"
            )
            self._schedule_reinit(reason=f"Failed recovery [{trigger}]: {e}")
        finally:
            self._recovery_gate.release()

    # ------------------------------------------------------------------
    # Timer Coalescing (Step 5)
    # ------------------------------------------------------------------
    def _schedule_reinit(self, reason):
        """Schedule ONE pending reinit timer (coalesced)."""
        delay = min(2 ** self.retry_count, 60)
        with self._pending_reinit_lock:
            if self._pending_reinit_timer is not None:
                self._pending_reinit_timer.cancel()
                logger.debug("event=reinit_timer_replaced old_timer_cancelled=true")
            timer = threading.Timer(
                delay, self.request_recovery, kwargs={"trigger": "scheduled_reinit", "reason": reason}
            )
            timer.daemon = True
            timer.name = "ReinitTimer"
            self._pending_reinit_timer = timer
            timer.start()
        logger.debug(f"event=reinit_scheduled delay={delay}s reason={reason}")

    def _reinitialize_camera(self, reason="Unknown"):
        """
        Internal recovery action.  Do NOT call directly from hot paths;
        use request_recovery() instead.
        """
        # Keep existing reinit_lock for internal reentrancy safety
        if not self.reinit_lock.acquire(blocking=False):
            logger.debug(
                "Reinitialization is already being performed in another thread."
            )
            return

        try:
            logger.debug(f"Reinitializing camera due to: {reason}")
            if self.retry_count >= 5:
                logger.debug(
                    "Maximum retry attempts reached. Scheduling longer delay."
                )
                self.retry_count = 0
                self._schedule_reinit(reason="Retry after max attempts")
                return

            self.retry_count += 1
            logger.debug(f"Reinitialization attempt {self.retry_count}/5.")

            # Stop current capture and threads
            self.stop()
            logger.debug("Waiting 2 seconds before reinitialization attempt.")
            time.sleep(2)

            if self.reader_thread and self.reader_thread.is_alive():
                logger.warning(
                    "Reader thread is still alive after stop(); retrying later."
                )
                self._schedule_reinit(
                    reason="Reader thread alive after stop during reinit"
                )
                return

            self.stop_event.clear()
            self.stop_flag = False
            self._setup_capture(
                require_initial_frame=True,
                initial_frame_wait_sec=self._recovery_initial_frame_wait_sec,
            )
            self._start_reader_thread()
            self._start_health_check_thread()
            logger.debug("Reinitialization successful.")
            self.retry_count = 0
        except Exception as e:
            logger.debug(f"Reinitialization failed: {e}")
            raise  # let request_recovery handle scheduling
        finally:
            self.reinit_lock.release()

    def get_frame(self):
        """
        Returns the most recent frame, dropping any older frames in the queue.
        Ensures consumers always see the latest frame and do not build backlog.
        """
        latest = None
        try:
            while True:
                latest = self.q.get_nowait()
        except queue.Empty:
            pass
        return latest

    def stop(self):
        """Stops the video stream and releases resources."""
        logger.debug("Releasing resources...")
        self.stop_event.set()
        self.stop_flag = True

        # Stop FFmpeg first so blocking pipe reads in reader thread are unblocked.
        self._terminate_ffmpeg_process(reason="stop()")
        self._cleanup_all_children()

        # Cancel any pending reinit timer (Step 5)
        with self._pending_reinit_lock:
            if self._pending_reinit_timer is not None:
                self._pending_reinit_timer.cancel()
                self._pending_reinit_timer = None

        if self.reader_thread and self.reader_thread.is_alive():
            if self.reader_thread != threading.current_thread():
                logger.debug("Joining reader thread...")
                self.reader_thread.join(timeout=5)
                if self.reader_thread.is_alive():
                    logger.debug("Reader thread did not terminate within timeout.")
            else:
                logger.debug("Skipping join on current thread (reader thread).")

        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        if (
            self.health_check_thread
            and self.health_check_thread != threading.current_thread()
        ):
            logger.debug("Joining health check thread...")
            self.health_check_thread.join(timeout=5)
            if self.health_check_thread.is_alive():
                logger.debug("Health check thread did not terminate within timeout.")
        else:
            logger.debug("Skipping join on current thread (health check thread).")

        if self.cap:
            logger.debug("Releasing OpenCV VideoCapture.")
            self.cap.release()
            self.cap = None

        self.reader_thread = None
        self.health_check_thread = None

        logger.debug("Resources released.")

    def _register_instance_for_shutdown(self):
        """
        Registers instance for global cleanup routines (atexit / signals).
        """
        VideoCapture._instances.add(self)
        if VideoCapture._shutdown_hooks_registered:
            return
        VideoCapture._shutdown_hooks_registered = True
        atexit.register(VideoCapture._terminate_all_instances)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)

                def handler(signum, frame, prev=previous):
                    VideoCapture._terminate_all_instances()
                    if callable(prev) and prev not in (
                        signal.SIG_IGN,
                        signal.SIG_DFL,
                    ):
                        try:
                            prev(signum, frame)
                        except Exception:
                            pass

                signal.signal(sig, handler)
            except Exception as signal_error:
                logger.debug(
                    f"Could not register shutdown handler for {sig}: {signal_error}"
                )

    @classmethod
    def _terminate_all_instances(cls):
        """Cleans up all known FFmpeg processes and registry children (e.g., at atexit)."""
        for instance in list(cls._instances):
            try:
                instance._terminate_ffmpeg_process(reason="shutdown hook")
                instance._cleanup_all_children()
            except Exception as cleanup_error:
                logger.debug(
                    f"Failed to terminate FFmpeg during shutdown: {cleanup_error}"
                )

    def _make_ffmpeg_preexec_fn(self):
        """
        Creates preexec function that starts its own session and sets a
        parent-death signal handler on Linux.
        """
        if os.name != "posix":
            return None

        def preexec():
            try:
                os.setsid()  # own process group for targeted signals
            except Exception:
                pass
            self._set_parent_death_signal()

        return preexec

    def _can_use_ffmpeg_preexec_fn(self):
        """
        preexec_fn runs between fork() and exec() and can deadlock in
        multithreaded Python processes. Only use it in single-thread mode.
        """
        if os.name != "posix":
            return False
        if threading.active_count() > 1:
            if not self._preexec_skip_logged:
                logger.debug(
                    "Skipping FFmpeg preexec_fn because multiple Python threads are active."
                )
                self._preexec_skip_logged = True
            return False
        return True

    def _set_parent_death_signal(self):
        """
        Sets PR_SET_PDEATHSIG on Linux so that FFmpeg is terminated
        if the Python process dies unexpectedly.
        """
        if os.name != "posix":
            return False
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            return True
        except Exception:
            return False

    @property
    def resolution(self):
        """Returns the resolution of the video stream."""
        return (self.stream_width, self.stream_height)
