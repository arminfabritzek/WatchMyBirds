# ------------------------------------------------------------------------------
# Video Capture Class for Input Streams
# camera/video_capture.py
# ------------------------------------------------------------------------------
# video_capture.py
from config import load_config

config = load_config()

import subprocess
import cv2
import queue
import threading
from threading import Event
import time
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

    def __init__(self, source, debug=False, auto_start=True):
        """
        Initializes the VideoCapture object and detects the stream type.
        """
        self.source = source
        self.debug = debug
        self.stop_flag = False
        self.stop_event = Event()
        self.cap = None
        self.ffmpeg_process = None
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

        logger.debug(
            f"Initialized VideoCapture with source: {self.source}, stream_type: {self.stream_type}"
        )
        if auto_start:
            self.start()

    def start(self):
        """Starts the video stream and background threads."""
        if self.is_running():
            logger.info("VideoCapture already running.")
            return
        if self.stream_type == self.RTSP:
            self._get_stream_resolution_ffprobe()
            logger.info(f"Initial resolution: {self.resolution}")
        self._setup_capture()
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

    def _setup_capture(self):
        logger.debug("Setting up video capture...")
        if self.stream_type == self.RTSP:
            try:
                self._setup_ffmpeg()
                self.backend = self.BACKEND_FFMPEG
                logger.info("Using FFmpeg backend successfully.")
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

        test_frame = None
        try:
            if self.backend == self.BACKEND_FFMPEG:
                logger.info("Waiting up to 5s for FFmpeg to produce the first frame...")
                for _ in range(10):
                    test_frame = self._read_ffmpeg_frame()
                    if test_frame is not None:
                        logger.info("Received initial frame from FFmpeg.")
                        break
                    time.sleep(0.5)
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                test_frame = frame if ret else None
        except Exception as e:
            logger.warning(f"Error testing frame availability: {e}")

        if test_frame is None:
            now = time.time()
            if now - self.last_codec_switch_time < self.backend_switch_cooldown:
                logger.info("Backend switch cooldown active, skipping immediate switch.")
                return
            logger.warning("Initial test frame still missing. Proceeding anyway to allow stream startup on slow devices (e.g., NAS).")
            if self.backend == self.BACKEND_FFMPEG:
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
        logger.info("Trying OpenCV backend for RTSP stream...")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("OpenCV could not open RTSP stream")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"OpenCV RTSP setup successful: {self.stream_width}x{self.stream_height}"
        )
        self.backend = self.BACKEND_OPENCV

    def _setup_ffmpeg(self):
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
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**8,
                )
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
                time.sleep(2**attempt)  # Exponential backoff
        error_msg = "Failed to start FFmpeg after multiple attempts."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _log_ffmpeg_errors(self):
        """
        Continuously logs FFmpeg's stderr for debugging purposes, only if debug mode is enabled.
        """
        logger.debug("Starting FFmpeg stderr logging.")
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b""):
                if line:
                    logger.debug(f"FFmpeg STDERR: {line.decode('utf-8').strip()}")
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
            return
        except Exception as ffprobe_error:
            logger.warning(f"FFprobe failed: {ffprobe_error}. Falling back to FFmpeg stderr parsing.")

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

        logger.debug(f"Running FFmpeg command for resolution fallback: {' '.join(ffmpeg_cmd)}")

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
                logger.info(f"FFmpeg fallback successfully detected resolution: {self.stream_width}x{self.stream_height}")
                logger.debug(f"Detected stream resolution via FFmpeg fallback: {self.stream_width}x{self.stream_height}")
            else:
                raise RuntimeError("FFmpeg did not return a parsable resolution.")
        except Exception as ffmpeg_error:
            logger.error(f"Failed to get stream resolution using FFmpeg fallback: {ffmpeg_error}")
            raise RuntimeError("Unable to determine stream resolution using ffprobe or ffmpeg.")

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
        logger.info(
            f"Detected HTTP stream resolution: {self.stream_width}x{self.stream_height}"
        )

    def _setup_webcam(self):
        """
        Sets up OpenCV's VideoCapture for webcam streams.
        """
        logger.info(f"Setting up Webcam with source index: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open webcam with index: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Detected Webcam resolution: {self.stream_width}x{self.stream_height}"
        )

    def _start_reader_thread(self):
        """
        Starts the reader thread if it's not already running.
        Ensures only one thread is active at a time to avoid conflicts.
        """
        if self.reader_thread and self.reader_thread.is_alive():
            logger.info("Reader thread already running; skipping reinitialization")

        else:
            # Add a short delay to allow FFmpeg to initialize properly
            logger.info("Starting reader thread...")
            time.sleep(2)
            self.reader_thread = threading.Thread(
                target=self._reader, daemon=True, name="ReaderThread"
            )
            self.reader_thread.start()
            logger.info("Reader thread started successfully.")

    def _start_health_check_thread(self):
        """
        Starts a health check thread to monitor FFmpeg subprocess.
        """
        if self.stream_type != self.RTSP:
            logger.info(
                "Health check thread is only applicable for RTSP streams. Skipping."
            )
            return

        if self.health_check_thread and self.health_check_thread.is_alive():
            logger.info(
                "Health check thread already running; skipping reinitialization."
            )
        else:
            logger.info("Starting health check thread...")
            self.health_check_thread = threading.Thread(
                target=self._health_check, daemon=True, name="HealthCheckThread"
            )
            self.health_check_thread.start()
            logger.info("Health check thread started successfully.")

    def _health_check(self):
        """
        Periodically checks if the stream (FFmpeg for RTSP, or OpenCV for HTTP) is alive.
        If not, attempts to reinitialize.
        """
        logger.info("Health check thread is running.")
        try:
            while not self.stop_event.is_set():
                if self.stream_type == self.RTSP:
                    if self.ffmpeg_process:
                        retcode = self.ffmpeg_process.poll()
                        if retcode is not None:
                            stderr_output = (
                                self.ffmpeg_process.stderr.read().decode().strip()
                            )
                            logger.error(
                                f"FFmpeg subprocess terminated with return code {retcode}. STDERR: {stderr_output}"
                            )
                            self._reinitialize_camera(
                                reason="RTSP FFmpeg subprocess terminated unexpectedly."
                            )
                        else:
                            if time.time() - self.last_frame_time > 10:
                                logger.error(
                                    "No frame received for over 5 seconds; triggering reinitialization."
                                )
                                self._reinitialize_camera(reason="RTSP stream stale.")
                elif self.stream_type == self.HTTP:
                    if not self.cap or not self.cap.isOpened():
                        logger.error(
                            "HTTP stream is not opened. Triggering reinitialization."
                        )
                        self._reinitialize_camera(reason="HTTP stream not opened.")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Health check encountered error: {e}")
        logger.info("Health check thread is stopping.")

    def _reader(self):
        """
        Continuously reads frames from the video source and places only the latest frame in the queue.
        Terminates cleanly if stop_event is set.
        """
        logger.info("Reader thread is running.")
        try:
            read_counter = 0
            last_log_time = time.time()
            while not self.stop_event.is_set():
                start_read = time.time()
                try:
                    frame = self._read_frame()
                    duration = time.time() - start_read
                    read_counter += 1

                    # Slow down reading to match configured STREAM_FPS
                    if hasattr(config, "STREAM_FPS") and config["STREAM_FPS"] > 0:
                        time.sleep(1.0 / config["STREAM_FPS"])

                    # Periodically log diagnostic information
                    if time.time() - last_log_time >= 15:
                        logger.debug(f"Diagnostics: {read_counter} reads in last 5s, last read duration={duration:.4f}s")
                        read_counter = 0
                        last_log_time = time.time()

                    if self.stop_event.is_set():
                        logger.info("Stop event set. Reader thread is terminating.")
                        break
                    if frame is not None:
                        self.consecutive_none_frames = 0
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
                        self.consecutive_none_frames += 1
                        logger.warning(f"Frame not available. Consecutive count: {self.consecutive_none_frames}")
                        if self.stream_type == self.RTSP and self.consecutive_none_frames >= self.max_none_frames_before_reconnect:
                            now = time.time()
                            if now - self.last_codec_switch_time > self.codec_switch_cooldown:
                                logger.info("Exceeded max consecutive None frames. Attempting codec switch recovery.")
                                self._handle_codec_switch()
                                self.last_codec_switch_time = now
                            else:
                                logger.info("Skipping codec switch, still in cooldown period.")
                            self.consecutive_none_frames = 0
                        elif self.stream_type != self.RTSP and self.consecutive_none_frames >= self.max_none_frames_before_reconnect:
                            self._reinitialize_camera(reason="Multiple None frames received.")
                            self.consecutive_none_frames = 0
                except Exception as e:
                    logger.error(f"Error reading frame: {e}")
                    if self.stop_event.is_set():
                        break
                    self._reinitialize_camera()
        finally:
            logger.info("Reader thread has exited.")

    def _read_frame(self):
        if self.stream_type == self.RTSP and self.backend == self.BACKEND_FFMPEG:
            return self._read_ffmpeg_frame()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.failed_reads = 0
                return frame
            else:
                self.failed_reads += 1
                logger.warning(
                    f"OpenCV failed to read frame ({self.failed_reads} consecutive failures)"
                )
                if self.failed_reads >= 3:
                    logger.error(
                        "Too many OpenCV read failures, switching to FFmpeg fallback."
                    )
                    self._switch_to_ffmpeg()
                return None
        else:
            logger.error("HTTP/RTSP capture not opened.")
            # If RTSP stream, handle possible codec switch gracefully
            if self.stream_type == self.RTSP:
                logger.info("Attempting fast codec switch recovery in _read_frame.")
                self._handle_codec_switch()
            else:
                self._reinitialize_camera(reason="Capture not opened.")
            return None

    def _switch_to_ffmpeg(self):
        """Releases OpenCV backend and initializes FFmpeg."""
        now = time.time()
        if now - self.last_codec_switch_time < self.backend_switch_cooldown:
            logger.info("Backend switch cooldown active, skipping FFmpeg switch.")
            return
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.backend = self.BACKEND_FFMPEG
            self._setup_ffmpeg()
            logger.info("Successfully switched to FFmpeg backend.")
            self.last_codec_switch_time = now
        except Exception as e:
            logger.error(f"Failed to switch to FFmpeg: {e}")

    def _handle_codec_switch(self):
        """
        Handles RTSP codec switches gracefully without triggering full reinitialization loops.
        """
        logger.info("Detected potential codec switch. Attempting fast reconnection...")
        try:
            # Release resources
            if self.cap:
                self.cap.release()
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()

            self.cap = None
            self.ffmpeg_process = None

            # Short pause to let codec change settle
            time.sleep(1)

            # Re-detect resolution
            try:
                self._get_stream_resolution_ffprobe()
            except Exception as e:
                logger.warning(f"FFprobe failed during codec switch: {e}")

            # Reconnect quickly
            self._setup_capture()
            self._start_reader_thread()
            logger.info("Fast reconnection after codec switch successful.")
        except Exception as e:
            logger.error(f"Codec switch handling failed: {e}")
            self._reinitialize_camera(reason="Codec switch recovery failed")

    def _read_ffmpeg_frame(self):
        frame_size = (
            self.stream_width * self.stream_height * 3
        )  # For bgr24 (3 bytes per pixel)
        try:
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                actual_size = len(raw_frame)
                # New block: handle empty frame
                if actual_size == 0:
                    now = time.time()
                    if now - self.last_codec_switch_time > self.codec_switch_cooldown:
                        logger.warning("Empty frame detected - possible codec switch.")
                        self._handle_codec_switch()
                        self.last_codec_switch_time = now
                    else:
                        logger.info("Skipping codec switch on empty frame, still in cooldown period.")
                    return None
                logger.warning(
                    f"FFmpeg produced incomplete frame: expected {frame_size} bytes, got {actual_size} bytes."
                )
                # Try to infer resolution from actual bytes if possible
                if actual_size % 3 == 0:
                    pixels = actual_size // 3
                    new_height = int(np.sqrt(pixels))
                    new_width = (
                        pixels // new_height if new_height else self.stream_width
                    )
                    if (
                        new_width > 0
                        and new_height > 0
                        and (
                            new_width != self.stream_width
                            or new_height != self.stream_height
                        )
                    ):
                        logger.info(
                            f"Adjusting resolution dynamically from {self.stream_width}x{self.stream_height} "
                            f"to {new_width}x{new_height} based on FFmpeg output."
                        )
                        self.stream_width = new_width
                        self.stream_height = new_height
                        frame_size = actual_size
                        try:
                            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                                (self.stream_height, self.stream_width, 3)
                            )
                            self.last_frame_time = time.time()
                            return frame
                        except Exception as reshape_err:
                            logger.error(
                                f"Failed to reshape frame with inferred resolution {new_width}x{new_height}: {reshape_err}"
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

    def _schedule_reinit(self, reason):
        delay = 2**self.retry_count
        logger.info(f"Scheduling reinitialization in {delay} seconds due to: {reason}")
        threading.Timer(
            delay, self._reinitialize_camera, kwargs={"reason": reason}
        ).start()

    def _reinitialize_camera(self, reason="Unknown"):
        """
        Attempts to reinitialize the camera after a failure.

        :param reason: The reason triggering reinitialization.
        """
        # Try to get the lock to avoid parallel reinitializations.
        if not self.reinit_lock.acquire(blocking=False):
            logger.info(
                "Reinitialization is already being performed in another thread."
            )
            return

        try:
            logger.info(f"Reinitializing camera due to: {reason}")
            if self.retry_count >= 5:
                logger.info(
                    "Maximum retry attempts reached. Scheduling longer delay before next attempt."
                )
                # Longer delay before retrying again (e.g. 60 seconds)
                threading.Timer(
                    60,
                    self._reinitialize_camera,
                    kwargs={"reason": "Retry after long delay"},
                ).start()
                # Reset the retry counter after scheduling a longer delay
                self.retry_count = 0
                return

            self.retry_count += 1
            logger.info(f"Reinitialization attempt {self.retry_count}/5.")

            # Stop current capture and threads
            self.stop()
            logger.info("Waiting 2 seconds before reinitialization attempt.")
            time.sleep(2)

            self.stop_event.clear()  # Reset stop_event for new threads
            self._setup_capture()
            self._start_reader_thread()
            self._start_health_check_thread()
            logger.info("Reinitialization successful.")
            self.retry_count = 0  # Reset retry count on success
        except Exception as e:
            logger.info(f"Reinitialization failed: {e}")
            logger.info("Scheduling next reinitialization attempt.")
            self._schedule_reinit(reason=f"Failed to reinitialize: {e}")
        finally:
            self.reinit_lock.release()

    def get_frame(self):
        try:
            frame = self.q.get(timeout=0.3)
            return frame
        except queue.Empty:
            logger.info("Queue is empty, unable to retrieve frame.")
            return None

    def stop(self):
        """Stops the video stream and releases resources."""
        logger.info("Releasing resources...")
        self.stop_event.set()
        self.stop_flag = True

        if self.reader_thread and self.reader_thread.is_alive():
            if self.reader_thread != threading.current_thread():
                logger.info("Joining reader thread...")
                self.reader_thread.join(timeout=5)
                if self.reader_thread.is_alive():
                    logger.info("Reader thread did not terminate within timeout.")
            else:
                logger.info("Skipping join on current thread (reader thread).")

        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        if (
            self.health_check_thread
            and self.health_check_thread != threading.current_thread()
        ):
            logger.info("Joining health check thread...")
            self.health_check_thread.join(timeout=5)
            if self.health_check_thread.is_alive():
                logger.info("Health check thread did not terminate within timeout.")
        else:
            logger.info("Skipping join on current thread (health check thread).")

        if self.ffmpeg_process:
            logger.info("Terminating FFmpeg process...")
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
                logger.info("FFmpeg process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.info(
                    "FFmpeg process did not terminate in time. Killing process..."
                )
                self.ffmpeg_process.kill()
                logger.info("FFmpeg process killed.")
            self.ffmpeg_process = None

        if self.cap:
            logger.info("Releasing OpenCV VideoCapture.")
            self.cap.release()
            self.cap = None

        logger.info("Resources released.")

    @property
    def resolution(self):
        """Returns the resolution of the video stream."""
        return (self.stream_width, self.stream_height)
