# ------------------------------------------------------------------------------
# Video Capture Class for Input Streams
# ------------------------------------------------------------------------------
# video_capture.py
import os
import subprocess
import cv2
import queue
import threading
from threading import Event
import time
import numpy as np
import logging
from dotenv import load_dotenv
load_dotenv()

# Read the debug flag from the environment variable (default: False)
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG if _debug else logging.INFO)

logger.info(f"Debug mode is {'enabled' if _debug else 'disabled'}.")
print(f"DEBUG_MODE environment variable: {os.getenv('DEBUG_MODE')}")
print(f"Debug mode in code: {_debug}")


class VideoCapture:
    """
    Handles video input streams from various sources (RTSP, WebRTC, Webcam, FFmpeg).
    """
    def __init__(self, source, debug=False):
        """
        Initializes the VideoCapture object and detects the stream type.
        """
        self.source = source
        self.debug = debug  # Initialize the debug attribute
        self.stop_flag = False
        self.stop_event = Event()  # Event to signal the thread to stop
        self.cap = None
        self.ffmpeg_process = None  # Used for FFmpeg streams
        self.q = queue.Queue(maxsize=10)  # Queue for storing frames
        self.reader_thread = None  # Initialize the reader_thread attribute
        self.last_frame_time = time.time()
        self.health_check_thread = None  # Initialize the health_check_thread attribute
        self.retry_count = 0  # Initialize retry count for reinitialization
        self.stream_type = self._detect_stream_type()

        self.reinit_lock = threading.Lock()
        self.reinitializing = False

        logger.debug(f"Initialized VideoCapture with source: {self.source}, stream_type: {self.stream_type}")

        if self.stream_type == "rtsp":
            self._get_stream_resolution_ffprobe()  # Detect resolution before setting up

        self._setup_capture()
        self._start_reader_thread()
        self._start_health_check_thread()

    def _log(self, message, level=logging.DEBUG):
        """
        Logs a message at the given level, only if debug mode is enabled.
        """
        if self.debug:
            logger.log(level, message)

    def _detect_stream_type(self):
        """
        Automatically detects the stream type based on the source string.

        :return: Detected stream type as a string ("rtsp", "http", "webcam").
        """
        self._log("Detect stream type...", level=logging.DEBUG)
        if isinstance(self.source, str):
            if self.source.startswith("rtsp://"):
                self._log("Stream type detected as RTSP.", level=logging.DEBUG)
                return "rtsp"
            elif self.source.startswith(("http://", "https://")):
                self._log("Stream type detected as HTTP.", level=logging.DEBUG)
                return "http"
        elif isinstance(self.source, int):  # Integer sources are treated as webcams
            self._log("Stream type detected as Webcam.", level=logging.DEBUG)
            return "webcam"
        error_msg = f"Unable to determine stream type for source: {self.source}"
        self._log(error_msg, level=logging.ERROR)
        raise ValueError(f"Unable to determine stream type for source: {self.source}")


    def _setup_capture(self):
        """
        Sets up the video capture based on the detected stream type.
        """
        self._log("Setting up video capture...", level=logging.DEBUG)
        if self.stream_type == "rtsp":
            self._setup_ffmpeg()
        elif self.stream_type == "http":
            self._setup_http()
        elif self.stream_type == "webcam":
            self._setup_webcam()
        else:
            error_msg = f"Unsupported stream type: {self.stream_type}"
            self._log(error_msg, level=logging.ERROR)
            raise ValueError(f"Unsupported stream type: {self.stream_type}")
        self._log("Video capture setup completed.", level=logging.DEBUG)

    def _setup_ffmpeg(self):
        ffmpeg_cmd = [
            "ffmpeg",
            "-probesize", "50000000",
            "-analyzeduration", "10000000",
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "pipe:"
        ]

        self._log(f"FFmpeg command: {' '.join(ffmpeg_cmd)}", level=logging.DEBUG)

        for attempt in range(5):  # Retry up to 5 times
            try:
                self._log(f"Starting FFmpeg process (attempt {attempt + 1})...", level=logging.DEBUG)
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
                )
                self._log("FFmpeg process started successfully.", level=logging.DEBUG)

                # Start logging FFmpeg errors if in debug mode
                if self.debug:
                    threading.Thread(target=self._log_ffmpeg_errors, daemon=True).start()

                # Verify FFmpeg process is running
                if self.ffmpeg_process.poll() is not None:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    error_msg = f"FFmpeg process terminated prematurely. STDERR: {stderr_output}"
                    self._log(error_msg, level=logging.ERROR)
                    raise RuntimeError(error_msg)

                return  # Exit loop on success
            except Exception as e:
                self._log(f"Failed to start FFmpeg process: {e}. Retrying...", level=logging.ERROR)
                time.sleep(2 ** attempt)  # Exponential backoff
        error_msg = "Failed to start FFmpeg after multiple attempts."
        self._log(error_msg, level=logging.ERROR)
        raise RuntimeError(error_msg)

    def _log_ffmpeg_errors(self):
        """
        Continuously logs FFmpeg's stderr for debugging purposes, only if debug mode is enabled.
        """
        self._log("Starting FFmpeg stderr logging.", level=logging.DEBUG)
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                if line:
                    logger.debug(f"FFmpeg STDERR: {line.decode('utf-8').strip()}")
                if self.stop_event.is_set():
                    self._log("Stop event set. Ending FFmpeg stderr logging.", level=logging.DEBUG)
                    break
        except Exception as e:
            logger.error(f"Exception while logging FFmpeg stderr: {e}")
        self._log("FFmpeg stderr logging thread terminated.", level=logging.DEBUG)

    def _get_stream_resolution_ffprobe(self):
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            self.source
        ]

        self._log(f"Running FFprobe command: {' '.join(ffprobe_cmd)}", level=logging.DEBUG)

        try:
            output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT, timeout=30).decode().strip()
            self._log(f"FFprobe output: {output}", level=logging.DEBUG)
            # Parse the FFprobe output
            width, height = map(int, output.split('\n'))
            self._log(f"Detected stream resolution: {width}x{height}", level=logging.DEBUG)

            # Set the resolution as instance attributes
            self.stream_width = width
            self.stream_height = height

        except subprocess.TimeoutExpired:
            self._log("FFprobe command timed out. Cannot reach the RTSP stream.", level=logging.DEBUG)
            raise RuntimeError("FFprobe timed out while trying to get stream resolution.")
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode().strip()
            self._log(f"FFprobe failed with error: {error_output}", level=logging.DEBUG)
            raise RuntimeError("Failed to get stream resolution using FFprobe.")
        except ValueError as ve:
            self._log(f"Error parsing FFprobe output: {ve}", level=logging.DEBUG)
            raise RuntimeError("Failed to parse stream resolution from FFprobe output.")
        except Exception as e:
            self._log(f"Unexpected error during FFprobe: {e}", level=logging.DEBUG)
            raise RuntimeError("An unexpected error occurred while getting stream resolution.")

    def _setup_http(self):
        """
        Sets up OpenCV's VideoCapture for HTTP streams.
        """
        self._log(f"Setting up HTTP stream with source: {self.source}", level=logging.DEBUG)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open HTTP stream: {self.source}"
            self._log(error_msg, level=logging.ERROR)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log(f"Detected HTTP stream resolution: {self.stream_width}x{self.stream_height}")

    def _setup_webcam(self):
        """
        Sets up OpenCV's VideoCapture for webcam streams.
        """
        self._log(f"Setting up Webcam with source index: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Failed to open webcam with index: {self.source}"
            self._log(error_msg, level=logging.ERROR)
            raise RuntimeError(error_msg)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # Retrieve resolution using OpenCV
        self.stream_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stream_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log(f"Detected Webcam resolution: {self.stream_width}x{self.stream_height}")

    def _start_reader_thread(self):
        """
        Starts the reader thread if it's not already running.
        Ensures only one thread is active at a time to avoid conflicts.
        """
        if self.reader_thread and self.reader_thread.is_alive():
            self._log("Reader thread already running; skipping reinitialization")

        else:
            # Add a short delay to allow FFmpeg to initialize properly
            self._log("Starting reader thread...")
            time.sleep(2)
            self.reader_thread = threading.Thread(target=self._reader, daemon=True, name="ReaderThread")
            self.reader_thread.start()
            self._log("Reader thread started successfully.")

    def _start_health_check_thread(self):
        """
        Starts a health check thread to monitor FFmpeg subprocess.
        """
        if self.stream_type != "rtsp":
            self._log("Health check thread is only applicable for RTSP streams. Skipping.")
            return

        if self.health_check_thread and self.health_check_thread.is_alive():
            self._log("Health check thread already running; skipping reinitialization.")
        else:
            self._log("Starting health check thread...")
            self.health_check_thread = threading.Thread(target=self._health_check, daemon=True, name="HealthCheckThread")
            self.health_check_thread.start()
            self._log("Health check thread started successfully.")

    def _health_check(self):
        """
        Periodically checks if the stream (FFmpeg for RTSP, or OpenCV for HTTP) is alive.
        If not, attempts to reinitialize.
        """
        self._log("Health check thread is running.")
        while not self.stop_event.is_set():
            if self.stream_type == "rtsp":
                if self.ffmpeg_process:
                    retcode = self.ffmpeg_process.poll()
                    if retcode is not None:
                        stderr_output = self.ffmpeg_process.stderr.read().decode().strip()
                        self._log(f"FFmpeg subprocess terminated with return code {retcode}. STDERR: {stderr_output}",
                                  level=logging.ERROR)
                        self._reinitialize_camera(reason="RTSP FFmpeg subprocess terminated unexpectedly.")
                    else:
                        # Check if frames have been read recently.
                        if time.time() - self.last_frame_time > 10:  # 10-second threshold to wait until reinitialization.
                            self._log("No frame received for over 10 seconds; triggering reinitialization.",
                                      level=logging.ERROR)
                            self._reinitialize_camera(reason="RTSP stream stale.")
            elif self.stream_type == "http":
                if not self.cap or not self.cap.isOpened():
                    self._log("HTTP stream is not opened. Triggering reinitialization.", level=logging.ERROR)
                    self._reinitialize_camera(reason="HTTP stream not opened.")
            # You can also add a similar check for webcams if needed.
            time.sleep(5)  # Check every 5 seconds
        self._log("Health check thread is stopping.")

    def _reader(self):
        """
        Continuously reads frames from the video source and places only the latest frame in the queue.
        Terminates cleanly if stop_event is set.
        """
        self._log("Reader thread is running.")
        while not self.stop_event.is_set():  # Check for thread termination
            try:
                frame = self._read_frame()
                if self.stop_event.is_set():  # Recheck stop_event after potentially long operations
                    self._log("Stop event set. Reader thread is terminating.")
                    break
                if frame is not None:
                    # Clear the queue before adding the new frame
                    with self.q.mutex:
                        self.q.queue.clear()
                    self.q.put(frame, timeout=0.1)  # Always have the latest frame in the queue
                    # self._log("Replaced the queue with the latest frame.")
                else:
                    self._log("Frame not available, triggering reinitialization.")
                    self._reinitialize_camera(reason="Received None frame.")
            except Exception as e:
                self._log(f"Error reading frame: {e}", level=logging.ERROR)
                if self.stop_event.is_set():
                    break
                self._reinitialize_camera()  # Reinitialize in case of error
        self._log("Reader thread has exited.")

    def _read_frame(self):
        if self.stream_type == "rtsp":
            return self._read_ffmpeg_frame()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self._log("HTTP capture failed to read frame.", level=logging.ERROR)
                self._reinitialize_camera(reason="HTTP capture failed to read frame.")
                return None
        else:
            self._log("HTTP capture is not opened.", level=logging.ERROR)
            self._reinitialize_camera(reason="HTTP capture not opened.")
            return None

    def _read_ffmpeg_frame(self):
        frame_size = self.stream_width * self.stream_height * 3  # For bgr24 (3 bytes per pixel)

        try:
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            # self._log(f"Read {len(raw_frame)} bytes from FFmpeg", level=logging.DEBUG)
            if len(raw_frame) != frame_size:
                self._log(f"FFmpeg produced incomplete frame: expected {frame_size} bytes, got {len(raw_frame)} bytes.", level=logging.ERROR)
                return None  # Skip incomplete frames
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.stream_height, self.stream_width, 3))
            self.last_frame_time = time.time()  # Update timestamp on successful frame read
            # self._log("Frame read successfully from FFmpeg.", level=logging.DEBUG)
            return frame
        except Exception as e:
            self._log(f"Error reading frame from FFmpeg: {e}", level=logging.ERROR)
            return None

    def _schedule_reinit(self, reason):
        delay = 2 ** self.retry_count
        self._log(f"Scheduling reinitialization in {delay} seconds due to: {reason}")
        threading.Timer(delay, self._reinitialize_camera, kwargs={'reason': reason}).start()

    def _reinitialize_camera(self, reason="Unknown"):
        """
        Attempts to reinitialize the camera after a failure.

        :param reason: The reason triggering reinitialization.
        """
        # Try to get the lock to avoid parallel reinitializations.
        if not self.reinit_lock.acquire(blocking=False):
            self._log("Reinitialization is already being performed in another thread.")
            return

        # Check whether a reinitialization is already activated.
        if self.reinitializing:
            self._log("Reinitialization is already in progress. Skip retry.")
            self.reinit_lock.release()
            return

        self.reinitializing = True
        try:
            self._log(f"Reinitializing camera due to: {reason}")
            if self.retry_count >= 5:
                self._log("Maximum retry attempts reached. Scheduling longer delay before next attempt.")
                # Longer delay before retrying again (e.g. 60 seconds)
                threading.Timer(60, self._reinitialize_camera, kwargs={'reason': "Retry after long delay"}).start()
                # Reset the retry counter after scheduling a longer delay
                self.retry_count = 0
                return

            self.retry_count += 1
            self._log(f"Reinitialization attempt {self.retry_count}/5.")

            # Stop current capture and threads
            self.release()
            self._log("Waiting 2 seconds before reinitialization attempt.")
            time.sleep(2)

            self.stop_event.clear()  # Reset stop_event for new threads
            self._setup_capture()
            self._start_reader_thread()
            self._start_health_check_thread()
            self._log("Reinitialization successful.")
            self.retry_count = 0  # Reset retry count on success
        except Exception as e:
            self._log(f"Reinitialization failed: {e}")
            self._log("Scheduling next reinitialization attempt.")
            self._schedule_reinit(reason=f"Failed to reinitialize: {e}")
        finally:
            self.reinitializing = False
            self.reinit_lock.release()

    def get_frame(self):
        if self.q.empty():
            self._log("Frame queue is empty.")
            return None
        try:
            frame = self.q.get_nowait()
            self._log("Retrieved a frame from the queue.")
            return frame
        except queue.Empty:
            self._log("Queue is empty, unable to retrieve frame.")
            return None

    def release(self):
        """
        Releases the video capture resources.
        """
        self._log("Releasing resources...")
        self.stop_event.set()  # Signal the thread to stop
        self.stop_flag = True

        # Stop the reader thread if it's not the current thread
        if self.reader_thread and self.reader_thread.is_alive():
            if self.reader_thread != threading.current_thread():
                self._log("Joining reader thread...")
                self.reader_thread.join(timeout=5)  # Wait for the thread to exit
                if self.reader_thread.is_alive():
                    self._log("Reader thread did not terminate within timeout.")
            else:
                self._log("Skipping join on current thread (reader thread).")

        # Clear the queue
        with self.q.mutex:
            self.q.queue.clear()
            self._log("Frame queue cleared.")

        # Stop the health check thread
        if self.health_check_thread and self.health_check_thread.is_alive():
            self._log("Joining health check thread...")
            self.health_check_thread.join(timeout=5)
            if self.health_check_thread.is_alive():
                self._log("Health check thread did not terminate within timeout.")

        # Release FFmpeg process
        if self.ffmpeg_process:
            self._log("Terminating FFmpeg process...")
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)  # Wait for FFmpeg to exit
                self._log("FFmpeg process terminated gracefully.")
            except subprocess.TimeoutExpired:
                self._log("FFmpeg process did not terminate in time. Killing process...")
                self.ffmpeg_process.kill()
                self._log("FFmpeg process killed.")

        # Release OpenCV capture if used
        if self.cap:
            self._log("Releasing OpenCV VideoCapture.")
            self.cap.release()

        self._log("Resources released.")

    @property
    def resolution(self):
        """
        Property to get the resolution of the video stream.
        """
        return (self.stream_width, self.stream_height)
