# ------------------------------------------------------------------------------
# Video Capture Class for Input Streams
# ------------------------------------------------------------------------------
# video_capture.py


import subprocess
import cv2
import queue
import threading
from threading import Event
import time
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)  # Default to INFO level
logger = logging.getLogger(__name__)

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
        self.stream_type = self._detect_stream_type()
        self._setup_capture()
        self._start_reader_thread()

    def _log(self, message):
        """Logs a message if debug mode is enabled."""
        if self.debug:
            logger.debug(message)

    def _detect_stream_type(self):
        """
        Automatically detects the stream type based on the source string.

        :return: Detected stream type as a string ("rtsp", "http", "webcam").
        """
        self._log("Detect stream type...")
        if isinstance(self.source, str):
            if self.source.startswith("rtsp://"):
                return "rtsp"
            elif self.source.startswith(("http://", "https://")):
                return "http"
        elif isinstance(self.source, int):  # Integer sources are treated as webcams
            return "webcam"
        raise ValueError(f"Unable to determine stream type for source: {self.source}")


    def _setup_capture(self):
        """
        Sets up the video capture based on the detected stream type.
        """
        if self.stream_type == "rtsp":
            self._setup_ffmpeg()
        elif self.stream_type == "http":
            self._setup_http()
        elif self.stream_type == "webcam":
            self._setup_webcam()
        else:
            raise ValueError(f"Unsupported stream type: {self.stream_type}")

    def _setup_rtsp(self):
        self._log("Setting up RTSP stream...")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

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

        for attempt in range(5):  # Retry up to 5 times
            try:
                self._log(f"Starting FFmpeg process (attempt {attempt + 1})...")

                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
                )
                self._log("FFmpeg process started successfully.")

                threading.Thread(target=self._log_ffmpeg_errors, daemon=True).start()
                return  # Exit loop on success
            except Exception as e:
                self._log(f"Failed to start FFmpeg process: {e}. Retrying...")

                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("Failed to start FFmpeg after multiple attempts.")

    def _log_ffmpeg_errors(self):
        """
        Continuously logs FFmpeg's stderr for debugging purposes, only if debug mode is enabled.
        """
        if not self.debug:  # Skip logging if debug is False
            return

        for line in self.ffmpeg_process.stderr:
            logger.debug(f"FFmpeg STDERR: {line.decode('utf-8').strip()}")

    def _setup_http(self):
        """
        Sets up OpenCV's VideoCapture for HTTP streams.
        """
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5-second timeout
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5-second read timeout

    def _setup_webcam(self):
        """
        Sets up OpenCV's VideoCapture for webcam streams.
        """
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)


    def _start_reader_thread(self):
        """
        Starts the reader thread if it's not already running.
        Ensures only one thread is active at a time to avoid conflicts.
        """
        if self.reader_thread and self.reader_thread.is_alive():
            self._log("Reader thread already running; skipping reinitialization")

        else:
            # Add a short delay to allow FFmpeg to initialize properly
            time.sleep(2)
            self.reader_thread = threading.Thread(target=self._reader, daemon=True)
            self.reader_thread.start()
            self._log("Reader thread started successfully.")


    def _reader(self):
        """
        Continuously reads frames from the video source and places them in the queue.
        Terminates cleanly if stop_event is set.
        """
        while not self.stop_event.is_set():  # Check for thread termination
            try:
                frame = self._read_frame()
                if self.stop_event.is_set():  # Recheck stop_event after potentially long operations
                    break
                if frame is not None:
                    if not self.q.full():
                        self.q.put(frame, timeout=0.1)  # Keep the queue filled with new frames
                else:
                    self._log("Frame not available, retrying...")
                    self._reinitialize_camera()  # Trigger reinitialization in case of failure
            except Exception as e:
                print(f"Error reading frame: {e}")
                if self.stop_event.is_set():
                    break
                self._reinitialize_camera()  # Reinitialize in case of error


    def _read_frame(self):
        """
        Reads a single frame based on the stream type.
        """
        if self.stream_type == "rtsp":
            return self._read_ffmpeg_frame()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None


    def _read_ffmpeg_frame(self):
        width, height = 1920, 1080  # Update these values to match your stream resolution
        frame_size = width * height * 3  # For bgr24 (3 bytes per pixel)

        try:
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                self._log("FFmpeg did not produce a complete frame...")

                return None  # Skip incomplete frames
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            return frame
        except Exception as e:
            self._log(f"Error reading frame from FFmpeg: {e}")
            return None


    def _reinitialize_camera(self):
        if threading.current_thread() == self.reader_thread:
            self._log("Warning: Skipping reinitialization from within the reader thread.")

            self.stop_event.set()  # Ensure the thread stops cleanly
            return

        self._log("Reinitializing camera...")

        self.stop_event.set()
        self.release()
        time.sleep(2)

        try:
            self.retry_count += 1
            self._setup_capture()
            self._start_reader_thread()
            self._log(f"Reinitialization complete. Attempt {self.retry_count}")

            self.retry_count = 0  # Reset retry count on success
        except Exception as e:
            self._log(f"Failed to reinitialize: {e}")

            if self.retry_count < 5:
                self._reinitialize_camera()
            else:
                self._log("Max retries reached. Giving up on reinitialization.")

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
        self.stop_event.set()  # Signal the thread to stop
        self.stop_flag = True

        # Stop the reader thread
        if self.reader_thread and self.reader_thread.is_alive():
            self._log("Joining reader thread...")

            self.reader_thread.join(timeout=5)  # Wait for the thread to exit

        # Clear the queue
        with self.q.mutex:
            self.q.queue.clear()

        # Release FFmpeg process
        if self.ffmpeg_process:
            self._log("Terminating FFmpeg process...")

            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)  # Wait for FFmpeg to exit
            except subprocess.TimeoutExpired:
                self._log("FFmpeg process did not terminate. Killing process...")

                self.ffmpeg_process.kill()

        # Release OpenCV capture if used
        if self.cap:
            self.cap.release()

        self._log("Resources released.")
