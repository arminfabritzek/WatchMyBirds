import subprocess
import cv2
import queue
import threading
from threading import Event
import time
import numpy as np


class VideoCapture:
    """
    Handles video input streams from various sources (RTSP, WebRTC, Webcam, FFmpeg).
    """
    def __init__(self, source):
        """
        Initializes the VideoCapture object and detects the stream type.
        """
        self.source = source
        self.stop_flag = False
        self.stop_event = Event()  # Event to signal the thread to stop
        self.cap = None
        self.ffmpeg_process = None  # Used for FFmpeg streams
        self.q = queue.Queue(maxsize=10)  # Queue for storing frames
        self.stream_type = self._detect_stream_type()
        self._setup_capture()
        self._start_reader_thread()


    def _detect_stream_type(self):
        """
        Automatically detects the stream type based on the source string.

        :return: Detected stream type as a string ("rtsp", "http", "webcam").
        """
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
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def _setup_ffmpeg(self):
        ffmpeg_cmd = [
            "ffmpeg",
            "-probesize", "50000000",  # Increase probesize for complex streams
            "-analyzeduration", "10000000",  # Increase analyze duration
            "-rtsp_transport", "tcp",  # Use TCP transport for RTSP
            "-i", self.source,  # Input stream URL
            "-f", "rawvideo",  # Output raw video
            "-pix_fmt", "bgr24",  # Keep pixel format as bgr24
            "-an",  # Disable audio
            "pipe:"  # Output to pipe
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8
            )
            print("FFmpeg process started successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg process: {e}")

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
        # Add a short delay to allow FFmpeg to initialize properly
        time.sleep(4)
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def _reader(self):
        """
        Background thread to read frames continuously and keep only the latest frame.
        """
        while not self.stop_flag:
            try:
                frame = self._read_frame()
                if frame is not None:
                    # Discard old frames if queue is full
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(frame, timeout=0.1)
            except Exception as e:
                print(f"Error reading frame: {e}")
                self._reinitialize_camera()

    def _reader(self):
        while not self.stop_event.is_set():  # Exit if stop_event is set
            try:
                frame = self._read_frame()
                if frame is not None:
                    # Discard old frames if queue is full
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(frame, timeout=0.1)
                else:
                    # Attempt to reinitialize only if stop_event is not set
                    if not self.stop_event.is_set():
                        print("No frame available. Attempting to reinitialize camera...")
                        self._reinitialize_camera()
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error reading frame: {e}")
                break

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
                raise ValueError("Failed to read a full frame from FFmpeg.")
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            return frame
        except ValueError as e:
            print(f"Error reading frame: {e}")
            return None

    def _reinitialize_camera(self):
        """
        Reinitializes the camera with exponential backoff after failures.
        """
        self.stop_flag = True
        self.release()
        time.sleep(2)  # Short delay before reinitialization
        self._setup_capture()
        self._start_reader_thread()

    def get_frame(self):
        if self.q.empty():
            return None
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def release(self):
        """
        Releases the video capture resources.
        """
        self.stop_event.set()  # Signal the thread to stop
        self.stop_flag = True
        if self.reader_thread.is_alive():
            self.reader_thread.join()  # Ensure the reader thread exits
        if self.cap:
            self.cap.release()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
        print("Resources released.")