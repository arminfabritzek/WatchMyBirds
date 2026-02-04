"""
Capture Service - Video Frame Acquisition.

Implements CaptureInterface by wrapping the existing VideoCapture.
Provides a clean interface for the detection pipeline.
"""

import threading
import time

import numpy as np

from camera.video_capture import VideoCapture
from config import get_config
from detectors.interfaces.capture import CaptureInterface
from logging_config import get_logger

logger = get_logger(__name__)


class CaptureService(CaptureInterface):
    """
    Handles video frame acquisition from various sources.

    Wraps the existing VideoCapture with a clean interface.
    Features:
    - Automatic source switching
    - Frame buffering with timestamps
    - Graceful error handling
    """

    def __init__(
        self,
        video_source: str | None = None,
        debug: bool = False,
        video_capture: VideoCapture | None = None,
    ):
        """
        Initialize the capture service.

        Args:
            video_source: Video source (from config if not provided).
            debug: Enable debug mode.
            video_capture: Optional existing VideoCapture instance.
        """
        self._config = get_config()
        self._source = video_source or self._config.get("VIDEO_SOURCE", "0")
        self._debug = debug or self._config.get("DEBUG_MODE", False)

        self._capture = video_capture
        self._initialized = False

        # Frame buffer
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float = 0.0
        self._frame_lock = threading.Lock()

        if self._capture is not None:
            self._initialized = True

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of video capture."""
        if self._initialized and self._capture is not None:
            return True

        try:
            self._capture = VideoCapture(
                self._source, debug=self._debug, auto_start=False
            )
            self._initialized = True
            logger.info(f"VideoCapture initialized: {self._source}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            return False

    def start(self) -> None:
        """Starts the frame acquisition."""
        if not self._ensure_initialized():
            logger.error("Cannot start: VideoCapture not initialized")
            return

        try:
            self._capture.start()
            logger.info("VideoCapture started")
        except Exception as e:
            logger.error(f"Failed to start video capture: {e}")

    def stop(self) -> None:
        """Stops the frame acquisition and releases resources."""
        if self._capture is None:
            return

        try:
            self._capture.stop_event.set()
            if hasattr(self._capture, "cap") and self._capture.cap:
                self._capture.cap.release()
            if (
                hasattr(self._capture, "ffmpeg_process")
                and self._capture.ffmpeg_process
            ):
                self._capture._terminate_ffmpeg_process()
            logger.info("VideoCapture stopped")
        except Exception as e:
            logger.error(f"Error stopping video capture: {e}")

        self._capture = None
        self._initialized = False

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0

    def get_frame(self) -> np.ndarray | None:
        """Returns the most recent frame."""
        if not self._ensure_initialized():
            return None

        try:
            frame = self._capture.get_frame()
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    self._latest_timestamp = time.time()
            return frame
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def get_frame_with_timestamp(self) -> tuple[np.ndarray | None, float]:
        """Returns the most recent frame with its capture timestamp."""
        frame = self.get_frame()
        with self._frame_lock:
            return frame, self._latest_timestamp

    def is_active(self) -> bool:
        """Checks if capture is currently active and receiving frames."""
        if not self._initialized or self._capture is None:
            return False

        with self._frame_lock:
            # Active if we've received a frame in the last 5 seconds
            return (time.time() - self._latest_timestamp) < 5.0

    def update_source(self, source: str) -> None:
        """
        Updates the video source at runtime.

        Args:
            source: New video source (path, URL, or device index as string).
        """
        logger.info(f"Updating video source: {self._source} -> {source}")

        # Clear frame buffer
        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0

        # Stop current capture
        self.stop()

        # Update source and reinitialize
        self._source = source

        if self._ensure_initialized():
            self.start()

    @property
    def source(self) -> str:
        """Returns the current video source."""
        return self._source

    def get_buffered_frame(self) -> tuple[np.ndarray | None, float]:
        """
        Returns the last buffered frame without fetching new one.

        Useful for the detection loop to avoid competing with frame thread.

        Returns:
            Tuple of (frame copy, timestamp). Frame is None if no buffer.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._latest_timestamp
            return None, 0.0

    def update_buffer(self) -> bool:
        """
        Updates the internal frame buffer from video capture.

        Returns:
            True if a new frame was captured.
        """
        frame = self.get_frame()
        return frame is not None
