"""
Capture Interface - Video Frame Acquisition.

Defines the contract for video frame acquisition from various sources
(webcam, RTSP stream, file, etc.).
"""

from abc import ABC, abstractmethod

import numpy as np


class CaptureInterface(ABC):
    """
    Interface for video frame acquisition.

    Implementations should handle:
    - Connection to video source
    - Frame buffering
    - Source switching at runtime
    - Graceful degradation on errors
    """

    @abstractmethod
    def start(self) -> None:
        """
        Starts the frame acquisition.

        Should be non-blocking; actual capture runs in background thread.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stops the frame acquisition and releases resources.

        Should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """
        Returns the most recent frame.

        Returns:
            np.ndarray: BGR image, or None if no frame available.
        """
        pass

    @abstractmethod
    def get_frame_with_timestamp(self) -> tuple[np.ndarray | None, float]:
        """
        Returns the most recent frame with its capture timestamp.

        Returns:
            Tuple of (frame, timestamp). Frame is None if unavailable.
            Timestamp is Unix time (time.time()).
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """
        Checks if capture is currently active and receiving frames.

        Returns:
            True if capture is running and frames are being received.
        """
        pass

    @abstractmethod
    def update_source(self, source: str) -> None:
        """
        Updates the video source at runtime.

        Args:
            source: New video source (path, URL, or device index as string).

        Should handle:
        - Stopping current capture gracefully
        - Clearing frame buffers
        - Starting new capture
        """
        pass

    @property
    @abstractmethod
    def source(self) -> str:
        """Returns the current video source."""
        pass
