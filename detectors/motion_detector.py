# detectors/motion_detector.py
import cv2
import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)


class MotionDetector:
    """
    A lightweight motion detector using Frame Differencing.
    Designed to act as a pre-filter for heavier object detection models.
    """

    def __init__(self, sensitivity=500, debug=False):
        """
        Args:
            sensitivity (int): Minimum contour area to trigger motion.
                               Lower = more sensitive.
            debug (bool): Enable debug logging.
        """
        self.sensitivity = sensitivity
        self.debug = debug
        self.previous_frame_gray = None
        self.kernel = np.ones((5, 5), np.uint8)

    def reset(self):
        """Resets the motion detector state. Call when video source changes."""
        self.previous_frame_gray = None
        logger.debug("MotionDetector state reset")

    def detect(self, frame):
        """
        Detects motion in the provided frame compared to the previous frame.

        Args:
            frame (np.array): The current BGR frame.

        Returns:
            bool: True if motion is detected, False otherwise.
        """
        import time

        if frame is None:
            return False

        # Convert to grayscale and blur to remove noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame if first run
        if self.previous_frame_gray is None:
            self.previous_frame_gray = gray
            self._stats_start = time.time()
            self._stats_motion = 0
            self._stats_total = 0
            return False  # Can't detect motion on first frame

        # Compute absolute difference
        frame_diff = cv2.absdiff(self.previous_frame_gray, gray)

        # Threshold the difference (pixel value > 25 becomes 255/white)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, self.kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.sensitivity:
                motion_detected = True
                if area > max_area:
                    max_area = area

        # Update previous frame
        self.previous_frame_gray = gray

        # Periodic logging (every 30 seconds)
        self._stats_total = getattr(self, "_stats_total", 0) + 1
        if motion_detected:
            self._stats_motion = getattr(self, "_stats_motion", 0) + 1

        stats_start = getattr(self, "_stats_start", time.time())
        if time.time() - stats_start >= 30:
            ratio = (
                (self._stats_motion / self._stats_total * 100)
                if self._stats_total > 0
                else 0
            )
            logger.info(
                f"[MOTION] {self._stats_motion}/{self._stats_total} frames with motion ({ratio:.0f}%) in last 30s"
            )
            self._stats_start = time.time()
            self._stats_motion = 0
            self._stats_total = 0

        return motion_detected
