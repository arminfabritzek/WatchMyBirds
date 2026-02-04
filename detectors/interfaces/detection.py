"""
Detection Interface - Object Detection.

Defines the contract for object detection on video frames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """
    Result of an object detection operation.

    Attributes:
        detected: Whether any objects were detected above threshold.
        original_frame: The frame that was analyzed (may have annotations).
        detections: List of detection dictionaries with bbox and confidence.
        model_id: Identifier of the model used.
        inference_time_ms: Time taken for inference in milliseconds.
    """

    detected: bool
    original_frame: np.ndarray | None
    detections: list[dict] = field(default_factory=list)
    model_id: str = ""
    inference_time_ms: int = 0

    # Detection dict structure:
    # {
    #     "x1": int, "y1": int, "x2": int, "y2": int,  # Bounding box
    #     "confidence": float,  # Detection confidence
    #     "class_name": str,  # Detected class (e.g., "bird")
    # }


class DetectionInterface(ABC):
    """
    Interface for object detection.

    Implementations should handle:
    - Model loading (lazy or eager)
    - Inference on frames
    - Confidence thresholding
    - Error recovery (reinitialize on failure)
    """

    @abstractmethod
    def detect(
        self, frame: np.ndarray, confidence_threshold: float, save_threshold: float
    ) -> DetectionResult:
        """
        Performs object detection on a frame.

        Args:
            frame: BGR image to analyze.
            confidence_threshold: Minimum confidence for a detection to be valid.
            save_threshold: Minimum confidence for a detection to be saved.

        Returns:
            DetectionResult with detected objects.
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Returns the model identifier.

        Returns:
            String identifying the model (path, name, or version).
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Checks if the detector is ready for inference.

        Returns:
            True if model is loaded and ready.
        """
        pass

    @abstractmethod
    def reinitialize(self) -> bool:
        """
        Reinitializes the detector after an error.

        Returns:
            True if reinitialization was successful.
        """
        pass
