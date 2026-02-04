"""
Detection Service - Object Detection.

Implements DetectionInterface by wrapping the existing Detector.
Provides a clean interface for the detection pipeline.
"""

import os
import time

import numpy as np

from config import get_config
from detectors.detector import Detector
from detectors.interfaces.detection import DetectionInterface, DetectionResult
from logging_config import get_logger

logger = get_logger(__name__)


class DetectionService(DetectionInterface):
    """
    Handles object detection on video frames.

    Wraps the existing Detector with a clean interface.
    Features:
    - Lazy detector initialization
    - Automatic reinitialization on errors
    - Inference timing
    """

    def __init__(
        self,
        model_choice: str | None = None,
        debug: bool = False,
        detector: Detector | None = None,
    ):
        """
        Initialize the detection service.

        Args:
            model_choice: Model to use (from config if not provided).
            debug: Enable debug mode.
            detector: Optional existing detector instance.
        """
        self._config = get_config()
        self._model_choice = model_choice or self._config.get(
            "DETECTOR_MODEL_CHOICE", "yolov8n"
        )
        self._debug = debug or self._config.get("DEBUG_MODE", False)

        self._detector = detector
        self._model_id = ""
        self._initialized = False

        if self._detector is not None:
            self._initialized = True
            self._update_model_id()

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the detector."""
        if self._initialized and self._detector is not None:
            return True

        try:
            self._detector = Detector(
                model_choice=self._model_choice, debug=self._debug
            )
            self._update_model_id()
            self._initialized = True
            logger.info(f"Detector initialized: {self._model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False

    def _update_model_id(self) -> None:
        """Extract model ID from detector."""
        if self._detector:
            self._model_id = getattr(self._detector, "model_id", "") or ""
            if not self._model_id and hasattr(self._detector, "model_path"):
                self._model_id = os.path.basename(self._detector.model_path)

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
        if not self._ensure_initialized():
            return DetectionResult(
                detected=False,
                original_frame=None,
                detections=[],
                model_id=self._model_id,
                inference_time_ms=0,
            )

        start_time = time.time()

        try:
            object_detected, original_frame, detection_info_list = (
                self._detector.detect_objects(
                    frame,
                    confidence_threshold=confidence_threshold,
                    save_threshold=save_threshold,
                )
            )

            inference_time_ms = int((time.time() - start_time) * 1000)

            return DetectionResult(
                detected=object_detected,
                original_frame=original_frame,
                detections=detection_info_list,
                model_id=self._model_id,
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            inference_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Detection error: {e}")

            return DetectionResult(
                detected=False,
                original_frame=None,
                detections=[],
                model_id=self._model_id,
                inference_time_ms=inference_time_ms,
            )

    def get_model_id(self) -> str:
        """Returns the model identifier."""
        return self._model_id

    def is_ready(self) -> bool:
        """Checks if the detector is ready for inference."""
        return self._initialized and self._detector is not None

    def reinitialize(self) -> bool:
        """
        Reinitializes the detector after an error.

        Returns:
            True if reinitialization was successful.
        """
        logger.info("Reinitializing detector...")
        self._detector = None
        self._initialized = False
        return self._ensure_initialized()
