"""
Classification Service - Bird Species Classification.

Implements ClassificationInterface by wrapping the existing ImageClassifier.
Provides a clean interface for the detection pipeline.
"""

import cv2
import numpy as np

from detectors.classifier import ImageClassifier
from detectors.interfaces.classification import (
    ClassificationInterface,
    ClassificationResult,
)
from logging_config import get_logger

logger = get_logger(__name__)


class ClassificationService(ClassificationInterface):
    """
    Handles bird species classification on image crops.

    Wraps the existing ImageClassifier with a clean interface.
    Features:
    - Lazy model loading (on first use)
    - RGB and BGR input support
    - Automatic crop preprocessing
    """

    def __init__(self, classifier: ImageClassifier = None):
        """
        Initialize the classification service.

        Args:
            classifier: Optional existing classifier instance.
                       Creates new one if not provided.
        """
        self._classifier = classifier or ImageClassifier()

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies a bird species from an RGB image crop.

        Args:
            crop: RGB image crop of the detected bird.

        Returns:
            ClassificationResult with species name and confidence.
        """
        try:
            _, _, class_name, confidence = self._classifier.predict_from_image(crop)

            return ClassificationResult(
                class_name=class_name,
                confidence=confidence,
                model_id=self.get_model_id(),
            )
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                class_name="",
                confidence=0.0,
                model_id=self.get_model_id(),
            )

    def classify_from_bgr(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies a bird species from a BGR image crop.

        Convenience method that handles BGR to RGB conversion.

        Args:
            crop: BGR image crop of the detected bird.

        Returns:
            ClassificationResult with species name and confidence.
        """
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return self.classify(crop_rgb)
        except Exception as e:
            logger.error(f"Classification error (BGR conversion): {e}")
            return ClassificationResult(
                class_name="",
                confidence=0.0,
                model_id=self.get_model_id(),
            )

    def get_model_id(self) -> str:
        """
        Returns the model identifier.

        Returns:
            String identifying the model (path, name, or version).
        """
        return getattr(self._classifier, "model_id", "") or ""

    def is_ready(self) -> bool:
        """
        Checks if the classifier is ready for inference.

        Returns:
            True if model is loaded and ready.
        """
        return getattr(self._classifier, "_initialized", False)
