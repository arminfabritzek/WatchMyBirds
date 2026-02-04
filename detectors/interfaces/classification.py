"""
Classification Interface - Bird Species Classification.

Defines the contract for classifying bird species from image crops.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationResult:
    """
    Result of a classification operation.

    Attributes:
        class_name: Latin name of the classified species (e.g., "Parus_major").
        confidence: Classification confidence (0.0 to 1.0).
        model_id: Identifier of the model used.
    """

    class_name: str
    confidence: float
    model_id: str = ""


class ClassificationInterface(ABC):
    """
    Interface for bird species classification.

    Implementations should handle:
    - Model loading (lazy or eager)
    - Image preprocessing
    - Top-1 prediction with confidence
    """

    @abstractmethod
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies a bird species from an image crop.

        Args:
            crop: RGB image crop of the detected bird.

        Returns:
            ClassificationResult with species name and confidence.
        """
        pass

    @abstractmethod
    def classify_from_bgr(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies a bird species from a BGR image crop.

        Convenience method that handles BGR to RGB conversion.

        Args:
            crop: BGR image crop of the detected bird.

        Returns:
            ClassificationResult with species name and confidence.
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
        Checks if the classifier is ready for inference.

        Returns:
            True if model is loaded and ready.
        """
        pass
