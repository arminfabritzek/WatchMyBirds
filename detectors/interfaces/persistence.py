"""
Persistence Interface - Image and Detection Storage.

Defines the contract for persisting images, detections, and thumbnails.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class ImagePersistenceResult:
    """
    Result of an image persistence operation.

    Attributes:
        success: Whether the image was saved successfully.
        original_path: Path to the original JPEG image.
        optimized_path: Path to the optimized WebP image.
        base_filename: The base filename (e.g., "20240120_120000_123456.jpg").
        date_str: The date string for folder organization (e.g., "2024-01-20").
    """

    success: bool
    original_path: Path | None = None
    optimized_path: Path | None = None
    base_filename: str = ""
    date_str: str = ""


@dataclass
class DetectionPersistenceResult:
    """
    Result of a detection persistence operation.

    Attributes:
        success: Whether the detection was saved successfully.
        detection_id: Database ID of the inserted detection.
        thumbnail_path: Path to the generated thumbnail.
        thumbnail_filename: Filename of the thumbnail.
    """

    success: bool
    detection_id: int = 0
    thumbnail_path: Path | None = None
    thumbnail_filename: str = ""


@dataclass
class DetectionData:
    """
    Data for a single detection to be persisted.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixels.
        confidence: Detection confidence.
        class_name: Detected class name (e.g., "bird").
        cls_class_name: Classification result (Latin name).
        cls_confidence: Classification confidence.
        score: Combined detection+classification score.
        agreement_score: Agreement between detection and classification.
    """

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str = ""
    cls_class_name: str = ""
    cls_confidence: float = 0.0
    score: float = 0.0
    agreement_score: float = 0.0


class PersistenceInterface(ABC):
    """
    Interface for image and detection persistence.

    Implementations should handle:
    - Original and optimized image saving
    - EXIF metadata injection
    - Detection database records
    - Thumbnail generation
    """

    @abstractmethod
    def save_image(
        self,
        frame: np.ndarray,
        capture_time: datetime,
        detector_model_id: str,
        classifier_model_id: str,
        source_id: int,
        location_config: dict | None = None,
        exif_gps_enabled: bool = True,
    ) -> ImagePersistenceResult:
        """
        Saves original and optimized versions of an image.

        Args:
            frame: BGR image to save.
            capture_time: When the image was captured.
            detector_model_id: ID of the detection model.
            classifier_model_id: ID of the classification model.
            source_id: Database ID of the video source.
            location_config: Optional GPS coordinates for EXIF.
            exif_gps_enabled: Whether to include GPS in EXIF.

        Returns:
            ImagePersistenceResult with paths and success status.
        """
        pass

    @abstractmethod
    def save_detection(
        self,
        image_filename: str,
        detection: DetectionData,
        frame: np.ndarray,
        detector_model_id: str,
        classifier_model_id: str,
        crop_index: int,
    ) -> DetectionPersistenceResult:
        """
        Saves a detection with thumbnail and database record.

        Args:
            image_filename: Filename of the parent image.
            detection: Detection data to persist.
            frame: Original frame for thumbnail generation.
            detector_model_id: ID of the detection model.
            classifier_model_id: ID of the classification model.
            crop_index: Index of this detection (for thumbnail naming).

        Returns:
            DetectionPersistenceResult with database ID and paths.
        """
        pass

    @abstractmethod
    def generate_thumbnail(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        output_path: Path,
        size: int = 256,
    ) -> bool:
        """
        Generates a square thumbnail from a detection bbox.

        Args:
            frame: Original frame.
            bbox: Bounding box as (x1, y1, x2, y2).
            output_path: Where to save the thumbnail.
            size: Target thumbnail size (square).

        Returns:
            True if thumbnail was generated successfully.
        """
        pass
