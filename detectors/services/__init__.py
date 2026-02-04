"""
Detection Pipeline Services.

This package contains concrete implementations of the pipeline interfaces.
Each service encapsulates a specific responsibility and can be tested independently.

ARCHITECTURE:
- Services implement interfaces from detectors/interfaces/
- Services may use utils/ for low-level operations
- DetectionManager orchestrates these services
"""

from detectors.services.capture_service import CaptureService
from detectors.services.classification_service import ClassificationService
from detectors.services.detection_service import DetectionService
from detectors.services.notification_service import NotificationService
from detectors.services.persistence_service import PersistenceService

__all__ = [
    "CaptureService",
    "ClassificationService",
    "DetectionService",
    "NotificationService",
    "PersistenceService",
]
