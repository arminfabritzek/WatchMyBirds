"""
Detection Pipeline Interfaces.

This package defines the abstract interfaces for all components
of the detection pipeline. These interfaces enable:
- Clear service boundaries
- Dependency injection
- Independent testing of each component
- Future extensibility

ARCHITECTURE:
- DetectionManager only coordinates these interfaces
- Concrete implementations live in services/
- No direct dependencies between implementations
"""

from detectors.interfaces.capture import CaptureInterface
from detectors.interfaces.classification import (
    ClassificationInterface,
    ClassificationResult,
)
from detectors.interfaces.detection import DetectionInterface, DetectionResult
from detectors.interfaces.notification import (
    NotificationInterface,
    SpeciesInfo,
)
from detectors.interfaces.persistence import (
    DetectionPersistenceResult,
    ImagePersistenceResult,
    PersistenceInterface,
)

__all__ = [
    # Interfaces
    "CaptureInterface",
    "DetectionInterface",
    "ClassificationInterface",
    "PersistenceInterface",
    "NotificationInterface",
    # Data Classes
    "DetectionResult",
    "ClassificationResult",
    "ImagePersistenceResult",
    "DetectionPersistenceResult",
    "SpeciesInfo",
]
