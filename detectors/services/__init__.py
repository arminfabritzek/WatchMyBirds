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
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.detection_service import DetectionService
from detectors.services.notification_service import NotificationService
from detectors.services.persistence_service import PersistenceService
from detectors.services.scoring_pipeline import ScoringResult, compute_detection_signals

__all__ = [
    "CaptureService",
    "ClassificationService",
    "DetectionService",
    "NotificationService",
    "PersistenceService",
    "DecisionPolicyService",
    "ScoringResult",
    "compute_detection_signals",
]
