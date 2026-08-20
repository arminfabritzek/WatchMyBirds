"""Pluggable species-classifier backends."""

from detectors.classifier_backends.base import (
    ClassifierBackend,
    ClassifierBackendPrediction,
)
from detectors.classifier_backends.factory import (
    CLASSIFIER_BACKEND_INAT,
    CLASSIFIER_BACKEND_WMB,
    build_classifier_backend,
)

__all__ = [
    "CLASSIFIER_BACKEND_INAT",
    "CLASSIFIER_BACKEND_WMB",
    "ClassifierBackend",
    "ClassifierBackendPrediction",
    "build_classifier_backend",
]
