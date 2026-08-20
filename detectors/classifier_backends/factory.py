"""Classifier backend selection from WatchMyBirds configuration."""

from __future__ import annotations

from typing import Any

from detectors.classifier import ImageClassifier
from detectors.classifier_backends.base import ClassifierBackend
from detectors.classifier_backends.inat_tflite import (
    INaturalistTFLiteBirdClassifierBackend,
)
from detectors.classifier_backends.wmb_onnx import WMBOnnxClassifierBackend

CLASSIFIER_BACKEND_WMB = "wmb_onnx"
CLASSIFIER_BACKEND_INAT = "inat_tflite"
SUPPORTED_CLASSIFIER_BACKENDS = frozenset(
    {CLASSIFIER_BACKEND_WMB, CLASSIFIER_BACKEND_INAT}
)


def build_classifier_backend(
    config: dict[str, Any],
    *,
    wmb_classifier: ImageClassifier | None = None,
) -> ClassifierBackend:
    backend_name = (
        str(config.get("CLASSIFIER_BACKEND", CLASSIFIER_BACKEND_INAT)).strip().lower()
    )
    if backend_name == CLASSIFIER_BACKEND_WMB:
        return WMBOnnxClassifierBackend(wmb_classifier)
    if backend_name == CLASSIFIER_BACKEND_INAT:
        cpu_limit = int(config.get("CPU_LIMIT", 0) or 0)
        return INaturalistTFLiteBirdClassifierBackend(
            model_base_path=str(config.get("MODEL_BASE_PATH", "./data/models")),
            num_threads=cpu_limit or None,
        )
    raise ValueError(f"Unsupported classifier backend: {backend_name!r}")
