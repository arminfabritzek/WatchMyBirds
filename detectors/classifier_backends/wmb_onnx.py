"""Adapter for WatchMyBirds' existing ONNX classifier."""

from __future__ import annotations

import numpy as np

from detectors.classifier import ImageClassifier
from detectors.classifier_backends.base import (
    ClassifierBackend,
    ClassifierBackendPrediction,
)


class WMBOnnxClassifierBackend(ClassifierBackend):
    """Preserve the legacy ONNX behavior behind the backend contract."""

    backend_name = "wmb_onnx"

    def __init__(self, classifier: ImageClassifier | None = None) -> None:
        self.classifier = classifier if classifier is not None else ImageClassifier()

    def predict_rgb(
        self,
        image: np.ndarray,
        *,
        top_k: int = 5,
    ) -> ClassifierBackendPrediction:
        indices, confidences, class_name, confidence = (
            self.classifier.predict_from_image(image, top_k=top_k)
        )
        decision = getattr(self.classifier, "last_decision", None) or {}
        decision_level = str(decision.get("level", "species"))
        shown_label = str(decision.get("label") or "") if decision else class_name
        shown_confidence = float(confidence)
        if decision_level == "genus":
            shown_confidence = float(decision.get("prob", confidence))

        classes = self.classifier.classes or []
        return ClassifierBackendPrediction(
            class_name=shown_label,
            confidence=shown_confidence,
            model_id=self.get_model_id(),
            top_k_classes=[classes[int(index)] for index in indices],
            top_k_confidences=[float(value) for value in confidences],
            decision_level=decision_level,
            raw_species_name=str(decision.get("raw_species", class_name)),
        )

    def get_model_id(self) -> str:
        return getattr(self.classifier, "model_id", "") or ""

    def is_ready(self) -> bool:
        return bool(getattr(self.classifier, "_initialized", False))
