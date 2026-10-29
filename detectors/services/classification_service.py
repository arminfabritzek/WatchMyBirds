"""Backend-neutral bird species classification service."""

import json
from dataclasses import replace
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from core.species_geography_runtime import GeographicWeighting
from detectors.classifier import ImageClassifier
from detectors.classifier_backends.base import ClassifierBackend
from detectors.classifier_backends.wmb_onnx import WMBOnnxClassifierBackend
from detectors.interfaces.classification import (
    ClassificationInterface,
    ClassificationResult,
)
from logging_config import get_logger

logger = get_logger(__name__)


class ClassificationService(ClassificationInterface):
    """
    Handles bird species classification on image crops.

    Wraps a classifier backend with a stable pipeline interface.
    Features:
    - Lazy model loading (on first use)
    - RGB and BGR input support
    - Automatic crop preprocessing
    """

    def __init__(
        self,
        classifier: ImageClassifier | None = None,
        *,
        backend: ClassifierBackend | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the classification service.

        Args:
            classifier: Optional legacy ONNX classifier instance.
            backend: Optional explicit backend. Takes precedence over classifier.
        """
        self._backend = (
            backend
            if backend is not None
            else WMBOnnxClassifierBackend(
                classifier if classifier is not None else ImageClassifier()
            )
        )
        self._geography = GeographicWeighting(config) if config is not None else None

    @property
    def backend(self) -> ClassifierBackend:
        return self._backend

    def replace_backend(self, backend: ClassifierBackend) -> None:
        """Atomically route future classifications to a new backend."""

        self._backend = backend
        try:
            from utils.species_names import clear_species_name_caches

            clear_species_name_caches()
        except Exception as exc:
            logger.debug("Species-name cache refresh skipped: %s", exc)

    def classify(
        self,
        crop: np.ndarray,
        *,
        observed_at: datetime | None = None,
        location: dict[str, float] | None = None,
    ) -> ClassificationResult:
        """
        Classifies a bird species from an RGB image crop.

        Args:
            crop: RGB image crop of the detected bird.

        Returns:
            ClassificationResult with species name and confidence.
        """
        try:
            prediction = self._backend.predict_rgb(crop)

            result = ClassificationResult(
                class_name=prediction.class_name,
                confidence=prediction.confidence,
                model_id=prediction.model_id,
                top_k_classes=prediction.top_k_classes,
                top_k_confidences=prediction.top_k_confidences,
                decision_level=prediction.decision_level,
                raw_species_name=prediction.raw_species_name,
                stage_timings_ms=dict(getattr(prediction, "stage_timings_ms", {})),
            )
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                class_name="",
                confidence=0.0,
                model_id=self.get_model_id(),
            )
        try:
            return self.apply_geography(result, observed_at, location=location)
        except Exception as exc:
            logger.warning(
                "Retaining image classification after geography error: %s", exc
            )
            return result

    def apply_geography(
        self,
        result: ClassificationResult,
        observed_at: datetime | None,
        *,
        location: dict[str, float] | None = None,
    ) -> ClassificationResult:
        if (
            self._geography is None
            or result.decision_level != "species"
            or result.class_name in {"", "background", "non_bird"}
            or len(result.top_k_classes) != len(result.top_k_confidences)
        ):
            return result
        candidates = list(
            zip(result.top_k_classes, result.top_k_confidences, strict=False)
        )
        if not candidates or candidates[0][0] != result.class_name:
            return result
        assessed = self._geography.assess(candidates, observed_at, location=location)
        if assessed is None:
            return result
        assessment, provenance = assessed
        selected = assessment.suggested_species
        if selected in {"background", "non_bird"}:
            return result
        evidence = {
            **assessment.to_dict(),
            **provenance,
            "raw_candidates": candidates,
            "raw_confidence": result.confidence,
        }
        return replace(
            result,
            class_name=selected,
            confidence=dict(candidates)[selected],
            raw_species_name=result.raw_species_name or result.class_name,
            geography_json=json.dumps(evidence, allow_nan=False),
            storage_alternatives=[
                (name, score) for name, score in candidates if name != selected
            ],
        )

    def classify_from_bgr(
        self,
        crop: np.ndarray,
        *,
        observed_at: datetime | None = None,
        location: dict[str, float] | None = None,
    ) -> ClassificationResult:
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
            return self.classify(crop_rgb, observed_at=observed_at, location=location)
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
        return self._backend.get_model_id()

    def is_ready(self) -> bool:
        """
        Checks if the classifier is ready for inference.

        Returns:
            True if model is loaded and ready.
        """
        return self._backend.is_ready()
