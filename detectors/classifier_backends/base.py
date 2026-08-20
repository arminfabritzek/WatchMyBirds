"""Shared contract for crop-classification runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClassifierBackendPrediction:
    """Backend-neutral species prediction returned to the pipeline service."""

    class_name: str
    confidence: float
    model_id: str
    top_k_classes: list[str] = field(default_factory=list)
    top_k_confidences: list[float] = field(default_factory=list)
    decision_level: str = "species"
    raw_species_name: str = ""
    common_name: str = ""


class ClassifierBackend(ABC):
    """A classifier runtime with model-owned preprocessing and decoding."""

    backend_name: str

    @abstractmethod
    def predict_rgb(
        self,
        image: np.ndarray,
        *,
        top_k: int = 5,
    ) -> ClassifierBackendPrediction:
        """Classify one RGB bird crop."""

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the loaded model's stable provenance identifier."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the runtime has completed lazy initialization."""
