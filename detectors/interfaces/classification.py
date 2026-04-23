"""
Classification Interface - Bird Species Classification.

Defines the contract for classifying bird species from image crops.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class DecisionState(StrEnum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


@dataclass
class DecisionResult:
    decision_state: DecisionState
    bbox_quality: float | None = None
    species_conf: float | None = None
    unknown_score: float | None = None
    reasons_json: str = "[]"
    policy_version: str = "1.0"


@dataclass
class ClassificationResult:
    """
    Result of a classification operation.

    Attributes:
        class_name: Label to show downstream. For legacy (pre-config)
            classifiers and for species-level confident predictions
            this is the Latin species name (e.g. "Parus_major"). For
            configured classifiers that fall back to genus level it is
            the genus token (e.g. "Sylvia_sp."). Empty string when the
            decision layer rejects the prediction.
        confidence: Classification confidence (0.0 to 1.0). When the
            decision level is ``genus``, this is the summed probability
            mass over sibling species, not the top-1 probability.
        model_id: Identifier of the model used.
        top_k_classes / top_k_confidences: raw top-K slice from the
            classifier softmax (species-level ordering regardless of
            decision level).
        decision_level: One of ``"species"``, ``"genus"``, or
            ``"reject"``. Defaults to ``"species"`` so results built
            without a decision layer (legacy path or error path) keep
            behaving like today's code expects.
        raw_species_name: The top-1 species label from the softmax,
            regardless of what decision level was chosen. Useful for
            logging / debugging even when the shown label is genus or
            empty.
    """

    class_name: str
    confidence: float
    model_id: str = ""
    top_k_classes: list[str] = field(default_factory=list)
    top_k_confidences: list[float] = field(default_factory=list)
    decision_level: str = "species"
    raw_species_name: str = ""


def compute_unknown_score(top_k_confidences: list[float]) -> float:
    """
    Computes an unknown/out-of-distribution score from top-k class probabilities.

    Uses two complementary signals:
    - **Margin**: Difference between top-1 and top-2 confidence.
      A small margin means the model is unsure which class it is.
    - **Entropy**: Shannon entropy of the top-k distribution (normalized).
      High entropy means the probability mass is spread across many classes.

    Returns:
        Score in [0.0, 1.0]. Higher means more likely unknown/OOD.
    """
    if not top_k_confidences or len(top_k_confidences) < 1:
        # No data at all → unknown by definition
        return 1.0

    top1 = top_k_confidences[0]

    # Single class or no second class → use inverse of top-1 confidence
    if len(top_k_confidences) < 2:
        return float(np.clip(1.0 - top1, 0.0, 1.0))

    # Margin: small margin → high unknown score
    top2 = top_k_confidences[1]
    margin = top1 - top2
    # Normalize margin to [0, 1]: margin of 0 → unknown_score=1, margin >=0.5 → 0
    margin_score = float(np.clip(1.0 - margin / 0.5, 0.0, 1.0))

    # Entropy (normalized Shannon entropy over top-k)
    probs = np.array(top_k_confidences, dtype=np.float64)
    probs = probs / (probs.sum() + 1e-12)  # re-normalize top-k to sum=1
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs) + 1e-12)
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Blend: 60% margin, 40% entropy
    unknown = 0.6 * margin_score + 0.4 * norm_entropy
    return float(np.clip(unknown, 0.0, 1.0))


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
