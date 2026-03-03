from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult, DecisionState
from detectors.services.decision_policy_service import DecisionPolicyService
from web.services.analysis_service import _build_detection_payload


@pytest.fixture
def mock_detection_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BBOX_QUALITY_THRESHOLD", "0.5")
    monkeypatch.setenv("SPECIES_CONF_THRESHOLD", "0.7")

    manager = DetectionManager()

    # Mock services
    manager.crop_service = MagicMock()
    manager.classification_service = MagicMock()
    manager.decision_policy_service = DecisionPolicyService()

    return manager


def test_deep_scan_uses_same_policy_engine(mock_detection_manager):
    # Setup dummy data for high species conf
    raw_detection = {
        "x1": 500,
        "y1": 300,
        "x2": 700,
        "y2": 500,
        "confidence": 0.85,
        "class_name": "bird",
        "method": "test",
    }
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    mock_detection_manager.crop_service.create_classification_crop.return_value = (
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    mock_detection_manager.classification_service.classify.return_value = (
        ClassificationResult(
            class_name="parus_major",
            confidence=0.9,
            model_id="test_model",
            top_k_confidences=[0.9, 0.03, 0.02, 0.01, 0.01],
        )
    )

    # Run payload builder
    payload, model_id = _build_detection_payload(
        mock_detection_manager, frame, raw_detection
    )

    assert payload.decision_state == DecisionState.CONFIRMED
    assert payload.decision_reasons == "[]"
    assert "decision_policy" in payload.policy_version


def test_deep_scan_unknown_path_persists_reason_codes(mock_detection_manager):
    raw_detection = {
        "x1": 500,
        "y1": 300,
        "x2": 700,
        "y2": 500,
        "confidence": 0.85,
        "class_name": "bird",
        "method": "test",
    }
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    mock_detection_manager.crop_service.create_classification_crop.return_value = (
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    mock_detection_manager.classification_service.classify.return_value = (
        ClassificationResult(
            class_name="parus_major",
            confidence=0.2,
            model_id="test_model",
            top_k_confidences=[0.2, 0.18, 0.15, 0.12, 0.10],
        )
    )

    # Run payload builder
    payload, model_id = _build_detection_payload(
        mock_detection_manager, frame, raw_detection
    )

    assert payload.decision_state == DecisionState.UNKNOWN
    assert "LOW_SPECIES_CONF" in payload.decision_reasons
    assert "HIGH_UNKNOWN_SCORE" in payload.decision_reasons
