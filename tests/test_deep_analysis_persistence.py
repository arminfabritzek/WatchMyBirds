from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult
from detectors.services.capability_registry import build_default_registry
from detectors.services.classification_service import ClassificationService
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.persistence_service import PersistenceService
from detectors.services.temporal_decision_service import TemporalDecisionService
from web.services.analysis_service import _build_detection_payload

HERON = "Nycticorax_nycticorax"
NUTHATCH = "Sitta_europaea"


@pytest.mark.parametrize("geographic", [False, True])
def test_deep_analysis_persists_raw_provenance_and_exact_top5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, geographic: bool
) -> None:
    import config

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "_CONFIG", None)
    service = PersistenceService()
    service._db_conn.execute(
        "INSERT INTO images(filename,timestamp) VALUES ('bird.jpg','20261010_120000_000001')"
    )
    service._db_conn.commit()
    monkeypatch.setattr(service, "generate_thumbnail", lambda *args, **kwargs: True)
    manager = DetectionManager.__new__(DetectionManager)
    manager.config = {}
    manager.classifier_model_id = "model-1"
    manager.SAVE_RESOLUTION_CROP = 224
    manager.crop_service = Mock()
    manager.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )
    candidates = [
        (HERON, 0.9),
        (NUTHATCH, 0.04),
        ("Parus_major", 0.03),
        ("Cyanistes_caeruleus", 0.02),
        ("Dendrocopos_major", 0.01),
    ]
    manager.classification_service = Mock()
    classification = ClassificationResult(
        HERON,
        0.9,
        "model-1",
        [s for s, _ in candidates],
        [p for _, p in candidates],
        raw_species_name=HERON,
        decision_level="species",
    )
    if geographic:
        classifier = ClassificationService(
            config={
                "LOCATION_DATA_CONFIGURED": True,
                "LOCATION_DATA": {"latitude": 0, "longitude": 0},
            }
        )
        classifier._geography._model = Mock()
        classifier._geography._model.predict.return_value = (
            {HERON: 0, NUTHATCH: 0.9},
            {"model_id": "geo-test"},
        )
        classification = classifier.apply_geography(
            classification, datetime(2026, 1, 1)
        )
    manager.classification_service.classify.return_value = classification
    manager.decision_policy_service = DecisionPolicyService()
    manager.temporal_decision_service = TemporalDecisionService()
    manager.capability_registry = build_default_registry()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    payload, model_id = _build_detection_payload(
        manager,
        frame,
        {"x1": 10, "y1": 10, "x2": 150, "y2": 150, "confidence": 0.95},
    )
    result = service.save_detection(
        "bird.jpg", payload, frame, "detector-1", model_id, 0
    )
    assert result.success
    row = service._db_conn.execute(
        "SELECT raw_species_name,decision_level,geography_json,species_source "
        "FROM detections WHERE detection_id=?",
        (result.detection_id,),
    ).fetchone()
    assert row["raw_species_name"] == HERON
    assert row["decision_level"] == "species"
    rows = service._db_conn.execute(
        "SELECT cls_class_name,cls_confidence FROM classifications WHERE detection_id=? ORDER BY rank",
        (result.detection_id,),
    ).fetchall()
    expected = (
        [candidates[1], candidates[0], *candidates[2:]] if geographic else candidates
    )
    assert [tuple(row) for row in rows] == expected
    assert row["species_source"] == ("geographic" if geographic else "model_top1")
    if geographic:
        assert json.loads(row["geography_json"])["raw_candidates"] == [
            list(c) for c in candidates
        ]
    service.close()
