from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.classifier_backends.factory import (
    CLASSIFIER_BACKEND_INAT,
    CLASSIFIER_BACKEND_WMB,
    build_classifier_backend,
)
from detectors.classifier_backends.inat_tflite import (
    INaturalistTFLiteBirdClassifierBackend,
)
from detectors.classifier_backends.wmb_onnx import WMBOnnxClassifierBackend
from detectors.services.classification_service import ClassificationService


def test_factory_keeps_existing_wmb_classifier_instance() -> None:
    classifier = MagicMock()
    backend = build_classifier_backend(
        {"CLASSIFIER_BACKEND": CLASSIFIER_BACKEND_WMB},
        wmb_classifier=classifier,
    )

    assert isinstance(backend, WMBOnnxClassifierBackend)
    assert backend.classifier is classifier


def test_wmb_backend_preserves_genus_decision() -> None:
    classifier = MagicMock()
    classifier.predict_from_image.return_value = (
        np.array([1, 0]),
        np.array([0.45, 0.35]),
        "Sylvia_atricapilla",
        0.45,
    )
    classifier.classes = ["Sylvia_borin", "Sylvia_atricapilla"]
    classifier.last_decision = {
        "level": "genus",
        "label": "Sylvia_sp.",
        "prob": 0.80,
        "raw_species": "Sylvia_atricapilla",
    }
    classifier.model_id = "wmb-test"
    backend = WMBOnnxClassifierBackend(classifier)

    prediction = backend.predict_rgb(np.zeros((8, 8, 3), dtype=np.uint8))

    assert prediction.class_name == "Sylvia_sp."
    assert prediction.confidence == 0.80
    assert prediction.decision_level == "genus"
    assert prediction.raw_species_name == "Sylvia_atricapilla"
    assert prediction.top_k_classes == ["Sylvia_atricapilla", "Sylvia_borin"]


def test_factory_builds_inaturalist_backend(tmp_path) -> None:
    backend = build_classifier_backend(
        {
            "CLASSIFIER_BACKEND": CLASSIFIER_BACKEND_INAT,
            "MODEL_BASE_PATH": str(tmp_path),
        }
    )

    assert isinstance(backend, INaturalistTFLiteBirdClassifierBackend)


def test_factory_defaults_to_inaturalist_backend(tmp_path) -> None:
    backend = build_classifier_backend({"MODEL_BASE_PATH": str(tmp_path)})

    assert isinstance(backend, INaturalistTFLiteBirdClassifierBackend)


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported classifier backend"):
        build_classifier_backend({"CLASSIFIER_BACKEND": "mystery"})


def test_classification_service_maps_backend_prediction() -> None:
    backend = MagicMock()
    backend.predict_rgb.return_value.class_name = "Cyanocitta_cristata"
    backend.predict_rgb.return_value.common_name = "Blue Jay"
    backend.predict_rgb.return_value.confidence = 0.75
    backend.predict_rgb.return_value.model_id = "inat-test"
    backend.predict_rgb.return_value.top_k_classes = ["Cyanocitta_cristata"]
    backend.predict_rgb.return_value.top_k_confidences = [0.75]
    backend.predict_rgb.return_value.decision_level = "species"
    backend.predict_rgb.return_value.raw_species_name = "Cyanocitta_cristata"
    backend.get_model_id.return_value = "inat-test"
    service = ClassificationService(backend=backend)

    result = service.classify(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.class_name == "Cyanocitta_cristata"
    assert result.model_id == "inat-test"
    assert result.top_k_confidences == [0.75]


def test_classifier_registry_reports_active_inat_backend(tmp_path, monkeypatch) -> None:
    from web.services import model_registry_service

    inat_dir = tmp_path / "classifier" / "inat_bird_mobilenet_v2"
    inat_dir.mkdir(parents=True)
    (inat_dir / "mobilenet_v2_1.0_224_inat_bird_quant.tflite").write_bytes(b"x")
    (inat_dir / "inat_bird_labels.txt").write_text("background\n", encoding="utf-8")
    monkeypatch.setattr(
        model_registry_service,
        "get_config",
        lambda: {
            "CLASSIFIER_BACKEND": CLASSIFIER_BACKEND_INAT,
            "MODEL_BASE_PATH": str(tmp_path),
        },
    )

    payload = model_registry_service.build_classifier_registry_payload(None)

    assert payload["backend"]["active"] == CLASSIFIER_BACKEND_INAT
    inat = next(
        option
        for option in payload["backend"]["options"]
        if option["id"] == CLASSIFIER_BACKEND_INAT
    )
    assert inat["installed"] is True
    assert inat["num_bird_taxa"] == 964
