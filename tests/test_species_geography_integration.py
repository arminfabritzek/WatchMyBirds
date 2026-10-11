from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import piexif
import pytest

from core.species_geography_runtime import (
    GeographicWeighting,
    ensure_geo_model_files,
    geographic_exif_context,
)
from detectors.interfaces.classification import ClassificationResult
from detectors.services.classification_service import ClassificationService
from utils.db.connection import _init_schema


@pytest.fixture
def raw_prediction() -> ClassificationResult:
    return ClassificationResult(
        "A",
        0.7,
        "image-model",
        ["A", "B", "C", "D", "E"],
        [0.7, 0.2, 0.05, 0.03, 0.02],
        raw_species_name="A",
    )


def configured_service(raw: ClassificationResult) -> ClassificationService:
    backend = Mock()
    backend.predict_rgb.return_value = raw
    service = ClassificationService(
        backend=backend,
        config={
            "LOCATION_DATA_CONFIGURED": True,
            "LOCATION_DATA": {"latitude": 1, "longitude": 2},
        },
    )
    service._geography._model = Mock()
    service._geography._model.predict.return_value = (
        {"A": 0.001, "B": 0.9, "C": 0.1, "D": 0.1, "E": 0.1},
        {"model_id": "geo-model", "latitude": 1, "longitude": 2, "week": 1},
    )
    return service


def test_geography_selects_real_candidate_without_boosting_confidence(
    raw_prediction: ClassificationResult,
) -> None:
    service = configured_service(raw_prediction)
    result = service.classify(
        np.zeros((8, 8, 3), np.uint8), observed_at=datetime(2026, 1, 1)
    )
    assert result.class_name == "B"
    assert result.confidence == 0.2
    assert result.raw_species_name == "A"
    assert result.model_id == "image-model"
    assert result.top_k_classes == raw_prediction.top_k_classes
    assert result.top_k_confidences == raw_prediction.top_k_confidences
    assert result.storage_alternatives == [
        ("A", 0.7),
        ("C", 0.05),
        ("D", 0.03),
        ("E", 0.02),
    ]
    evidence = json.loads(result.geography_json)
    assert evidence["raw_candidates"] == [
        [n, p]
        for n, p in zip(
            raw_prediction.top_k_classes, raw_prediction.top_k_confidences, strict=True
        )
    ]
    assert evidence["raw_confidence"] == 0.7
    assert "latitude" not in evidence and "longitude" not in evidence
    assert raw_prediction.class_name == "A" and raw_prediction.geography_json is None


def test_incomplete_taxonomy_preserves_the_complete_image_prediction(
    raw_prediction: ClassificationResult,
) -> None:
    service = configured_service(raw_prediction)
    service._geography._model.predict.return_value = ({"A": 0.001, "B": 0.9}, {})
    result = service.apply_geography(raw_prediction, datetime(2026, 1, 1))
    assert result.class_name == raw_prediction.class_name
    assert result.confidence == raw_prediction.confidence
    assert result.top_k_classes == raw_prediction.top_k_classes
    assert result.top_k_confidences == raw_prediction.top_k_confidences
    evidence = json.loads(result.geography_json)
    assert evidence["weighting_applied"] is False
    assert evidence["unmapped_species"] == ["C", "D", "E"]


def test_different_locations_and_seasons_use_the_same_policy(
    raw_prediction: ClassificationResult,
) -> None:
    service = configured_service(raw_prediction)

    def occurrence(lat: float, lon: float, stamp: datetime) -> tuple[dict, dict]:
        return (
            {
                "A": 0.001 if lat < 0 or stamp.month == 12 else 0.9,
                "B": 0.9,
                "C": 0.1,
                "D": 0.1,
                "E": 0.1,
            },
            {},
        )

    service._geography._model.predict.side_effect = occurrence
    assert (
        service.apply_geography(
            raw_prediction,
            datetime(2026, 6, 1),
            location={"latitude": 20, "longitude": 10},
        ).class_name
        == "A"
    )
    assert (
        service.apply_geography(
            raw_prediction,
            datetime(2026, 6, 1),
            location={"latitude": -20, "longitude": 10},
        ).class_name
        == "B"
    )
    assert (
        service.apply_geography(
            raw_prediction,
            datetime(2026, 12, 1),
            location={"latitude": 20, "longitude": 10},
        ).class_name
        == "B"
    )


@pytest.mark.parametrize(
    "level,name",
    [
        ("genus", "A_sp."),
        ("reject", ""),
        ("species_review", "A"),
        ("species", "background"),
        ("species", "non_bird"),
    ],
)
def test_geography_preserves_non_species_and_absence_decisions(
    raw_prediction: ClassificationResult,
    level: str,
    name: str,
) -> None:
    service = configured_service(raw_prediction)
    raw = replace(raw_prediction, decision_level=level, class_name=name)
    assert service.apply_geography(raw, datetime(2026, 1, 1)) is raw
    service._geography._model.predict.assert_not_called()


@pytest.mark.parametrize(
    "updates,stamp",
    [
        ({"SPECIES_GEOGRAPHY_ENABLED": False}, datetime(2026, 1, 1)),
        ({"LOCATION_DATA_CONFIGURED": False}, datetime(2026, 1, 1)),
        ({"LOCATION_DATA": {"latitude": 91, "longitude": 0}}, datetime(2026, 1, 1)),
        (
            {"LOCATION_DATA": {"latitude": float("nan"), "longitude": 0}},
            datetime(2026, 1, 1),
        ),
        ({"LOCATION_DATA": None}, datetime(2026, 1, 1)),
        ({}, None),
    ],
)
def test_missing_or_invalid_context_preserves_image_prediction(
    raw_prediction: ClassificationResult,
    updates: dict,
    stamp: datetime | None,
) -> None:
    service = configured_service(raw_prediction)
    service._geography._config.update(updates)
    assert service.apply_geography(raw_prediction, stamp) is raw_prediction
    service._geography._model.predict.assert_not_called()


def test_geo_failure_preserves_image_and_backs_off(
    raw_prediction: ClassificationResult,
) -> None:
    service = configured_service(raw_prediction)
    service._geography._model.predict.side_effect = ValueError("invalid output")
    crop = np.zeros((8, 8, 3), np.uint8)
    for _ in range(2):
        assert (
            service.classify(crop, observed_at=datetime(2026, 1, 1)).class_name == "A"
        )
    assert service._geography._model.predict.call_count == 1
    service._geography.assess = Mock(side_effect=RuntimeError("unexpected"))
    assert service.classify(crop, observed_at=datetime(2026, 1, 1)).class_name == "A"


def test_malformed_topk_does_not_lose_predictions(
    raw_prediction: ClassificationResult,
) -> None:
    service = configured_service(raw_prediction)
    for raw in (
        replace(raw_prediction, top_k_confidences=[0.7]),
        replace(raw_prediction, top_k_classes=["A"] * 5),
    ):
        assert service.apply_geography(raw, datetime(2026, 1, 1)) is raw
    service._geography._model.predict.assert_not_called()


def test_default_location_never_triggers_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.species_geography_runtime as runtime

    download = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(runtime, "ensure_geo_model_files", download)
    geo = GeographicWeighting(
        {"LOCATION_DATA": {"latitude": 52.516, "longitude": 13.377}}
    )
    assert geo.assess([("A", 0.8)], datetime(2026, 1, 1)) is None
    download.assert_not_called()


def test_public_artifacts_are_pinned_atomic_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.species_geography_runtime as runtime

    bodies = {
        runtime.GEO_MODEL_FILENAME: b"model",
        runtime.GEO_LABELS_FILENAME: b"labels",
    }
    monkeypatch.setattr(
        runtime, "GEO_MODEL_SHA256", hashlib.sha256(b"model").hexdigest()
    )
    monkeypatch.setattr(
        runtime, "GEO_LABELS_SHA256", hashlib.sha256(b"labels").hexdigest()
    )

    def get(url: str, **kwargs: object) -> Mock:
        name = (
            runtime.GEO_MODEL_FILENAME
            if url.endswith(".onnx")
            else runtime.GEO_LABELS_FILENAME
        )
        assert "%2B" in url
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [bodies[name]]
        return response

    request = Mock(side_effect=get)
    monkeypatch.setattr(runtime.requests, "get", request)
    paths = ensure_geo_model_files(tmp_path)
    assert [p.read_bytes() for p in paths] == [b"model", b"labels"]
    assert ensure_geo_model_files(tmp_path) == paths and request.call_count == 2
    paths[0].write_bytes(b"corrupted")
    bodies[runtime.GEO_MODEL_FILENAME] = b"wrong download"
    with pytest.raises(ValueError, match="checksum"):
        ensure_geo_model_files(tmp_path)
    assert paths[0].read_bytes() == b"corrupted"
    assert list(tmp_path.glob("*.download")) == []


def exif_sample() -> bytes:
    return piexif.dump(
        {
            "Exif": {piexif.ExifIFD.DateTimeOriginal: "2023:12:31 23:59:00"},
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: "S",
                piexif.GPSIFD.GPSLatitude: [(33, 1), (30, 1), (0, 1)],
                piexif.GPSIFD.GPSLongitudeRef: "W",
                piexif.GPSIFD.GPSLongitude: [(70, 1), (15, 1), (0, 1)],
            },
        }
    )


def test_import_exif_requires_actual_date_and_coordinates() -> None:
    assert geographic_exif_context(exif_sample()) == (
        datetime(2023, 12, 31, 23, 59),
        {"latitude": -33.5, "longitude": -70.25},
    )
    for value in (
        None,
        b"broken",
        piexif.dump({"Exif": {piexif.ExifIFD.DateTimeOriginal: "2023:12:31 23:59:00"}}),
    ):
        assert geographic_exif_context(value) == (None, None)


def test_deep_scan_uses_capture_origin_and_never_station_location_for_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.services import analysis_service

    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    conn.execute(
        "INSERT INTO sources(source_id,name,type) VALUES (99,'Imported','folder_upload')"
    )
    conn.execute(
        "INSERT INTO images(filename,timestamp,source_id) VALUES ('20200101_123000.jpg','20200101_123000',99)"
    )

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(analysis_service.db_service, "closing_connection", connection)
    monkeypatch.setattr(analysis_service, "config", {"LOCATION_DATA_CONFIGURED": True})
    path = tmp_path / "image.jpg"
    import cv2

    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    assert analysis_service._image_geography("20200101_123000.jpg", str(path)) == (
        None,
        None,
    )
    piexif.insert(exif_sample(), str(path))
    assert analysis_service._image_geography(
        "20200101_123000.jpg", str(path)
    ) == geographic_exif_context(exif_sample())
    conn.execute(
        "UPDATE images SET source_id=(SELECT source_id FROM sources WHERE type='ipcam' LIMIT 1)"
    )
    monkeypatch.setitem(analysis_service.config, "LOCATION_DATA_CONFIGURED", False)
    assert analysis_service._image_geography("20200101_123000.jpg", str(path)) == (
        None,
        None,
    )
    conn.close()


def test_explicit_location_and_disable_setting_survive_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import config

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("LOCATION_DATA", raising=False)
    monkeypatch.delenv("SPECIES_GEOGRAPHY_ENABLED", raising=False)
    monkeypatch.setattr(config, "_CONFIG", None)
    assert config.get_config()["LOCATION_DATA_CONFIGURED"] is False
    config.update_runtime_settings(
        {"LOCATION_DATA": "0, 0", "SPECIES_GEOGRAPHY_ENABLED": False}
    )
    monkeypatch.setattr(config, "_CONFIG", None)
    result = config.get_config()
    assert result["LOCATION_DATA_CONFIGURED"] is True
    assert result["LOCATION_DATA"] == {"latitude": 0.0, "longitude": 0.0}
    assert result["SPECIES_GEOGRAPHY_ENABLED"] is False
    assert "LOCATION_DATA_CONFIGURED" not in (tmp_path / "settings.yaml").read_text()


@pytest.mark.parametrize("with_exif", [False, True])
def test_ingest_persists_top5_and_uses_photo_geography_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_prediction: ClassificationResult,
    with_exif: bool,
) -> None:
    import cv2

    from utils import ingest

    output = tmp_path / "output"
    monkeypatch.setattr(
        ingest, "config", {"OUTPUT_DIR": str(output), "SAVE_THRESHOLD": 0.1}
    )
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    source = conn.execute("SELECT source_id FROM sources LIMIT 1").fetchone()[0]
    image = tmp_path / "photo.jpg"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    if with_exif:
        piexif.insert(exif_sample(), str(image))
    detector = Mock(model_id="detector")
    detector.detect_objects.return_value = (
        True,
        None,
        [
            {
                "x1": 10,
                "y1": 10,
                "x2": 60,
                "y2": 60,
                "confidence": 0.9,
                "class_name": "bird",
            },
        ],
    )
    classifier = configured_service(raw_prediction)
    classifier.backend.get_model_id.return_value = "image-model"
    assert (
        ingest.ingest_file(conn, str(image), source, detector, classifier) == "ingested"
    )
    rows = conn.execute(
        "SELECT cls_class_name,cls_confidence FROM classifications ORDER BY rank"
    ).fetchall()
    assert len(rows) == 5 and len({r[0] for r in rows}) == 5
    assert rows[0][0] == ("B" if with_exif else "A")
    assert {r[0]: r[1] for r in rows} == dict(
        zip(raw_prediction.top_k_classes, raw_prediction.top_k_confidences, strict=True)
    )
    if with_exif:
        assert classifier._geography._model.predict.call_args.args == (
            -33.5,
            -70.25,
            datetime(2023, 12, 31, 23, 59),
        )
    else:
        classifier._geography._model.predict.assert_not_called()
    detection = conn.execute(
        "SELECT raw_species_name,geography_json FROM detections"
    ).fetchone()
    assert detection["raw_species_name"] == "A"
    assert bool(detection["geography_json"]) is with_exif
    assert (
        ingest.ingest_file(conn, str(image), source, detector, classifier) == "skipped"
    )
    conn.close()
