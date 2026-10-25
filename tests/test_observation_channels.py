"""Capture admission agrees across persisted history and outbound channels."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from core.automatic_station_report import build_report
from detectors.interfaces.classification import ClassificationResult
from tests.test_notification_gate_smoothing import (
    _base_manager as _base_manager,
)
from tests.test_notification_gate_smoothing import (
    _run_one_job,
)
from utils.db import insert_classification, insert_detection, insert_image
from utils.db.connection import _init_schema
from utils.db.detections import (
    fetch_count_last_24h,
    fetch_day_count,
    fetch_detections_for_gallery,
    fetch_detections_last_24h,
    fetch_species_covers,
    fetch_species_page,
    fetch_species_story_board_candidates,
)


@pytest.mark.parametrize(
    "species,confidence,level,occurrence,admitted,reason",
    [
        ("Parus_major", 0.95, "species", None, True, None),
        ("Parus_major", 0.60, "species", None, False, "LOW_DISPLAY_SCORE"),
        ("Parus_major", 0.95, "species", 0, False, "ZERO_GEOGRAPHIC_OCCURRENCE"),
        ("Parus_major", 0.95, "species", 0.0001, True, None),
        ("Parus_sp.", 0.95, "genus", None, False, "NO_NAMED_BIRD_SPECIES"),
        ("squirrel", 0.95, "species", None, False, "NO_NAMED_BIRD_SPECIES"),
        ("bird", 0.95, "species", None, False, "NO_NAMED_BIRD_SPECIES"),
        ("", 0.95, "species", None, False, "MISSING_SPECIES"),
        ("Parus_major", 0.95, None, None, False, "NO_SPECIES_LEVEL_DECISION"),
    ],
)
def test_capture_species_agrees_in_every_channel(
    _base_manager,
    monkeypatch: pytest.MonkeyPatch,
    species: str,
    confidence: float,
    level: str | None,
    occurrence: float | None,
    admitted: bool,
    reason: str | None,
) -> None:
    manager = _base_manager
    # Isolate the display floor from the independent species-confidence floor.
    manager.decision_policy_service.species_threshold = 0.5
    manager.notification_service.should_send.return_value = True
    geography = json.dumps(
        {"ranking": [{"species": species, "occurrence_score": occurrence}]}
    )
    manager.classification_service.classify.return_value = ClassificationResult(
        class_name=species,
        confidence=confidence,
        decision_level=level,
        geography_json=geography,
        top_k_confidences=[confidence, 0.01],
    )
    bus = MagicMock()
    monkeypatch.setattr("utils.live_event_bus.get_bus", lambda: bus)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)

    def persist(**kwargs: object) -> object:
        detection = kwargs["detection"]
        filename = "20260101_120000.jpg"
        insert_image(conn, {"filename": filename, "timestamp": "20260101_120000"})
        did = insert_detection(
            conn,
            {
                "image_filename": filename,
                "od_class_name": detection.class_name,
                "od_confidence": detection.confidence,
                "score": detection.score,
                "decision_state": detection.decision_state,
                "decision_level": detection.decision_level,
                "decision_reasons": detection.decision_reasons,
                "bbox_quality": detection.bbox_quality,
                "unknown_score": detection.unknown_score,
                "bbox_x": 0.2,
                "bbox_y": 0.2,
                "bbox_w": 0.2,
                "bbox_h": 0.2,
            },
        )
        insert_classification(
            conn,
            {
                "detection_id": did,
                "cls_class_name": species,
                "cls_confidence": confidence,
                "rank": 1,
            },
        )
        conn.execute("UPDATE detections SET quality_gallery_ok=0")
        return manager.persistence_service.save_detection.return_value

    manager.persistence_service.save_detection.side_effect = persist
    try:
        _run_one_job(manager)
        assert conn.execute("SELECT decision_level FROM detections").fetchone()[0]
        expected = [species] if admitted else []
        for rows in (
            fetch_detections_for_gallery(conn),
            fetch_species_covers(conn),
            fetch_species_page(conn, species or "Unknown_species")[0],
            fetch_detections_last_24h(conn, "20260101"),
            fetch_species_story_board_candidates(conn),
            build_report(conn, {})["species"],
        ):
            assert [row["species_key"] for row in rows] == expected
        from utils.daily_report import _fetch_species_best_photos

        monkeypatch.setattr("utils.daily_report.os.path.isfile", lambda path: True)
        assert [
            row["species"] for row in _fetch_species_best_photos(conn, "2026-01-01")
        ] == expected
        assert fetch_day_count(conn, "2026-01-01") == int(admitted)
        assert fetch_count_last_24h(conn, "20260101") == int(admitted)
        event = bus.publish.call_args.args[0]
        assert event["species_latin"] == (species if admitted else "")
        assert bool(event["species_common"]) == admitted
        if admitted:
            assert (
                manager.mqtt_detection_service.publish_detection.call_args.kwargs[
                    "latin_name"
                ]
                == species
            )
            assert (
                manager.notification_service.create_species_info.call_args.kwargs[
                    "latin_name"
                ]
                == species
            )
            manager.notification_service.queue_detection.assert_called_once()
        else:
            manager.mqtt_detection_service.publish_detection.assert_not_called()
            manager.notification_service.queue_detection.assert_not_called()
            persisted = conn.execute(
                "SELECT decision_reasons FROM detections"
            ).fetchone()[0]
            assert reason in json.loads(persisted)
    finally:
        conn.close()
