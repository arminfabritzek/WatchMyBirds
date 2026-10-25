from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from core.species_geography import rank_geographically
from core.species_geography_runtime import LocalGeoModel, capture_datetime
from scripts.audit_species_geography import audit
from utils.db.connection import _init_schema


def test_geography_changes_only_ranking_and_preserves_original_scores() -> None:
    candidates = [("Nycticorax_nycticorax", 0.8), ("Sitta_europaea", 0.15)]
    result = rank_geographically(
        candidates, {"Nycticorax_nycticorax": 0.0, "Sitta_europaea": 0.9}
    )
    assert result.raw_species == "Nycticorax_nycticorax"
    assert result.suggested_species == "Sitta_europaea"
    assert result.ranking[1].weighted_score > 0
    assert {row.species: row.image_score for row in result.ranking} == dict(candidates)
    assert {row.species for row in result.ranking} == set(dict(candidates))


def test_missing_geography_or_taxonomy_preserves_image_scores() -> None:
    candidates = [("Sitta_europaea", 0.6), ("Parus_major", 0.2)]
    for occurrence in (None, {}, {"unrelated_species": 0.1}):
        result = rank_geographically(candidates, occurrence)
        assert result.suggested_species == candidates[0][0]
        assert [row.weighted_score for row in result.ranking] == [0.6, 0.2]


def test_tied_scores_preserve_original_order() -> None:
    result = rank_geographically([("A", 0.4), ("B", 0.4)], {"A": 0.1, "B": 0.1})
    assert result.suggested_species == "A"


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [("A", 0)],
        [("A", -1)],
        [("A", float("nan"))],
        [("A", 0.8), ("B", 0.8)],
        [("A", 0.5), ("A", 0.1)],
    ],
)
def test_invalid_image_candidates_are_rejected(
    candidates: list[tuple[str, float]],
) -> None:
    with pytest.raises(ValueError):
        rank_geographically(candidates)


@pytest.mark.parametrize("prior", [-0.1, 1.1, float("nan")])
def test_invalid_occurrence_scores_are_rejected(prior: float) -> None:
    with pytest.raises(ValueError):
        rank_geographically([("A", 0.8)], {"A": prior})


def test_geo_model_checksums_prevent_untrusted_model_execution(tmp_path: Path) -> None:
    path = tmp_path / "bad.onnx"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        LocalGeoModel(path, path)


def test_capture_time_uses_explicit_local_calendar() -> None:
    assert capture_datetime(
        "20261010_120000_000001", "Europe/Berlin"
    ).utcoffset() == timedelta(hours=2)
    assert capture_datetime("2026-09-30T23:30:00+00:00", "Europe/Berlin").month == 10


def test_geo_model_calendar_bins_and_cache_do_not_leak_mutations() -> None:
    geo = LocalGeoModel.__new__(LocalGeoModel)
    geo._session = Mock()
    geo._session.get_inputs.return_value = [Mock(name="input")]
    geo._session.run.return_value = [np.array([[0.7, 0.2]])]
    geo._labels = ["A", "B"]
    geo._cache = {}
    geo._lock = threading.Lock()
    scores, provenance = geo.predict(-33.0, 151.0, datetime(2026, 12, 31, tzinfo=UTC))
    assert provenance["week"] == 48
    assert list(geo._session.run.call_args.args[1].values())[0].tolist() == [
        [-33.0, 151.0, 48.0]
    ]
    scores["A"] = 0
    assert geo.predict(-33.0, 151.0, datetime(2026, 12, 31, tzinfo=UTC))[0]["A"] == 0.7
    assert geo._session.run.call_count == 1
    with pytest.raises(ValueError, match="coordinates"):
        geo.predict(91, 151, datetime.now(UTC))


def test_audit_is_read_only_and_human_answers_do_not_change_ranking(
    tmp_path: Path,
) -> None:
    database = tmp_path / "images.db"
    with sqlite3.connect(database) as conn:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO images(filename,timestamp) VALUES ('bird.jpg','20261010_120000_000001')"
        )
        conn.execute(
            "INSERT INTO detections(detection_id,image_filename,status,bbox_x,bbox_y,bbox_w,bbox_h) "
            "VALUES (1,'bird.jpg','active',0.1,0.1,0.5,0.5)"
        )
        conn.executemany(
            "INSERT INTO classifications(detection_id,cls_class_name,cls_confidence,rank) VALUES (1,?,?,?)",
            [("A", 0.8, 1), ("B", 0.15, 2)],
        )
    geo = Mock()
    geo.predict.return_value = ({"A": 0, "B": 0.9}, {"model_id": "test"})
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    output = tmp_path / "audit.db"
    report = audit(
        database,
        output,
        geo=geo,
        latitude=0,
        longitude=0,
        timezone="UTC",
        focus_species="A",
    )
    assert report["counts"]["assessed"] == 1
    assert report["focused_cases"][0]["suggested_species"] == "B"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    with sqlite3.connect(output) as conn:
        details = json.loads(
            conn.execute("SELECT result_json FROM assessments").fetchone()[0]
        )
    assert details["raw_candidates"] == [["A", 0.8], ["B", 0.15]]
    with pytest.raises(ValueError, match="new file"):
        audit(database, output, geo=geo, latitude=0, longitude=0, timezone="UTC")
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename="bird.jpg",
                detection_id=1,
                species_identity="corrected",
                species_key="A",
            ),
            LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui"),
        )
    second = tmp_path / "audit-labeled.db"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    report = audit(
        database,
        second,
        geo=geo,
        latitude=0,
        longitude=0,
        timezone="UTC",
        focus_species="A",
    )
    assert report["focused_cases"][0]["human_species"] == "A"
    assert report["focused_cases"][0]["suggested_species"] == "B"
    assert report["counts"]["correct_to_wrong"] == 1
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
