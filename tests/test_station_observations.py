from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from core.human_label_core import (
    HumanAnswer,
    LabelProvenance,
    record_human_answer,
    retract_species_identity,
)
from utils.db.connection import _init_schema
from utils.db.detections import (
    fetch_detections_for_gallery,
    fetch_observation_page,
    fetch_species_covers,
    fetch_species_page,
)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    _init_schema(connection)
    yield connection
    connection.close()


def seed(
    conn: sqlite3.Connection,
    identifier: int = 101,
    species: str | None = "Himantopus_leucocephalus",
    quality: int = 0,
    state: str = "confirmed",
    level: str | None = "species",
) -> str:
    filename = f"20261013_082211_{identifier}.jpg"
    conn.execute(
        "INSERT OR IGNORE INTO images(filename, timestamp) VALUES (?, ?)",
        (filename, "20261013_082211"),
    )
    conn.execute(
        """INSERT INTO detections(detection_id, image_filename, od_class_name, score,
        bbox_x, bbox_y, bbox_w, bbox_h, decision_state, decision_level, quality_gallery_ok, status)
        VALUES (?, ?, 'bird', 0.95, 0.1, 0.1, 0.3, 0.3, ?, ?, ?, 'active')""",
        (identifier, filename, state, level, quality),
    )
    from detectors.services.decision_policy_service import DecisionPolicyService

    reasons = DecisionPolicyService().admission_reasons(
        score=0.95, species_key=species, geography_json=None
    )
    if reasons and state == "confirmed":
        conn.execute(
            "UPDATE detections SET decision_state='uncertain', decision_reasons=? WHERE detection_id=?",
            (json.dumps(reasons), identifier),
        )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank) VALUES (?, ?, 0.95, 1)",
        (identifier, species),
    )
    return filename


def answer(
    conn: sqlite3.Connection,
    identifier: int,
    filename: str,
    state: str = "corrected",
    species: str | None = "Pica_pica",
) -> None:
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=filename,
            detection_id=identifier,
            species_identity=state,
            species_key=species,
        ),
        provenance(),
    )


def provenance() -> LabelProvenance:
    return LabelProvenance(
        "test", "test", "normal_correction", "watchmybirds_ui", "test"
    )


@pytest.mark.parametrize(
    ("identifier", "raw", "current"),
    [
        (101, "Himantopus_leucocephalus", "Pica_pica"),
        (102, "Picus_viridis", "Picus_viridis"),
        (3, "Poecile_carolinensis", "Poecile_palustris"),
    ],
)
def test_human_species_and_history_survive_photo_quality(
    conn: sqlite3.Connection, identifier: int, raw: str, current: str
) -> None:
    filename = seed(conn, identifier, raw)
    answer(conn, identifier, filename, species=current)
    assert [r["detection_id"] for r in fetch_detections_for_gallery(conn)] == [
        identifier
    ]
    for threshold in (0.0, 1.0):
        covers = fetch_species_covers(conn, min_score=threshold)
        assert [(r["species_key"], r["human_days"]) for r in covers] == [(current, 1)]
        rows, total, _ = fetch_species_page(conn, current, min_score=threshold)
        assert total == 1 and rows[0]["detection_id"] == identifier
        assert rows[0]["cls_class_name"] == raw
    conn.execute("UPDATE images SET original_present = 0")
    assert fetch_species_covers(conn)[0]["missing_originals"] == 1


def test_auto_admission_is_independent_of_photo_threshold(
    conn: sqlite3.Connection,
) -> None:
    seed(conn, species="Pica_pica", quality=1)
    assert fetch_species_covers(conn, min_score=1)[0]["automatic_count"] == 1
    conn.execute("UPDATE detections SET decision_level = 'species_review'")
    assert fetch_species_covers(conn) == []
    assert fetch_observation_page(conn, answered=False)[1] == 1
    conn.execute("UPDATE detections SET decision_level = 'species'")
    assert fetch_species_covers(conn)[0]["automatic_count"] == 1


@pytest.mark.parametrize("state", ["unknown", "wrong"])
def test_answered_unknown_or_refuted_proposal_stays_completed(
    conn: sqlite3.Connection, state: str
) -> None:
    filename = seed(conn)
    answer(conn, 101, filename, state=state, species=None)
    assert not any(
        r["species_key"] != "Unknown_species" for r in fetch_species_covers(conn)
    )
    from core.automatic_station_report import build_report

    assert fetch_detections_for_gallery(conn) == []
    assert build_report(conn, {})["summary"]["identified_objects"] == 0
    facts = list(
        conn.execute("SELECT fact_type, answer_value FROM current_human_label_facts")
    )
    assert (
        ("bird_presence", "present") in [tuple(r) for r in facts]
        if state == "unknown"
        else len(facts) == 1
    )
    assert fetch_observation_page(conn, answered=False)[1] == 0
    assert fetch_observation_page(conn, answered=True)[1] == 1
    assert fetch_species_page(conn, "Unknown_species")[1] == (
        1 if state == "unknown" else 0
    )
    conn.execute(
        "UPDATE detections SET decision_state = 'confirmed', decision_level = 'species'"
    )
    assert fetch_observation_page(conn, answered=False)[1] == 0
    assert fetch_species_page(conn, "Himantopus_leucocephalus")[1] == 0


def test_retraction_removes_human_claim_without_resurrecting_history(
    conn: sqlite3.Connection,
) -> None:
    filename = seed(conn, species="Parus_major", quality=1)
    answer(conn, 101, filename)
    retract_species_identity(
        conn, image_filename=filename, detection_id=101, provenance=provenance()
    )
    row = fetch_species_covers(conn)[0]
    assert row["species_key"] == "Parus_major" and row["human_count"] == 0


def test_period_keeps_archive_species_without_counting_them(
    conn: sqlite3.Connection,
) -> None:
    filename = seed(conn)
    answer(conn, 101, filename)
    row = fetch_species_covers(conn, selection={"start": "2026-10-14"})[0]
    assert (
        row["observation_count"] == 0 and row["archive_last_seen"] == "20261013_082211"
    )
    assert (
        fetch_species_page(conn, "Pica_pica", selection={"start": "2026-10-14"})[1] == 0
    )


def test_legacy_species_is_not_human_evidence(conn: sqlite3.Connection) -> None:
    seed(conn)
    conn.execute(
        "UPDATE detections SET manual_species_override='Pica_pica', species_source='manual'"
    )
    assert fetch_species_covers(conn)[0]["species_key"] == "Himantopus_leucocephalus"


def test_old_human_species_without_presence_is_usable_without_backfill(
    conn: sqlite3.Connection,
) -> None:
    from core.human_label_core import append_fact, ensure_object_subject

    filename = seed(conn)
    subject = ensure_object_subject(conn, 101)
    append_fact(
        conn,
        subject_id=subject,
        fact_type="species_identity",
        answer_value="corrected",
        species_key="Pica_pica",
        provenance=provenance(),
    )
    assert fetch_species_covers(conn)[0]["human_count"] == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE fact_type='bird_presence'"
        ).fetchone()[0]
        == 0
    )
    assert fetch_species_page(conn, "Pica_pica")[0][0]["image_filename"] == filename


def test_object_absence_overrules_species_without_reopening_review(
    conn: sqlite3.Connection,
) -> None:
    from core.human_label_core import append_fact, ensure_object_subject

    filename = seed(conn)
    answer(conn, 101, filename)
    append_fact(
        conn,
        subject_id=ensure_object_subject(conn, 101),
        fact_type="bird_presence",
        answer_value="absent",
        provenance=provenance(),
    )
    assert fetch_species_covers(conn) == []
    assert fetch_observation_page(conn, answered=False)[1] == 0


@pytest.mark.parametrize(
    "species", ["Passer_sp", "Passer_sp.", "Unknown_species", "bird"]
)
def test_genus_and_unknown_proposals_are_not_named_species(
    conn: sqlite3.Connection, species: str | None
) -> None:
    seed(conn, species=species)
    assert fetch_species_covers(conn) == []
    pending, total, _ = fetch_observation_page(conn, answered=False)
    assert total == 1 and pending[0]["review_reason"]


def test_object_absence_preserves_other_birds_in_same_image(
    conn: sqlite3.Connection,
) -> None:
    filename = seed(conn, identifier=1, species="Parus_major", quality=1)
    conn.execute(
        """INSERT INTO detections(detection_id, image_filename, od_class_name, score,
        decision_state, decision_level, status) VALUES (2, ?, 'bird', .9, 'confirmed', 'species', 'active')""",
        (filename,),
    )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, cls_confidence, rank) VALUES (2, 'Cyanistes_caeruleus', .9, 1)"
    )
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=filename, detection_id=1, object_bird_presence="absent"
        ),
        provenance(),
    )
    assert [r["species_key"] for r in fetch_species_covers(conn)] == [
        "Cyanistes_caeruleus"
    ]


def test_geographic_doubt_requires_review_until_human_identification(
    conn: sqlite3.Connection,
) -> None:
    from detectors.services.decision_policy_service import DecisionPolicyService

    filename = seed(conn, species="Pica_pica", quality=1)
    geography = json.dumps(
        {"ranking": [{"species": "Pica_pica", "occurrence_score": 0}]}
    )
    reasons = DecisionPolicyService().admission_reasons(
        score=0.95, species_key="Pica_pica", geography_json=geography
    )
    conn.execute(
        "UPDATE detections SET geography_json=?, decision_state='uncertain', decision_reasons=?",
        (geography, json.dumps(reasons)),
    )
    assert fetch_species_covers(conn) == []
    assert fetch_observation_page(conn, answered=False)[0][0]["review_reason"]
    assert fetch_observation_page(conn, answered=False)[1] == 1
    answer(conn, 101, filename, species="Pica_pica")
    assert fetch_species_covers(conn)[0]["human_count"] == 1
    assert fetch_observation_page(conn, answered=False)[1] == 0


def test_unknown_answer_can_later_be_identified_without_changing_siblings(
    conn: sqlite3.Connection,
) -> None:
    first = seed(conn, identifier=1, species="Parus_major", quality=1)
    second = seed(conn, identifier=2, species="Parus_major", state="uncertain")
    answer(conn, 1, first, state="unknown", species=None)
    assert fetch_species_page(conn, "Unknown_species")[1] == 1
    assert [
        r["detection_id"] for r in fetch_observation_page(conn, answered=False)[0]
    ] == [2]
    answer(conn, 1, first, species="Pica_pica")
    assert fetch_species_page(conn, "Unknown_species")[1] == 0
    assert fetch_species_page(conn, "Pica_pica")[1] == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM human_label_facts WHERE answer_value='unknown'"
        ).fetchone()[0]
        == 1
    )
    assert (
        conn.execute(
            "SELECT species_source FROM detections WHERE image_filename=?", (second,)
        ).fetchone()[0]
        is None
    )


def test_manual_bird_has_history_without_synthetic_model_scores(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    from core.human_label_core import BBox
    from core.manual_object_core import ManualObjectDraft, create_manual_object

    filename = seed(conn, identifier=1, state="uncertain")
    original = tmp_path / "original.jpg"
    original.write_bytes(b"local fixture")
    create_manual_object(
        conn,
        ManualObjectDraft(
            filename, BBox(0.5, 0.5, 0.2, 0.2), "Pica_pica", "manual-station-test"
        ),
        provenance(),
        original_path=original,
        locale="DE",
    )
    row = fetch_species_page(conn, "Pica_pica")[0][0]
    assert row["manual_object_id"] and row["score"] is None
    assert row["cls_class_name"] is None and row["origin"] == "human"
    assert fetch_species_covers(conn)[0]["human_days"] == 1


def test_current_canonical_species_overrules_stale_compatibility_source(
    conn: sqlite3.Connection,
) -> None:
    filename = seed(conn, quality=1)
    answer(conn, 101, filename, species="Pica_pica")
    conn.execute(
        "UPDATE detections SET species_source='manual_wrong', manual_species_override=NULL"
    )
    assert fetch_detections_for_gallery(conn)[0]["species_key"] == "Pica_pica"
    assert fetch_species_covers(conn)[0]["species_key"] == "Pica_pica"


def test_multiple_objects_do_not_duplicate_missing_original_count(
    conn: sqlite3.Connection,
) -> None:
    filename = seed(conn, identifier=1, species="Parus_major", quality=1)
    seed(conn, identifier=2, species="Parus_major", quality=1)
    conn.execute(
        "UPDATE detections SET image_filename=? WHERE detection_id=2", (filename,)
    )
    conn.execute("UPDATE images SET original_present=0")
    row = fetch_species_covers(conn)[0]
    assert row["observation_count"] == 2 and row["missing_originals"] == 1


@pytest.mark.parametrize(
    "column,value",
    [
        ("quality_gallery_ok", 0),
        ("bbox_quality", 0.1),
        ("od_confidence", 0),
        ("unknown_score", 0.95),
        ("score", 0.01),
    ],
)
def test_readers_keep_the_capture_decision(
    conn: sqlite3.Connection, column: str, value: float
) -> None:
    from config import get_config
    from core.automatic_station_report import build_report
    from utils.db.detections import fetch_detections_last_24h, fetch_gallery_days

    seed(conn, species="Parus_major")
    conn.execute(f"UPDATE detections SET {column} = ?", (value,))
    cfg = get_config()
    old = cfg["GALLERY_DISPLAY_THRESHOLD"]
    try:
        cfg["GALLERY_DISPLAY_THRESHOLD"] = 1.0
        assert fetch_detections_for_gallery(conn)[0]["species_key"] == "Parus_major"
        assert (
            fetch_detections_last_24h(conn, "20261001")[0]["species_key"]
            == "Parus_major"
        )
        assert fetch_species_covers(conn)[0]["automatic_count"] == 1
        assert fetch_gallery_days(conn)[1] == 1
        assert build_report(conn, {})["summary"]["identified_objects"] == 1
        assert fetch_observation_page(conn, answered=False)[1] == 0
    finally:
        cfg["GALLERY_DISPLAY_THRESHOLD"] = old


def test_unresolved_capture_has_one_gate_and_human_override(
    conn: sqlite3.Connection,
) -> None:
    from core.automatic_station_report import build_report
    from utils.db.detections import fetch_detections_last_24h, fetch_gallery_days

    filename = seed(conn, species="Parus_major", state="uncertain")
    assert not fetch_detections_for_gallery(conn)
    assert not fetch_detections_last_24h(conn, "20261001")
    assert not fetch_species_covers(conn)
    assert fetch_gallery_days(conn)[1] == 0
    assert build_report(conn, {})["summary"]["identified_objects"] == 0
    assert fetch_observation_page(conn, answered=False)[1] == 1
    answer(conn, 101, filename, species="Pica_pica")
    assert fetch_detections_for_gallery(conn)[0]["species_key"] == "Pica_pica"
    assert fetch_detections_last_24h(conn, "20261001")[0]["species_key"] == "Pica_pica"
    assert fetch_gallery_days(conn)[1] == 1
    assert build_report(conn, {})["summary"]["identified_objects"] == 1
    assert fetch_observation_page(conn, answered=False)[1] == 0


def test_refutation_does_not_blacklist_other_well_supported_objects(
    conn: sqlite3.Connection,
) -> None:
    first = seed(conn, identifier=1, species="Parus_major", quality=1)
    second = seed(conn, identifier=2, species="Parus_major", quality=1)
    answer(conn, 1, first, state="wrong", species=None)
    cover = fetch_species_covers(conn)[0]
    assert cover["automatic_count"] == 1 and cover["warning"]
    assert fetch_observation_page(conn, answered=False)[1] == 0
    answer(conn, 2, second, species="Parus_major")
    assert fetch_species_covers(conn)[0]["human_count"] == 1
    assert fetch_observation_page(conn, answered=False)[1] == 0


@pytest.mark.parametrize("whole_image", [False, True])
def test_no_bird_completes_previously_identified_object(
    conn: sqlite3.Connection,
    whole_image: bool,
) -> None:
    filename = seed(conn, identifier=42)
    answer(conn, 42, filename, species="Pica_pica")
    record_human_answer(
        conn,
        HumanAnswer(filename, image_bird_presence="absent")
        if whole_image
        else HumanAnswer(filename, detection_id=42, object_bird_presence="absent"),
        provenance(),
    )
    assert fetch_observation_page(conn, answered=False)[1] == 0
    rows, total, _ = fetch_observation_page(conn, answered=True)
    assert total == 1 and rows[0]["presence"] == "absent"
    assert fetch_species_covers(conn) == []
    assert fetch_detections_for_gallery(conn) == []
    from core.automatic_station_report import build_report

    assert build_report(conn, {})["summary"]["identified_objects"] == 0
    assert conn.execute(
        "SELECT answer_value FROM current_human_label_facts WHERE detection_id=42 AND fact_type='species_identity'"
    ).fetchone()[0] in {"corrected", "confirmed"}
    if whole_image:
        record_human_answer(
            conn, HumanAnswer(filename, image_bird_presence="present"), provenance()
        )
    answer(conn, 42, filename, species="Picus_viridis")
    assert fetch_species_covers(conn)[0]["species_key"] == "Picus_viridis"
    assert fetch_observation_page(conn, answered=False)[1] == 0


@pytest.mark.parametrize("resolution", ["object_absent", "image_present"])
def test_image_training_statement_does_not_change_object_answer(
    conn: sqlite3.Connection, resolution: str
) -> None:
    from core.human_label_core import append_fact, ensure_object_subject

    filename = seed(conn, identifier=44)
    record_human_answer(
        conn, HumanAnswer(filename, image_bird_presence="absent"), provenance()
    )
    append_fact(
        conn,
        subject_id=ensure_object_subject(conn, 44),
        fact_type="species_identity",
        answer_value="corrected",
        species_key="Passer_domesticus",
        provenance=provenance(),
    )
    assert fetch_observation_page(conn, answered=False)[1] == 0
    record_human_answer(
        conn,
        HumanAnswer(
            filename,
            detection_id=44,
            object_bird_presence="absent" if resolution == "object_absent" else None,
            image_bird_presence="present" if resolution == "image_present" else None,
        ),
        provenance(),
    )
    assert fetch_observation_page(conn, answered=False)[1] == 0
    assert fetch_observation_page(conn, answered=True)[1] == 1
    assert fetch_species_covers(conn) == []
