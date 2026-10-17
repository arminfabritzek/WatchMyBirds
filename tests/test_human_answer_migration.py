"""Upgrade endpoint-generated answer history without reviving withdrawn answers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from core.human_label_core import (
    LabelProvenance,
    append_fact,
    ensure_image_subject,
    ensure_object_subject,
    retract_answer,
)
from utils.db.connection import (
    human_answers_migrated,
    migrate_human_answers,
)
from utils.db.detections import (
    fetch_detections_for_gallery,
    fetch_observation_page,
    fetch_species_covers,
)

FIXTURE = Path(__file__).parent / "fixtures" / "human_answers_0_5_9.sql"


@pytest.fixture
def legacy() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(FIXTURE.read_text())
    conn.execute("ALTER TABLE detections ADD COLUMN trashed_at TEXT")
    from utils.db.observations import recreate_observations_view

    recreate_observations_view(conn)
    conn.commit()
    yield conn
    conn.close()


def provenance(source_ref: str = "test-withdrawal") -> LabelProvenance:
    return LabelProvenance(
        "synthetic", "0.5.9", "normal_correction", "watchmybirds_ui", source_ref
    )


def counts(conn: sqlite3.Connection) -> tuple[int, int, int]:
    return tuple(
        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("images", "detections", "human_label_facts")
    )


def snapshot(conn: sqlite3.Connection) -> list[tuple]:
    return [
        tuple(row)
        for row in conn.execute(
            """SELECT object_key, species_key,
        active AND NOT answered AND NOT admitted AS review_open, answered
        FROM current_observations ORDER BY object_key"""
        )
    ]


def test_endpoint_fixture_migrates_once_and_preserves_box_decisions(
    legacy: sqlite3.Connection,
) -> None:
    before = counts(legacy)
    result = migrate_human_answers(legacy)
    assert human_answers_migrated(legacy)
    assert counts(legacy)[:2] == before[:2]
    assert result == {
        "withdrawn_facts": 4,
        "image_object_conflicts": 1,
        "image_manual_conflicts": 1,
        "reset_species_history": 5,
        "reset_conflict": 1,
        "changed_decisions": 9,
        "policy_LOW_DISPLAY_SCORE": 1,
        "policy_ZERO_GEOGRAPHIC_OCCURRENCE": 1,
        "policy_NO_NAMED_BIRD_SPECIES": 2,
        "imported_object_absence": 2,
        "restored_absence_objects": 2,
        "dated_trash_objects": 1,
    }
    assert 9 in {
        row["detection_id"] for row in fetch_observation_page(legacy, answered=False)[0]
    }
    for did in (3, 4, 5):
        row = legacy.execute(
            "SELECT decision_state, species_key, active, admitted FROM current_observations WHERE detection_id=?",
            (did,),
        ).fetchone()
        assert tuple(row) == ("confirmed", "Parus_major", 1, 1)
    after = list(legacy.iterdump())
    assert migrate_human_answers(legacy) == {}
    assert list(legacy.iterdump()) == after
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM detections WHERE status='rejected'"
        ).fetchone()[0]
        == 1
    )
    assert legacy.execute(
        "SELECT trashed_at FROM detections WHERE detection_id=10"
    ).fetchone()[0]
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM classifications WHERE status='rejected'"
        ).fetchone()[0]
        == 1
    )


def test_species_answer_reset_preserves_observation_surfaces(
    legacy: sqlite3.Connection,
) -> None:
    migrate_human_answers(legacy)
    final_decisions = [
        tuple(row)
        for row in legacy.execute(
            "SELECT detection_id, decision_state, decision_reasons FROM detections"
        )
    ]
    baseline = sqlite3.connect(":memory:")
    baseline.executescript(FIXTURE.read_text())
    for did, state, reasons in baseline.execute(
        "SELECT detection_id,decision_state,decision_reasons FROM detections"
    ):
        if did in {1, 2, 3, 4, 5, 6, 7, 11}:
            legacy.execute(
                "UPDATE detections SET decision_state=?,decision_reasons=? WHERE detection_id=?",
                (state, reasons, did),
            )
    baseline.close()
    from core.automatic_station_report import build_report

    report_before = build_report(legacy, {})["summary"]
    before = snapshot(legacy)
    gallery = [r["detection_id"] for r in fetch_detections_for_gallery(legacy)]
    species = [
        (r["species_key"], r["observation_count"]) for r in fetch_species_covers(legacy)
    ]
    review = fetch_observation_page(legacy, answered=False)[1]
    for did, state, reasons in final_decisions:
        legacy.execute(
            "UPDATE detections SET decision_state=?,decision_reasons=? WHERE detection_id=?",
            (state, reasons, did),
        )
    assert build_report(legacy, {})["summary"] == report_before
    assert snapshot(legacy) == before
    assert [r["detection_id"] for r in fetch_detections_for_gallery(legacy)] == gallery
    assert [
        (r["species_key"], r["observation_count"]) for r in fetch_species_covers(legacy)
    ] == species
    assert fetch_observation_page(legacy, answered=False)[1] == review


def test_upgraded_answer_withdrawal_cannot_recover_legacy_species(
    legacy: sqlite3.Connection,
) -> None:
    from core.automatic_station_report import build_report
    from core.metadata_export_core import _rows_for_image
    from utils.db.detections import fetch_sibling_detections

    migrate_human_answers(legacy)
    report_before = build_report(legacy, {})
    filename = "20260101_120000_confirmed.jpg"
    assert (
        legacy.execute(
            "SELECT manual_species_override FROM detections WHERE detection_id=1"
        ).fetchone()[0]
        == "Cyanistes_caeruleus"
    )
    retract_answer(
        legacy,
        image_filename=filename,
        detection_id=1,
        fact_type="species_identity",
        provenance=provenance(),
    )
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id=1 AND fact_type IN ('species_identity','bird_presence')"
        ).fetchone()[0]
        == 0
    )
    row = legacy.execute(
        "SELECT species_key, active, answered, admitted FROM current_observations WHERE detection_id=1"
    ).fetchone()
    assert tuple(row) == ("Unknown_species", 1, 0, 0)
    assert 1 not in [r["detection_id"] for r in fetch_detections_for_gallery(legacy)]
    assert any(
        r["detection_id"] == 1
        for r in fetch_observation_page(legacy, answered=False)[0]
    )
    # The other current correction remains; withdrawing one does not hide another bird.
    assert (
        next(
            r
            for r in fetch_species_covers(legacy)
            if r["species_key"] == "Cyanistes_caeruleus"
        )["observation_count"]
        == 1
    )
    report_after = build_report(legacy, {})
    assert report_after["summary"]["open_objects"] == (
        report_before["summary"]["open_objects"] + 1
    )
    assert report_after["summary"]["identified_objects"] == (
        report_before["summary"]["identified_objects"] - 1
    )
    assert (
        next(
            row["observation_count"]
            for row in report_after["species"]
            if row["species_key"] == "Cyanistes_caeruleus"
        )
        == 1
    )
    for row in [
        *fetch_sibling_detections(legacy, filename, include_rejected=True),
        *_rows_for_image(legacy, filename),
    ]:
        assert row["human_species_key"] is None
        assert row["cls_class_name"] == "Parus_major"


def test_existing_presence_conflict_withdraws_only_bird_answers(
    legacy: sqlite3.Connection,
) -> None:
    filename = "20260101_120000_confirmed.jpg"
    subject = ensure_object_subject(legacy, 1)
    append_fact(
        legacy,
        subject_id=subject,
        fact_type="bbox_quality",
        answer_value="suitable",
        provenance=provenance("box-answer"),
    )
    # Older station data can contain contradictions that current writers reject.
    with patch("core.human_label_core._reject_presence_conflict"):
        append_fact(
            legacy,
            subject_id=ensure_image_subject(legacy, filename),
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance("review:no-bird"),
        )
    legacy.commit()
    result = migrate_human_answers(legacy)
    assert result["image_object_conflicts"] == 2
    assert result["reset_conflict"] == 2
    assert result["withdrawn_facts"] == 7
    assert [
        r[0]
        for r in legacy.execute(
            "SELECT fact_type FROM current_human_label_facts WHERE detection_id=1"
        )
    ] == ["bbox_quality"]
    row = legacy.execute(
        "SELECT decision_state, answered, admitted FROM current_observations WHERE detection_id=1"
    ).fetchone()
    assert tuple(row) == ("uncertain", 0, 0)
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM human_label_facts WHERE assertion_state='retracted' AND source_kind='watchmybirds_migration'"
        ).fetchone()[0]
        == 7
    )


def test_old_column_conflict_is_not_imported(legacy: sqlite3.Connection) -> None:
    legacy.execute(
        "UPDATE detections SET manual_species_override='Cyanistes_caeruleus',species_source='manual' WHERE detection_id=14"
    )
    legacy.execute(
        "UPDATE images SET review_status='no_bird' WHERE filename='20260101_120000_automatic.jpg'"
    )
    legacy.commit()
    result = migrate_human_answers(legacy)
    assert result["legacy_conflicts"] == 1
    assert result["reset_conflict"] == 2
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id=14"
        ).fetchone()[0]
        == 0
    )
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE image_filename='20260101_120000_automatic.jpg'"
        ).fetchone()[0]
        == 0
    )
    assert (
        legacy.execute(
            "SELECT decision_state FROM detections WHERE detection_id=14"
        ).fetchone()[0]
        == "uncertain"
    )


@pytest.mark.parametrize("source", ["manual", "manual_unknown"])
def test_image_fact_and_legacy_species_conflict_completes_migration(
    legacy: sqlite3.Connection, source: str
) -> None:
    filename = "20260101_120000_automatic.jpg"
    append_fact(
        legacy,
        subject_id=ensure_image_subject(legacy, filename),
        fact_type="bird_presence",
        answer_value="absent",
        provenance=provenance("image-answer"),
    )
    legacy.execute(
        "UPDATE detections SET manual_species_override=?,species_source=? WHERE detection_id=14",
        ("Pica_pica" if source == "manual" else None, source),
    )
    legacy.commit()
    result = migrate_human_answers(legacy)
    assert human_answers_migrated(legacy)
    assert result["reset_conflict"] >= 1
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE image_filename=?",
            (filename,),
        ).fetchone()[0]
        == 0
    )
    assert 14 in {
        row["detection_id"] for row in fetch_observation_page(legacy, answered=False)[0]
    }
    after = list(legacy.iterdump())
    assert migrate_human_answers(legacy) == {}
    assert list(legacy.iterdump()) == after


@pytest.mark.parametrize(
    "species_answer", ["confirmed", "corrected", "unknown", "wrong"]
)
@pytest.mark.parametrize("presence_first", [True, False])
def test_conflict_withdraws_all_bird_axes_from_independent_clicks(
    legacy: sqlite3.Connection, species_answer: str, presence_first: bool
) -> None:
    filename = "20260101_120000_automatic.jpg"
    subject = ensure_object_subject(legacy, 14)
    answers = [("bird_presence", "present"), ("species_identity", species_answer)]
    if not presence_first:
        answers.reverse()
    for axis, value in answers:
        append_fact(
            legacy,
            subject_id=subject,
            fact_type=axis,
            answer_value=value,
            species_key="Parus_major" if value in {"confirmed", "corrected"} else None,
            provenance=provenance("independent-" + axis),
        )
    append_fact(
        legacy,
        subject_id=subject,
        fact_type="bbox_quality",
        answer_value="suitable",
        provenance=provenance("box"),
    )
    with patch("core.human_label_core._reject_presence_conflict"):
        append_fact(
            legacy,
            subject_id=ensure_image_subject(legacy, filename),
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance("image"),
        )
    legacy.commit()
    migrate_human_answers(legacy)
    assert [
        tuple(row)
        for row in legacy.execute(
            "SELECT fact_type,answer_value FROM current_human_label_facts WHERE detection_id=14"
        )
    ] == [("bbox_quality", "suitable")]
    assert 14 in {
        r["detection_id"] for r in fetch_observation_page(legacy, answered=False)[0]
    }


def test_mixed_legacy_database_completes_and_restarts_without_duplicates(
    legacy: sqlite3.Connection, tmp_path: Path
) -> None:
    from tests.test_station_observations import seed

    cases = [
        ("fact", "presence"),
        ("fact", "species"),
        ("fact", "unknown"),
        ("fact", "manual"),
        ("fact", "manual_unknown"),
        ("legacy", "presence"),
        ("legacy", "species"),
        ("legacy", "manual"),
        ("legacy", "manual_unknown"),
        ("fact", "drawn"),
        ("legacy", "drawn"),
    ]
    open_ids = {2, 9, 15, 16, 17, 18}
    preserved_positive_ids = set()
    absent_ids = {8, 12, 13}
    for index, (image_source, object_source) in enumerate(cases):
        first = 100 + index * 4
        filename = seed(legacy, identifier=first, species="Parus_major", quality=1)
        for did in range(first + 1, first + 4):
            other = seed(legacy, identifier=did, species="Parus_major", quality=1)
            legacy.execute(
                "UPDATE detections SET image_filename=? WHERE detection_id=?",
                (filename, did),
            )
            legacy.execute("DELETE FROM images WHERE filename=?", (other,))
        subject = ensure_object_subject(legacy, first + 1)
        if object_source in {"presence", "species", "unknown"}:
            append_fact(
                legacy,
                subject_id=subject,
                fact_type="bird_presence"
                if object_source == "presence"
                else "species_identity",
                answer_value={
                    "presence": "present",
                    "species": "corrected",
                    "unknown": "unknown",
                }[object_source],
                species_key="Parus_major" if object_source == "species" else None,
                provenance=provenance("object"),
            )
        elif object_source in {"manual", "manual_unknown"}:
            legacy.execute(
                "UPDATE detections SET manual_species_override=?,species_source=? WHERE detection_id=?",
                (
                    "Parus_major" if object_source == "manual" else None,
                    object_source,
                    first + 1,
                ),
            )
        else:
            legacy.execute(
                """INSERT INTO manual_objects(image_filename,bbox_x,bbox_y,bbox_w,bbox_h,
                species_state,created_at,updated_at) VALUES (?,0.1,0.1,0.2,0.2,'unknown','20260101','20260101')""",
                (filename,),
            )
        append_fact(
            legacy,
            subject_id=subject,
            fact_type="bbox_quality",
            answer_value="suitable",
            provenance=provenance("box"),
        )
        append_fact(
            legacy,
            subject_id=ensure_object_subject(legacy, first + 2),
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance("sibling"),
        )
        withdrawn = ensure_object_subject(legacy, first + 3)
        append_fact(
            legacy,
            subject_id=withdrawn,
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance("withdrawn-sibling"),
        )
        append_fact(
            legacy,
            subject_id=withdrawn,
            fact_type="bird_presence",
            answer_value=None,
            assertion_state="retracted",
            provenance=provenance("withdrawn-sibling"),
        )
        if image_source == "fact":
            with patch("core.human_label_core._reject_presence_conflict"):
                append_fact(
                    legacy,
                    subject_id=ensure_image_subject(legacy, filename),
                    fact_type="bird_presence",
                    answer_value="absent",
                    provenance=provenance("image"),
                )
        else:
            legacy.execute(
                "UPDATE images SET review_status='no_bird' WHERE filename=?",
                (filename,),
            )
        open_ids.update({first, first + 3})
        absent_ids.add(first + 2)
        if image_source == "legacy" and object_source == "species":
            preserved_positive_ids.add(first + 1)
        elif not (image_source == "legacy" and object_source == "presence"):
            open_ids.add(first + 1)
    legacy.commit()
    before = counts(legacy)
    original_facts = [
        tuple(r)
        for r in legacy.execute("SELECT * FROM human_label_facts ORDER BY fact_id")
    ]
    manual_before = [
        tuple(r)
        for r in legacy.execute(
            "SELECT * FROM manual_objects ORDER BY manual_object_id"
        )
    ]
    result = migrate_human_answers(legacy)
    assert human_answers_migrated(legacy)
    assert counts(legacy)[:2] == before[:2]
    assert [
        tuple(r)
        for r in legacy.execute(
            "SELECT * FROM human_label_facts ORDER BY fact_id LIMIT ?",
            (len(original_facts),),
        )
    ] == original_facts
    assert [
        tuple(r)
        for r in legacy.execute(
            "SELECT * FROM manual_objects ORDER BY manual_object_id"
        )
    ] == manual_before
    assert {
        r["detection_id"]
        for r in fetch_observation_page(legacy, answered=False, per_page=100)[0]
    } == open_ids
    for did in open_ids:
        assert (
            legacy.execute(
                "SELECT decision_state FROM detections WHERE detection_id=?", (did,)
            ).fetchone()[0]
            == "uncertain"
        )
    assert {
        r[0]
        for r in legacy.execute(
            "SELECT detection_id FROM current_human_label_facts WHERE scope='object' AND fact_type='bird_presence' AND answer_value='absent'"
        )
    } == absent_ids
    assert {
        r[0]
        for r in legacy.execute(
            "SELECT detection_id FROM current_human_label_facts WHERE detection_id>=100 AND fact_type='species_identity'"
        )
    } == preserved_positive_ids
    assert legacy.execute(
        "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id>=100 AND fact_type='bbox_quality'"
    ).fetchone()[0] == len(cases)
    assert before == (31, 62, 79)
    assert counts(legacy) == (31, 62, 94)
    assert len(open_ids) == 37
    done, total, _ = fetch_observation_page(legacy, answered=True, per_page=100)
    assert total == 36
    assert {
        f"detection:{row[0]}"
        for row in legacy.execute(
            "SELECT DISTINCT detection_id FROM current_human_label_facts WHERE scope='object'"
        )
    } <= {row["object_key"] for row in done}
    assert result == {
        "withdrawn_facts": 13,
        "image_object_conflicts": 6,
        "image_manual_conflicts": 2,
        "legacy_conflicts": 2,
        "legacy_object_conflicts": 2,
        "legacy_manual_conflicts": 1,
        "reset_species_history": 5,
        "changed_decisions": 41,
        "reset_conflict": 33,
        "policy_LOW_DISPLAY_SCORE": 1,
        "policy_ZERO_GEOGRAPHIC_OCCURRENCE": 1,
        "policy_NO_NAMED_BIRD_SPECIES": 2,
        "imported_object_absence": 2,
        "restored_absence_objects": 2,
        "dated_trash_objects": 1,
    }
    db_path = tmp_path / "images.db"
    with sqlite3.connect(db_path) as destination:
        legacy.backup(destination)
    after = list(legacy.iterdump())
    with sqlite3.connect(db_path) as restarted:
        assert human_answers_migrated(restarted)
        assert migrate_human_answers(restarted) == {}
        assert list(restarted.iterdump()) == after


def test_interruption_rolls_back_facts_decisions_status_and_marker(
    legacy: sqlite3.Connection,
) -> None:
    before = list(legacy.iterdump())
    original = append_fact
    calls = 0

    def interrupted(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    with patch("core.human_label_core.append_fact", side_effect=interrupted):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            migrate_human_answers(legacy)
    assert list(legacy.iterdump()) == before
    assert not human_answers_migrated(legacy)
    assert migrate_human_answers(legacy)["imported_object_absence"] == 2
    assert human_answers_migrated(legacy)


def test_withdrawn_legacy_axes_are_never_imported(legacy: sqlite3.Connection) -> None:
    legacy.execute(
        "UPDATE detections SET manual_species_override='Pica_pica',species_source='manual' WHERE detection_id=2"
    )
    legacy.execute(
        "UPDATE detections SET manual_bbox_review='correct' WHERE detection_id=4"
    )
    legacy.commit()
    migrate_human_answers(legacy)
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id=2 AND fact_type='species_identity'"
        ).fetchone()[0]
        == 0
    )
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id=4 AND fact_type='bbox_quality'"
        ).fetchone()[0]
        == 0
    )


def test_legacy_only_species_import_withdraws_derived_presence(
    legacy: sqlite3.Connection,
) -> None:
    legacy.execute(
        "UPDATE detections SET manual_species_override='Pica_pica',species_source='manual' WHERE detection_id=14"
    )
    legacy.commit()
    assert migrate_human_answers(legacy)["reset_legacy_species"] == 1
    retract_answer(
        legacy,
        image_filename="20260101_120000_automatic.jpg",
        detection_id=14,
        fact_type="species_identity",
        provenance=provenance(),
    )
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE detection_id=14"
        ).fetchone()[0]
        == 0
    )


def test_migrated_full_image_withdrawal_withdraws_object_absences(
    legacy: sqlite3.Connection,
) -> None:
    migrate_human_answers(legacy)
    retract_answer(
        legacy,
        image_filename="20260101_120000_full_image.jpg",
        detection_id=None,
        fact_type="bird_presence",
        provenance=provenance(),
    )
    assert (
        legacy.execute(
            "SELECT COUNT(*) FROM current_human_label_facts WHERE image_filename='20260101_120000_full_image.jpg'"
        ).fetchone()[0]
        == 0
    )


def test_retention_is_blocked_before_migration_even_when_enabled(
    legacy, tmp_path
) -> None:
    from core.retention_core import execute_plan

    with pytest.raises(RuntimeError, match="migration is incomplete"):
        execute_plan(legacy, str(tmp_path), {"AUTOMATIC_DELETION_ENABLED": True})
    assert list(tmp_path.iterdir()) == []


def test_existing_empty_database_disables_old_retention_once(
    monkeypatch, tmp_path
) -> None:
    from config import get_config
    from core.settings_core import take_retention_update_notice
    from utils.db import connection
    from utils.settings import load_settings_yaml, save_settings_yaml

    sqlite3.connect(tmp_path / "images.db").close()
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(get_config(), "RETENTION_POSTURE", "conservative")
    monkeypatch.setitem(get_config(), "RETENTION_ENABLED", True)
    monkeypatch.setitem(get_config(), "RETENTION_AUTO_ENABLED", True)
    save_settings_yaml(
        {"RETENTION_POSTURE": "conservative", "RETENTION_AUTO_ENABLED": True},
        str(tmp_path),
    )
    with connection.closing_connection() as conn:
        assert human_answers_migrated(conn)
    settings = load_settings_yaml(str(tmp_path))
    assert settings["AUTOMATIC_DELETION_ENABLED"] is False
    assert settings["AUTOMATIC_DELETION_ENABLED_AT"] is None
    assert take_retention_update_notice() is True
    assert take_retention_update_notice() is False
    settings["RETENTION_POSTURE"] = "conservative"
    save_settings_yaml(settings, str(tmp_path))
    connection._schema_initialized_paths.clear()
    connection.get_connection().close()
    assert load_settings_yaml(str(tmp_path))["RETENTION_POSTURE"] == "conservative"


def test_missing_migration_marker_is_retried_on_next_start(
    monkeypatch, tmp_path
) -> None:
    from config import get_config
    from utils.db import connection

    conn = sqlite3.connect(tmp_path / "images.db")
    conn.executescript(FIXTURE.read_text())
    conn.close()
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    with patch.object(
        connection, "migrate_human_answers", side_effect=RuntimeError("interrupted")
    ):
        conn = connection.get_connection()
        assert not human_answers_migrated(conn)
        conn.close()
    connection._schema_initialized_paths.clear()
    conn = connection.get_connection()
    assert human_answers_migrated(conn)
    conn.close()


def test_current_readers_cannot_read_retired_answer_columns(
    legacy: sqlite3.Connection,
) -> None:
    from core.automatic_station_report import build_report
    from core.metadata_export_core import _rows_for_image
    from utils.db.detections import fetch_sibling_detections
    from utils.db.trash import fetch_trash_items

    migrate_human_answers(legacy)
    forbidden = {
        "manual_species_override",
        "species_source",
        "review_status",
        "manual_bbox_review",
    }
    reads = []

    def authorize(
        action: int,
        table: str | None,
        column: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        if (
            action == sqlite3.SQLITE_READ
            and table in {"detections", "images"}
            and column in forbidden
        ):
            reads.append((table, column))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    legacy.set_authorizer(authorize)
    try:
        fetch_detections_for_gallery(legacy)
        fetch_species_covers(legacy)
        fetch_observation_page(legacy, answered=False)
        fetch_observation_page(legacy, answered=True)
        build_report(legacy, {})
        fetch_sibling_detections(
            legacy, "20260101_120000_confirmed.jpg", include_rejected=True
        )
        _rows_for_image(legacy, "20260101_120000_confirmed.jpg")
        fetch_trash_items(legacy)
        assert reads == []
    finally:
        legacy.set_authorizer(None)


def test_wrong_species_and_image_absence_are_independent(
    legacy: sqlite3.Connection,
) -> None:
    filename = "20260101_120000_automatic.jpg"
    legacy.execute(
        "UPDATE detections SET species_source='manual_wrong' WHERE detection_id=14"
    )
    legacy.execute(
        "UPDATE images SET review_status='no_bird' WHERE filename=?", (filename,)
    )
    legacy.commit()
    result = migrate_human_answers(legacy)
    assert result.get("legacy_conflicts", 0) == 0
    assert {
        tuple(row)
        for row in legacy.execute(
            "SELECT fact_type,answer_value FROM current_human_label_facts WHERE detection_id=14"
        )
    } == {("species_identity", "wrong"), ("bird_presence", "absent")}


def test_trash_deletion_is_blocked_before_migration(legacy: sqlite3.Connection) -> None:
    from core.detections_core import hard_delete_detections_with_conn

    before = list(legacy.iterdump())
    with patch("core.detections_core.gc_hard_delete_detections") as remove:
        with pytest.raises(ValueError, match="migration must finish"):
            hard_delete_detections_with_conn(legacy, [8])
        remove.assert_not_called()
    assert list(legacy.iterdump()) == before
