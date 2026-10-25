"""Species withdrawal removes the whole human action and preserves model decisions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from config import get_config
from core.human_label_core import (
    HumanAnswer,
    HumanLabelError,
    LabelProvenance,
    record_human_answer,
    retract_species_identity,
)
from utils.db import connection as db_connection

AI_SPECIES = "Cyanistes_caeruleus"


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[sqlite3.Connection]:
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    connection = db_connection.get_connection()
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> dict[str, object]:
    filename = "20260921_124151_424618.jpg"
    conn.execute(
        "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
        (filename, "2026-09-21T12:41:51+00:00"),
    )
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, raw_species_name, decision_state, decision_level,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            0.34,
            0.44,
            0.09,
            0.12,
            "bird",
            AI_SPECIES,
            "uncertain",
            None,
            "2026-09-21T12:41:52+00:00",
        ),
    )
    detection_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank)"
        " VALUES (?, ?, ?, 1)",
        (detection_id, AI_SPECIES, 0.94),
    )
    conn.commit()
    return {"filename": filename, "detection_id": detection_id}


@pytest.fixture
def provenance() -> LabelProvenance:
    return LabelProvenance(
        installation_id="0123456789abcdef0123456789abcdef",
        app_version="0.5.8",
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref="image-correction:undo",
        created_at="2026-09-21T13:00:00+00:00",
    )


def _row(conn: sqlite3.Connection, detection_id: int) -> dict:
    return dict(
        conn.execute(
            """
            SELECT raw_species_name, manual_species_override, species_source,
                   decision_state, decision_level
            FROM detections WHERE detection_id = ?
            """,
            (detection_id,),
        ).fetchone()
    )


def _facts(conn: sqlite3.Connection, detection_id: int) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT f.fact_type, f.answer_value, f.species_key, f.assertion_state
        FROM current_human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE s.detection_id = ?
        """,
        (detection_id,),
    ).fetchall()
    return {r["fact_type"]: dict(r) for r in rows}


def _confirm(conn, seeded, provenance) -> None:
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            object_bird_presence="present",
            species_identity="confirmed",
            species_key=AI_SPECIES,
        ),
        provenance,
    )
    conn.commit()


def test_confirmation_writes_facts_without_mirroring(conn, seeded, provenance):
    """Baseline: what the undo has to reverse."""
    _confirm(conn, seeded, provenance)

    row = _row(conn, int(seeded["detection_id"]))
    assert row["manual_species_override"] is None
    assert row["species_source"] is None
    assert row["decision_state"] == "uncertain"
    assert row["decision_level"] is None
    assert set(_facts(conn, int(seeded["detection_id"]))) == {
        "bird_presence",
        "species_identity",
    }


def test_undo_clears_the_manual_override_and_source(conn, seeded, provenance):
    _confirm(conn, seeded, provenance)

    retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    row = _row(conn, int(seeded["detection_id"]))
    assert not row["manual_species_override"]
    assert not row["species_source"]


def test_undo_preserves_the_decision_axes(conn, seeded, provenance):
    _confirm(conn, seeded, provenance)

    retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    row = _row(conn, int(seeded["detection_id"]))
    assert row["decision_state"] != "confirmed"
    assert row["decision_level"] != "species"


def test_the_model_proposal_survives_both_directions(conn, seeded, provenance):
    """The AI prediction is immutable history; the undo restores it by itself."""
    _confirm(conn, seeded, provenance)
    retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    row = _row(conn, int(seeded["detection_id"]))
    assert row["raw_species_name"] == AI_SPECIES
    cls = conn.execute(
        "SELECT cls_class_name FROM classifications WHERE detection_id = ?",
        (int(seeded["detection_id"]),),
    ).fetchone()
    assert cls["cls_class_name"] == AI_SPECIES


def test_undo_retracts_the_fact_rather_than_deleting_it(conn, seeded, provenance):
    """History is kept: the answer is withdrawn, not erased.

    ``current_human_label_facts`` shows asserted facts only, so a withdrawn
    species correctly disappears from it while both rows survive in the
    append-only base table.
    """
    _confirm(conn, seeded, provenance)
    fact_id = retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    assert fact_id is not None
    assert "species_identity" not in _facts(conn, int(seeded["detection_id"])), (
        "a withdrawn answer must not remain a current fact"
    )

    history = conn.execute(
        """
        SELECT f.assertion_state, f.answer_value
        FROM human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE s.detection_id = ? AND f.fact_type = 'species_identity'
        ORDER BY f.fact_id
        """,
        (int(seeded["detection_id"]),),
    ).fetchall()
    assert [r["assertion_state"] for r in history] == ["asserted", "retracted"]
    assert history[0]["answer_value"] == "confirmed"
    assert history[1]["answer_value"] is None


def test_undo_withdraws_presence_written_by_the_same_click(conn, seeded, provenance):
    _confirm(conn, seeded, provenance)
    retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    facts = _facts(conn, int(seeded["detection_id"]))
    assert "bird_presence" not in facts


def test_undo_without_a_previous_answer_is_a_no_op(conn, seeded, provenance):
    """A row nobody answered has nothing to take back."""
    fact_id = retract_species_identity(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )
    conn.commit()

    assert fact_id is None
    row = _row(conn, int(seeded["detection_id"]))
    assert row["raw_species_name"] == AI_SPECIES


def test_undo_rejects_a_detection_from_another_image(conn, seeded, provenance):
    _confirm(conn, seeded, provenance)

    with pytest.raises(HumanLabelError):
        retract_species_identity(
            conn,
            image_filename="20260101_000000_other.jpg",
            detection_id=int(seeded["detection_id"]),
            provenance=provenance,
        )
