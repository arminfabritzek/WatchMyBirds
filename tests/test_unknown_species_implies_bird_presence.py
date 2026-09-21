"""An answered species axis implies the box holds a bird.

A person who answers *any* species question -- "yes that is a Nuthatch",
"no it is a Great Tit", or "there is a bird here but I cannot name it" --
has already asserted that the box contains a bird. Leaving
``bird_presence`` unset in that case made the row fail OD readiness with
``object_bird_presence_unknown``, discarding a box whose *geometry* the
human had implicitly endorsed.

The species answer and the presence answer stay independent facts; this
only fills the presence axis the answer already entails.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from config import get_config
from core.human_label_core import (
    HumanAnswer,
    LabelProvenance,
    object_training_readiness,
    record_human_answer,
)
from utils.db import connection as db_connection


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[sqlite3.Connection]:
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    connection = db_connection.get_connection()
    yield connection
    connection.close()


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> dict[str, int | str]:
    filename = "20260920_094825_673647.jpg"
    conn.execute(
        "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
        (filename, "2026-09-20T09:48:25+00:00"),
    )
    cursor = conn.execute(
        """
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, raw_species_name, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            0.1,
            0.2,
            0.3,
            0.4,
            "bird",
            "Sitta_europaea",
            "2026-09-20T09:48:26+00:00",
        ),
    )
    conn.commit()
    return {"filename": filename, "detection_id": int(cursor.lastrowid)}


@pytest.fixture
def provenance() -> LabelProvenance:
    return LabelProvenance(
        installation_id="0123456789abcdef0123456789abcdef",
        app_version="0.6.0",
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref="review:unknown-species",
        created_at="2026-09-20T09:50:00+00:00",
    )


def _object_facts(conn: sqlite3.Connection, detection_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT f.fact_type, f.answer_value, f.species_key
        FROM current_human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE s.detection_id = ?
        """,
        (detection_id,),
    ).fetchall()
    return [
        {
            "scope": "object",
            "fact_type": row["fact_type"],
            "answer_value": row["answer_value"],
            "species_key": row["species_key"],
        }
        for row in rows
    ]


def test_unknown_species_records_a_bird_presence_fact(conn, seeded, provenance):
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            species_identity="unknown",
        ),
        provenance,
    )
    conn.commit()

    facts = {
        f["fact_type"]: f["answer_value"]
        for f in _object_facts(conn, int(seeded["detection_id"]))
    }
    assert facts.get("bird_presence") == "present", (
        "'I see a bird but cannot name the species' asserts a bird is present"
    )
    assert facts.get("species_identity") == "unknown"


def test_unknown_species_no_longer_blocks_od_on_the_presence_axis(
    conn, seeded, provenance
):
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            species_identity="unknown",
        ),
        provenance,
    )
    conn.commit()

    readiness = object_training_readiness(
        _object_facts(conn, int(seeded["detection_id"]))
    )
    assert "object_bird_presence_unknown" not in readiness["od"]["reasons"]
    assert "object_bird_absent" not in readiness["od"]["reasons"]


def test_cls_stays_excluded_for_the_withdrawn_species(conn, seeded, provenance):
    """Filling the presence axis must not make the withdrawn species usable."""
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            species_identity="unknown",
        ),
        provenance,
    )
    conn.commit()

    readiness = object_training_readiness(
        _object_facts(conn, int(seeded["detection_id"]))
    )
    assert readiness["cls"]["ready"] is False
    assert "species_unknown" in readiness["cls"]["reasons"]


def test_an_explicit_absent_answer_is_never_overridden(conn, seeded, provenance):
    """A person who says "not a bird" outranks the species-axis implication."""
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            object_bird_presence="absent",
            species_identity="unknown",
        ),
        provenance,
    )
    conn.commit()

    facts = {
        f["fact_type"]: f["answer_value"]
        for f in _object_facts(conn, int(seeded["detection_id"]))
    }
    assert facts.get("bird_presence") == "absent"


def test_confirming_a_species_also_asserts_presence(conn, seeded, provenance):
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            species_identity="confirmed",
            species_key="Sitta_europaea",
        ),
        provenance,
    )
    conn.commit()

    facts = {
        f["fact_type"]: f["answer_value"]
        for f in _object_facts(conn, int(seeded["detection_id"]))
    }
    assert facts.get("bird_presence") == "present"
