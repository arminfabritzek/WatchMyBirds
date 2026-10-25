"""Safety contract for the derived-bird_presence backfill.

The script writes to a production database, so what it must *not* touch
matters more than what it writes: an explicit human answer on the
presence axis is never overwritten, and a subject that already carries a
presence fact is never given a second one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from config import get_config
from core.human_label_core import (
    DERIVED_FROM_SPECIES_ANSWER,
    HumanAnswer,
    LabelProvenance,
    append_fact,
    ensure_image_subject,
    ensure_object_subject,
    record_human_answer,
)
from scripts.backfill_derived_bird_presence import backfill
from utils.db import connection as db_connection


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[sqlite3.Connection]:
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    connection = db_connection.get_connection()
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


@pytest.fixture
def provenance() -> LabelProvenance:
    return LabelProvenance(
        installation_id="0123456789abcdef0123456789abcdef",
        app_version="0.5.7",
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref="review:legacy",
        created_at="2026-09-01T10:00:00+00:00",
    )


def _seed_detection(conn: sqlite3.Connection, filename: str) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO images (filename, timestamp) VALUES (?, ?)",
        (filename, "2026-09-01T10:00:00+00:00"),
    )
    cur = conn.execute(
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
            "Parus_major",
            "2026-09-01T10:00:01+00:00",
        ),
    )
    return int(cur.lastrowid)


def _legacy_species_only(
    conn: sqlite3.Connection,
    detection_id: int,
    provenance: LabelProvenance,
    answer_value: str = "corrected",
) -> int:
    """A row as it looked before the presence axis was derived."""
    subject_id = ensure_object_subject(conn, detection_id)
    append_fact(
        conn,
        subject_id=subject_id,
        fact_type="species_identity",
        answer_value=answer_value,
        species_key="Parus_major" if answer_value == "corrected" else None,
        provenance=provenance,
    )
    return subject_id


def _facts(conn: sqlite3.Connection, detection_id: int) -> dict[str, dict]:
    rows = conn.execute(
        """
        SELECT f.fact_type, f.answer_value, f.source_ref, f.installation_id,
               f.created_at
        FROM current_human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE s.detection_id = ?
        """,
        (detection_id,),
    ).fetchall()
    return {row["fact_type"]: dict(row) for row in rows}


def test_dry_run_reports_candidates_without_writing(conn, provenance):
    det = _seed_detection(conn, "20260901_100000_a.jpg")
    _legacy_species_only(conn, det, provenance)
    conn.commit()

    stats = backfill(conn, apply=False)

    assert stats["candidates"] == 1
    assert "bird_presence" not in _facts(conn, det)


def test_apply_writes_the_derived_presence_fact(conn, provenance):
    det = _seed_detection(conn, "20260901_100000_b.jpg")
    _legacy_species_only(conn, det, provenance)
    conn.commit()

    backfill(conn, apply=True)
    conn.commit()

    presence = _facts(conn, det)["bird_presence"]
    assert presence["answer_value"] == "present"
    assert DERIVED_FROM_SPECIES_ANSWER in presence["source_ref"]


def test_backfill_preserves_the_original_provenance(conn, provenance):
    """Attribution must survive: the fact belongs to whoever answered."""
    det = _seed_detection(conn, "20260901_100000_c.jpg")
    _legacy_species_only(conn, det, provenance)
    conn.commit()

    backfill(conn, apply=True)
    conn.commit()

    presence = _facts(conn, det)["bird_presence"]
    assert presence["installation_id"] == provenance.installation_id
    assert presence["created_at"] == provenance.created_at


def test_an_explicit_absent_answer_is_never_overwritten(conn, provenance):
    """The decisive safety guard."""
    det = _seed_detection(conn, "20260901_100000_d.jpg")
    subject_id = _legacy_species_only(conn, det, provenance)
    append_fact(
        conn,
        subject_id=subject_id,
        fact_type="bird_presence",
        answer_value="absent",
        provenance=provenance,
    )
    conn.commit()

    stats = backfill(conn, apply=True)
    conn.commit()

    assert stats["candidates"] == 0
    assert _facts(conn, det)["bird_presence"]["answer_value"] == "absent"


def test_rows_already_carrying_presence_are_left_alone(conn, provenance):
    det = _seed_detection(conn, "20260901_100000_e.jpg")
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="20260901_100000_e.jpg",
            detection_id=det,
            species_identity="confirmed",
            species_key="Parus_major",
        ),
        provenance,
    )
    conn.commit()

    stats = backfill(conn, apply=True)
    conn.commit()

    assert stats["candidates"] == 0
    presence = conn.execute(
        """
        SELECT COUNT(*) c FROM current_human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE s.detection_id = ? AND f.fact_type = 'bird_presence'
        """,
        (det,),
    ).fetchone()["c"]
    assert presence == 1, "the backfill must not add a second presence fact"


def test_running_twice_is_idempotent(conn, provenance):
    det = _seed_detection(conn, "20260901_100000_f.jpg")
    _legacy_species_only(conn, det, provenance)
    conn.commit()

    backfill(conn, apply=True)
    conn.commit()
    second = backfill(conn, apply=True)
    conn.commit()

    assert second["candidates"] == 0


def test_an_image_level_no_bird_answer_blocks_that_row_only(conn, provenance):
    """A cross-scope contradiction is reported, never silently resolved.

    Someone relabelled a box and later marked the whole frame "no bird".
    Both are real answers and the database keeps no record of which one
    they meant to stand, so the backfill skips that row and leaves the
    rest of the batch untouched.
    """
    conflicted = _seed_detection(conn, "20260901_100000_h.jpg")
    _legacy_species_only(conn, conflicted, provenance)
    image_subject = ensure_image_subject(conn, "20260901_100000_h.jpg")
    append_fact(
        conn,
        subject_id=image_subject,
        fact_type="bird_presence",
        answer_value="absent",
        provenance=provenance,
    )

    clean = _seed_detection(conn, "20260901_100000_i.jpg")
    _legacy_species_only(conn, clean, provenance)
    conn.commit()

    stats = backfill(conn, apply=True)
    conn.commit()

    assert stats["conflicted"] == 1
    assert stats["written"] == 1
    assert "bird_presence" not in _facts(conn, conflicted)
    assert _facts(conn, clean)["bird_presence"]["answer_value"] == "present"


def test_an_unknown_species_answer_is_also_backfilled(conn, provenance):
    det = _seed_detection(conn, "20260901_100000_g.jpg")
    _legacy_species_only(conn, det, provenance, answer_value="unknown")
    conn.commit()

    backfill(conn, apply=True)
    conn.commit()

    assert _facts(conn, det)["bird_presence"]["answer_value"] == "present"
