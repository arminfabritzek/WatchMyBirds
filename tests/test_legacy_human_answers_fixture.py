"""Verify the human-answer history captured from release 0.5.9 endpoints."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def legacy_answers() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript((FIXTURES / "human_answers_0_5_9.sql").read_text())
    yield conn
    conn.close()


def test_legacy_fixture_integrity(legacy_answers: sqlite3.Connection) -> None:
    assert legacy_answers.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert legacy_answers.execute("PRAGMA foreign_key_check").fetchall() == []
    manifest = json.loads((FIXTURES / "human_answers_0_5_9.json").read_text())
    assert manifest["source_commit"] == "f39d9dd5"
    assert len(manifest["cases"]) == 20
    assert len(manifest["requests"]) == 20
    assert all(request["status"] == "success" for request in manifest["requests"])


@pytest.mark.parametrize("name", ["box_only", "box_withdrawn", "box_corrected"])
def test_legacy_box_answers_preserved_automatic_decisions(
    legacy_answers: sqlite3.Connection, name: str
) -> None:
    filename = f"20260101_120000_{name}.jpg"
    row = legacy_answers.execute(
        "SELECT detection_id, decision_state, decision_level FROM detections WHERE image_filename=?",
        (filename,),
    ).fetchone()
    assert tuple(row)[1:] == ("confirmed", "species")
    facts = legacy_answers.execute(
        """SELECT f.fact_type FROM human_label_facts f
        JOIN label_subjects s USING(subject_id) WHERE s.detection_id=?""",
        (row[0],),
    ).fetchall()
    assert facts
    assert {fact[0] for fact in facts} <= {"bbox_quality", "bbox_correction"}


def test_legacy_withdrawal_retains_history(legacy_answers: sqlite3.Connection) -> None:
    rows = legacy_answers.execute(
        """SELECT f.assertion_state FROM human_label_facts f
        JOIN label_subjects s USING(subject_id)
        WHERE s.image_filename='20260101_120000_withdrawn.jpg'
        AND f.fact_type='species_identity' ORDER BY f.fact_id"""
    ).fetchall()
    assert [row[0] for row in rows] == ["asserted", "retracted"]


def test_legacy_manual_object_can_conflict_with_image_absence(
    legacy_answers: sqlite3.Connection,
) -> None:
    row = legacy_answers.execute(
        """SELECT m.status, m.species_key, f.answer_value
        FROM manual_objects m JOIN current_human_label_facts f USING(image_filename)
        WHERE m.image_filename='20260101_120000_manual_full_image.jpg'
        AND f.scope='image' AND f.fact_type='bird_presence'"""
    ).fetchone()
    assert tuple(row) == ("active", "Parus_major", "absent")


def test_legacy_object_absence_is_in_trash(legacy_answers: sqlite3.Connection) -> None:
    row = legacy_answers.execute(
        """SELECT d.status, c.status, f.answer_value
        FROM detections d JOIN classifications c USING(detection_id)
        JOIN current_human_label_facts f USING(detection_id)
        WHERE d.image_filename='20260101_120000_object_absent.jpg'
        AND f.scope='object' AND f.fact_type='bird_presence'"""
    ).fetchone()
    assert tuple(row) == ("rejected", "rejected", "absent")
