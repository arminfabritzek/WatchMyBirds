"""Regression: an explicit human "species unknown" must not fall back to AI.

``core.human_label_core`` records a ``species_identity='unknown'`` answer by
clearing ``manual_species_override`` and stamping
``species_source='manual_unknown'``. The species fallback chain used to walk
straight past the cleared override into the classifier's own ``cls_class_name``
and redisplay the species the human had just withdrawn.
"""

from __future__ import annotations

import sqlite3

import pytest

from utils.db.detections import (
    effective_species_sql,
    effective_species_sql_for_columns,
)
from utils.species_names import UNKNOWN_SPECIES_KEY, species_key_from_candidates

AI_SPECIES = "Sitta_europaea"


@pytest.mark.parametrize("source", ["manual_unknown", "manual_wrong"])
def test_python_helper_does_not_resurrect_ai_species(source: str) -> None:
    assert (
        species_key_from_candidates(
            manual_override=None,
            cls_class_name=AI_SPECIES,
            od_class_name="bird",
            species_source=source,
        )
        == UNKNOWN_SPECIES_KEY
    )


@pytest.mark.parametrize(
    ("source", "override", "expected"),
    [
        ("model_top1", None, AI_SPECIES),  # untouched AI proposal
        ("manual", "Parus_major", "Parus_major"),  # human-confirmed species
        (None, None, AI_SPECIES),  # legacy row without species_source
        ("", None, AI_SPECIES),
    ],
)
def test_python_helper_leaves_other_rows_alone(
    source: str | None, override: str | None, expected: str
) -> None:
    """The guard must be narrow: only explicit unknown/wrong answers."""
    assert (
        species_key_from_candidates(
            manual_override=override,
            cls_class_name=AI_SPECIES,
            od_class_name="bird",
            species_source=source,
        )
        == expected
    )


def _fixture_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            raw_species_name TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            od_class_name TEXT,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER,
            status TEXT DEFAULT 'active'
        );
        """
    )
    rows = [
        # (id, override, species_source) - all share the same AI top-1
        (1, None, "model_top1"),
        (2, None, "manual_unknown"),
        (3, "Parus_major", "manual"),
    ]
    for det_id, override, source in rows:
        conn.execute(
            "INSERT INTO detections (detection_id, image_filename,"
            " raw_species_name, manual_species_override, species_source,"
            " od_class_name) VALUES (?,?,?,?,?,?)",
            (det_id, "f.jpg", AI_SPECIES, override, source, "bird"),
        )
        conn.execute(
            "INSERT INTO classifications (detection_id, cls_class_name,"
            " cls_confidence, rank, status) VALUES (?,?,?,?,?)",
            (det_id, AI_SPECIES, 0.92, 1, "active"),
        )
    conn.commit()
    return conn


def _effective(conn: sqlite3.Connection, sql: str) -> dict[int, str]:
    query = f"SELECT d.detection_id AS i, {sql} AS s FROM detections d"
    return {r["i"]: r["s"] for r in conn.execute(query)}


def test_sql_mirrors_the_python_guard() -> None:
    """The SQL surface (gallery, species pages) must agree with Python."""
    conn = _fixture_conn()
    try:
        got = _effective(conn, effective_species_sql("d"))
    finally:
        conn.close()
    assert got[1] == AI_SPECIES, "untouched AI proposal changed"
    assert got[2] == UNKNOWN_SPECIES_KEY, "human unknown resurrected as AI species"
    assert got[3] == "Parus_major", "human-confirmed species changed"


def test_sql_for_columns_degrades_without_species_source() -> None:
    """Legacy schemas lacking species_source must still resolve, not crash."""
    conn = _fixture_conn()
    try:
        legacy = effective_species_sql_for_columns(
            "d",
            detection_columns={"manual_species_override", "od_class_name"},
            classification_columns={"cls_class_name", "rank", "status"},
        )
        got = _effective(conn, legacy)
        assert got[2] == AI_SPECIES  # no species_source column -> old behaviour

        modern = effective_species_sql_for_columns(
            "d",
            detection_columns={
                "manual_species_override",
                "od_class_name",
                "species_source",
            },
            classification_columns={"cls_class_name", "rank", "status"},
        )
        assert _effective(conn, modern)[2] == UNKNOWN_SPECIES_KEY
    finally:
        conn.close()


def test_effective_species_sql_prepares_against_legacy_schema() -> None:
    """A DB predating ``species_source`` must still prepare and resolve.

    SQLite validates every column reference at prepare time, so an
    unconditional ``d.species_source`` reference raised OperationalError on
    older installs and on the minimal schemas some tests build.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            manual_species_override TEXT,
            od_class_name TEXT,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER,
            status TEXT DEFAULT 'active'
        );
        """
    )
    conn.execute(
        "INSERT INTO detections (detection_id, image_filename,"
        " manual_species_override, od_class_name) VALUES (1, 'f.jpg', NULL, 'bird')"
    )
    conn.execute(
        "INSERT INTO classifications (detection_id, cls_class_name,"
        " cls_confidence, rank, status) VALUES (1, ?, 0.9, 1, 'active')",
        (AI_SPECIES,),
    )
    conn.commit()
    try:
        # Passing the connection lets the helper detect the missing column.
        got = _effective(conn, effective_species_sql("d", conn))
        assert got[1] == AI_SPECIES
    finally:
        conn.close()


def test_review_does_not_preselect_ai_species_after_human_unknown() -> None:
    """An explicit unknown must not arm Approve with the model's guess.

    ``_resolve_review_selected_species`` previously saw only an empty
    ``manual_species_override`` and could not tell "nobody answered yet" from
    "a human answered unknown", so it pre-selected the classifier's top-1.
    That both mislabelled the header and left the withdrawn species one
    Approve click from becoming a positive label again.
    """
    from web.blueprints.review import _resolve_review_selected_species

    quick = [{"source": "cls", "scientific": AI_SPECIES, "common": "Kleiber"}]
    names = {AI_SPECIES: "Kleiber", UNKNOWN_SPECIES_KEY: "Vogel (Art unklar)"}

    # unanswered row -> the AI proposal is a helpful default
    assert _resolve_review_selected_species(
        quick,
        manual_species_override=None,
        common_names=names,
        species_source="model_top1",
    ) == (AI_SPECIES, "Kleiber", "default")

    # explicit human unknown -> nothing pre-selected
    species, common, origin = _resolve_review_selected_species(
        quick,
        manual_species_override=None,
        common_names=names,
        species_source="manual_unknown",
    )
    assert species is None, "AI species was re-armed after a human unknown"
    assert origin == "manual_unknown"
    assert common == "Vogel (Art unklar)"

    # a human-confirmed species still wins
    assert _resolve_review_selected_species(
        quick,
        manual_species_override="Parus_major",
        common_names={**names, "Parus_major": "Kohlmeise"},
        species_source="manual",
    ) == ("Parus_major", "Kohlmeise", "manual")
