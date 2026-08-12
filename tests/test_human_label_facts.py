"""Real-SQLite contracts for canonical human label facts."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from config import get_config
from core.human_label_core import (
    BBox,
    HumanAnswer,
    HumanLabelError,
    LabelProvenance,
    append_fact,
    ensure_image_subject,
    ensure_object_subject,
    get_or_create_labeling_installation_id,
    object_training_readiness,
    record_human_answer,
    retract_bbox_quality,
)
from utils.db import connection as db_connection
from utils.restore import _merge_human_label_tables
from utils.settings import load_settings_yaml


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[sqlite3.Connection]:
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    connection = db_connection.get_connection()
    yield connection
    connection.close()


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> dict[str, int | str]:
    filename = "20260810_081500_000001.jpg"
    conn.execute(
        """
        INSERT INTO images (
            filename, timestamp, detector_model_id, classifier_model_id
        ) VALUES (?, ?, ?, ?)
        """,
        (filename, "2026-08-10T08:15:00+00:00", "det-image-v1", "cls-image-v2"),
    )
    cursor = conn.execute(
        """
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            od_class_name, od_model_id, raw_species_name,
            detector_model_version, classifier_model_version,
            frame_width, frame_height, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            0.1,
            0.2,
            0.3,
            0.4,
            "bird",
            "det-row-v1",
            "Parus_major",
            "det-row-v1",
            "cls-row-v2",
            1920,
            1080,
            "2026-08-10T08:15:01+00:00",
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
        source_ref="review:item-1",
        created_at="2026-08-10T08:20:00+00:00",
    )


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def test_fresh_database_has_fact_schema_and_current_view(
    conn: sqlite3.Connection,
) -> None:
    assert {"label_subjects", "human_label_facts"} <= _table_names(conn)
    view = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' "
        "AND name = 'current_human_label_facts'"
    ).fetchone()
    assert view is not None

    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert {
        "idx_label_subjects_image_unique",
        "idx_label_subjects_detection_unique",
        "idx_human_label_facts_root_unique",
        "idx_human_label_facts_successor_unique",
    } <= indexes


def test_existing_legacy_rows_upgrade_without_backfill_or_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    db_path = tmp_path / "images.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            uri TEXT,
            config_json TEXT,
            active INTEGER DEFAULT 1
        );
        INSERT INTO sources(name, type) VALUES ('Default Camera', 'ipcam');
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            source_id INTEGER,
            content_hash TEXT
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            created_at TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            manual_bbox_review TEXT
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            rank INTEGER DEFAULT 1
        );
        CREATE TABLE training_exports (
            detection_id INTEGER PRIMARY KEY,
            export_status TEXT NOT NULL
        );
        INSERT INTO images (
            filename, timestamp, review_status, source_id, content_hash
        ) VALUES ('legacy.jpg', 'old-time', 'no_bird', 1, 'legacy-hash');
        INSERT INTO detections (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            created_at, manual_species_override, species_source,
            manual_bbox_review
        ) VALUES (
            'legacy.jpg', 0.11, 0.22, 0.33, 0.44,
            'old-created', 'Parus_major', 'manual', 'correct'
        );
        INSERT INTO classifications(detection_id, cls_class_name, rank)
        VALUES (1, 'Parus_major', 1);
        INSERT INTO training_exports VALUES (1, 'exported');
        """
    )
    legacy.commit()
    legacy.close()

    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    upgraded = db_connection.get_connection()
    try:
        image = upgraded.execute(
            "SELECT timestamp, review_status FROM images WHERE filename='legacy.jpg'"
        ).fetchone()
        detection = upgraded.execute(
            """
            SELECT bbox_x, bbox_y, bbox_w, bbox_h,
                   manual_species_override, species_source, manual_bbox_review
            FROM detections WHERE detection_id=1
            """
        ).fetchone()
        assert tuple(image) == ("old-time", "no_bird")
        assert tuple(detection) == (
            0.11,
            0.22,
            0.33,
            0.44,
            "Parus_major",
            "manual",
            "correct",
        )
        assert upgraded.execute("SELECT COUNT(*) FROM label_subjects").fetchone()[0] == 0
        assert upgraded.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 0
        assert upgraded.execute(
            "SELECT export_status FROM training_exports WHERE detection_id=1"
        ).fetchone()[0] == "exported"
    finally:
        upgraded.close()


def test_object_subject_snapshots_original_proposal_once(
    conn: sqlite3.Connection, seeded: dict[str, int | str]
) -> None:
    subject_id = ensure_object_subject(conn, int(seeded["detection_id"]))
    assert ensure_object_subject(conn, int(seeded["detection_id"])) == subject_id

    subject = conn.execute(
        """
        SELECT scope, image_filename, detection_id,
               proposal_bbox_x, proposal_bbox_y, proposal_bbox_w, proposal_bbox_h,
               proposal_coordinate_space, proposal_species_key,
               detector_model_version, classifier_model_version
        FROM label_subjects WHERE subject_id = ?
        """,
        (subject_id,),
    ).fetchone()
    assert tuple(subject) == (
        "object",
        seeded["filename"],
        seeded["detection_id"],
        0.1,
        0.2,
        0.3,
        0.4,
        "frame_fraction_xywh_v1",
        "Parus_major",
        "det-row-v1",
        "cls-row-v2",
    )


def test_object_training_readiness_keeps_od_and_cls_independent() -> None:
    facts = [
        {"scope": "object", "fact_type": "bird_presence", "answer_value": "present"},
        {"scope": "object", "fact_type": "bbox_quality", "answer_value": "unsuitable"},
        {
            "scope": "object",
            "fact_type": "species_identity",
            "answer_value": "corrected",
            "species_key": "Parus_major",
        },
    ]

    readiness = object_training_readiness(facts)

    assert readiness["od"] == {"ready": False, "reasons": ["bbox_unsuitable"]}
    assert readiness["cls"] == {"ready": True, "reasons": []}


def test_object_training_readiness_reports_unknown_axes() -> None:
    readiness = object_training_readiness([])

    assert readiness["od"] == {
        "ready": False,
        "reasons": ["object_bird_presence_unknown", "bbox_quality_unknown"],
    }
    assert readiness["cls"] == {
        "ready": False,
        "reasons": ["object_bird_presence_unknown", "species_identity_unknown"],
    }


def test_independent_image_and_object_facts_coexist(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    image_subject = ensure_image_subject(conn, str(seeded["filename"]))
    object_subject = ensure_object_subject(conn, int(seeded["detection_id"]))

    append_fact(
        conn,
        subject_id=image_subject,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )
    append_fact(
        conn,
        subject_id=object_subject,
        fact_type="bbox_quality",
        answer_value="unsuitable",
        provenance=provenance,
    )
    conn.commit()

    current = conn.execute(
        """
        SELECT scope, fact_type, answer_value
        FROM current_human_label_facts
        ORDER BY scope, fact_type
        """
    ).fetchall()
    assert [tuple(row) for row in current] == [
        ("image", "bird_presence", "present"),
        ("object", "bbox_quality", "unsuitable"),
    ]


def test_supersession_and_retraction_preserve_history(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    subject_id = ensure_object_subject(conn, int(seeded["detection_id"]))
    first = append_fact(
        conn,
        subject_id=subject_id,
        fact_type="species_identity",
        answer_value="confirmed",
        species_key="Parus_major",
        provenance=provenance,
    )
    second = append_fact(
        conn,
        subject_id=subject_id,
        fact_type="species_identity",
        answer_value="corrected",
        species_key="Cyanistes_caeruleus",
        provenance=provenance,
    )
    retracted = append_fact(
        conn,
        subject_id=subject_id,
        fact_type="species_identity",
        answer_value=None,
        assertion_state="retracted",
        provenance=provenance,
    )
    conn.commit()

    rows = conn.execute(
        """
        SELECT fact_id, assertion_state, answer_value, supersedes_fact_id
        FROM human_label_facts ORDER BY fact_id
        """
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (first, "asserted", "confirmed", None),
        (second, "asserted", "corrected", first),
        (retracted, "retracted", None, second),
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM current_human_label_facts"
    ).fetchone()[0] == 0


def test_bbox_quality_retraction_clears_fact_and_legacy_projection(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    asserted_id = record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            bbox_quality="suitable",
        ),
        provenance,
    )[0]

    retraction_id = retract_bbox_quality(
        conn,
        image_filename=str(seeded["filename"]),
        detection_id=int(seeded["detection_id"]),
        provenance=provenance,
    )

    assert retraction_id is not None
    current_count = conn.execute(
        """
        SELECT COUNT(*) FROM current_human_label_facts
        WHERE detection_id = ? AND fact_type = 'bbox_quality'
        """,
        (int(seeded["detection_id"]),),
    ).fetchone()[0]
    history = conn.execute(
        """
        SELECT fact_id, assertion_state, supersedes_fact_id
        FROM human_label_facts
        WHERE fact_type = 'bbox_quality'
        ORDER BY fact_id
        """
    ).fetchall()
    projection = conn.execute(
        """
        SELECT manual_bbox_review, bbox_reviewed_at
        FROM detections WHERE detection_id = ?
        """,
        (int(seeded["detection_id"]),),
    ).fetchone()
    assert current_count == 0
    assert [tuple(row) for row in history] == [
        (asserted_id, "asserted", None),
        (retraction_id, "retracted", asserted_id),
    ]
    assert tuple(projection) == (None, None)


def test_corrected_bbox_does_not_modify_proposal_geometry(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    subject_id = ensure_object_subject(conn, int(seeded["detection_id"]))
    append_fact(
        conn,
        subject_id=subject_id,
        fact_type="bbox_correction",
        answer_value="provided",
        bbox=BBox(x=0.15, y=0.25, w=0.2, h=0.3),
        provenance=provenance,
    )
    conn.commit()

    proposal = conn.execute(
        "SELECT proposal_bbox_x, proposal_bbox_y, proposal_bbox_w, proposal_bbox_h "
        "FROM label_subjects WHERE subject_id = ?",
        (subject_id,),
    ).fetchone()
    detection = conn.execute(
        "SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM detections WHERE detection_id = ?",
        (seeded["detection_id"],),
    ).fetchone()
    assert tuple(proposal) == (0.1, 0.2, 0.3, 0.4)
    assert tuple(detection) == (0.1, 0.2, 0.3, 0.4)


@pytest.mark.parametrize(
    ("scope", "fact_type", "answer_value"),
    [
        ("image", "bbox_quality", "suitable"),
        ("object", "detector_miss", "reported"),
    ],
)
def test_invalid_scope_type_combinations_fail_at_sqlite_layer(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
    scope: str,
    fact_type: str,
    answer_value: str,
) -> None:
    subject_id = (
        ensure_image_subject(conn, str(seeded["filename"]))
        if scope == "image"
        else ensure_object_subject(conn, int(seeded["detection_id"]))
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO human_label_facts (
                subject_id, fact_type, assertion_state, answer_value,
                context, source_kind, installation_id, app_version, created_at
            ) VALUES (?, ?, 'asserted', ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id,
                fact_type,
                answer_value,
                provenance.context,
                provenance.source_kind,
                provenance.installation_id,
                provenance.app_version,
                provenance.created_at,
            ),
        )


def test_presence_conflicts_require_explicit_resolution(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    image_subject = ensure_image_subject(conn, str(seeded["filename"]))
    object_subject = ensure_object_subject(conn, int(seeded["detection_id"]))
    append_fact(
        conn,
        subject_id=object_subject,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )

    with pytest.raises(HumanLabelError, match="conflict"):
        append_fact(
            conn,
            subject_id=image_subject,
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance,
        )


def test_hard_delete_detection_keeps_image_facts(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    image_subject = ensure_image_subject(conn, str(seeded["filename"]))
    object_subject = ensure_object_subject(conn, int(seeded["detection_id"]))
    append_fact(
        conn,
        subject_id=image_subject,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )
    append_fact(
        conn,
        subject_id=object_subject,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )
    conn.commit()

    conn.execute(
        "DELETE FROM detections WHERE detection_id = ?", (seeded["detection_id"],)
    )
    conn.commit()

    rows = conn.execute(
        "SELECT scope FROM current_human_label_facts"
    ).fetchall()
    assert [row[0] for row in rows] == ["image"]


def test_provenance_round_trips_with_subject_model_versions(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    subject_id = ensure_object_subject(conn, int(seeded["detection_id"]))
    append_fact(
        conn,
        subject_id=subject_id,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )
    conn.commit()

    row = conn.execute(
        """
        SELECT installation_id, app_version, context, source_kind, source_ref,
               detector_model_version, classifier_model_version
        FROM current_human_label_facts
        """
    ).fetchone()
    assert tuple(row) == (
        provenance.installation_id,
        provenance.app_version,
        provenance.context,
        provenance.source_kind,
        provenance.source_ref,
        "det-row-v1",
        "cls-row-v2",
    )


def test_provenance_rejects_unversioned_source_kind(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
) -> None:
    invalid = LabelProvenance(
        installation_id="0123456789abcdef0123456789abcdef",
        app_version="0.6.0",
        context="normal_correction",
        source_kind="ad_hoc_script",
    )

    with pytest.raises(HumanLabelError, match="invalid source_kind"):
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename=str(seeded["filename"]),
                image_bird_presence="present",
            ),
            invalid,
        )


def test_labeling_identity_is_stable_and_independent_from_telemetry(tmp_path) -> None:
    first = get_or_create_labeling_installation_id(str(tmp_path))
    second = get_or_create_labeling_installation_id(str(tmp_path))

    assert first == second
    assert len(first) == 32
    assert all(character in "0123456789abcdef" for character in first)
    settings = load_settings_yaml(str(tmp_path))
    assert settings["labeling_installation_id"] == first
    assert "telemetry_installation_id" not in settings


def test_merge_restore_remaps_subject_and_supersession_ids(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
    tmp_path,
) -> None:
    backup_path = tmp_path / "backup-labels.db"
    backup = sqlite3.connect(backup_path)
    backup.execute("PRAGMA foreign_keys=ON")
    db_connection._init_schema(backup)
    backup.execute(
        """
        INSERT INTO images(filename, timestamp)
        VALUES ('backup-name.jpg', '2026-08-09T10:00:00+00:00')
        """
    )
    old_detection_id = int(
        backup.execute(
            """
            INSERT INTO detections(
                image_filename, bbox_x, bbox_y, bbox_w, bbox_h, created_at
            ) VALUES ('backup-name.jpg', 0.1, 0.2, 0.3, 0.4, 'old')
            """
        ).lastrowid
    )
    old_subject_id = ensure_object_subject(backup, old_detection_id)
    append_fact(
        backup,
        subject_id=old_subject_id,
        fact_type="bbox_quality",
        answer_value="unsuitable",
        provenance=provenance,
    )
    append_fact(
        backup,
        subject_id=old_subject_id,
        fact_type="bbox_quality",
        answer_value="suitable",
        provenance=provenance,
    )
    backup.commit()
    backup.close()

    conn.execute("ATTACH DATABASE ? AS backup", (str(backup_path),))
    result: dict[str, list] = {"warnings": [], "conflicts": []}
    try:
        imported = _merge_human_label_tables(
            conn,
            image_mapping={"backup-name.jpg": str(seeded["filename"])},
            detection_mapping={old_detection_id: int(seeded["detection_id"])},
            result=result,
        )
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE backup")

    assert imported == {"subjects": 1, "facts": 2, "conflicts": 0}
    assert result["conflicts"] == []
    rows = conn.execute(
        """
        SELECT fact_id, answer_value, supersedes_fact_id
        FROM human_label_facts
        ORDER BY fact_id
        """
    ).fetchall()
    assert rows[0][1] == "unsuitable"
    assert rows[0][2] is None
    assert rows[1][1] == "suitable"
    assert rows[1][2] == rows[0][0]
    assert conn.execute(
        "SELECT answer_value FROM current_human_label_facts"
    ).fetchone()[0] == "suitable"


def test_merge_restore_reports_fact_axis_conflict_without_guessing(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
    tmp_path,
) -> None:
    destination_subject = ensure_object_subject(conn, int(seeded["detection_id"]))
    append_fact(
        conn,
        subject_id=destination_subject,
        fact_type="bird_presence",
        answer_value="present",
        provenance=provenance,
    )
    conn.commit()

    backup_path = tmp_path / "backup-conflict.db"
    backup = sqlite3.connect(backup_path)
    backup.execute("PRAGMA foreign_keys=ON")
    db_connection._init_schema(backup)
    backup.execute("INSERT INTO images(filename) VALUES ('source.jpg')")
    old_detection_id = int(
        backup.execute(
            """
            INSERT INTO detections(
                image_filename, bbox_x, bbox_y, bbox_w, bbox_h, created_at
            ) VALUES ('source.jpg', 0.1, 0.2, 0.3, 0.4, 'old')
            """
        ).lastrowid
    )
    old_subject = ensure_object_subject(backup, old_detection_id)
    append_fact(
        backup,
        subject_id=old_subject,
        fact_type="bird_presence",
        answer_value="absent",
        provenance=provenance,
    )
    backup.commit()
    backup.close()

    conn.execute("ATTACH DATABASE ? AS backup", (str(backup_path),))
    result: dict[str, list] = {"warnings": [], "conflicts": []}
    try:
        imported = _merge_human_label_tables(
            conn,
            image_mapping={"source.jpg": str(seeded["filename"])},
            detection_mapping={old_detection_id: int(seeded["detection_id"])},
            result=result,
        )
    finally:
        conn.execute("DETACH DATABASE backup")

    assert imported == {"subjects": 0, "facts": 0, "conflicts": 1}
    assert result["conflicts"][0]["type"] == "human_label_fact_axis"
    assert conn.execute(
        "SELECT answer_value FROM current_human_label_facts"
    ).fetchone()[0] == "present"


def test_one_human_answer_writes_independent_facts_and_legacy_projection(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    fact_ids = record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            image_bird_presence="present",
            object_bird_presence="present",
            bbox_quality="suitable",
            species_identity="corrected",
            species_key="Cyanistes_caeruleus",
        ),
        provenance,
    )
    conn.commit()

    assert len(fact_ids) == 4
    facts = conn.execute(
        """
        SELECT scope, fact_type, answer_value, species_key
        FROM current_human_label_facts
        ORDER BY scope, fact_type
        """
    ).fetchall()
    assert [tuple(row) for row in facts] == [
        ("image", "bird_presence", "present", None),
        ("object", "bbox_quality", "suitable", None),
        ("object", "bird_presence", "present", None),
        ("object", "species_identity", "corrected", "Cyanistes_caeruleus"),
    ]
    image_status = conn.execute(
        "SELECT review_status FROM images WHERE filename = ?", (seeded["filename"],)
    ).fetchone()[0]
    legacy = conn.execute(
        """
        SELECT status, manual_bbox_review, manual_species_override,
               species_source, decision_state, decision_level
        FROM detections WHERE detection_id = ?
        """,
        (seeded["detection_id"],),
    ).fetchone()
    assert image_status == "confirmed_bird"
    assert tuple(legacy) == (
        "active",
        "correct",
        "Cyanistes_caeruleus",
        "manual",
        "confirmed",
        "species",
    )


def test_object_rejection_projection_does_not_mark_whole_image_no_bird(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            detection_id=int(seeded["detection_id"]),
            object_bird_presence="absent",
        ),
        provenance,
    )
    conn.commit()

    assert conn.execute(
        "SELECT status FROM detections WHERE detection_id = ?",
        (seeded["detection_id"],),
    ).fetchone()[0] == "rejected"
    assert conn.execute(
        "SELECT review_status FROM images WHERE filename = ?",
        (seeded["filename"],),
    ).fetchone()[0] == "untagged"


def test_explicit_image_no_bird_projects_without_object_answer(
    conn: sqlite3.Connection,
    seeded: dict[str, int | str],
    provenance: LabelProvenance,
) -> None:
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename=str(seeded["filename"]),
            image_bird_presence="absent",
        ),
        provenance,
    )
    conn.commit()

    assert conn.execute(
        "SELECT review_status FROM images WHERE filename = ?",
        (seeded["filename"],),
    ).fetchone()[0] == "no_bird"
    assert conn.execute(
        "SELECT COUNT(*) FROM current_human_label_facts WHERE scope='object'"
    ).fetchone()[0] == 0
