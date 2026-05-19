"""Tests for the reject-audit log (A1a, plan 2026-05-19).

The audit log replaces full persistence (image + crop + detection row)
for detections the classifier routes to ``decision_level='reject'``.
These tests pin the SQL-layer contract: schema acceptance, idempotent
inserts, malformed inputs handled, indexable cluster columns.

The detection-manager wiring (lazy save_image, audit-vs-keeper branch)
is tested at the unit-mock layer in
``tests/test_detection_manager_reject_audit.py``.
"""

from __future__ import annotations

import sqlite3

from utils.db.reject_audit import insert_reject_audit


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE reject_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            frame_timestamp TEXT NOT NULL,
            frame_width INTEGER,
            frame_height INTEGER,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            raw_species_name TEXT,
            top1_prob REAL,
            decision_state TEXT,
            decision_reasons TEXT,
            detector_model_id TEXT,
            classifier_model_id TEXT
        );
        CREATE INDEX idx_reject_audit_created_at
            ON reject_audit(created_at DESC);
        CREATE INDEX idx_reject_audit_raw_species
            ON reject_audit(raw_species_name);
        CREATE INDEX idx_reject_audit_bbox
            ON reject_audit(bbox_x, bbox_y);
        """
    )
    return conn


def _full_row(**overrides) -> dict:
    """Realistic Troglodytes-on-tree-branch FP payload."""
    base = {
        "frame_timestamp": "20260519_211700_273402",
        "frame_width": 2560,
        "frame_height": 1920,
        "bbox_x": 0.57,
        "bbox_y": 0.15,
        "bbox_w": 0.10,
        "bbox_h": 0.30,
        "od_class_name": "bird",
        "od_confidence": 0.43,
        "raw_species_name": "Troglodytes_troglodytes",
        "top1_prob": 0.22,
        "decision_state": "uncertain",
        "decision_reasons": '["below_review_threshold"]',
        "detector_model_id": "20260513_yolox_s_locator_640_mosaic0p75_v2_coco",
        "classifier_model_id": "20260427_143835",
    }
    base.update(overrides)
    return base


def test_insert_returns_positive_audit_id():
    conn = _make_conn()
    audit_id = insert_reject_audit(conn, _full_row())
    assert audit_id > 0


def test_insert_persists_all_fields():
    conn = _make_conn()
    audit_id = insert_reject_audit(conn, _full_row())
    row = dict(
        conn.execute(
            "SELECT * FROM reject_audit WHERE audit_id = ?", (audit_id,)
        ).fetchone()
    )
    assert row["frame_timestamp"] == "20260519_211700_273402"
    assert row["bbox_x"] == 0.57
    assert row["bbox_y"] == 0.15
    assert row["raw_species_name"] == "Troglodytes_troglodytes"
    assert row["od_confidence"] == 0.43
    assert row["top1_prob"] == 0.22
    assert row["decision_state"] == "uncertain"


def test_insert_accepts_partial_payload():
    """Non-bird path (cls_result is None) sends a sparse row."""
    conn = _make_conn()
    audit_id = insert_reject_audit(
        conn,
        {
            "frame_timestamp": "20260519_211700_273402",
            "frame_width": 2560,
            "frame_height": 1920,
            "bbox_x": 0.0,
            "bbox_y": 0.5,
            "bbox_w": 0.1,
            "bbox_h": 0.46,
            "od_class_name": "hedgehog",
            "od_confidence": 0.41,
            # raw_species_name / top1_prob omitted (non-bird → no cls)
        },
    )
    row = dict(
        conn.execute(
            "SELECT * FROM reject_audit WHERE audit_id = ?", (audit_id,)
        ).fetchone()
    )
    assert row["od_class_name"] == "hedgehog"
    assert row["raw_species_name"] is None
    assert row["top1_prob"] is None


def test_insert_assigns_created_at_when_missing():
    """Default created_at = NOW(UTC) ISO so cluster queries by time work."""
    conn = _make_conn()
    audit_id = insert_reject_audit(conn, _full_row())
    row = conn.execute(
        "SELECT created_at FROM reject_audit WHERE audit_id = ?",
        (audit_id,),
    ).fetchone()
    assert row["created_at"]
    # ISO 8601 with timezone marker (UTC + or trailing Z)
    assert "T" in row["created_at"]
    assert "+" in row["created_at"] or row["created_at"].endswith("Z")


def test_insert_preserves_provided_created_at():
    """When caller supplies created_at, the insert must not overwrite it."""
    conn = _make_conn()
    audit_id = insert_reject_audit(
        conn, _full_row(created_at="2026-05-19T19:00:00+00:00")
    )
    row = conn.execute(
        "SELECT created_at FROM reject_audit WHERE audit_id = ?",
        (audit_id,),
    ).fetchone()
    assert row["created_at"] == "2026-05-19T19:00:00+00:00"


def test_cluster_query_by_bbox_position():
    """The static-FP cluster query (the whole point of this table)."""
    conn = _make_conn()
    # 50 rejects at the same bbox = static background object
    for _ in range(50):
        insert_reject_audit(conn, _full_row())
    # 3 rejects at a different position = unrelated drift
    for _ in range(3):
        insert_reject_audit(
            conn,
            _full_row(bbox_x=0.20, bbox_y=0.80),
        )
    clusters = conn.execute(
        """
        SELECT bbox_x, bbox_y, COUNT(*) AS n
        FROM reject_audit
        GROUP BY bbox_x, bbox_y
        ORDER BY n DESC;
        """
    ).fetchall()
    assert clusters[0]["bbox_x"] == 0.57
    assert clusters[0]["n"] == 50
    assert clusters[1]["n"] == 3


def test_many_inserts_are_independent():
    """50 inserts should produce 50 distinct audit_ids (no UNIQUE coupling)."""
    conn = _make_conn()
    ids = [insert_reject_audit(conn, _full_row()) for _ in range(50)]
    assert len(set(ids)) == 50


def test_decision_reasons_stored_as_text():
    """decision_reasons is the JSON blob from the scoring pipeline.

    Stored as TEXT, not deserialised — the audit row is a snapshot, the
    consumer parses the JSON when it needs structured access.
    """
    conn = _make_conn()
    audit_id = insert_reject_audit(
        conn,
        _full_row(decision_reasons='["below_review_threshold","top1_lt_review"]'),
    )
    row = conn.execute(
        "SELECT decision_reasons FROM reject_audit WHERE audit_id = ?",
        (audit_id,),
    ).fetchone()
    assert row["decision_reasons"] == (
        '["below_review_threshold","top1_lt_review"]'
    )
