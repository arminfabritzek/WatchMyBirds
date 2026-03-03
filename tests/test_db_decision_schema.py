import json

import pytest

from utils.db.connection import closing_connection
from utils.db.detections import insert_detection


# Reset global schema initialized flag for tests in this module so they can have a fresh schema.
@pytest.fixture(autouse=True)
def wipe_schema_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config
    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_additive_decision_columns_created(tmp_path):
    with closing_connection() as conn:
        cur = conn.execute("PRAGMA table_info(detections);")
        cols = {row["name"]: row["type"] for row in cur.fetchall()}

        # Check that the new decision fields are present
        assert "decision_state" in cols
        assert "bbox_quality" in cols
        assert "unknown_score" in cols
        assert "decision_reasons" in cols
        assert "policy_version" in cols


def test_insert_detection_with_decision_fields(tmp_path):
    with closing_connection() as conn:
        # Need an image to ensure foreign key on image_filename is satisfied
        conn.execute(
            "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
            ("test.jpg", "20260303_120000"),
        )

        # Build complete row with decision fields
        det_row = {
            "image_filename": "test.jpg",
            "bbox_x": 0.1,
            "bbox_y": 0.1,
            "bbox_w": 0.5,
            "bbox_h": 0.5,
            "od_class_name": "bird",
            "od_confidence": 0.9,
            "decision_state": "confirmed",
            "bbox_quality": 0.95,
            "unknown_score": 0.05,
            "decision_reasons": json.dumps(["HIGH_SPECIES_CONF", "GOOD_BBOX"]),
            "policy_version": "v1",
        }

        det_id = insert_detection(conn, det_row)

        row = conn.execute(
            "SELECT * FROM detections WHERE detection_id = ?", (det_id,)
        ).fetchone()

        assert row is not None
        assert row["decision_state"] == "confirmed"
        assert row["bbox_quality"] == 0.95
        assert row["unknown_score"] == 0.05
        assert "HIGH_SPECIES_CONF" in row["decision_reasons"]
        assert row["policy_version"] == "v1"


def test_insert_detection_without_decision_fields_backcompat(tmp_path):
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
            ("test2.jpg", "20260303_120000"),
        )

        # Build row without the new fields as legacy code might do
        det_row = {
            "image_filename": "test2.jpg",
            "bbox_x": 0.2,
            "bbox_y": 0.2,
            "bbox_w": 0.4,
            "bbox_h": 0.4,
            "od_class_name": "bird",
            "od_confidence": 0.8,
        }

        det_id = insert_detection(conn, det_row)

        row = conn.execute(
            "SELECT * FROM detections WHERE detection_id = ?", (det_id,)
        ).fetchone()

        assert row is not None
        # Must be NULL by default when not supplied, avoiding integrity errors
        assert row["decision_state"] is None
        assert row["bbox_quality"] is None
        assert row["unknown_score"] is None
        assert row["decision_reasons"] is None
        assert row["policy_version"] is None
