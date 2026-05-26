"""Schema + write-path tests for PTZ context on images rows.

Verifies the additive columns added for the gallery thumbnail bias
: ptz_origin,
ptz_preset_token, ptz_zone, ptz_state, ptz_camera_id, plus the v2-
reserved coordinate slots ptz_pan, ptz_tilt, ptz_zoom, ptz_position_at.
"""

import pytest

from utils.db.connection import closing_connection
from utils.db.images import insert_image


@pytest.fixture(autouse=True)
def wipe_schema_cache(monkeypatch, tmp_path):
    """Use a temp OUTPUT_DIR so each test gets a fresh schema."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_additive_ptz_columns_created(tmp_path):
    with closing_connection() as conn:
        cur = conn.execute("PRAGMA table_info(images);")
        cols = {row["name"]: row["type"] for row in cur.fetchall()}

        assert "ptz_origin" in cols
        assert "ptz_preset_token" in cols
        assert "ptz_zone" in cols
        assert "ptz_state" in cols
        assert "ptz_camera_id" in cols
        # v2-reserved slots: present in v1 schema, populated NULL.
        assert "ptz_pan" in cols
        assert "ptz_tilt" in cols
        assert "ptz_zoom" in cols
        assert "ptz_position_at" in cols


def test_ptz_origin_partial_index_exists(tmp_path):
    with closing_connection() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_images_ptz_origin';"
        )
        assert cur.fetchone() is not None, "ptz_origin partial index missing"


def test_insert_image_with_ptz_preset_writes_columns(tmp_path):
    with closing_connection() as conn:
        insert_image(
            conn,
            {
                "filename": "20260515_120000_000000.jpg",
                "timestamp": "20260515_120000_000000",
                "detector_model_id": "yolox-tiny",
                "classifier_model_id": "classifier-v1",
                "source_id": 1,
                "ptz_origin": "preset",
                "ptz_preset_token": "Preset002",
                "ptz_zone": "center",
                "ptz_state": "tracking",
                "ptz_camera_id": 0,
            },
        )
        row = conn.execute(
            "SELECT ptz_origin, ptz_preset_token, ptz_zone, ptz_state, "
            "ptz_camera_id, ptz_pan, ptz_tilt, ptz_zoom, ptz_position_at "
            "FROM images WHERE filename = ?;",
            ("20260515_120000_000000.jpg",),
        ).fetchone()

        assert row["ptz_origin"] == "preset"
        assert row["ptz_preset_token"] == "Preset002"
        assert row["ptz_zone"] == "center"
        assert row["ptz_state"] == "tracking"
        assert row["ptz_camera_id"] == 0
        # v2-reserved slots stay NULL in v1.
        assert row["ptz_pan"] is None
        assert row["ptz_tilt"] is None
        assert row["ptz_zoom"] is None
        assert row["ptz_position_at"] is None


def test_insert_image_without_ptz_keys_is_back_compatible(tmp_path):
    """Legacy callers (ingest pipelines, fixtures) pass no PTZ keys.

    The row must insert cleanly with all PTZ columns NULL — meaning
    "we do not know the PTZ state for this frame".
    """
    with closing_connection() as conn:
        insert_image(
            conn,
            {
                "filename": "legacy.jpg",
                "timestamp": "20260515_120000_000000",
                "source_id": 1,
            },
        )
        row = conn.execute(
            "SELECT ptz_origin, ptz_preset_token, ptz_zone, ptz_state, "
            "ptz_camera_id FROM images WHERE filename = 'legacy.jpg';"
        ).fetchone()

        assert row["ptz_origin"] is None
        assert row["ptz_preset_token"] is None
        assert row["ptz_zone"] is None
        assert row["ptz_state"] is None
        assert row["ptz_camera_id"] is None


def test_insert_image_overview_origin_round_trips(tmp_path):
    """Frames at overview rest should be discoverable as 'overview'."""
    with closing_connection() as conn:
        insert_image(
            conn,
            {
                "filename": "overview.jpg",
                "timestamp": "20260515_120000_000000",
                "source_id": 1,
                "ptz_origin": "overview",
                "ptz_state": "overview",
                "ptz_preset_token": "Preset005",
                "ptz_camera_id": 0,
            },
        )
        row = conn.execute(
            "SELECT ptz_origin, ptz_state FROM images WHERE filename = 'overview.jpg';"
        ).fetchone()

        assert row["ptz_origin"] == "overview"
        assert row["ptz_state"] == "overview"
