"""Regression tests for utils.restore._merge_database.

Covers GitHub issue #139's restore-merge findings:
- same filename, different content across two collections must not
  attach the incoming image's detections/labels to the unrelated
  pre-existing image that happens to share the name;
- detections and human-label facts follow the correct image identity
  after merge;
- repeated imports of the same backup do not duplicate rows;
- a DB-level failure raises instead of being reported as success.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from config import get_config
from tests.conftest import reset_config_in_place
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image


def _make_db(tmp_path: Path, name: str) -> Path:
    """Build a fresh, schema-initialized SQLite DB at ``tmp_path/name``."""
    output_dir = tmp_path / name
    output_dir.mkdir()
    db_connection._schema_initialized_paths.clear()
    with patch.dict(get_config(), {"OUTPUT_DIR": str(output_dir)}):
        conn = db_connection.get_connection()
        conn.close()
    return output_dir / "images.db"


def _seed_image_with_detection(
    db_path: Path,
    *,
    filename: str,
    content_hash: str,
    species: str = "Parus_major",
) -> int:
    """Insert one image + detection + classification into ``db_path``."""
    output_dir = db_path.parent
    db_connection._schema_initialized_paths.clear()
    with patch.dict(get_config(), {"OUTPUT_DIR": str(output_dir)}):
        conn = db_connection.get_connection()
        insert_image(
            conn,
            {
                "filename": filename,
                "timestamp": "2026-08-20T10:00:00+00:00",
                "content_hash": content_hash,
            },
        )
        detection_id = insert_detection(
            conn,
            {
                "image_filename": filename,
                "bbox_x": 0.1,
                "bbox_y": 0.1,
                "bbox_w": 0.2,
                "bbox_h": 0.2,
                "od_class_name": "bird",
                "od_confidence": 0.9,
                "od_model_id": "yolo-test",
                "created_at": "2026-08-20T10:00:01+00:00",
                "score": 0.95,
                "raw_species_name": species,
                "thumbnail_path": filename.replace(".jpg", "_crop_1.webp"),
            },
        )
        insert_classification(
            conn,
            {
                "detection_id": detection_id,
                "cls_class_name": species,
                "cls_confidence": 0.95,
                "cls_model_id": "cls-test",
                "rank": 1,
                "created_at": "2026-08-20T10:00:02+00:00",
            },
        )
        conn.commit()
        conn.close()
    return detection_id


@pytest.fixture
def live_db(tmp_path, monkeypatch):
    """Points the process config at a fresh, schema-initialized live DB."""
    output_dir = tmp_path / "live"
    output_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    reset_config_in_place()
    db_connection._schema_initialized_paths.clear()
    conn = db_connection.get_connection()
    conn.close()
    yield output_dir / "images.db"


def test_same_filename_different_content_does_not_cross_attach(
    live_db, tmp_path
) -> None:
    """Two collections' same-named, different-content image must not merge.

    Stage 5 (file import) would rename the incoming file on disk; this
    test asserts the DB merge honors that rename via filename_rename_map
    instead of attaching the incoming detection to the pre-existing,
    unrelated image row that happens to share the original filename.
    """
    from utils.restore import _merge_database

    shared_name = "20260820_100000_bird.jpg"

    live_detection_id = _seed_image_with_detection(
        live_db, filename=shared_name, content_hash="hash-live-content"
    )

    backup_db = _make_db(tmp_path, "backup")
    backup_detection_id = _seed_image_with_detection(
        backup_db,
        filename=shared_name,
        content_hash="hash-different-content",
        species="Cyanistes_caeruleus",
    )

    renamed_name = "20260820_100000_bird__conflict_abcd1234.jpg"
    result = _merge_database(backup_db, filename_rename_map={shared_name: renamed_name})

    assert result["stats"]["images_imported"] == 1
    assert result["stats"]["detections_imported"] == 1

    conn = sqlite3.connect(live_db)
    conn.row_factory = sqlite3.Row

    live_row = conn.execute(
        "SELECT * FROM images WHERE filename = ?", (shared_name,)
    ).fetchone()
    assert live_row["content_hash"] == "hash-live-content"

    imported_row = conn.execute(
        "SELECT * FROM images WHERE filename = ?", (renamed_name,)
    ).fetchone()
    assert imported_row is not None
    assert imported_row["content_hash"] == "hash-different-content"

    live_detections = conn.execute(
        "SELECT detection_id, image_filename FROM detections WHERE image_filename = ?",
        (shared_name,),
    ).fetchall()
    assert [row["detection_id"] for row in live_detections] == [live_detection_id]

    imported_detections = conn.execute(
        "SELECT detection_id, image_filename, raw_species_name "
        "FROM detections WHERE image_filename = ?",
        (renamed_name,),
    ).fetchall()
    assert len(imported_detections) == 1
    assert imported_detections[0]["raw_species_name"] == "Cyanistes_caeruleus"
    assert imported_detections[0]["detection_id"] != live_detection_id
    assert imported_detections[0]["detection_id"] != backup_detection_id

    conn.close()


def test_repeated_import_does_not_duplicate(live_db, tmp_path) -> None:
    from utils.restore import _merge_database

    filename = "20260820_110000_bird.jpg"
    backup_db = _make_db(tmp_path, "backup2")
    _seed_image_with_detection(backup_db, filename=filename, content_hash="hash-repeat")

    first = _merge_database(backup_db, filename_rename_map={})
    second = _merge_database(backup_db, filename_rename_map={})

    assert first["stats"]["images_imported"] == 1
    assert second["stats"]["images_imported"] == 0
    assert second["stats"]["images_skipped"] == 1

    conn = sqlite3.connect(live_db)
    image_count = conn.execute(
        "SELECT COUNT(*) FROM images WHERE filename = ?", (filename,)
    ).fetchone()[0]
    detection_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE image_filename = ?", (filename,)
    ).fetchone()[0]
    conn.close()

    assert image_count == 1
    assert detection_count == 1


def test_database_failure_raises_instead_of_reporting_success(
    live_db, tmp_path
) -> None:
    """A merge that fails mid-way must raise, not report a false success.

    Injects a failure in the human-label merge step, which runs inside
    the same transaction after images/detections have been inserted.
    Asserts the exception propagates (previously it was caught,
    appended to result["warnings"], and the function returned normally
    with `completed=True`) and that the preceding writes in this
    attempt are rolled back rather than left partially committed.
    """
    from utils import restore as restore_module

    filename = "20260820_120000_bird.jpg"
    backup_db = _make_db(tmp_path, "backup3")
    _seed_image_with_detection(backup_db, filename=filename, content_hash="hash-fail")

    def _boom(*_args, **_kwargs):
        raise sqlite3.OperationalError("simulated failure during label merge")

    with patch.object(restore_module, "_merge_human_label_tables", side_effect=_boom):
        with pytest.raises(sqlite3.OperationalError):
            restore_module._merge_database(backup_db, filename_rename_map={})

    conn = sqlite3.connect(live_db)
    # The images/detections inserts that preceded the failing step in
    # this same transaction must have been rolled back -- no partial merge.
    image_count = conn.execute(
        "SELECT COUNT(*) FROM images WHERE filename = ?", (filename,)
    ).fetchone()[0]
    detection_count = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE image_filename = ?", (filename,)
    ).fetchone()[0]
    conn.close()

    assert image_count == 0
    assert detection_count == 0
