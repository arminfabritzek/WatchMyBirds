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


def test_failed_original_mapping_aborts_merge(live_db, tmp_path):
    from utils.restore import _merge_database

    backup = _make_db(tmp_path, "failed-media")
    name = "20260820_120000_bird.jpg"
    _seed_image_with_detection(backup, filename=name, content_hash="incoming")
    with pytest.raises(ValueError, match="Original import failed"):
        _merge_database(backup, filename_rename_map={name: None})
    with sqlite3.connect(live_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_live_database_replacement_is_refused(live_db, tmp_path):
    from utils.path_manager import PathManager
    from utils.restore import _replace_database, restore_from_archive

    before = live_db.read_bytes()
    with pytest.raises(RuntimeError, match="offline recovery"):
        _replace_database(tmp_path / "backup.db", PathManager(str(live_db.parent)))
    result = list(
        restore_from_archive(tmp_path / "backup.tar.gz", db_strategy="replace")
    )
    assert result[-1]["error"]
    assert "offline recovery" in result[-1]["error"]
    assert live_db.read_bytes() == before


def test_conflict_file_is_never_overwritten(tmp_path):
    import hashlib

    from utils.path_manager import PathManager
    from utils.restore import _generate_conflict_filename, _import_original_file

    pm = PathManager(str(tmp_path / "live"))
    name = "20260820_120000_bird.jpg"
    original = pm.get_original_path(name)
    original.parent.mkdir(parents=True)
    original.write_bytes(b"local")
    source = tmp_path / "incoming" / "2026-08-20" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"incoming")
    conflict = original.with_name(
        _generate_conflict_filename(name, hashlib.sha256(b"incoming").hexdigest())
    )
    conflict.write_bytes(b"different preexisting original")
    with pytest.raises(ValueError, match="Conflicting original"):
        _import_original_file(source, source.parent.parent, pm)
    assert conflict.read_bytes() == b"different preexisting original"


def test_archive_merge_keeps_files_detections_and_labels_together(
    live_db, tmp_path, monkeypatch
):
    import hashlib
    import tarfile

    from core.human_label_core import (
        LabelProvenance,
        append_fact,
        ensure_object_subject,
    )
    from utils import restore
    from utils.path_manager import PathManager

    filename = "20260820_100000_bird.jpg"
    pm = PathManager(str(live_db.parent))
    local = pm.get_original_path(filename)
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local bird")
    _seed_image_with_detection(
        live_db,
        filename=filename,
        content_hash=hashlib.sha256(b"local bird").hexdigest(),
    )
    backup = _make_db(tmp_path, "incoming")
    detection_id = _seed_image_with_detection(
        backup,
        filename=filename,
        content_hash=hashlib.sha256(b"incoming bird").hexdigest(),
    )
    with sqlite3.connect(backup) as conn:
        subject = ensure_object_subject(conn, detection_id)
        append_fact(
            conn,
            subject_id=subject,
            fact_type="bbox_quality",
            answer_value="suitable",
            provenance=LabelProvenance(
                installation_id="incoming",
                app_version="test",
                context="normal_correction",
                source_kind="watchmybirds_ui",
            ),
        )
    conn.close()
    image = tmp_path / "incoming.jpg"
    image.write_bytes(b"incoming bird")
    archive_path = pm.get_restore_tmp_dir() / "collection.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(backup, arcname="images.db")
        archive.add(image, arcname=f"originals/2026-08-20/{filename}")
    monkeypatch.setattr(restore, "get_path_manager", lambda: pm)
    monkeypatch.setattr(restore, "get_db_path", lambda: str(live_db))
    for _ in range(2):
        result = list(restore.restore_from_archive(archive_path))[-1]
        assert result["error"] is None, result
    with sqlite3.connect(live_db) as conn:
        rows = conn.execute("SELECT filename, content_hash FROM images").fetchall()
        assert len(rows) == 2
        for name, digest in rows:
            assert (
                hashlib.sha256(pm.get_original_path(name).read_bytes()).hexdigest()
                == digest
            )
        labeled_filename, thumbnail = conn.execute("""
            SELECT d.image_filename, d.thumbnail_path FROM human_label_facts f
            JOIN label_subjects s ON s.subject_id=f.subject_id
            JOIN detections d ON d.detection_id=s.detection_id
        """).fetchone()
        assert labeled_filename != filename
        assert pm.get_original_path(labeled_filename).read_bytes() == b"incoming bird"
        assert thumbnail is None
        assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == 1
