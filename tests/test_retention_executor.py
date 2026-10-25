from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from core.deletion_plan import plan_deletions
from core.human_label_core import HumanAnswer, record_human_answer
from core.retention_core import _ROW_KEYS, execute_plan, settings_for_plan
from core.training_export import create_export
from tests.retention_helpers import seed_image
from tests.test_photo_deletion_plan import PROVENANCE
from tests.test_photo_deletion_plan import case as case
from utils.db.connection import HUMAN_ANSWERS_MIGRATION
from utils.path_manager import PathManager

Case = tuple[sqlite3.Connection, PathManager]

NOW = datetime(2026, 10, 17, 12, tzinfo=UTC)
ENABLED = NOW - timedelta(days=15)
SETTINGS = {
    "AUTOMATIC_DELETION_ENABLED": True,
    "AUTOMATIC_DELETION_ENABLED_AT": ENABLED.isoformat(),
    "OLD_BIRD_PHOTOS": "keep_previews",
}


def media(paths: PathManager) -> dict[str, str]:
    return {
        p.relative_to(paths.base_dir).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for root in (paths.originals_dir, paths.derivatives_dir)
        for p in root.rglob("*")
        if p.is_file()
    }


def rows(conn: sqlite3.Connection) -> set[tuple[str, int | str]]:
    return {
        (table, row[0])
        for table, key in _ROW_KEYS.items()
        for row in conn.execute(f"SELECT {key} FROM {table}")
    }


def add(
    conn: sqlite3.Connection, paths: PathManager, suffix: str, **kwargs: Any
) -> tuple[str, int | None]:
    name = f"20240101_120000_{suffix}.jpg"
    seed_image(conn, name, paths.base_dir, **kwargs)
    original = paths.get_original_path(name)
    if original.exists():
        Image.new("RGB", (200, 200), (100, 130, 90)).save(original)
    conn.execute(
        "UPDATE detections SET od_class_name='Parus_major', frame_width=200, frame_height=200 WHERE image_filename=?",
        (name,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT detection_id FROM detections WHERE image_filename=?", (name,)
    ).fetchone()
    return name, row[0] if row else None


def answer(
    conn: sqlite3.Connection, name: str, identifier: int | None, **kwargs: Any
) -> None:
    record_human_answer(conn, HumanAnswer(name, identifier, **kwargs), PROVENANCE)
    conn.commit()


@pytest.mark.parametrize("option", ["keep_previews", "delete_all"])
def test_accelerated_run_matches_exact_plan_and_preserves_answers(
    case: Case, option: str
) -> None:
    conn, paths = case
    bird, _ = add(conn, paths, "bird")
    favorite, _ = add(conn, paths, "favorite", favorite=True)
    waiting, did = add(conn, paths, "waiting")
    answer(conn, waiting, did, object_bird_presence="present")
    pending, did = add(conn, paths, "pending")
    answer(conn, pending, did, object_bird_presence="absent")
    create_export(
        conn,
        paths=paths,
        labeling_installation_id="0" * 32,
        selection={f"detection:{did}"},
    )
    answered, did = add(conn, paths, "answered")
    answer(conn, answered, did, object_bird_presence="absent")
    eid, _ = create_export(
        conn,
        paths=paths,
        labeling_installation_id="0" * 32,
        selection={f"detection:{did}"},
    )
    conn.execute(
        "UPDATE labeling_exports SET received_at=? WHERE export_id=?",
        ((NOW - timedelta(days=40)).isoformat(), eid),
    )
    conn.commit()
    mixed, did = add(conn, paths, "mixed", detections=2, review_status="untagged")
    conn.execute("UPDATE detections SET is_favorite=1 WHERE detection_id=?", (did,))
    conn.commit()
    opened, _ = add(conn, paths, "open", review_status="untagged")
    orphan, _ = add(conn, paths, "orphan", detections=0)
    trash, did = add(conn, paths, "trash", favorite=True)
    answer(conn, trash, did, object_bird_presence="absent")
    conn.execute(
        "UPDATE detections SET status='rejected', trashed_at=? WHERE detection_id=?",
        ((NOW - timedelta(days=30)).isoformat(), did),
    )
    conn.execute(
        "INSERT INTO rescan_proposals(job_id,image_filename) VALUES ('test',?)",
        (trash,),
    )
    conn.execute("INSERT INTO bird_add_requests VALUES ('test',?,'{}','{}')", (trash,))
    conn.commit()
    conn.execute(
        "CREATE TABLE training_exports(export_id INTEGER PRIMARY KEY, detection_id INTEGER REFERENCES detections(detection_id) ON DELETE CASCADE)"
    )
    for name in (trash, opened, bird):
        conn.execute(
            "INSERT INTO training_exports(detection_id) SELECT detection_id FROM detections WHERE image_filename=?",
            (name,),
        )
    legacy_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='training_exports'"
    ).fetchone()[0]
    conn.commit()
    for table in ("labeling_exports", "labeling_export_members"):
        assert not {
            row[2] for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        } & {"detections", "label_subjects", "images"}
    settings = {**SETTINGS, "OLD_BIRD_PHOTOS": option}
    plans = list(
        plan_deletions(
            conn,
            settings_for_plan(settings),
            NOW,
            paths=paths,
            automatic_deletion_enabled_at=ENABLED,
        )
    )
    expected_files = {
        f.reference for p in plans for f in p.files if f.present and f.decision.due(NOW)
    }
    expected_rows = {
        (r.table, r.key) for p in plans for r in p.rows if r.decision.due(NOW)
    }
    before_files, before_rows = media(paths), rows(conn)
    exports = [
        list(map(tuple, conn.execute(f"SELECT * FROM {t}")))
        for t in ("labeling_exports", "labeling_export_members")
    ]
    facts = dict(
        conn.execute(
            "SELECT fact_id, subject_id FROM human_label_facts WHERE subject_id IN (SELECT subject_id FROM label_subjects WHERE image_filename=?)",
            (answered,),
        )
    )
    result = execute_plan(conn, str(paths.base_dir), settings, NOW, batch_size=2)
    assert result["errors"] == 0, result
    assert conn.execute("SELECT COUNT(*) FROM training_exports").fetchone()[0] == 1
    assert (
        legacy_schema
        == conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='training_exports'"
        ).fetchone()[0]
    )
    after_files, after_rows = media(paths), rows(conn)
    assert set(before_files) - set(after_files) == expected_files
    assert before_rows - after_rows == expected_rows
    assert result["deleted"] == len(expected_files)
    assert result["deleted_rows"] == len(expected_rows)
    assert all(
        before_files[key] == value
        for key, value in after_files.items()
        if key in before_files
    )
    assert exports == [
        list(map(tuple, conn.execute(f"SELECT * FROM {t}")))
        for t in ("labeling_exports", "labeling_export_members")
    ]
    assert facts == dict(
        conn.execute(
            "SELECT fact_id, subject_id FROM human_label_facts WHERE subject_id IN (SELECT subject_id FROM label_subjects WHERE image_filename=?)",
            (answered,),
        )
    )
    for name in (favorite, waiting, pending, mixed):
        assert paths.get_original_path(name).exists()
    assert paths.get_derivative_path(bird, "optimized").exists() == (
        option == "keep_previews"
    )
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    again = execute_plan(conn, str(paths.base_dir), settings, NOW)
    assert again["deleted"] == again["deleted_rows"] == again["errors"] == 0


def test_trash_deadline_restore_and_grace(case: Case) -> None:
    conn, paths = case
    opened, _ = add(conn, paths, "open", review_status="untagged")
    trash, did = add(conn, paths, "trash", favorite=True)
    conn.execute(
        "UPDATE detections SET status='rejected',trashed_at=? WHERE detection_id=?",
        ((NOW - timedelta(days=29)).isoformat(), did),
    )
    conn.commit()
    settings = {**SETTINGS, "AUTOMATIC_DELETION_ENABLED_AT": NOW.isoformat()}
    assert execute_plan(conn, str(paths.base_dir), settings, NOW)["deleted"] == 0
    conn.execute(
        "UPDATE detections SET status='active',trashed_at=NULL WHERE detection_id=?",
        (did,),
    )
    conn.commit()
    assert (
        execute_plan(
            conn, str(paths.base_dir), settings, NOW + timedelta(days=13, hours=23)
        )["deleted"]
        == 0
    )
    result = execute_plan(conn, str(paths.base_dir), settings, NOW + timedelta(days=14))
    assert result["deleted"] > 0
    assert not paths.get_original_path(opened).exists()
    assert paths.get_original_path(trash).exists()


def test_switch_off_blocks_everything_and_missing_marker_blocks(case: Case) -> None:
    conn, paths = case
    add(conn, paths, "old", review_status="untagged")
    before = media(paths), rows(conn)
    assert execute_plan(
        conn,
        str(paths.base_dir),
        {**SETTINGS, "AUTOMATIC_DELETION_ENABLED": False},
        NOW,
    )["disabled"]
    assert before == (media(paths), rows(conn))
    conn.execute("DELETE FROM _migration_state WHERE key=?", (HUMAN_ANSWERS_MIGRATION,))
    conn.commit()
    with pytest.raises(RuntimeError, match="migration"):
        execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert before == (media(paths), rows(conn))


def test_rechecks_favorite_committed_after_enumeration(case: Case) -> None:
    conn, paths = case
    name, did = add(conn, paths, "candidate", review_status="untagged")
    from core import retention_core

    original = retention_core.image_batches

    def concurrent_batches(*args: Any, **kwargs: Any) -> Iterator[list[dict[str, Any]]]:
        for batch in original(*args, **kwargs):
            with sqlite3.connect(paths.get_database_path()) as other:
                other.execute(
                    "UPDATE detections SET is_favorite=1 WHERE detection_id=?", (did,)
                )
            yield batch

    with patch.object(retention_core, "image_batches", concurrent_batches):
        result = execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert result["deleted"] == result["deleted_rows"] == 0
    assert paths.get_original_path(name).exists()


def test_disable_between_batches_stops_before_next_photo(case: Case) -> None:
    conn, paths = case
    for index in range(3):
        add(conn, paths, f"open{index}", review_status="untagged")
    settings = dict(SETTINGS)

    def stop_after_batch(progress: dict[str, Any]) -> None:
        settings["AUTOMATIC_DELETION_ENABLED"] = False

    result = execute_plan(
        conn,
        str(paths.base_dir),
        settings,
        NOW,
        batch_size=1,
        progress_callback=stop_after_batch,
    )
    assert result["processed"] == 1 and result["disabled"]
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2


def test_missing_files_do_not_block_row_cleanup(case: Case) -> None:
    conn, paths = case
    name, _ = add(
        conn,
        paths,
        "missing",
        review_status="untagged",
        write_original=False,
        write_optimized=False,
        write_thumbs=False,
    )
    result = execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert result["missing"] > 0 and result["errors"] == 0
    assert not conn.execute("SELECT 1 FROM images WHERE filename=?", (name,)).fetchone()


def test_file_failure_keeps_rows_and_retry_finishes(case: Case) -> None:
    conn, paths = case
    add(conn, paths, "failed", review_status="untagged")
    before = rows(conn)
    with patch("core.retention_core.os.unlink", side_effect=PermissionError):
        assert execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)["errors"] == 1
    assert rows(conn) == before
    assert execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)["errors"] == 0
    assert not rows(conn)


def test_missing_bird_previews_are_regenerated_before_original_expires(
    case: Case,
) -> None:
    conn, paths = case
    name, _ = add(conn, paths, "bird", write_optimized=False, write_thumbs=False)
    result = execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert result["errors"] == 0, result
    assert not paths.get_original_path(name).exists()
    assert paths.get_derivative_path(name, "optimized").exists()
    assert paths.get_derivative_path(Path(name).stem + "_crop_1.webp").exists()


def test_symlink_escape_keeps_outside_file_and_rows(case: Case, tmp_path: Path) -> None:
    conn, paths = case
    name, _ = add(conn, paths, "unsafe", review_status="untagged")
    original = paths.get_original_path(name)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"keep")
    original.unlink()
    original.symlink_to(outside)
    before = rows(conn)
    assert execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)["errors"] == 1
    assert outside.read_bytes() == b"keep" and rows(conn) == before


def test_bounded_batches_and_cooperative_stop(case: Case) -> None:
    conn, paths = case
    conn.executemany(
        "INSERT INTO images(filename,timestamp) VALUES (?,?)",
        [(f"20240101_120000_{i:04}.jpg", "20240101_120000") for i in range(1001)],
    )
    conn.commit()
    batches: list[int] = []
    result = execute_plan(
        conn,
        str(paths.base_dir),
        SETTINGS,
        NOW,
        batch_size=17,
        progress_callback=lambda progress: batches.append(progress["processed"]),
        stop_requested=lambda: bool(batches),
    )
    assert batches[0] == 17 and result["stopped"]
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 984


def test_released_shared_photo_cleans_up_unreferenced_crops(case: Case) -> None:
    conn, paths = case
    name, did = add(conn, paths, "siblings", detections=2, review_status="untagged")
    conn.execute("UPDATE detections SET is_favorite=1 WHERE detection_id=?", (did,))
    conn.commit()
    execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert len(list(paths.thumbs_dir.rglob("*.webp"))) == 2
    conn.execute("UPDATE detections SET is_favorite=0 WHERE detection_id=?", (did,))
    conn.commit()
    result = execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert result["errors"] == 0
    assert not media(paths)


def test_directory_swap_cannot_redirect_file_deletion(
    case: Case, tmp_path: Path
) -> None:
    conn, paths = case
    name, _ = add(conn, paths, "swap", review_status="untagged")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / name).write_bytes(b"keep")
    from core import retention_core

    checked = retention_core._checked_media
    switched = False

    def swap(path_manager: PathManager, reference: str, category: str) -> Path:
        nonlocal switched
        result = checked(path_manager, reference, category)
        if category == "original" and not switched:
            switched = True
            result.parent.rename(paths.base_dir / "moved-originals")
            result.parent.symlink_to(outside, target_is_directory=True)
        return result

    before = rows(conn)
    with patch.object(retention_core, "_checked_media", swap):
        result = execute_plan(conn, str(paths.base_dir), SETTINGS, NOW)
    assert result["errors"] == 1 and rows(conn) == before
    assert (outside / name).read_bytes() == b"keep"
