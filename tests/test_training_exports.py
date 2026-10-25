from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sqlite3
import tracemalloc
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import patch

import pytest

from config import get_config
from core import training_retention
from core.canonical_dataset import CanonicalDataset
from core.human_label_core import BBox, HumanAnswer, record_human_answer, retract_answer
from core.manual_object_core import (
    ManualObjectDraft,
    create_manual_object,
    update_manual_object,
)
from core.training_export import (
    create_export,
    load_export,
    receive_export,
    waiting_revisions,
)
from tests.test_canonical_dataset_bundle import canonical_case as canonical_case
from tests.test_restore_database_merge import _make_db
from utils import backup
from utils.db import connection as db_connection
from utils.path_manager import PathManager
from utils.restore import _merge_database
from web.services.canonical_dataset_service import stream_canonical_bundle


def waiting(conn: sqlite3.Connection, paths: PathManager) -> list[dict[str, object]]:
    return waiting_revisions(
        conn, media_exists=lambda name: paths.get_original_path(name).is_file()
    )


def export(
    conn: sqlite3.Connection, paths: PathManager, **kwargs
) -> tuple[str, CanonicalDataset]:
    return create_export(
        conn,
        paths=paths,
        labeling_installation_id="0123456789abcdef0123456789abcdef",
        **kwargs,
    )


def test_absence_overrides_old_species_in_export(canonical_case) -> None:
    conn, paths, provenance, _, _, ids = canonical_case
    name = "20260810_090200_cls.jpg"
    record_human_answer(
        conn, HumanAnswer(name, ids[0], object_bird_presence="absent"), provenance
    )
    conn.commit()
    _, bundle = export(conn, paths)
    obj = next(row for row in bundle.objects if row["detection_id"] == ids[0])
    assert obj["presence"] == "absent" and obj["species_key"] is None
    assert any(f["species_key"] == "Cyanistes_caeruleus" for f in bundle.facts)
    assert not any(row["detection_id"] == ids[0] for row in bundle.classifier_ready)
    assert not any(row["detection_id"] == ids[0] for row in bundle.coco["annotations"])


def test_migration_answers_wait_withdrawals_and_missing_originals_do_not(
    canonical_case,
) -> None:
    conn, paths, provenance, ids, _, _ = canonical_case
    migrated = dataclasses.replace(provenance, source_kind="watchmybirds_migration")
    name = "20260810_090000_partial.jpg"
    record_human_answer(
        conn, HumanAnswer(name, ids[1], object_bird_presence="absent"), migrated
    )
    conn.commit()
    subject = conn.execute(
        "SELECT subject_id FROM label_subjects WHERE detection_id=?", (ids[1],)
    ).fetchone()[0]
    assert any(row["object_id"] == subject for row in waiting(conn, paths))
    retract_answer(
        conn,
        image_filename=name,
        detection_id=ids[1],
        fact_type="bird_presence",
        provenance=migrated,
    )
    conn.commit()
    assert not any(
        row["kind"] == "subject" and row["object_id"] == subject
        for row in waiting(conn, paths)
    )
    paths.get_original_path(name).unlink()
    assert not any(row["image_filename"] == name for row in waiting(conn, paths))
    assert name not in training_retention.is_export_relevant_any(conn, [name])


def test_repeat_download_uses_original_revisions_and_receipt_is_per_export(
    canonical_case,
) -> None:
    conn, paths, provenance, _, _, ids = canonical_case
    first, bundle = export(conn, paths)
    original_bytes = b"".join(
        stream_canonical_bundle(bundle, path_resolver=paths.get_original_path)
    )
    assert not waiting(conn, paths)
    name = "20260810_090200_cls.jpg"
    record_human_answer(
        conn,
        HumanAnswer(
            name, ids[0], species_identity="corrected", species_key="Pica_pica"
        ),
        provenance,
    )
    conn.commit()
    assert [row["image_filename"] for row in waiting(conn, paths)] == [name]
    saved, _ = load_export(conn, first)
    assert (
        b"".join(stream_canonical_bundle(saved, path_resolver=paths.get_original_path))
        == original_bytes
    )
    assert saved.classifier_ready[0]["species_key"] == "Cyanistes_caeruleus"
    receive_export(conn, first)
    assert [row["image_filename"] for row in waiting(conn, paths)] == [name]
    second, _ = export(conn, paths)
    receive_export(conn, first)
    assert training_retention.protection_sources_any(conn, bundle.media_filenames) == {
        name: {"unconfirmed_export"}
    }
    at = conn.execute(
        "SELECT received_at FROM labeling_exports WHERE export_id=?", (first,)
    ).fetchone()[0]
    receive_export(conn, first)
    assert (
        conn.execute(
            "SELECT received_at FROM labeling_exports WHERE export_id=?", (first,)
        ).fetchone()[0]
        == at
    )
    receive_export(conn, second)
    assert not training_retention.is_export_relevant_any(conn, bundle.media_filenames)


def test_selection_includes_unanswered_siblings_manual_objects_and_image_facts(
    canonical_case,
) -> None:
    conn, paths, provenance, ids, _, _ = canonical_case
    name = "20260810_090000_partial.jpg"
    manual, _ = create_manual_object(
        conn,
        ManualObjectDraft(name, BBox(0.4, 0.4, 0.2, 0.2), None, "test-add"),
        provenance,
        original_path=paths.get_original_path(name),
        locale="DE",
    )
    conn.commit()
    _, bundle = export(conn, paths, selection={f"detection:{ids[0]}"})
    assert bundle.media_filenames == [name]
    assert {row["object_key"] for row in bundle.objects} == {
        f"detection:{identifier}" for identifier in ids
    } | {f"manual:{manual['manual_object_id']}"}
    assert bundle.manual_objects and bundle.manual_revisions
    assert bundle.objects[1]["bbox"] is not None
    assert {row["image_filename"] for row in waiting(conn, paths)} == {
        "20260810_090100_negative.jpg",
        "20260810_090200_cls.jpg",
    }
    with pytest.raises(ValueError):
        export(conn, paths, selection={"detection:999999"})


def _measure_stream_in_child(
    bundle: CanonicalDataset, output_dir: Path, result: Connection
) -> None:
    """Isolate process-wide tracing from other tests' background workers."""
    paths = PathManager(str(output_dir))
    before = {p for p in paths.base_dir.rglob("*") if p.is_file()}
    tracemalloc.start()
    total = largest = 0
    try:
        for chunk in stream_canonical_bundle(
            bundle, path_resolver=paths.get_original_path
        ):
            total += len(chunk)
            largest = max(largest, len(chunk))
            assert {p for p in paths.base_dir.rglob("*") if p.is_file()} == before
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    with result:
        result.send((total, largest, peak))


def test_stream_memory_is_bounded_and_no_archive_is_written(canonical_case) -> None:
    conn, paths, *_ = canonical_case
    _, bundle = export(conn, paths)
    for name in bundle.media_filenames:
        with paths.get_original_path(name).open("wb") as output:
            for _ in range(96):
                output.write(os.urandom(64 * 1024))
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_measure_stream_in_child, args=(bundle, paths.base_dir, sender)
    )
    worker.start()
    sender.close()
    try:
        worker.join(timeout=30)
        assert not worker.is_alive(), "Export memory measurement timed out"
        assert worker.exitcode == 0
        assert receiver.poll(), "Export memory measurement returned no result"
        total, largest, peak = receiver.recv()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        receiver.close()
        worker.close()
    assert total > 18 * 1024 * 1024
    assert largest < 128 * 1024
    assert peak < 4 * 1024 * 1024


def test_backup_and_merge_preserve_exports_revisions_manual_objects_and_trash(
    canonical_case, tmp_path: Path
) -> None:
    conn, paths, provenance, ids, _, _ = canonical_case
    name = "20260810_090000_partial.jpg"
    manual, _ = create_manual_object(
        conn,
        ManualObjectDraft(name, BBox(0.4, 0.4, 0.2, 0.2), None, "test-merge"),
        provenance,
        original_path=paths.get_original_path(name),
        locale="DE",
    )
    conn.execute(
        "UPDATE detections SET status='rejected', trashed_at='2026-10-01T12:00:00+00:00' WHERE detection_id=?",
        (ids[1],),
    )
    conn.commit()
    first, _ = export(conn, paths, selection={f"detection:{ids[0]}"})
    receive_export(conn, first)
    second, second_bundle = export(conn, paths)
    source_zip = b"".join(
        stream_canonical_bundle(second_bundle, path_resolver=paths.get_original_path)
    )
    snapshot = tmp_path / "snapshot.db"
    with patch.object(
        backup, "get_db_path", return_value=str(paths.base_dir / "images.db")
    ):
        assert backup._create_db_snapshot(snapshot)
    with sqlite3.connect(snapshot) as saved:
        assert saved.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 2
        assert saved.execute(
            "SELECT received_at FROM labeling_exports WHERE export_id=?", (first,)
        ).fetchone()[0]
        assert (
            saved.execute(
                "SELECT trashed_at FROM detections WHERE detection_id=?", (ids[1],)
            ).fetchone()[0]
            == "2026-10-01T12:00:00+00:00"
        )
    target = _make_db(tmp_path, "destination")
    target_paths = PathManager(str(target.parent))
    rename = "20260810_090000_renamed.jpg"
    for old in second_bundle.media_filenames + [name]:
        dest = target_paths.get_original_path(rename if old == name else old)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(paths.get_original_path(old).read_bytes())
    with sqlite3.connect(target) as dest:
        dest.execute(
            "INSERT INTO images(filename, timestamp) VALUES ('occupied.jpg', '2026-10-01')"
        )
        dest.execute("INSERT INTO detections(image_filename) VALUES ('occupied.jpg')")
    with patch.dict(get_config(), {"OUTPUT_DIR": str(target.parent)}):
        _merge_database(snapshot, filename_rename_map={name: rename})
        _merge_database(snapshot, filename_rename_map={name: rename})
        db_connection._schema_initialized_paths.clear()
        with db_connection.closing_connection() as merged:
            assert (
                merged.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0]
                == 2
            )
            assert (
                merged.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 1
            )
            assert (
                merged.execute(
                    "SELECT COUNT(*) FROM manual_object_revisions"
                ).fetchone()[0]
                == 1
            )
            assert (
                merged.execute(
                    "SELECT trashed_at FROM detections WHERE image_filename=? AND status='rejected'",
                    (rename,),
                ).fetchone()[0]
                == "2026-10-01T12:00:00+00:00"
            )
            assert not waiting(merged, target_paths)
            saved, mapping = load_export(merged, second)
            assert (
                b"".join(
                    stream_canonical_bundle(
                        saved,
                        path_resolver=lambda filename: target_paths.get_original_path(
                            mapping.get(filename, filename)
                        ),
                    )
                )
                == source_zip
            )
            assert training_retention.is_export_relevant_any(
                merged, saved.media_filenames
            )
            receive_export(merged, second)
            assert not training_retention.is_export_relevant_any(
                merged, saved.media_filenames
            )
            local_manual = merged.execute("SELECT * FROM manual_objects").fetchone()
            update_manual_object(
                merged,
                manual_object_id=local_manual["manual_object_id"],
                image_filename=rename,
                expected_revision=local_manual["revision"],
                provenance=provenance,
                locale="DE",
                bbox=BBox(0.45, 0.45, 0.2, 0.2),
                original_path=target_paths.get_original_path(rename),
            )
            merged.commit()
        _merge_database(snapshot, filename_rename_map={name: rename})
        with db_connection.closing_connection() as merged:
            assert (
                merged.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 1
            )
            assert (
                merged.execute(
                    "SELECT COUNT(*) FROM manual_object_revisions"
                ).fetchone()[0]
                == 2
            )
            assert {row["image_filename"] for row in waiting(merged, target_paths)} == {
                rename
            }


@pytest.mark.parametrize("mode", ["traversal", "symlink"])
def test_export_refuses_media_outside_originals(
    canonical_case, tmp_path: Path, mode: str
) -> None:
    conn, paths, *_ = canonical_case
    name = "20260810_090200_cls.jpg"
    original = paths.get_original_path(name)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"private")
    if mode == "symlink":
        original.unlink()
        original.symlink_to(outside)
    else:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "UPDATE label_subjects SET image_filename='../outside.jpg' WHERE image_filename=?",
            (name,),
        )
        conn.commit()
    with pytest.raises(ValueError, match="Original is outside|Invalid original"):
        export(conn, paths)
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
