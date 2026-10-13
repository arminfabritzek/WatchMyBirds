"""Storage accounting uses measured files and preserves the retention contract."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from core.retention_core import build_plan, resolve_posture_settings
from core.storage_accounting import inspect_storage
from core.storage_scan import HISTORY_LIMIT, StorageScan, growth_summary
from tests.labeling_helpers import _reset_test_config
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection
from utils.db.storage import database_summary, open_readonly
from utils.path_manager import PathManager

NOW = datetime(2026, 10, 13, tzinfo=UTC)
SETTINGS = dict(
    RETENTION_POSTURE="conservative",
    RETENTION_DAYS=90,
    RETENTION_PROTECT_FAVORITES=True,
)


@pytest.fixture
def output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return _reset_test_config(monkeypatch, tmp_path)


def scan(output: Path, **kwargs: Any) -> dict[str, Any]:
    return inspect_storage(
        PathManager(str(output)),
        SETTINGS,
        now=NOW,
        temp_root=output / "empty-temp",
        backup_root=kwargs.pop("backup_root", output / "empty-usb"),
        **kwargs,
    )


def test_measured_categories_protections_overlap_and_no_deletion(output: Path) -> None:
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_eligible.jpg", output, orig_bytes=100)
        seed_image(
            conn, "20260101_120001_favorite.jpg", output, favorite=True, orig_bytes=200
        )
        seed_image(
            conn, "20261012_120000_recent.jpg", output, favorite=True, orig_bytes=300
        )
        seed_image(
            conn,
            "20260101_120002_missingthumb.jpg",
            output,
            write_thumbs=False,
            orig_bytes=400,
        )
        seed_image(
            conn,
            "20260101_120003_unreviewed.jpg",
            output,
            review_status="untagged",
            orig_bytes=500,
        )
        seed_image(conn, "20260101_120004_missing.jpg", output, write_original=False)
        seed_image(
            conn,
            "20260101_120005_retired.jpg",
            output,
            write_original=False,
            original_present=0,
        )
        expected = build_plan(
            conn, str(output), resolve_posture_settings(SETTINGS), NOW
        )
    (output / "backup").mkdir()
    (output / "backup" / "snapshot.zip").write_bytes(b"b" * 40)
    (output / "images.pre-a2-cleanup-test.db").write_bytes(b"b" * 25)
    (output / "images.db.pre-repair-test").write_bytes(b"b" * 35)
    (output / "export.zip").write_bytes(b"a" * 50)
    (output / "extra.bin").write_bytes(b"e" * 60)
    before = {
        p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*.jpg")
    }
    result = scan(output)
    assert result["database_error"] is None, result
    p = result["protections"]
    assert p["eligible"]["bytes"] == expected.estimated_bytes == 100
    assert p["protected"]["bytes"] == 1400
    assert p["reasons"]["favorite"]["bytes"] == 500
    assert p["reasons"]["training_evidence"]["bytes"] == 500
    assert p["reasons"]["too_recent"]["bytes"] == 300
    assert p["reasons"]["unreviewed"]["bytes"] == 500
    assert p["reasons"]["missing_derivative"]["bytes"] == 400
    assert p["missing_originals"] == 1
    assert p["retired_originals"] == 1
    assert any("favorite" in key and "too_recent" in key for key in p["overlap_groups"])
    categories = result["inventory"]["categories"]
    assert categories["originals"]["bytes"] == 1500
    assert (
        categories["backups"]["bytes"] == 100 and categories["archives"]["bytes"] == 50
    )
    assert (
        sum(row["bytes"] for row in categories.values())
        == result["inventory"]["logical_bytes"]
    )
    assert result["database"]["records"]["images"] == 7
    assert result["database"]["allocated_bytes"] >= result["database"]["reusable_bytes"]
    assert before == {
        p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*.jpg")
    }
    with closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 7
        assert (
            conn.execute("SELECT SUM(original_present) FROM images").fetchone()[0] == 6
        )


def test_canonical_ready_unresolved_and_historical_evidence(output: Path) -> None:
    provenance = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_label.jpg", output)
        detection = conn.execute("SELECT detection_id FROM detections").fetchone()[0]
        record_human_answer(
            conn,
            HumanAnswer(
                "20260101_120000_label.jpg",
                detection,
                object_bird_presence="present",
                bbox_quality="suitable",
                species_identity="corrected",
                species_key="Parus_major",
            ),
            provenance,
        )
        conn.execute("UPDATE detections SET frame_width=640,frame_height=480")
        conn.commit()
    ready = scan(output)
    assert ready["database_error"] is None, ready
    assert ready["protections"]["reasons"]["training_ready"]["files"] == 1
    assert ready["protections"]["reasons"]["training_evidence"]["files"] == 1
    with closing_connection() as conn:
        record_human_answer(
            conn,
            HumanAnswer(
                "20260101_120000_label.jpg", detection, species_identity="wrong"
            ),
            provenance,
        )
    later = scan(output)
    assert later["protections"]["reasons"]["historical_facts"]["files"] == 1
    assert later["protections"]["reasons"]["training_unresolved"]["files"] == 1
    assert later["protections"]["eligible"]["files"] == 0


def test_hardlinks_are_deduplicated_and_not_promised_as_freed_blocks(
    output: Path,
) -> None:
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_link.jpg", output, orig_bytes=4096)
    original = next((output / "originals").rglob("*.jpg"))
    (output / "backup").mkdir()
    os.link(original, output / "backup" / "shared.jpg")
    result = scan(output)
    assert result["protections"]["eligible"]["bytes"] == 4096
    assert result["protections"]["eligible"]["unshared_allocated_bytes"] == 0
    assert result["protections"]["shared_originals"] == 1
    assert result["inventory"]["allocated_bytes"] == sum(
        row["allocated_bytes"] for row in result["inventory"]["categories"].values()
    )


def test_changed_files_and_wal_are_reported(output: Path) -> None:
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_change.jpg", output)
    original = next((output / "originals").rglob("*.jpg"))
    called = False
    writer = sqlite3.connect(output / "images.db")
    writer.execute("PRAGMA journal_mode=WAL")

    def progress(value: dict[str, Any]) -> None:
        nonlocal called
        if value.get("phase") == "protection" and not called:
            called = True
            original.write_bytes(b"changed")
            writer.execute("UPDATE images SET review_status='untagged'")
            writer.commit()

    result = scan(output, progress=progress)
    writer.close()
    assert result["status"] == "partial"
    assert result["files_changed"] and result["database_changed"]
    assert not result["protections"]["current_estimate"]
    assert result["inventory"]["categories"]["originals"]["bytes"] == 7


def test_symlinks_and_untrusted_filenames_cannot_escape_scope(
    output: Path, tmp_path: Path
) -> None:
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images(filename,timestamp) VALUES ('../../outside.jpg','20260101_120000')"
        )
    secret = tmp_path / "outside.jpg"
    secret.write_bytes(b"private")
    (output / "linked").symlink_to(tmp_path, target_is_directory=True)
    result = scan(output)
    assert result["status"] == "partial"
    assert result["protections"]["unsafe_originals"] == 1
    assert result["inventory"]["skipped_entries"] == 1
    assert "outside.jpg" not in json.dumps(result)
    assert secret.read_bytes() == b"private"


def test_dbstat_unavailable_reusable_pages_and_readonly_source(output: Path) -> None:
    with closing_connection() as conn:
        conn.execute("CREATE TABLE scratch(value BLOB)")
        conn.executemany("INSERT INTO scratch VALUES (?)", [(b"x" * 10000,)] * 100)
        conn.commit()
        conn.execute("DELETE FROM scratch")
        conn.commit()

    class WithoutDbstat(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            if "dbstat" in sql:
                raise sqlite3.OperationalError("no such table: dbstat")
            return super().execute(sql, parameters)

    with sqlite3.connect(output / "images.db", factory=WithoutDbstat) as conn:
        conn.row_factory = sqlite3.Row
        summary = database_summary(conn)
        assert summary["object_sizes_status"] == "unavailable"
        assert summary["reusable_pages"] > 0
    conn = open_readonly(output / "images.db")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO images(filename) VALUES ('forbidden')")
    conn.close()


def test_permission_failures_are_partial_counts(output: Path) -> None:
    with closing_connection():
        pass

    def denied(*args: Any, **kwargs: Any) -> Iterator[Any]:
        kwargs["onerror"](PermissionError("private path"))
        return iter(())

    with patch("core.storage_accounting.os.walk", side_effect=denied):
        result = scan(output)
    assert result["status"] == "partial"
    assert result["errors"] and "private path" not in json.dumps(result)


def test_disabled_retention_distinguishes_current_from_potential(output: Path) -> None:
    with closing_connection() as conn:
        seed_image(conn, "20260101_120000_off.jpg", output)
    result = inspect_storage(
        PathManager(str(output)),
        {**SETTINGS, "RETENTION_POSTURE": "off"},
        now=NOW,
        temp_root=output / "empty",
    )
    assert result["protections"]["eligible"]["bytes"] == 0
    assert result["protections"]["eligible_if_enabled"]["bytes"] == 1000


def test_one_background_scan_history_bound_restart_and_growth(output: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    at = NOW

    def inspector(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(3)
        return dict(
            status="complete",
            finished_at=at.isoformat(),
            files_changed=False,
            inventory=dict(
                scopes=[dict(scope="output", state="measured", device=1)],
                errors=[],
                skipped_entries=1,
                output_bytes=100,
            ),
        )

    service = StorageScan(
        PathManager(str(output)), lambda: SETTINGS, inspector=inspector
    )
    assert service.start() and entered.wait(3)
    assert not service.start()
    assert service.status()["status"] == "running"
    release.set()
    service.wait(3)
    assert service.status()["status"] == "complete"
    for index in range(35):
        at = NOW + timedelta(hours=index + 1)
        assert service.start()
        service.wait(3)
    assert len(service.status()["history"]) == HISTORY_LIMIT
    loaded = StorageScan(str(output), lambda: SETTINGS)
    assert len(loaded.status()["history"]) == HISTORY_LIMIT
    assert loaded.status()["growth"]["bytes_per_day"] == 0
    assert str(output) not in json.dumps(loaded.status())
    samples = [
        dict(at=NOW.isoformat(), bytes=100, scope="one"),
        dict(at=(NOW + timedelta(days=3)).isoformat(), bytes=400, scope="one"),
    ]
    assert growth_summary(samples)["collection_gap"]
    assert growth_summary(samples)["bytes_per_day"] == 100
    samples[-1]["scope"] = "other"
    assert growth_summary(samples)["status"] == "scope_changed"


def test_scoped_canonical_batches_match_full_export(output: Path) -> None:
    from core.canonical_dataset import build_canonical_dataset

    provenance = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")
    with closing_connection() as conn:
        for index in range(203):
            filename = f"20260101_120000_{index:04d}.jpg"
            seed_image(conn, filename, output)
            if index in {0, 199, 200, 202}:
                detection = conn.execute(
                    "SELECT detection_id FROM detections WHERE image_filename=?",
                    (filename,),
                ).fetchone()[0]
                record_human_answer(
                    conn,
                    HumanAnswer(
                        filename,
                        detection,
                        object_bird_presence="present",
                        bbox_quality="suitable" if index < 200 else "unsuitable",
                    ),
                    provenance,
                )
        conn.execute("UPDATE detections SET frame_width=640,frame_height=480")
        conn.commit()
        full = build_canonical_dataset(conn, media_exists=lambda _name: True)
        ready = {
            row["image_filename"]
            for row in full.manifest
            if row["decision"] == "included"
            and row["view"] in {"od_positive", "od_negative", "cls_positive"}
        }
        unresolved = {
            row["image_filename"]
            for row in full.manifest
            if row["decision"] == "included" and row["view"] == "unresolved"
        }
    result = scan(output)
    assert result["database_error"] is None
    reasons = result["protections"]["reasons"]
    assert reasons["training_ready"]["files"] == len(ready) == 2
    assert reasons["training_unresolved"]["files"] == len(unresolved) == 4
    assert reasons["canonical_facts"]["files"] == 4
    assert result["database"]["records"]["images"] == 203


def test_external_export_and_backup_scopes_are_measured_separately(
    output: Path, tmp_path: Path
) -> None:
    with closing_connection():
        pass
    exports = tmp_path / "temp"
    export = exports / "canonical-bundle-test"
    export.mkdir(parents=True)
    (export / "bundle.zip").write_bytes(b"e" * 123)
    (exports / "unrelated-private.txt").write_bytes(b"private")
    backup = tmp_path / "usb"
    backup.mkdir()
    (backup / "snapshot.tar").write_bytes(b"b" * 234)
    result = inspect_storage(
        PathManager(str(output)),
        SETTINGS,
        now=NOW,
        temp_root=exports,
        backup_root=backup,
    )
    categories = result["inventory"]["categories"]
    assert categories["temporary_exports"]["bytes"] == 123
    assert categories["backups"]["bytes"] == 234
    assert result["temporary_export_lifecycle"] == "unavailable"
    assert "private" not in json.dumps(result)
    assert (
        result["inventory"]["output_bytes"]
        == result["inventory"]["logical_bytes"] - 357
    )


def test_inactive_automount_is_not_touched(output: Path, tmp_path: Path) -> None:
    from core.storage_accounting import unmounted_automount

    with closing_connection():
        pass
    (output.parent / "code-sync-backups").mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:0 / / rw - ext4 /dev/root rw\n2 1 0:27 / /mnt/wmb-backup rw - autofs systemd-1 rw\n"
    )
    root = Path("/mnt/wmb-backup/snapshots")
    assert unmounted_automount(root, mountinfo)
    assert not unmounted_automount(output, mountinfo)
    with mountinfo.open("a") as stream:
        stream.write("3 2 8:1 / /mnt/wmb-backup rw - ext4 /dev/usb rw\n")
    assert not unmounted_automount(root, mountinfo)
    original_stat = Path.stat

    def checked_stat(path: Path, **kwargs: Any) -> os.stat_result:
        assert not str(path).startswith("/mnt/wmb-backup"), "automount was touched"
        return original_stat(path, **kwargs)

    with (
        patch(
            "core.storage_accounting.unmounted_automount",
            side_effect=lambda path: path == root,
        ),
        patch.object(Path, "stat", checked_stat),
    ):
        result = scan(output, backup_root=root)
    assert result["status"] == "complete"
    scope = next(s for s in result["inventory"]["scopes"] if s["scope"] == "usb_backup")
    assert scope["state"] == "unavailable" and scope["reason"] == "inactive_automount"
