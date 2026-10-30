from __future__ import annotations

import json
import os
import threading
import zipfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from core.human_label_core import HumanAnswer, record_human_answer
from core.training_export import receive_export, waiting_revisions
from core.training_export_jobs import TrainingExportJobs
from tests.test_canonical_dataset_bundle import canonical_case as canonical_case


@pytest.fixture
def jobs(canonical_case: Any) -> Any:
    _, paths, *_ = canonical_case
    manager = TrainingExportJobs(
        str(paths.base_dir), photos_per_part=1, reserve_bytes=0
    )
    yield manager
    manager.close()


def complete(manager: TrainingExportJobs, job_id: str) -> dict:
    manager.wait(10)
    state = manager.status(job_id)
    assert state["status"] == "complete", state
    return state


def test_full_export_is_split_and_receipts_remain_per_part(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, *_ = canonical_case
    state = complete(jobs, jobs.start())
    assert state["processed"] == state["total"] == 3
    assert state["part_count"] == 3
    assert not list(paths.base_dir.rglob("snapshot.db"))
    assert not waiting_revisions(conn, media_exists=lambda _: True)
    for part in state["parts"]:
        artifact = jobs.open_download(part["export_id"])
        assert artifact is not None
        path, lease = artifact
        try:
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                snapshot = json.loads(archive.read("bundle.json"))["snapshot"]
                assert snapshot["export_job_id"] == state["job_id"]
                assert snapshot["part_number"] == part["part_number"]
                assert (
                    len(
                        [
                            name
                            for name in archive.namelist()
                            if name.startswith("media/")
                        ]
                    )
                    == 1
                )
        finally:
            lease.close()
        assert (
            conn.execute(
                "SELECT received_at FROM labeling_exports WHERE export_id=?",
                (part["export_id"],),
            ).fetchone()[0]
            is None
        )
    receive_export(conn, state["parts"][0]["export_id"])
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM labeling_exports WHERE received_at IS NULL"
        ).fetchone()[0]
        == 2
    )


def test_download_lease_blocks_removal_and_expiry(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, *_ = canonical_case
    state = complete(jobs, jobs.start())
    identifier = state["parts"][0]["export_id"]
    receive_export(conn, identifier)
    path, lease = jobs.open_download(identifier)
    with pytest.raises(ValueError, match="downloaded"):
        jobs.remove_archive(identifier)
    with jobs.connection() as live:
        live.execute(
            "UPDATE training_export_parts SET created_at='2000-01-01' WHERE export_id=?",
            (identifier,),
        )
    jobs.cleanup()
    assert path.is_file()
    lease.close()
    jobs.cleanup()
    assert not path.exists()


def test_cancelled_preparation_does_not_publish_or_release_protection(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    from core import training_export_jobs

    conn, paths, *_ = canonical_case
    started = threading.Event()
    release = threading.Event()
    original = training_export_jobs.stream_canonical_bundle

    def stream(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(5)
        yield from original(*args, **kwargs)

    with patch.object(training_export_jobs, "stream_canonical_bundle", stream):
        identifier = jobs.start()
        try:
            assert started.wait(3)
            with pytest.raises(ValueError, match="already preparing"):
                jobs.start()
            jobs.cancel(identifier)
        finally:
            release.set()
            jobs.wait(5)
    assert jobs.status(identifier)["status"] == "cancelled"
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
    assert len(waiting_revisions(conn, media_exists=lambda _: True)) > 0
    assert not list(paths.get_training_export_dir().rglob("*.zip"))
    assert not list(paths.get_training_export_dir().rglob("*.partial"))


def test_disk_full_pauses_and_retry_uses_same_snapshot(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, *_ = canonical_case
    with patch(
        "core.training_export_jobs.shutil.disk_usage",
        return_value=SimpleNamespace(free=0),
    ):
        identifier = jobs.start()
        jobs.wait(5)
    assert jobs.status(identifier)["status"] == "paused"
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
    jobs.retry(identifier)
    complete(jobs, identifier)


def test_parts_share_frozen_answers_after_resume(canonical_case: Any) -> None:
    conn, paths, provenance, _, _, ids = canonical_case
    jobs = TrainingExportJobs(
        str(paths.base_dir),
        photos_per_part=1,
        reserve_bytes=0,
        staging_bytes=7 * 1024**2,
    )
    try:
        for original in paths.originals_dir.rglob("*.jpg"):
            original.write_bytes(os.urandom(2 * 1024**2))
        identifier = jobs.start()
        jobs.wait(10)
        state = jobs.status(identifier)
        assert state["status"] == "paused", state
        assert state["part_count"] == 1
        name = "20260810_090200_cls.jpg"
        record_human_answer(
            conn,
            HumanAnswer(
                name, ids[0], species_identity="corrected", species_key="Pica_pica"
            ),
            provenance,
        )
        conn.commit()
        jobs.staging_bytes = 64 * 1024**2
        jobs.retry(identifier)
        state = complete(jobs, identifier)
        metadata = [
            json.loads(row[0])
            for row in conn.execute("SELECT bundle_json FROM labeling_exports")
        ]
        assert any(
            any(
                row.get("species_key") == "Cyanistes_caeruleus"
                for row in bundle["classifier_ready"]
            )
            for bundle in metadata
        )
        assert [
            row["image_filename"]
            for row in waiting_revisions(conn, media_exists=lambda _: True)
        ] == [name]
    finally:
        jobs.close()


def test_missing_originals_are_excluded_without_export_members(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, *_ = canonical_case
    name = "20260810_090100_negative.jpg"
    paths.get_original_path(name).unlink()
    state = complete(jobs, jobs.start())
    assert state["missing"] == 1
    assert state["part_count"] == 2
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM labeling_export_members WHERE image_filename=?",
            (name,),
        ).fetchone()[0]
        == 0
    )
    stored = tuple(
        conn.execute(
            "SELECT COUNT(*), SUM(length(bundle_json)) FROM labeling_exports"
        ).fetchone()
    )
    for _ in range(2):
        repeated = complete(jobs, jobs.start())
        assert repeated["part_count"] == 0
        assert repeated["missing"] == 1
        assert (
            tuple(
                conn.execute(
                    "SELECT COUNT(*), SUM(length(bundle_json)) FROM labeling_exports"
                ).fetchone()
            )
            == stored
        )
    assert len(list(paths.get_training_export_dir().rglob("*.zip"))) == 2

    paths.get_original_path(name).write_bytes(b"restored original")
    restored = complete(jobs, jobs.start())
    assert restored["part_count"] == 1
    assert restored["missing"] == 0
    assert (
        conn.execute(
            "SELECT image_filename FROM labeling_export_members WHERE export_id=?",
            (restored["parts"][0]["export_id"],),
        ).fetchone()[0]
        == name
    )


def test_only_missing_originals_create_no_export_artifacts(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, *_ = canonical_case
    for original in paths.originals_dir.rglob("*.jpg"):
        original.unlink()
    for _ in range(2):
        state = complete(jobs, jobs.start())
        assert state["missing"] == 3
        assert state["processed"] == state["total"] == state["part_count"] == 0
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
    assert not list(paths.get_training_export_dir().rglob("*.zip"))
    assert not list(paths.get_training_export_dir().rglob("snapshot.db"))


def test_explicit_selection_rejects_missing_originals(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, *_ = canonical_case
    selected = waiting_revisions(conn, media_exists=lambda _: True)[0]
    paths.get_original_path(selected["image_filename"]).unlink()
    with pytest.raises(ValueError, match="waiting for training"):
        jobs.start({selected["selection_key"]})


def test_original_lost_after_selection_is_excluded_once(
    jobs: TrainingExportJobs, canonical_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core import training_export_jobs

    conn, paths, *_ = canonical_case
    name = "20260810_090100_negative.jpg"
    build = training_export_jobs.build_canonical_dataset

    def lose_original(*args: Any, **kwargs: Any) -> Any:
        if name in kwargs["filenames"]:
            paths.get_original_path(name).unlink(missing_ok=True)
        return build(*args, **kwargs)

    monkeypatch.setattr(training_export_jobs, "build_canonical_dataset", lose_original)
    state = complete(jobs, jobs.start())
    assert state["missing"] == 1
    assert state["part_count"] == 3
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM labeling_export_members WHERE image_filename=?",
            (name,),
        ).fetchone()[0]
        == 0
    )
    metadata_size = conn.execute(
        "SELECT SUM(length(bundle_json)) FROM labeling_exports"
    ).fetchone()[0]

    repeated = complete(jobs, jobs.start())

    assert repeated["part_count"] == 0
    assert (
        conn.execute(
            "SELECT SUM(length(bundle_json)) FROM labeling_exports"
        ).fetchone()[0]
        == metadata_size
    )


def test_removed_archive_rebuild_keeps_exact_context_and_receipt(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, paths, provenance, _, _, ids = canonical_case
    state = complete(jobs, jobs.start())
    identifier = state["parts"][-1]["export_id"]
    path, lease = jobs.open_download(identifier)
    original = path.read_bytes()
    lease.close()
    receive_export(conn, identifier)
    receipt = conn.execute(
        "SELECT received_at FROM labeling_exports WHERE export_id=?", (identifier,)
    ).fetchone()[0]
    jobs.remove_archive(identifier)
    record_human_answer(
        conn,
        HumanAnswer(
            "20260810_090200_cls.jpg",
            ids[0],
            species_identity="corrected",
            species_key="Pica_pica",
        ),
        provenance,
    )
    conn.commit()
    complete(jobs, jobs.start(rebuild_export_id=identifier))
    path, lease = jobs.open_download(identifier)
    try:
        assert path.read_bytes() == original
    finally:
        lease.close()
    assert (
        conn.execute(
            "SELECT received_at FROM labeling_exports WHERE export_id=?", (identifier,)
        ).fetchone()[0]
        == receipt
    )
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 3


def test_restart_reuses_snapshot_after_interrupted_archive(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    from core import training_export_jobs

    conn, paths, provenance, _, _, ids = canonical_case
    started, release = threading.Event(), threading.Event()
    original = training_export_jobs.stream_canonical_bundle

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(5)
        yield from original(*args, **kwargs)

    with patch.object(training_export_jobs, "stream_canonical_bundle", interrupted):
        identifier = jobs.start()
        try:
            assert started.wait(3)
            jobs._stop.set()
        finally:
            release.set()
            jobs.wait(5)
    assert jobs.status(identifier)["status"] == "queued"
    record_human_answer(
        conn,
        HumanAnswer(
            "20260810_090200_cls.jpg",
            ids[0],
            species_identity="corrected",
            species_key="Pica_pica",
        ),
        provenance,
    )
    conn.commit()
    resumed = TrainingExportJobs(
        str(paths.base_dir), photos_per_part=1, reserve_bytes=0
    )
    try:
        complete(resumed, identifier)
        saved = [
            json.loads(row[0])
            for row in conn.execute("SELECT bundle_json FROM labeling_exports")
        ]
        assert any(
            row.get("species_key") == "Cyanistes_caeruleus"
            for bundle in saved
            for row in bundle["classifier_ready"]
        )
        assert [
            row["image_filename"]
            for row in waiting_revisions(conn, media_exists=lambda _: True)
        ] == ["20260810_090200_cls.jpg"]
    finally:
        resumed.close()


def test_write_exhaustion_leaves_no_partial_export_and_can_resume(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    import errno

    conn, paths, *_ = canonical_case

    def exhausted(*args: Any, **kwargs: Any) -> Any:
        yield b"partial ZIP"
        raise OSError(errno.ENOSPC, "disk full")

    with patch("core.training_export_jobs.stream_canonical_bundle", exhausted):
        identifier = jobs.start()
        jobs.wait(5)
    assert jobs.status(identifier)["status"] == "paused"
    assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
    assert not list(paths.get_training_export_dir().rglob("*.partial"))
    jobs.retry(identifier)
    complete(jobs, identifier)


def test_export_history_paging_never_loads_bundle_metadata(canonical_case: Any) -> None:
    from core.training_export import list_exports

    conn, *_ = canonical_case
    conn.executemany(
        "INSERT INTO labeling_exports(export_id,created_at,bundle_json) VALUES (?, ?, ?)",
        [
            (
                f"{number:032x}",
                "2026-10-30T00:00:00Z",
                "invalid metadata must not be read",
            )
            for number in range(45)
        ],
    )
    conn.commit()
    seen = []
    before = ""
    while page := list_exports(conn, limit=20, before=before):
        seen.extend(row["export_id"] for row in page)
        before = page[-1]["export_id"]
    assert len(seen) == len(set(seen)) == 45


def test_job_artifacts_reject_symlinked_directories(
    jobs: TrainingExportJobs, tmp_path: Any
) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    identifier = "a" * 32
    jobs.paths.get_training_export_dir().mkdir(exist_ok=True)
    jobs.paths.get_training_export_dir(identifier).symlink_to(
        target, target_is_directory=True
    )
    with pytest.raises(ValueError, match="Unsafe export job directory"):
        jobs._file(identifier, "snapshot.db")
    assert not list(target.iterdir())


def test_missing_staged_archive_can_be_rebuilt_without_receipt(
    jobs: TrainingExportJobs, canonical_case: Any
) -> None:
    conn, *_ = canonical_case
    state = complete(jobs, jobs.start())
    identifier = state["parts"][0]["export_id"]
    path, lease = jobs.open_download(identifier)
    expected = path.read_bytes()
    lease.close()
    path.unlink()
    jobs.cleanup()
    assert jobs.status(state["job_id"])["parts"][0]["available"] == 0
    complete(jobs, jobs.start(rebuild_export_id=identifier))
    path, lease = jobs.open_download(identifier)
    try:
        assert path.read_bytes() == expected
    finally:
        lease.close()
    assert (
        conn.execute(
            "SELECT received_at FROM labeling_exports WHERE export_id=?", (identifier,)
        ).fetchone()[0]
        is None
    )


def test_oversized_legacy_metadata_is_rejected_before_decoding(
    canonical_case: Any,
) -> None:
    from core.training_export import load_export

    conn, *_ = canonical_case
    conn.execute(
        "INSERT INTO labeling_exports(export_id,created_at,bundle_json) VALUES ('large','2026-10-30',zeroblob(?))",
        (33 * 1024**2,),
    )
    conn.commit()
    with pytest.raises(ValueError, match="metadata limit"):
        load_export(conn, "large")
