from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from core.deletion_plan import DeletionSettings, plan_deletions
from core.retention_core import reclaim_space
from core.storage_pressure import StoragePressure
from core.training_export import create_export
from tests.test_photo_deletion_plan import case as case
from tests.test_retention_executor import SETTINGS, add, answer, media
from utils.path_manager import PathManager

NOW = datetime(2026, 10, 17, 12, tzinfo=UTC)
Case = tuple[sqlite3.Connection, PathManager]


def test_low_space_overrides_new_activation_grace(case: Case) -> None:
    conn, paths = case
    name, _ = add(conn, paths, "open", review_status="untagged")
    normal = next(
        plan_deletions(
            conn,
            DeletionSettings(),
            NOW,
            paths=paths,
            automatic_deletion_enabled_at=NOW,
        )
    )
    assert all(
        item.decision.deadline == NOW + timedelta(days=14)
        and not item.decision.due(NOW)
        for item in normal.files + normal.rows
    )
    result = reclaim_space(
        conn,
        str(paths.base_dir),
        {**SETTINGS, "AUTOMATIC_DELETION_ENABLED_AT": NOW.isoformat()},
        free_bytes=lambda: 0,
        target_bytes=1,
        now=NOW,
    )
    assert result["deleted"] > 0 and result["deleted_rows"] > 0
    assert not conn.execute("SELECT 1 FROM images WHERE filename=?", (name,)).fetchone()


@pytest.mark.parametrize("option", ["keep_previews", "delete_all"])
def test_pressure_order_and_shared_photo_protection(case: Case, option: str) -> None:
    conn, paths = case
    bird, _ = add(conn, paths, "a_bird")
    opened, _ = add(conn, paths, "z_open", review_status="untagged")
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
    answer(conn, answered, did, object_bird_presence="present")
    eid, _ = create_export(
        conn,
        paths=paths,
        labeling_installation_id="0" * 32,
        selection={f"detection:{did}"},
    )
    conn.execute(
        "UPDATE labeling_exports SET received_at=? WHERE export_id=?",
        (NOW.isoformat(), eid),
    )
    mixed, did = add(conn, paths, "mixed", detections=2, review_status="untagged")
    conn.execute("UPDATE detections SET is_favorite=1 WHERE detection_id=?", (did,))
    conn.commit()
    mixed_training, did = add(
        conn, paths, "mixed_training", detections=2, review_status="untagged"
    )
    answer(conn, mixed_training, did, object_bird_presence="absent")
    before = media(paths)
    settings = {
        **SETTINGS,
        "OLD_BIRD_PHOTOS": option,
        "AUTOMATIC_DELETION_ENABLED_AT": NOW.isoformat(),
    }
    stages = []
    for stage in (1, 2, 3):
        plans = list(
            plan_deletions(
                conn,
                DeletionSettings(old_bird_photos=option),
                NOW,
                paths=paths,
                pressure_stage=stage,
            )
        )
        stages.append(
            {
                p.filename: {f.category for f in p.files if f.decision.due(NOW)}
                for p in plans
            }
        )
    assert stages[0][opened] >= {"original", "optimized", "thumbnail"}
    assert stages[0][bird] == stages[0][answered] == set()
    assert stages[1][bird] == stages[1][answered] == {"original"}
    assert ("optimized" in stages[2][bird]) == (option == "delete_all")
    for stage in stages:
        for name in (favorite, waiting, pending, mixed, mixed_training):
            assert stage[name] == set()
    result = reclaim_space(
        conn,
        str(paths.base_dir),
        settings,
        free_bytes=lambda: 0,
        target_bytes=1,
        now=NOW,
    )
    assert [s["stage"] for s in result["stages"]] == (
        [1, 2, 3] if option == "delete_all" else [1, 2]
    )
    assert result["errors"] == 0
    after = media(paths)
    for name in (favorite, waiting, pending, mixed, mixed_training):
        stem = Path(name).stem
        assert {k: v for k, v in after.items() if stem in k} == {
            k: v for k, v in before.items() if stem in k
        }
    assert not conn.execute(
        "SELECT 1 FROM images WHERE filename=?", (opened,)
    ).fetchone()
    for name in (bird, answered):
        assert conn.execute("SELECT 1 FROM images WHERE filename=?", (name,)).fetchone()
        assert not paths.get_original_path(name).exists()
        assert paths.get_derivative_path(name, "optimized").exists() == (
            option == "keep_previews"
        )


def test_pressure_stops_at_target_and_honors_switch(case: Case) -> None:
    conn, paths = case
    opened, _ = add(conn, paths, "open", review_status="untagged")
    before = media(paths)
    result = reclaim_space(
        conn,
        str(paths.base_dir),
        {**SETTINGS, "AUTOMATIC_DELETION_ENABLED": False},
        free_bytes=lambda: 0,
        target_bytes=1,
        now=NOW,
    )
    assert result["deleted"] == 0
    assert media(paths) == before
    free = lambda: int(not paths.get_original_path(opened).exists())
    result = reclaim_space(
        conn, str(paths.base_dir), SETTINGS, free_bytes=free, target_bytes=1, now=NOW
    )
    assert len(result["stages"]) == 1
    assert result["free_bytes"] == 1


def test_save_admission_is_cached_but_accounts_for_captures(tmp_path: Path) -> None:
    clock = [0.0]
    free = Mock(return_value=100)
    guard = StoragePressure(
        str(tmp_path), free_bytes=free, clock=lambda: clock[0], reserve_bytes=10
    )
    guard._stop.set()
    assert guard.allow_save(20)
    assert guard.allow_save(20)
    assert guard.allow_save(20)
    assert not guard.allow_save(20)
    assert guard.status()["paused"]
    assert free.call_count == 1
    clock[0] = 2.0
    assert guard.allow_save(20)
    assert not guard.status()["paused"]
    assert free.call_count == 2


def test_saving_waits_for_pressure_worker_even_if_external_space_recovers(
    tmp_path: Path,
) -> None:
    guard = StoragePressure(str(tmp_path), free_bytes=lambda: 1000, reserve_bytes=10)
    guard._worker = Mock()
    guard._worker.is_alive.return_value = True
    assert not guard.allow_save(20)
    assert "reclaiming unprotected photos" in guard.status()["message"]
    guard._worker.is_alive.return_value = False
    assert guard.allow_save(20)


def test_disabled_pressure_recovery_reports_stop_then_resumes(case: Case) -> None:
    _, paths = case
    free = [0]
    guard = StoragePressure(
        str(paths.base_dir),
        free_bytes=lambda: free[0],
        settings_provider=lambda: {"AUTOMATIC_DELETION_ENABLED": False},
        reserve_bytes=10,
    )
    guard.recover(30)
    assert "automatic deletion is off" in guard.status()["message"]
    free[0] = 100
    guard.recover(30)
    assert not guard.status()["paused"]


def test_storage_stop_prevents_image_files_and_database_rows(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    from detectors.services.persistence_service import PersistenceService

    conn, paths = case
    service = PersistenceService(str(paths.base_dir))
    monkeypatch.setattr(service.storage_pressure, "allow_save", lambda budget: False)
    before = media(paths)
    count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    result = service.save_image(
        np.zeros((100, 100, 3), dtype=np.uint8), NOW, "detector", "classifier", 1
    )
    assert not result.success
    assert media(paths) == before
    assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == count
    service.close()
