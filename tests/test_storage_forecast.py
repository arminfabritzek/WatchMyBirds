from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from core.deletion_plan import DEFAULT_SETTINGS, deletion_preview
from core.storage_accounting import inspect_storage
from core.storage_forecast import (
    FORECAST_WARNING_MONTHS,
    MONTH_DAYS,
    forecast_capacity,
)
from core.storage_scan import StorageScan
from tests.retention_helpers import seed_image
from tests.test_photo_deletion_plan import NOW
from tests.test_photo_deletion_plan import case as case
from utils.db.storage import open_readonly
from utils.path_manager import PathManager

Case = tuple[sqlite3.Connection, PathManager]


def history(conn: sqlite3.Connection, paths: PathManager, **kwargs: object) -> None:
    for age in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 10]:
        at = NOW - timedelta(days=age)
        seed_image(
            conn, at.strftime("%Y%m%d_%H%M%S") + ".jpg", paths.base_dir, **kwargs
        )
    conn.commit()


def project(
    case: Case,
    option: str = "keep_previews",
    *,
    enabled: bool = True,
    free: int = 100_000,
    database: dict | None = None,
) -> dict:
    conn, paths = case
    preview = deletion_preview(
        conn,
        replace(DEFAULT_SETTINGS, old_bird_photos=option),
        NOW,
        paths=paths,
        automatic_deletion_enabled_at=NOW - timedelta(days=30) if enabled else None,
    )
    return forecast_capacity(
        preview["forecast_inputs"],
        free_bytes=free,
        reserve_bytes=0,
        database=database or {},
    )


def test_growing_previews_against_finite_photo_retention(case: Case) -> None:
    conn, paths = case
    history(conn, paths)
    keep = project(case)
    delete = project(case, "delete_all")
    assert keep["status"] == "estimated"
    assert keep["net_bytes_per_day"] == 15 + 11
    assert keep["saved_bytes_per_day"] == 1026
    assert keep["months"] > 3
    assert not keep["warning"]
    assert delete["status"] == "not_foreseeable"
    assert delete["net_bytes_per_day"] == 0
    assert delete["months"] is None


def test_expiring_open_history_is_stable_but_switch_off_keeps_growing(
    case: Case,
) -> None:
    conn, paths = case
    history(conn, paths, review_status="untagged")
    assert project(case)["status"] == "not_foreseeable"
    off = project(case, enabled=False, free=80_000)
    assert off["status"] == "estimated"
    assert off["net_bytes_per_day"] == 1026
    assert off["warning"]


@pytest.mark.parametrize("option", ["keep_previews", "delete_all"])
def test_protected_history_grows_with_both_options(case: Case, option: str) -> None:
    conn, paths = case
    history(conn, paths, favorite=True)
    result = project(case, option)
    assert result["status"] == "estimated"
    assert result["net_bytes_per_day"] == 1026


def test_finite_retention_has_no_lasting_growth_even_with_little_free_space(
    case: Case,
) -> None:
    conn, paths = case
    history(conn, paths)
    result = project(case, "delete_all", free=1026 * 2)
    assert result["status"] == "not_foreseeable"
    assert result["months"] is None
    assert not result["warning"]


def test_metadata_growth_survives_photo_expiry(case: Case) -> None:
    conn, paths = case
    history(conn, paths)
    result = project(
        case,
        "delete_all",
        database={
            "records": {"images": 8, "detections": 8},
            "allocated_bytes": 3200,
            "reusable_bytes": 1600,
        },
    )
    assert result["metadata_bytes_per_day"] == 200
    assert result["net_bytes_per_day"] == 200
    assert result["status"] == "estimated"


def test_deadlines_reserve_and_warning_boundary() -> None:
    inputs = dict(
        sample_days=7,
        recent_bytes=700,
        kept_rows=0,
        permanent_bytes=700,
        reclaimable_bytes=3000,
    )
    result = forecast_capacity(inputs, free_bytes=2100, reserve_bytes=100, database={})
    assert result["months"] == pytest.approx(50 / MONTH_DAYS)
    assert result["warning"]
    inputs["reclaimable_bytes"] = 0
    boundary = 100 * MONTH_DAYS * FORECAST_WARNING_MONTHS
    assert not forecast_capacity(
        inputs, free_bytes=boundary, reserve_bytes=0, database={}
    )["warning"]
    assert (
        forecast_capacity(inputs, free_bytes=0, reserve_bytes=0, database={})["months"]
        == 0
    )


def test_empty_or_short_history_is_insufficient(case: Case) -> None:
    assert project(case)["status"] == "insufficient_history"
    conn, paths = case
    seed_image(conn, NOW.strftime("%Y%m%d_%H%M%S") + ".jpg", paths.base_dir)
    assert project(case)["status"] == "insufficient_history"


def test_forecast_collection_is_readonly(case: Case) -> None:
    conn, paths = case
    history(conn, paths)
    # SQLite can update shared-memory locks through a read-only connection.
    before = {
        p: p.read_bytes()
        for p in paths.base_dir.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }
    with open_readonly(paths.get_database_path()) as readonly:
        readonly.execute("PRAGMA query_only=ON")
        result = project((readonly, paths))
        assert result["status"] == "estimated"
        assert readonly.total_changes == 0
    assert before == {
        p: p.read_bytes()
        for p in paths.base_dir.rglob("*")
        if p.is_file() and not p.name.endswith("-shm")
    }


def test_scan_integrates_forecast_and_rejects_incomplete_measurement(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, paths = case
    history(conn, paths)
    settings = dict(
        AUTOMATIC_DELETION_ENABLED=True,
        AUTOMATIC_DELETION_ENABLED_AT=(NOW - timedelta(days=30)).isoformat(),
    )
    result = inspect_storage(
        paths,
        settings,
        now=NOW,
        temp_root=paths.base_dir / "empty",
        backup_root=paths.base_dir / "no-backup",
    )
    assert result["status"] == "complete"
    assert result["forecast"]["status"] == "estimated"
    assert result["forecast"]["sample_days"] == 7
    from core.storage_accounting import inventory

    def incomplete(*args: object, **kwargs: object) -> dict:
        measured = inventory(*args, **kwargs)
        measured["skipped_entries"] += 1
        return measured

    monkeypatch.setattr("core.storage_accounting.inventory", incomplete)
    partial = inspect_storage(
        paths,
        settings,
        now=NOW,
        temp_root=paths.base_dir / "empty",
        backup_root=paths.base_dir / "no-backup",
    )
    assert partial["status"] == "partial"
    assert partial["forecast"]["status"] == "unavailable"


def test_saved_forecast_requires_measurement_after_settings_change(case: Case) -> None:
    _, paths = case
    settings = {"OLD_BIRD_PHOTOS": "keep_previews"}
    service = StorageScan(paths, lambda: settings)
    service._state["result"] = dict(
        forecast=dict(status="estimated", months=2),
        deletion_preview=dict(settings=DEFAULT_SETTINGS.__dict__, enabled_at=None),
    )
    assert service.status()["result"]["forecast"]["status"] == "estimated"
    settings["OLD_BIRD_PHOTOS"] = "delete_all"
    assert service.status()["result"]["forecast"]["status"] == "settings_changed"
    settings["OLD_BIRD_PHOTOS"] = "keep_previews"
    settings.update(
        AUTOMATIC_DELETION_ENABLED=True, AUTOMATIC_DELETION_ENABLED_AT=NOW.isoformat()
    )
    assert service.status()["result"]["forecast"]["status"] == "settings_changed"
