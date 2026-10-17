from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from core.deletion_plan import (
    DEFAULT_SETTINGS,
    PhotoPlan,
    deletion_preview,
    plan_deletions,
)
from core.human_label_core import (
    BBox,
    HumanAnswer,
    LabelProvenance,
    record_human_answer,
)
from core.manual_object_core import ManualObjectDraft, create_manual_object
from core.training_export import create_export
from core.training_retention import protection_sources_any
from tests.labeling_helpers import _reset_test_config
from tests.retention_helpers import seed_image
from utils.db.connection import _init_schema, closing_connection
from utils.db.storage import open_readonly
from utils.path_manager import PathManager

NOW = datetime(2026, 10, 17, 12, tzinfo=UTC)
RECEIVED = datetime(2026, 10, 1, 12, tzinfo=UTC)
NAME = "20260101_120000_photo.jpg"
PROVENANCE = LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui")
Case = tuple[sqlite3.Connection, PathManager]


@pytest.fixture
def case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[sqlite3.Connection, PathManager]]:
    output = _reset_test_config(monkeypatch, tmp_path)
    with closing_connection() as conn:
        yield conn, PathManager(str(output))


def seed(conn: sqlite3.Connection, paths: PathManager, **kwargs: Any) -> list[int]:
    seed_image(conn, NAME, paths.base_dir, **kwargs)
    original = paths.get_original_path(NAME)
    if original.is_file():
        Image.new("RGB", (200, 200), (100, 130, 90)).save(original)
    conn.commit()
    return [
        row[0]
        for row in conn.execute(
            "SELECT detection_id FROM detections WHERE image_filename=?", (NAME,)
        )
    ]


def plan(conn: sqlite3.Connection, paths: PathManager, **kwargs: Any) -> PhotoPlan:
    return next(
        plan_deletions(
            conn,
            kwargs.pop("settings", DEFAULT_SETTINGS),
            kwargs.pop("now", NOW),
            paths=paths,
            automatic_deletion_enabled_at=kwargs.pop(
                "automatic_deletion_enabled_at", NOW - timedelta(days=14)
            ),
            **kwargs,
        )
    )


def answer(conn: sqlite3.Connection, identifier: int | None, **kwargs: Any) -> None:
    record_human_answer(conn, HumanAnswer(NAME, identifier, **kwargs), PROVENANCE)
    conn.commit()


def received(
    conn: sqlite3.Connection, paths: PathManager, at: datetime = RECEIVED
) -> str:
    export_id, _ = create_export(
        conn, paths=paths, labeling_installation_id="0123456789abcdef0123456789abcdef"
    )
    conn.execute(
        "UPDATE labeling_exports SET received_at=? WHERE export_id=?",
        (at.isoformat(), export_id),
    )
    conn.commit()
    return export_id


def test_trash_precedes_favorite_and_training_and_uses_move_date(case: Case) -> None:
    conn, paths = case
    identifier = seed(conn, paths, favorite=True)[0]
    answer(conn, identifier, object_bird_presence="absent")
    moved = NOW - timedelta(days=29)
    conn.execute(
        "UPDATE detections SET status='rejected', trashed_at=?", (moved.isoformat(),)
    )
    conn.commit()
    result = plan(conn, paths)
    assert result.training_sources == ("waiting_for_training",)
    assert {item.decision.rule for item in result.files + result.rows} == {1}
    assert all(
        item.decision.deadline == moved + timedelta(days=30)
        for item in result.files + result.rows
    )
    assert not any(item.decision.due(NOW) for item in result.files + result.rows)
    assert all(
        item.decision.due(NOW + timedelta(days=1))
        for item in result.files + result.rows
    )


def test_favorite_precedes_training(case: Case) -> None:
    conn, paths = case
    identifier = seed(conn, paths, favorite=True)[0]
    answer(conn, identifier, object_bird_presence="present")
    result = plan(conn, paths)
    assert result.training_sources == ("waiting_for_training",)
    assert all(
        item.decision.rule == 2 and item.decision.action == "keep"
        for item in result.files + result.rows
    )


def test_training_uses_shared_protection_for_waiting_unconfirmed_and_changes(
    case: Case,
) -> None:
    conn, paths = case
    identifier = seed(conn, paths)[0]
    answer(conn, identifier, bbox_quality="suitable")
    with patch(
        "core.deletion_plan.protection_sources_any", wraps=protection_sources_any
    ) as protection:
        result = plan(conn, paths)
        protection.assert_called_once()
    assert result.training_sources == ("waiting_for_training",)
    export_id, _ = create_export(
        conn, paths=paths, labeling_installation_id="0123456789abcdef0123456789abcdef"
    )
    assert plan(conn, paths).training_sources == ("unconfirmed_export",)
    conn.execute(
        "UPDATE labeling_exports SET received_at=? WHERE export_id=?",
        (RECEIVED.isoformat(), export_id),
    )
    conn.commit()
    result = plan(conn, paths)
    assert result.files[0].decision.rule == 4
    assert result.files[0].decision.deadline == RECEIVED + timedelta(days=90)
    answer(conn, identifier, bbox_quality="unsuitable")
    result = plan(conn, paths)
    assert result.training_sources == ("waiting_for_training",)
    assert all(item.decision.action == "keep" for item in result.files)


def test_current_no_bird_overrides_older_species_and_keeps_answer_rows(
    case: Case,
) -> None:
    conn, paths = case
    identifier = seed(conn, paths)[0]
    answer(conn, identifier, species_identity="confirmed", species_key="Parus_major")
    received(conn, paths, RECEIVED - timedelta(days=30))
    answer(conn, identifier, object_bird_presence="absent")
    received(conn, paths)
    result = plan(conn, paths)
    assert all(
        item.decision.rule == 4
        and item.decision.deadline == RECEIVED + timedelta(days=14)
        and item.decision.due(NOW)
        for item in result.files
    )
    assert all(row.decision.action == "keep" for row in result.rows)
    assert len([row for row in result.rows if row.table == "human_label_facts"]) == 3


def test_repeated_current_revision_uses_latest_receipt(case: Case) -> None:
    conn, paths = case
    ids = seed(conn, paths, detections=2)
    for identifier in ids:
        answer(conn, identifier, object_bird_presence="absent")
    received(conn, paths, RECEIVED - timedelta(days=30))
    answer(conn, ids[1], bbox_quality="unsuitable")
    received(conn, paths)
    result = plan(conn, paths)
    assert all(
        item.decision.deadline == RECEIVED + timedelta(days=14) for item in result.files
    )


@pytest.mark.parametrize("option", ["keep_previews", "delete_all"])
@pytest.mark.parametrize("answered", [False, True])
def test_bird_options_preserve_rows_and_start_at_capture_or_receipt(
    case: Case, option: str, answered: bool
) -> None:
    conn, paths = case
    identifier = seed(conn, paths)[0]
    if answered:
        answer(
            conn, identifier, species_identity="confirmed", species_key="Parus_major"
        )
        received(conn, paths)
    settings = replace(
        DEFAULT_SETTINGS, old_bird_photos=option, original_days=5, preview_days=10
    )
    result = plan(conn, paths, settings=settings)
    base = RECEIVED if answered else datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert result.files[0].decision.rule == (4 if answered else 5)
    assert result.files[0].decision.deadline == base + timedelta(days=5)
    assert result.files[0].decision.due(NOW)
    assert all(row.decision.action == "keep" for row in result.rows)
    for item in result.files[1:]:
        assert item.decision.action == (
            "keep" if option == "keep_previews" else "delete"
        )
        assert item.decision.deadline == (
            None if option == "keep_previews" else base + timedelta(days=10)
        )


@pytest.mark.parametrize("detections", [0, 1])
def test_open_cases_and_empty_images_expire_files_and_rows(
    case: Case, detections: int
) -> None:
    conn, paths = case
    seed(
        conn, paths, detections=detections, review_status="untagged", write_preview=True
    )
    result = plan(conn, paths)
    assert all(
        item.decision.rule == 6 and item.decision.due(NOW)
        for item in result.files + result.rows
    )


def test_one_favorite_on_multi_object_photo_keeps_every_file(case: Case) -> None:
    conn, paths = case
    ids = seed(conn, paths, detections=2, review_status="untagged")
    conn.execute("UPDATE detections SET is_favorite=1 WHERE detection_id=?", (ids[1],))
    conn.commit()
    result = plan(conn, paths)
    assert {item.rule for item in result.objects.values()} == {2, 6}
    assert all(item.decision.action == "keep" for item in result.files)
    rows = {row.key: row.decision for row in result.rows if row.table == "detections"}
    assert rows[ids[0]].due(NOW) and rows[ids[1]].action == "keep"
    assert result.rows[0].decision.action == "keep"


def test_multi_object_photo_uses_latest_deadline(case: Case) -> None:
    conn, paths = case
    ids = seed(conn, paths, detections=2)
    conn.execute(
        "UPDATE detections SET status='rejected', trashed_at=? WHERE detection_id=?",
        ((NOW - timedelta(days=2)).isoformat(), ids[0]),
    )
    conn.commit()
    result = plan(
        conn, paths, settings=replace(DEFAULT_SETTINGS, old_bird_photos="delete_all")
    )
    assert all(
        item.decision.deadline >= NOW + timedelta(days=28) for item in result.files
    )


def test_manual_bird_is_training_protected_and_then_follows_bird_policy(
    case: Case,
) -> None:
    conn, paths = case
    seed(conn, paths, detections=0)
    manual, _ = create_manual_object(
        conn,
        ManualObjectDraft(NAME, BBox(0.2, 0.2, 0.3, 0.3), "Parus_major", "test-manual"),
        PROVENANCE,
        original_path=paths.get_original_path(NAME),
        locale="DE",
    )
    conn.commit()
    assert plan(conn, paths).objects[f"manual:{manual['manual_object_id']}"].rule == 3
    received(conn, paths)
    result = plan(conn, paths)
    assert result.files[0].decision.rule == 4
    assert result.files[0].decision.deadline == RECEIVED + timedelta(days=90)
    assert all(row.decision.action == "keep" for row in result.rows)
    assert any(row.table == "manual_object_revisions" for row in result.rows)


def test_image_statement_counts_as_an_answer_without_objects(case: Case) -> None:
    conn, paths = case
    seed(conn, paths, detections=0)
    answer(conn, None, image_bird_presence="absent")
    assert plan(conn, paths).files[0].decision.rule == 3
    received(conn, paths)
    result = plan(conn, paths)
    assert all(
        item.decision.rule == 4
        and item.decision.deadline == RECEIVED + timedelta(days=14)
        for item in result.files
    )
    assert all(row.decision.action == "keep" for row in result.rows)


def test_missing_original_does_not_wait_or_lose_answer(case: Case) -> None:
    conn, paths = case
    identifier = seed(conn, paths, write_original=False)[0]
    answered_at = NOW - timedelta(days=13)
    record_human_answer(
        conn,
        HumanAnswer(NAME, identifier, object_bird_presence="absent"),
        replace(PROVENANCE, created_at=answered_at.isoformat()),
    )
    conn.commit()
    result = plan(conn, paths)
    assert not result.training_sources and not result.files[0].present
    assert all(
        item.decision.deadline == answered_at + timedelta(days=14)
        for item in result.files
    )
    assert all(row.decision.action == "keep" for row in result.rows)
    preview = deletion_preview(conn, DEFAULT_SETTINGS, NOW, paths=paths)
    assert preview["unknown_deadlines"] == 0 and preview["due_files"] == 0


@pytest.mark.parametrize("option", ["keep_previews", "delete_all"])
def test_bird_without_original_uses_latest_current_fact_date(
    case: Case, option: str
) -> None:
    conn, paths = case
    identifier = seed(conn, paths, write_original=False)[0]
    answered_at = NOW - timedelta(days=10)
    record_human_answer(
        conn,
        HumanAnswer(
            NAME, identifier, species_identity="confirmed", species_key="Parus_major"
        ),
        replace(PROVENANCE, created_at=(answered_at - timedelta(days=2)).isoformat()),
    )
    record_human_answer(
        conn,
        HumanAnswer(NAME, identifier, bbox_quality="suitable"),
        replace(PROVENANCE, created_at=(answered_at + timedelta(days=1)).isoformat()),
    )
    record_human_answer(
        conn,
        HumanAnswer(NAME, identifier, bbox_quality="unsuitable"),
        replace(PROVENANCE, created_at=answered_at.isoformat()),
    )
    conn.commit()
    settings = replace(
        DEFAULT_SETTINGS, old_bird_photos=option, original_days=5, preview_days=20
    )
    result = plan(conn, paths, settings=settings)
    assert not result.training_sources and not result.files[0].present
    assert result.files[0].decision.deadline == answered_at + timedelta(days=5)
    for item in result.files[1:]:
        assert item.decision.action == (
            "keep" if option == "keep_previews" else "delete"
        )
        assert item.decision.deadline == (
            None if option == "keep_previews" else answered_at + timedelta(days=20)
        )
    assert all(row.decision.action == "keep" for row in result.rows)


def test_image_statement_without_original_uses_current_statement_date(
    case: Case,
) -> None:
    conn, paths = case
    seed(conn, paths, detections=0, write_original=False, write_preview=True)
    answered_at = NOW - timedelta(days=15)
    record_human_answer(
        conn,
        HumanAnswer(NAME, None, image_bird_presence="absent"),
        replace(PROVENANCE, created_at=answered_at.isoformat()),
    )
    conn.commit()
    result = plan(conn, paths)
    assert not result.training_sources
    assert all(
        item.decision.deadline == answered_at + timedelta(days=14)
        and item.decision.due(NOW)
        for item in result.files
    )
    assert all(row.decision.action == "keep" for row in result.rows)


def test_missing_original_keeps_matching_receipt_as_deadline_start(case: Case) -> None:
    conn, paths = case
    identifier = seed(conn, paths)[0]
    record_human_answer(
        conn,
        HumanAnswer(NAME, identifier, object_bird_presence="absent"),
        replace(PROVENANCE, created_at=(RECEIVED - timedelta(days=20)).isoformat()),
    )
    conn.commit()
    received(conn, paths)
    paths.get_original_path(NAME).unlink()
    result = plan(conn, paths)
    assert all(
        item.decision.deadline == RECEIVED + timedelta(days=14) for item in result.files
    )


def test_missing_previews_do_not_protect_original_and_missing_files_allow_rows(
    case: Case,
) -> None:
    conn, paths = case
    seed(conn, paths, write_thumbs=False, write_optimized=False)
    assert plan(conn, paths).files[0].decision.due(NOW)
    conn.execute("UPDATE detections SET decision_state='uncertain'")
    conn.commit()
    result = plan(conn, paths)
    assert any(not item.present for item in result.files)
    assert all(row.decision.due(NOW) for row in result.rows)


@pytest.mark.parametrize("mode", ["symlink", "traversal"])
def test_unsafe_paths_are_never_scheduled(
    case: Case, tmp_path: Path, mode: str
) -> None:
    conn, paths = case
    seed(conn, paths, review_status="untagged")
    outside = tmp_path / "outside.webp"
    outside.write_bytes(b"private")
    if mode == "symlink":
        thumb = paths.get_derivative_path(NAME, "optimized")
        thumb.unlink()
        thumb.symlink_to(outside)
    else:
        conn.execute("UPDATE detections SET thumbnail_path='../outside.webp'")
        conn.commit()
    result = plan(conn, paths)
    unsafe = [item for item in result.files if item.error]
    assert unsafe and all(item.decision.action == "keep" for item in unsafe)
    assert outside.read_bytes() == b"private"


def test_planner_and_summary_are_readonly_and_ignore_legacy_settings(
    case: Case,
) -> None:
    conn, paths = case
    seed(conn, paths, review_status="untagged")
    conn.commit()
    before = {p: p.read_bytes() for p in paths.base_dir.rglob("*") if p.is_file()}
    with open_readonly(paths.get_database_path()) as readonly:
        readonly.execute("PRAGMA query_only=ON")
        first = list(
            plan_deletions(readonly, DEFAULT_SETTINGS, NOW, paths=paths, batch_size=1)
        )
        preview = deletion_preview(readonly, DEFAULT_SETTINGS, NOW, paths=paths)
        assert first == list(
            plan_deletions(readonly, DEFAULT_SETTINGS, NOW, paths=paths)
        )
        assert readonly.total_changes == 0
    assert preview["due_files"] == 0
    assert preview["upcoming"][0]["rule"] == 6
    assert before == {
        p: p.read_bytes() for p in paths.base_dir.rglob("*") if p.is_file()
    }


def test_migrated_release_fixture_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        (Path(__file__).parent / "fixtures/human_answers_0_5_9.sql").read_text()
    )
    paths = PathManager(str(output))
    for row in conn.execute("SELECT filename FROM images WHERE original_present=1"):
        original = paths.get_original_path(row[0])
        original.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 200)).save(original)
    _init_schema(conn, existing_installation=True)
    try:
        results = list(plan_deletions(conn, DEFAULT_SETTINGS, NOW, paths=paths))
        classes = Counter(photo.files[0].decision.rule for photo in results)
        assert classes == {1: 1, 3: 11, 4: 1, 5: 2, 6: 5}
        assert not any(
            photo.files[0].decision.due(NOW)
            for photo in results
            if photo.files[0].decision.rule in {1, 3}
        )
        assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
        missing = next(
            photo for photo in results if "missing_original" in photo.filename
        )
        assert not missing.files[0].present and not missing.training_sources
        answered_at = conn.execute(
            "SELECT MAX(created_at) FROM current_human_label_facts WHERE image_filename=?",
            (missing.filename,),
        ).fetchone()[0]
        assert missing.files[0].decision.deadline == datetime.fromisoformat(
            answered_at
        ) + timedelta(days=DEFAULT_SETTINGS.original_days)
        assert all(row.decision.action == "keep" for row in missing.rows)
        assert (
            sum(paths.get_original_path(photo.filename).is_file() for photo in results)
            == 19
        )
        assert sum(photo.files[0].decision.due(NOW) for photo in results) == 2
    finally:
        conn.close()


@pytest.mark.parametrize("detections", [0, 1])
@pytest.mark.parametrize("activation", [None, NOW, NOW - timedelta(days=7)])
def test_old_open_cases_wait_after_latest_activation(
    case: Case, detections: int, activation: datetime | None
) -> None:
    conn, paths = case
    seed(conn, paths, detections=detections, review_status="untagged")
    result = plan(conn, paths, automatic_deletion_enabled_at=activation)
    deadline = (activation or NOW) + timedelta(days=14)
    assert all(
        item.decision.rule == 6 and item.decision.deadline == deadline
        for item in result.files + result.rows
    )
    assert not any(item.decision.due(NOW) for item in result.files + result.rows)
    assert all(item.decision.due(deadline) for item in result.files + result.rows)
    preview = deletion_preview(
        conn,
        DEFAULT_SETTINGS,
        NOW,
        paths=paths,
        automatic_deletion_enabled_at=activation,
    )
    assert preview["due_files"] == preview["due_rows"] == 0
    assert datetime.fromisoformat(preview["upcoming"][0]["deadline"]) == deadline


def test_new_open_case_uses_capture_when_later_than_activation(case: Case) -> None:
    conn, paths = case
    seed(conn, paths, review_status="untagged")
    conn.execute("UPDATE images SET timestamp=?", (NOW.isoformat(),))
    conn.commit()
    result = plan(conn, paths, automatic_deletion_enabled_at=NOW - timedelta(days=10))
    assert all(
        item.decision.deadline == NOW + timedelta(days=14)
        for item in result.files + result.rows
    )


def test_activation_requires_timezone(case: Case) -> None:
    conn, paths = case
    with pytest.raises(ValueError, match="automatic_deletion_enabled_at"):
        list(
            plan_deletions(
                conn,
                DEFAULT_SETTINGS,
                NOW,
                paths=paths,
                automatic_deletion_enabled_at=NOW.replace(tzinfo=None),
            )
        )
