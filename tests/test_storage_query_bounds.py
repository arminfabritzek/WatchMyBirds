"""Small photo lookups must not scale with unrelated station answer history."""

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.canonical_dataset import build_canonical_dataset
from core.deletion_plan import _batch_crop_names
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from tests.labeling_helpers import _reset_test_config, _seed
from utils.db.connection import closing_connection
from utils.db.detections import fetch_detections_for_gallery
from utils.db.storage import (
    historical_images,
    image_batches,
    prepare_canonical_views,
    select_images,
)
from utils.path_manager import PathManager


def _work(conn: sqlite3.Connection, query: Callable[[], object]) -> int:
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    conn.set_progress_handler(progress, 1)
    try:
        query()
    finally:
        conn.set_progress_handler(None, 0)
    return steps


def test_selected_photo_work_stays_bounded_as_unrelated_answers_grow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    name = "20260101_120000_target.jpg"
    with closing_connection() as conn:
        did = _seed(conn, filename=name, timestamp="20260101_120000")
        lookups = [
            lambda: fetch_detections_for_gallery(
                conn, detection_ids=[did], visibility="selected"
            ),
            lambda: list(image_batches(conn, filenames=[name])),
            lambda: conn.execute(
                "SELECT object_key, admitted, presence, has_current_answer "
                "FROM current_observations WHERE image_filename=?",
                (name,),
            ).fetchall(),
            lambda: conn.execute(
                "SELECT 1 FROM detections WHERE thumbnail_path=? "
                "AND image_filename!=? LIMIT 1",
                ("missing.webp", name),
            ).fetchall(),
        ]
        before = [_work(conn, lookup) for lookup in lookups]
        conn.executemany(
            "INSERT INTO images(filename,timestamp) VALUES (?,?)",
            [
                (f"20200101_120000_unrelated_{i}.jpg", "20200101_120000")
                for i in range(2000)
            ],
        )
        conn.execute(
            """INSERT INTO detections(image_filename)
            SELECT filename FROM images WHERE filename != ?""",
            (name,),
        )
        conn.execute(
            """INSERT INTO label_subjects(scope,image_filename,detection_id,
            proposal_bbox_x,proposal_bbox_y,proposal_bbox_w,proposal_bbox_h,
            proposal_coordinate_space,created_at)
            SELECT 'object',image_filename,detection_id,0,0,1,1,
                'frame_fraction_xywh_v1','2020-01-01'
            FROM detections WHERE image_filename!=?""",
            (name,),
        )
        conn.execute("""INSERT INTO human_label_facts(subject_id,fact_type,
            assertion_state,answer_value,context,source_kind,installation_id,
            app_version,created_at)
            SELECT subject_id,'bird_presence','asserted','present',
                'normal_correction','test','test','test','2020-01-01'
            FROM label_subjects""")
        conn.commit()
        after = [_work(conn, lookup) for lookup in lookups]
        for small, large in zip(before, after, strict=True):
            assert large <= small * 4 + 1000, (before, after)


def test_crop_inventory_scans_once_and_preserves_literal_stems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = PathManager(str(tmp_path))
    names = ["20260101_120000_[bird].jpg", "20260101_120001_crop_bird.jpg"]
    folder = paths.get_derivative_path(names[0]).parent
    folder.mkdir(parents=True)
    expected = {name: {Path(name).stem + "_crop_7.webp"} for name in names}
    for filename in [
        *(v for values in expected.values() for v in values),
        "20260101_120000_bird_crop_9.webp",
        "unrelated_crop_1.webp",
    ]:
        (folder / filename).write_bytes(b"fixture")
    calls = []
    original = Path.glob

    def tracked(path: Path, pattern: str) -> Iterator[Path]:
        calls.append(path)
        return original(path, pattern)

    monkeypatch.setattr(Path, "glob", tracked)
    assert _batch_crop_names(paths, names) == expected
    assert calls == [folder]


@pytest.mark.parametrize("storage_views", [False, True])
def test_training_snapshot_work_is_bounded_by_selected_photos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, storage_views: bool
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    name = "20260101_120000_target.jpg"
    with closing_connection() as conn:
        did = _seed(conn, filename=name, timestamp="20260101_120000")
        record_human_answer(
            conn,
            HumanAnswer(name, did, object_bird_presence="present"),
            LabelProvenance("test", "test", "normal_correction", "watchmybirds_ui"),
        )
        conn.commit()
        if storage_views:
            prepare_canonical_views(conn)
            select_images(conn, [name])
        lookups = [
            lambda: build_canonical_dataset(
                conn,
                labeling_installation_id="test",
                filenames=None if storage_views else {name},
                media_exists=lambda _: False,
            ),
        ]
        if storage_views:
            lookups.append(lambda: historical_images(conn))
        before = [_work(conn, lookup) for lookup in lookups]
        conn.executemany(
            "INSERT INTO main.images(filename,timestamp) VALUES (?,?)",
            [
                (f"20200101_120000_unrelated_{i}.jpg", "20200101_120000")
                for i in range(2000)
            ],
        )
        conn.execute(
            "INSERT INTO main.detections(image_filename) "
            "SELECT filename FROM main.images WHERE filename != ?",
            (name,),
        )
        conn.execute(
            """INSERT INTO main.label_subjects(scope,image_filename,detection_id,
            proposal_bbox_x,proposal_bbox_y,proposal_bbox_w,proposal_bbox_h,
            proposal_coordinate_space,created_at)
            SELECT 'object',image_filename,detection_id,0,0,1,1,
                'frame_fraction_xywh_v1','2020-01-01'
            FROM main.detections WHERE image_filename!=?""",
            (name,),
        )
        conn.execute(
            """INSERT INTO main.human_label_facts(subject_id,fact_type,
            assertion_state,answer_value,context,source_kind,installation_id,
            app_version,created_at)
            SELECT subject_id,'bird_presence','asserted','present',
                'normal_correction','test','test','test','2020-01-01'
            FROM main.label_subjects WHERE image_filename!=?""",
            (name,),
        )
        conn.commit()
        after = [_work(conn, lookup) for lookup in lookups]
        for small, large in zip(before, after, strict=True):
            assert large <= small * 4 + 1000, (before, after)


def test_storyboard_tolerates_photo_deleted_between_ranking_and_hydration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from utils.db.detections import fetch_species_story_board_candidates

    _reset_test_config(monkeypatch, tmp_path)
    name = "20260101_120000_removed.jpg"
    with closing_connection() as conn:
        did = _seed(conn, filename=name, timestamp="20260101_120000")
        conn.execute(
            "UPDATE detections SET decision_state='confirmed', decision_level='species'"
        )
        conn.commit()
        assert fetch_species_story_board_candidates(conn)[0]["detection_id"] == did

        class ConcurrentRemoval:
            def execute(self, sql: str, params: object = ()) -> sqlite3.Cursor:
                if sql.startswith(
                    "SELECT * FROM current_observations WHERE image_filename IN"
                ):
                    conn.execute("DELETE FROM detections WHERE detection_id=?", (did,))
                return conn.execute(sql, params)

        assert fetch_species_story_board_candidates(ConcurrentRemoval()) == []
