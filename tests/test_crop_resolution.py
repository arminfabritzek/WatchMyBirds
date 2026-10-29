"""Older crops are re-rendered at the current size while their original exists."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from core import crop_resolution
from core.crop_refresh import current_detection_bbox, refresh_detection_thumbnail
from core.crop_resolution import upgrade_crops
from core.human_label_core import BBox, HumanAnswer, record_human_answer
from tests.test_canonical_dataset_bundle import canonical_case as canonical_case
from utils.image_ops import resize_square_crop
from utils.path_manager import PathManager
from web.services.human_label_service import refresh_answer_thumbnails
from web.services.nightly_jobs import crop_resolution_job
from web.services.nightly_jobs.crop_resolution_job import CropResolutionJob

ORIGINAL = "20260812_120000_000000.jpg"
CROP = "20260812_120000_000000_crop_1.webp"


@pytest.fixture
def station(tmp_path):
    paths = PathManager(str(tmp_path))
    frame = np.full((1440, 2560, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (1200, 600), (1700, 1100), (0, 200, 255), -1)
    original = paths.get_original_path(ORIGINAL)
    original.parent.mkdir(parents=True)
    cv2.imwrite(str(original), frame)
    thumb = paths.get_derivative_path(CROP, "thumb")
    thumb.parent.mkdir(parents=True)
    cv2.imwrite(str(thumb), np.zeros((256, 256, 3), dtype=np.uint8))

    conn = sqlite3.connect(paths.get_database_path())
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY, image_filename TEXT,
            thumbnail_path TEXT, bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO detections VALUES (1, ?, ?, 0.47, 0.42, 0.2, 0.35)",
        (ORIGINAL, CROP),
    )
    conn.commit()
    yield conn, paths, original, thumb
    conn.close()


def _side(path) -> int:
    with Image.open(path) as image:
        return min(image.size)


def test_small_crop_is_rendered_again_once(station):
    conn, paths, _, thumb = station

    first = upgrade_crops(conn, paths, size=512)
    second = upgrade_crops(conn, paths, size=512)

    assert first["upgraded"] == 1
    assert _side(thumb) == 512
    assert second["upgraded"] == 0
    assert second["done"] == 1


def test_crop_without_original_keeps_its_size(station):
    conn, paths, original, thumb = station
    original.unlink()
    before = thumb.read_bytes()

    result = upgrade_crops(conn, paths, size=512)

    assert result["without_original"] == 1
    assert result["upgraded"] == 0
    assert thumb.read_bytes() == before


def test_stop_request_leaves_crops_untouched(station):
    conn, paths, _, thumb = station

    result = upgrade_crops(conn, paths, size=512, stop_requested=lambda: True)

    assert result["stopped"] is True
    assert _side(thumb) == 256


@pytest.mark.parametrize("correction_first", [True, False])
def test_upgrade_preserves_correction_committed_during_render(
    canonical_case: Any, monkeypatch: pytest.MonkeyPatch, correction_first: bool
) -> None:
    conn, paths, provenance, ids, *_ = canonical_case
    name = "20260810_090000_partial.jpg"
    detection_id = ids[0]
    crop_name = "20260810_090000_partial_crop_1.webp"
    original = paths.get_original_path(name)
    thumb = paths.get_derivative_path(crop_name, "thumb")
    thumb.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((800, 1000, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (100, 160), (350, 400), (0, 200, 255), -1)
    assert cv2.imwrite(str(original), frame)
    assert cv2.imwrite(str(thumb), np.zeros((256, 256, 3), dtype=np.uint8))
    original_bytes = original.read_bytes()
    conn.execute(
        "UPDATE detections SET thumbnail_path=? WHERE detection_id=?",
        (crop_name, detection_id),
    )
    conn.commit()
    new_box = (0.05, 0.05, 0.12, 0.12)
    corrected: list[bytes] = []

    def correct_during_render(**kwargs: Any) -> bool:
        if not correction_first:
            rendered = refresh_detection_thumbnail(**kwargs)
        with sqlite3.connect(paths.get_database_path(), timeout=0.1) as writer:
            writer.row_factory = sqlite3.Row
            record_human_answer(
                writer,
                HumanAnswer(name, detection_id, bbox_correction=BBox(*new_box)),
                provenance,
            )
            writer.commit()
            refresh_answer_thumbnails(writer, [(name, detection_id)], paths)
        corrected.append(thumb.read_bytes())
        if correction_first:
            rendered = refresh_detection_thumbnail(**kwargs)
        return rendered

    monkeypatch.setattr(
        crop_resolution, "refresh_detection_thumbnail", correct_during_render
    )
    first = upgrade_crops(conn, paths)
    second = upgrade_crops(conn, paths)

    assert tuple(current_detection_bbox(conn, detection_id)) == new_box
    assert thumb.read_bytes() == corrected[0]
    assert first["upgraded"] == second["upgraded"] == 0
    assert first["errors"] == second["errors"] == 0
    assert original.read_bytes() == original_bytes
    assert list(thumb.parent.iterdir()) == [thumb]


def test_upgrade_does_not_restore_a_deleted_observation(
    station: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, paths, _, thumb = station

    def delete_during_render(**kwargs: Any) -> bool:
        conn.execute("DELETE FROM detections WHERE detection_id=1")
        conn.commit()
        thumb.unlink()
        return refresh_detection_thumbnail(**kwargs)

    monkeypatch.setattr(
        crop_resolution, "refresh_detection_thumbnail", delete_during_render
    )
    result = upgrade_crops(conn, paths)

    assert result["upgraded"] == result["errors"] == 0
    assert not list(thumb.parent.iterdir())


def test_failed_upgrade_publication_preserves_preview(
    station: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, paths, original, thumb = station
    previous, original_bytes = thumb.read_bytes(), original.read_bytes()
    replace = os.replace

    def fail_publication(source: Path, destination: Path) -> None:
        if destination == thumb:
            raise OSError("replacement refused")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publication)
    result = upgrade_crops(conn, paths)

    assert result["upgraded"] == 0
    assert result["errors"] == 1
    assert thumb.read_bytes() == previous
    assert original.read_bytes() == original_bytes
    assert list(thumb.parent.iterdir()) == [thumb]


def test_upgrade_serializes_publication_with_answer_writes(
    station: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, paths, _, thumb = station
    replace = os.replace
    publications = []

    def publish(source: Path, destination: Path) -> None:
        if destination == thumb:
            with sqlite3.connect(paths.get_database_path(), timeout=0) as writer:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    writer.execute("BEGIN IMMEDIATE")
            publications.append(destination)
        replace(source, destination)

    monkeypatch.setattr(os, "replace", publish)
    result = upgrade_crops(conn, paths)

    assert result["upgraded"] == 1
    assert publications == [thumb]
    assert not conn.in_transaction
    with sqlite3.connect(paths.get_database_path(), timeout=0) as writer:
        writer.execute("BEGIN IMMEDIATE")


def test_upgrade_leaves_callers_transaction_untouched(station: Any) -> None:
    conn, paths, _, thumb = station
    previous = thumb.read_bytes()
    conn.execute("UPDATE detections SET bbox_x=0.1 WHERE detection_id=1")

    with pytest.raises(ValueError, match="own transactions"):
        upgrade_crops(conn, paths)

    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT bbox_x FROM detections").fetchone()[0] == 0.47
    assert thumb.read_bytes() == previous


def test_enlarging_interpolates_instead_of_repeating_pixels():
    checker = (np.indices((4, 4)).sum(axis=0) % 2 * 255).astype(np.uint8)

    enlarged = resize_square_crop(checker, 8)

    assert set(np.unique(enlarged)) - {0, 255}


def test_job_reports_failed_renders(monkeypatch):
    monkeypatch.setattr(
        crop_resolution_job.crop_resolution,
        "run",
        lambda **kwargs: {"done": 1, "total": 1, "upgraded": 0, "errors": 1},
    )
    monkeypatch.setattr(crop_resolution_job, "update_progress", lambda *args: None)

    assert CropResolutionJob().run(threading.Event(), "test") == 1
