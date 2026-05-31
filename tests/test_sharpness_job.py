"""End-to-end test for the SharpnessJob against a real SQLite DB.

Builds a minimal images.db (just the columns SharpnessJob reads),
plants a couple of crop files in the expected derivatives layout,
runs the job, and asserts the DB rows are populated.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from web.services.nightly_jobs.sharpness_job import SharpnessJob


def _make_minimal_db(db_path: Path):
    """Build the smallest schema SharpnessJob needs.

    Mirrors the real WMB schema where `images.filename` is the text
    PK and `detections.image_filename` is the FK reference (not an
    integer image_id).
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            thumbnail_path TEXT,
            status TEXT DEFAULT 'active',
            sharpness_score REAL,
            crop_brightness REAL
        );
        """
    )
    conn.commit()
    return conn


def _write_crop(path: Path, size: int = 256, sharp: bool = True):
    """Write a checkerboard crop (sharp) or a uniform crop (blurry)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sharp:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        s = 16
        for y in range(0, size, s):
            for x in range(0, size, s):
                if ((y // s) + (x // s)) % 2 == 0:
                    img[y : y + s, x : x + s] = 255
    else:
        img = np.full((size, size, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Lay out a fake OUTPUT_DIR with crops_root and DB."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    db = output_dir / "images.db"
    crops_root = output_dir / "derivatives" / "thumbs"
    crops_root.mkdir(parents=True)

    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("WMB_DB_PATH", str(db))
    monkeypatch.setenv("WMB_CROPS_ROOT", str(crops_root))
    monkeypatch.setenv("WMB_SHARPNESS_BATCH_SIZE", "10")

    return {"output_dir": output_dir, "db": db, "crops_root": crops_root}


def test_run_scores_unscored_detections(env):
    conn = _make_minimal_db(env["db"])
    # Two detections, one sharp crop, one blurry.
    image_filename = "20260527_120000_cam0.jpg"
    day_dir = "2026-05-27"
    conn.execute(
        "INSERT INTO images(filename) VALUES (?)",
        (image_filename,),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, thumbnail_path) VALUES (?, ?, ?)",
        (1, image_filename, "crop_sharp.webp"),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, thumbnail_path) VALUES (?, ?, ?)",
        (2, image_filename, "crop_blurry.webp"),
    )
    conn.commit()

    _write_crop(env["crops_root"] / day_dir / "crop_sharp.webp", sharp=True)
    _write_crop(env["crops_root"] / day_dir / "crop_blurry.webp", sharp=False)

    job = SharpnessJob()
    stop_event = threading.Event()
    rc = job.run(stop_event, reason="unit-test")
    assert rc == 0

    rows = conn.execute(
        "SELECT detection_id, sharpness_score, crop_brightness "
        "FROM detections ORDER BY detection_id"
    ).fetchall()
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    assert by_id[1][0] is not None and by_id[1][0] > 100  # sharp
    assert by_id[2][0] is not None and by_id[2][0] < 10  # blurry
    # Brightness: sharp checkerboard ~127, blurry uniform = 128
    assert by_id[1][1] is not None
    assert 100 <= by_id[1][1] <= 160
    assert by_id[2][1] == pytest.approx(128.0, abs=1.0)


def test_re_run_does_not_overwrite_existing_scores(env):
    """Once a detection has a sharpness_score, the job skips it."""
    conn = _make_minimal_db(env["db"])
    image_filename = "20260527_120000_cam0.jpg"
    day_dir = "2026-05-27"
    conn.execute(
        "INSERT INTO images(filename) VALUES (?)",
        (image_filename,),
    )
    # Pre-populated row — must not be re-scored.
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, thumbnail_path, "
        "sharpness_score, crop_brightness) VALUES (?, ?, ?, 999.0, 50.0)",
        (1, image_filename, "crop_pre.webp"),
    )
    conn.commit()
    _write_crop(env["crops_root"] / day_dir / "crop_pre.webp", sharp=True)

    job = SharpnessJob()
    job.run(threading.Event(), reason="re-run")

    row = conn.execute(
        "SELECT sharpness_score, crop_brightness FROM detections WHERE detection_id=1"
    ).fetchone()
    assert row[0] == 999.0
    assert row[1] == 50.0


def test_missing_crop_file_is_skipped_not_errored(env, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="web.services.nightly_jobs.sharpness_job")

    conn = _make_minimal_db(env["db"])
    image_filename = "20260527_120000_cam0.jpg"
    conn.execute(
        "INSERT INTO images(filename) VALUES (?)",
        (image_filename,),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, thumbnail_path) VALUES (?, ?, ?)",
        (1, image_filename, "does_not_exist.webp"),
    )
    conn.commit()

    job = SharpnessJob()
    rc = job.run(threading.Event(), reason="missing-file-test")
    assert rc == 0

    row = conn.execute(
        "SELECT sharpness_score, crop_brightness FROM detections WHERE detection_id=1"
    ).fetchone()
    # Missing file → row stays NULL, no crash.
    assert row[0] is None
    assert row[1] is None


def test_stop_event_aborts_mid_batch(env):
    conn = _make_minimal_db(env["db"])
    image_filename = "20260527_120000_cam0.jpg"
    day_dir = "2026-05-27"
    conn.execute(
        "INSERT INTO images(filename) VALUES (?)",
        (image_filename,),
    )
    # Many crops so the stop has work to interrupt.
    for i in range(1, 30):
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, thumbnail_path) VALUES (?, ?, ?)",
            (i, image_filename, f"c_{i}.webp"),
        )
        _write_crop(env["crops_root"] / day_dir / f"c_{i}.webp", sharp=True)
    conn.commit()

    job = SharpnessJob()
    stop_event = threading.Event()
    stop_event.set()  # set IMMEDIATELY → loop should exit before any work

    rc = job.run(stop_event, reason="stopped-immediately")
    assert rc == 0

    scored = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE sharpness_score IS NOT NULL"
    ).fetchone()[0]
    # With stop set before the first iteration, zero rows are scored.
    assert scored == 0


def test_missing_db_returns_failure(env, tmp_path, monkeypatch):
    """A run against a non-existent DB returns rc=1 without crashing."""
    monkeypatch.setenv("WMB_DB_PATH", str(tmp_path / "nonexistent.db"))
    job = SharpnessJob()
    rc = job.run(threading.Event(), reason="missing-db")
    assert rc == 1
