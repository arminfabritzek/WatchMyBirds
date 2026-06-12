"""End-to-end test for the SharpnessJob against a real SQLite DB.

Builds a minimal images.db (just the columns SharpnessJob reads),
plants a couple of crop files in the expected derivatives layout,
runs the job, and asserts the DB rows are populated.
"""

from __future__ import annotations

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
            crop_brightness REAL,
            quality_gallery_ok INTEGER,
            is_gallery_eligible INTEGER DEFAULT 0
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


# ---------------------------------------------------------------------------
# Gallery quality floor (quality_gallery_ok eligibility step)
# ---------------------------------------------------------------------------


def _scored_db(scores: list[float]) -> sqlite3.Connection:
    """In-memory DB with N active detections carrying the given
    sharpness_scores (in detection_id order). quality_gallery_ok and
    is_gallery_eligible start at their defaults."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            status TEXT DEFAULT 'active',
            sharpness_score REAL,
            quality_gallery_ok INTEGER,
            is_gallery_eligible INTEGER DEFAULT 0
        );
        """
    )
    for i, s in enumerate(scores, start=1):
        conn.execute(
            "INSERT INTO detections(detection_id, status, sharpness_score) "
            "VALUES (?, 'active', ?)",
            (i, s),
        )
    conn.commit()
    return conn


def _set_floor(monkeypatch, *, bottom_pct: int, min_scored: int) -> None:
    import web.services.nightly_jobs.sharpness_job as mod

    monkeypatch.setattr(mod, "_gallery_quality_bottom_pct", lambda: bottom_pct)
    monkeypatch.setattr(mod, "_gallery_quality_min_scored", lambda: min_scored)


def test_eligibility_floors_bottom_percentile(monkeypatch):
    """With enough scored crops, the bottom ~15% get quality_gallery_ok=0
    and the rest get 1. The cut is station-relative."""
    _set_floor(monkeypatch, bottom_pct=15, min_scored=10)
    # 100 crops, scores 1..100. Bottom 15% → scores < the 15th-percentile
    # cutoff get 0.
    conn = _scored_db([float(i) for i in range(1, 101)])

    SharpnessJob()._recompute_gallery_eligibility(conn)

    hidden = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok = 0"
    ).fetchone()[0]
    shown = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok = 1"
    ).fetchone()[0]
    assert hidden + shown == 100
    # ~15 hidden (cutoff is the 15th lowest score; strictly-less-than).
    assert 10 <= hidden <= 20
    # The very blurriest crop is hidden, the sharpest is shown.
    assert (
        conn.execute(
            "SELECT quality_gallery_ok FROM detections WHERE detection_id=1"
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT quality_gallery_ok FROM detections WHERE detection_id=100"
        ).fetchone()[0]
        == 1
    )


def test_eligibility_below_min_scored_flags_all_visible(monkeypatch):
    """A tiny station (below GALLERY_QUALITY_MIN_SCORED) hides nothing —
    every scored crop is flagged visible."""
    _set_floor(monkeypatch, bottom_pct=15, min_scored=200)
    conn = _scored_db([float(i) for i in range(1, 51)])  # only 50 scored

    SharpnessJob()._recompute_gallery_eligibility(conn)

    hidden = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok = 0"
    ).fetchone()[0]
    shown = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok = 1"
    ).fetchone()[0]
    assert hidden == 0
    assert shown == 50


def test_eligibility_zero_pct_disables_floor(monkeypatch):
    """GALLERY_QUALITY_BOTTOM_PCT=0 disables the cut: all scored visible."""
    _set_floor(monkeypatch, bottom_pct=0, min_scored=10)
    conn = _scored_db([float(i) for i in range(1, 101)])

    SharpnessJob()._recompute_gallery_eligibility(conn)

    hidden = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE quality_gallery_ok = 0"
    ).fetchone()[0]
    assert hidden == 0


def test_eligibility_leaves_null_scores_untouched(monkeypatch):
    """Un-scored crops (sharpness_score NULL) keep quality_gallery_ok
    NULL — never bulk-flipped — so the reader's COALESCE shows them."""
    _set_floor(monkeypatch, bottom_pct=15, min_scored=10)
    conn = _scored_db([float(i) for i in range(1, 101)])
    # Add 5 un-scored rows.
    for did in range(101, 106):
        conn.execute(
            "INSERT INTO detections(detection_id, status, sharpness_score) "
            "VALUES (?, 'active', NULL)",
            (did,),
        )
    conn.commit()

    SharpnessJob()._recompute_gallery_eligibility(conn)

    null_rows = conn.execute(
        "SELECT COUNT(*) FROM detections "
        "WHERE sharpness_score IS NULL AND quality_gallery_ok IS NULL"
    ).fetchone()[0]
    assert null_rows == 5


def test_eligibility_does_not_touch_is_gallery_eligible(monkeypatch):
    """The quality floor and the AI-pick axis stay orthogonal: the
    eligibility step must never write is_gallery_eligible."""
    _set_floor(monkeypatch, bottom_pct=15, min_scored=10)
    conn = _scored_db([float(i) for i in range(1, 101)])
    # Mark a handful as AI-picks (the aesthetic tagger's column).
    conn.execute(
        "UPDATE detections SET is_gallery_eligible = 1 "
        "WHERE detection_id IN (1, 2, 99, 100)"
    )
    conn.commit()

    SharpnessJob()._recompute_gallery_eligibility(conn)

    still_picks = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE is_gallery_eligible = 1"
    ).fetchone()[0]
    assert still_picks == 4
    # And a floored-out AI-pick keeps its badge while being quality-hidden:
    # det 1 is the blurriest AND an AI-pick — both axes hold independently.
    row = conn.execute(
        "SELECT is_gallery_eligible, quality_gallery_ok "
        "FROM detections WHERE detection_id = 1"
    ).fetchone()
    assert row[0] == 1  # still an AI-pick
    assert row[1] == 0  # but quality-floored
