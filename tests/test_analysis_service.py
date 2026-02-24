import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from web.services import analysis_service


@contextmanager
def _connection_ctx(conn: sqlite3.Connection):
    yield conn


def _build_review_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            deep_scan_last_attempt_at TEXT,
            deep_scan_last_result TEXT,
            deep_scan_attempt_count INTEGER DEFAULT 0
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT
        );
        """
    )
    return conn


# ---------------------------------------------------------------------------
# check_deep_analysis_eligibility
# ---------------------------------------------------------------------------


def test_check_deep_analysis_eligibility_allows_untagged_orphan(monkeypatch):
    conn = _build_review_db()
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("img_a.jpg", "untagged"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    is_ok, reason = analysis_service.check_deep_analysis_eligibility("img_a.jpg")
    assert is_ok is True
    assert reason == ""


def test_check_deep_analysis_eligibility_rejects_image_with_detections(monkeypatch):
    conn = _build_review_db()
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("img_b.jpg", "untagged"),
    )
    conn.execute("INSERT INTO detections (image_filename) VALUES (?)", ("img_b.jpg",))
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    is_ok, reason = analysis_service.check_deep_analysis_eligibility("img_b.jpg")
    assert is_ok is False
    assert "without detections" in reason


def test_check_eligibility_rejects_none_result_without_force(monkeypatch):
    """Images with deep_scan_last_result='none' are ineligible by default."""
    conn = _build_review_db()
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("img_scanned.jpg", "untagged", "none"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    is_ok, reason = analysis_service.check_deep_analysis_eligibility("img_scanned.jpg")
    assert is_ok is False
    assert "no detections" in reason


def test_check_eligibility_allows_none_result_with_force(monkeypatch):
    """force=True bypasses the 'none' result exclusion."""
    conn = _build_review_db()
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("img_scanned.jpg", "untagged", "none"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    is_ok, reason = analysis_service.check_deep_analysis_eligibility(
        "img_scanned.jpg", force=True
    )
    assert is_ok is True
    assert reason == ""


def test_check_eligibility_allows_error_result(monkeypatch):
    """Images with deep_scan_last_result='error' are retried."""
    conn = _build_review_db()
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("img_err.jpg", "untagged", "error"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    is_ok, reason = analysis_service.check_deep_analysis_eligibility("img_err.jpg")
    assert is_ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# submit_analysis_job
# ---------------------------------------------------------------------------


def test_submit_analysis_job_skips_ineligible(monkeypatch):
    monkeypatch.setattr(
        analysis_service,
        "check_deep_analysis_eligibility",
        lambda filename, force=False: (False, "not eligible"),
    )
    calls = []
    monkeypatch.setattr(
        analysis_service.analysis_queue,
        "enqueue",
        lambda item: calls.append(item) or True,
    )

    ok = analysis_service.submit_analysis_job("img_c.jpg")
    assert ok is False
    assert calls == []


# ---------------------------------------------------------------------------
# process_deep_analysis_job
# ---------------------------------------------------------------------------


def test_process_deep_analysis_job_uses_standard_persistence_path(monkeypatch):
    # Eligibility + image path + image load
    monkeypatch.setattr(
        analysis_service,
        "check_deep_analysis_eligibility",
        lambda filename, force=False: (True, ""),
    )
    monkeypatch.setattr(
        analysis_service.gallery_service,
        "get_image_paths",
        lambda output_dir, filename: {"original": Path("/tmp/fake.jpg")},
    )
    monkeypatch.setattr(
        analysis_service.cv2,
        "imread",
        lambda p: np.zeros((100, 200, 3), dtype=np.uint8),
    )

    # Existing detection count lookup (used for crop index offset)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            review_status TEXT,
            deep_scan_last_attempt_at TEXT,
            deep_scan_last_result TEXT,
            deep_scan_attempt_count INTEGER DEFAULT 0
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT
        );
        """
    )
    # Insert an image row so recording works
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("img_d.jpg", "untagged"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    saved_calls = []

    class _FakeCropService:
        def create_classification_crop(self, **kwargs):
            return np.ones((32, 32, 3), dtype=np.uint8)

    class _FakeClassificationService:
        def classify(self, crop_rgb):
            return SimpleNamespace(
                class_name="Parus_major",
                confidence=0.8,
                model_id="cls_model_v1",
            )

    class _FakePersistenceService:
        def save_detection(
            self,
            image_filename,
            detection,
            frame,
            detector_model_id,
            classifier_model_id,
            crop_index,
        ):
            saved_calls.append(
                {
                    "image_filename": image_filename,
                    "detection": detection,
                    "detector_model_id": detector_model_id,
                    "classifier_model_id": classifier_model_id,
                    "crop_index": crop_index,
                }
            )
            return SimpleNamespace(success=True)

    class _FakeDetectionManager:
        SAVE_RESOLUTION_CROP = 512

        def __init__(self):
            self.crop_service = _FakeCropService()
            self.classification_service = _FakeClassificationService()
            self.persistence_service = _FakePersistenceService()
            self.classifier_model_id = ""

        def run_exhaustive_scan(self, frame):
            return [
                {
                    "x1": 10,
                    "y1": 15,
                    "x2": 50,
                    "y2": 65,
                    "confidence": 0.9,
                    "class_name": "bird",
                    "method": "tiled",
                }
            ]

    dm = _FakeDetectionManager()
    analysis_service.process_deep_analysis_job(dm, {"filename": "img_d.jpg"})

    assert len(saved_calls) == 1
    call = saved_calls[0]
    assert call["image_filename"] == "img_d.jpg"
    assert call["detector_model_id"] == "deep_scan_tiled"
    assert call["classifier_model_id"] == "cls_model_v1"
    assert call["crop_index"] == 1

    det = call["detection"]
    assert det.bbox == (10, 15, 50, 65)
    assert det.cls_class_name == "Parus_major"
    assert det.cls_confidence == 0.8
    assert det.score == pytest.approx(0.85)  # 0.5*0.9 + 0.5*0.8
    assert det.agreement_score == 0.8

    # Verify DB recording
    row = conn.execute(
        "SELECT deep_scan_last_result, deep_scan_attempt_count FROM images WHERE filename = ?",
        ("img_d.jpg",),
    ).fetchone()
    assert row["deep_scan_last_result"] == "found"
    assert row["deep_scan_attempt_count"] == 1


# ---------------------------------------------------------------------------
# _fetch_orphan_review_filenames  (nightly sweep filter)
# ---------------------------------------------------------------------------


def test_fetch_orphan_review_excludes_none_result(monkeypatch):
    """Nightly sweep must not re-queue images with deep_scan_last_result='none'."""
    conn = _build_review_db()
    # Orphan that was never scanned → eligible
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("orphan_new.jpg", "untagged"),
    )
    # Orphan already scanned with 'none' → excluded
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("orphan_scanned.jpg", "untagged", "none"),
    )
    # Orphan with 'error' → retried
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("orphan_error.jpg", "untagged", "error"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    filenames = analysis_service._fetch_orphan_review_filenames()
    assert "orphan_new.jpg" in filenames
    assert "orphan_error.jpg" in filenames
    assert "orphan_scanned.jpg" not in filenames


# ---------------------------------------------------------------------------
# count_deep_scan_candidates  (Scope 3)
# ---------------------------------------------------------------------------


def test_count_deep_scan_candidates(monkeypatch):
    conn = _build_review_db()
    # 2 eligible, 1 already scanned (excluded)
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("c1.jpg", "untagged"),
    )
    conn.execute(
        "INSERT INTO images (filename, review_status) VALUES (?, ?)",
        ("c2.jpg", "untagged"),
    )
    conn.execute(
        "INSERT INTO images (filename, review_status, deep_scan_last_result) VALUES (?, ?, ?)",
        ("c3.jpg", "untagged", "none"),
    )
    conn.commit()
    monkeypatch.setattr(
        analysis_service.db_service, "closing_connection", lambda: _connection_ctx(conn)
    )

    count = analysis_service.count_deep_scan_candidates()
    assert count == 2
