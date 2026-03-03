"""Tests for Offline Decision Matrix Evaluation Script (P2-03)."""

import csv
import sqlite3
from pathlib import Path

import pytest

from scripts.eval_decision_matrix import compute_metrics, export_candidates, main


@pytest.fixture
def sample_detections():
    """Provides a list of sample detection dicts."""
    return [
        {
            "detection_id": 1,
            "image_filename": "a.jpg",
            "od_class_name": "bird",
            "cls_class_name": "Parus_major",
            "od_confidence": 0.9,
            "cls_confidence": 0.85,
            "score": 0.87,
            "decision_state": "confirmed",
            "bbox_quality": 0.95,
            "unknown_score": 0.1,
            "decision_reasons": "[]",
            "policy_version": "v1",
            "created_at": "2026-03-01T10:00:00",
        },
        {
            "detection_id": 2,
            "image_filename": "b.jpg",
            "od_class_name": "bird",
            "cls_class_name": "Cyanistes_caeruleus",
            "od_confidence": 0.8,
            "cls_confidence": 0.4,
            "score": 0.6,
            "decision_state": "uncertain",
            "bbox_quality": 0.8,
            "unknown_score": 0.35,
            "decision_reasons": '["LOW_SPECIES_CONF"]',
            "policy_version": "v1",
            "created_at": "2026-03-01T10:01:00",
        },
        {
            "detection_id": 3,
            "image_filename": "c.jpg",
            "od_class_name": "bird",
            "cls_class_name": "",
            "od_confidence": 0.7,
            "cls_confidence": 0.2,
            "score": 0.45,
            "decision_state": "unknown",
            "bbox_quality": 0.6,
            "unknown_score": 0.85,
            "decision_reasons": '["HIGH_UNKNOWN_SCORE", "LOW_SPECIES_CONF"]',
            "policy_version": "v1",
            "created_at": "2026-03-01T10:02:00",
        },
        {
            "detection_id": 4,
            "image_filename": "d.jpg",
            "od_class_name": "bird",
            "cls_class_name": "",
            "od_confidence": 0.3,
            "cls_confidence": 0.1,
            "score": 0.2,
            "decision_state": "rejected",
            "bbox_quality": 0.2,
            "unknown_score": 0.3,
            "decision_reasons": '["LOW_BBOX_QUALITY", "LOW_SPECIES_CONF"]',
            "policy_version": "v1",
            "created_at": "2026-03-01T10:03:00",
        },
    ]


def test_compute_metrics_state_distribution(sample_detections):
    """Metrics should correctly count each decision state."""
    metrics = compute_metrics(sample_detections)

    assert metrics["total"] == 4
    assert metrics["states"]["confirmed"] == 1
    assert metrics["states"]["uncertain"] == 1
    assert metrics["states"]["unknown"] == 1
    assert metrics["states"]["rejected"] == 1


def test_compute_metrics_averages(sample_detections):
    """Average bbox quality and unknown score should be computed."""
    metrics = compute_metrics(sample_detections)

    assert metrics["avg_bbox_quality"] is not None
    assert 0.0 < metrics["avg_bbox_quality"] < 1.0
    assert metrics["avg_unknown_score"] is not None


def test_compute_metrics_reason_codes(sample_detections):
    """Reason code frequency should match the data."""
    metrics = compute_metrics(sample_detections)

    assert "LOW_SPECIES_CONF" in metrics["reason_codes"]
    assert metrics["reason_codes"]["LOW_SPECIES_CONF"] == 3  # appears in 3 detections


def test_compute_metrics_empty():
    """Empty detections should return zeros without crashing."""
    metrics = compute_metrics([])

    assert metrics["total"] == 0
    assert metrics["states"] == {}
    assert metrics["avg_bbox_quality"] is None


def test_export_candidates_csv(sample_detections, tmp_path):
    """Export should write uncertain/unknown rows to CSV."""
    csv_path = str(tmp_path / "candidates.csv")
    count = export_candidates(sample_detections, csv_path)

    assert count == 2  # uncertain + unknown

    # Verify CSV content
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    states = {row["decision_state"] for row in rows}
    assert states == {"uncertain", "unknown"}


def test_main_with_real_db(tmp_path):
    """
    Integration test: create a minimal DB, run main(), verify exit code.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            od_class_name TEXT,
            od_confidence REAL,
            score REAL,
            decision_state TEXT,
            bbox_quality REAL,
            unknown_score REAL,
            decision_reasons TEXT,
            policy_version TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER,
            cls_class_name TEXT,
            cls_confidence REAL,
            status TEXT DEFAULT 'active'
        );
        INSERT INTO detections VALUES
            (1, 'img1.jpg', 'bird', 0.9, 0.85, 'confirmed', 0.95, 0.1, '[]', 'v1', 'active', '2026-01-01');
        INSERT INTO detections VALUES
            (2, 'img2.jpg', 'bird', 0.7, 0.5, 'uncertain', 0.6, 0.4, '["LOW_SPECIES_CONF"]', 'v1', 'active', '2026-01-02');
    """
    )
    conn.commit()
    conn.close()

    # Run without export
    exit_code = main(["--db", str(db_path)])
    assert exit_code == 0

    # Run with JSON output
    exit_code = main(["--db", str(db_path), "--json"])
    assert exit_code == 0

    # Run with export
    csv_path = str(tmp_path / "export.csv")
    exit_code = main(["--db", str(db_path), "--export", csv_path])
    assert exit_code == 0
    assert Path(csv_path).exists()


def test_main_missing_db(tmp_path):
    """Main should return 1 when DB file doesn't exist."""
    exit_code = main(["--db", str(tmp_path / "nonexistent.db")])
    assert exit_code == 1
