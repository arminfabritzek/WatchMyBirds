from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _benchmark_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            review_status TEXT
        );
        CREATE TABLE detections (
            image_filename TEXT,
            status TEXT,
            od_class_name TEXT
        );
        CREATE TABLE label_subjects (
            subject_id INTEGER PRIMARY KEY,
            scope TEXT,
            image_filename TEXT
        );
        CREATE TABLE current_human_label_facts (
            subject_id INTEGER,
            fact_type TEXT,
            answer_value TEXT
        );
        """
    )
    return conn


def test_export_selection_keeps_declared_bird_fps_and_human_positives() -> None:
    exporter = _load_script("export_field_benchmark_set")
    conn = _benchmark_db()
    conn.executemany(
        "INSERT INTO images VALUES (?, ?)",
        [
            ("20260822_180000_000001.jpg", "untagged"),
            ("20260822_180000_000002.jpg", "untagged"),
            ("20260822_180000_000003.jpg", "confirmed"),
            ("20260821_120000_000001.jpg", "confirmed"),
        ],
    )
    conn.executemany(
        "INSERT INTO detections VALUES (?, ?, ?)",
        [
            ("20260822_180000_000001.jpg", "active", "bird"),
            ("20260822_180000_000002.jpg", "active", "person"),
            ("20260822_180000_000003.jpg", "active", "bird"),
        ],
    )
    conn.execute(
        "INSERT INTO label_subjects VALUES (?, ?, ?)",
        (1, "object", "20260821_120000_000001.jpg"),
    )
    conn.execute(
        "INSERT INTO current_human_label_facts VALUES (?, ?, ?)",
        (1, "bird_presence", "present"),
    )

    negatives, positives = exporter._fetch_names(
        conn,
        negative_from="20260822_000000_000000.jpg",
        negative_through="20260822_235959_999999.jpg",
        negative_limit=100,
        positive_limit=100,
    )

    assert negatives == ["20260822_180000_000001.jpg"]
    assert positives == ["20260821_120000_000001.jpg"]


def test_copy_group_refuses_to_mix_with_an_existing_snapshot(tmp_path: Path) -> None:
    exporter = _load_script("export_field_benchmark_set")
    destination = tmp_path / "negative"
    destination.mkdir()
    (destination / "old.jpg").write_bytes(b"old")

    with pytest.raises(ValueError, match="not empty"):
        exporter._copy_group(tmp_path, destination, [])


def test_score_variant_rejects_unreadable_image_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    _load_script("benchmark_detector_variants")
    gate = _load_script("benchmark_field_fp_gate")

    def incomplete_stats(_variant: object, frames: list[str]) -> dict[str, object]:
        return {
            "frames_processed": len(frames) - 1,
            "bird_frames": 0,
            "bird_frame_rate_pct": 0.0,
            "avg_latency_ms": 1.0,
        }

    monkeypatch.setattr(gate, "_benchmark_variant", incomplete_stats)

    with pytest.raises(ValueError, match="negative images"):
        gate._score_variant({"id": "model"}, ["a.jpg"], ["b.jpg"])


def test_wildlife_threshold_rows_show_recall_fp_tradeoff() -> None:
    benchmark = _load_script("benchmark_wildlife_detector")

    rows = benchmark._threshold_rows(
        model_name="mdv6",
        negative_scores=[0.05, 0.25, 0.8, 0.0],
        positive_scores=[0.15, 0.9],
        thresholds=[0.1, 0.5],
        negative_latency_ms=12.345,
        positive_latency_ms=23.456,
    )

    assert rows[0]["field_false_positive_rate_pct"] == 50.0
    assert rows[0]["field_recall_pct"] == 100.0
    assert rows[1]["field_false_positive_rate_pct"] == 25.0
    assert rows[1]["field_recall_pct"] == 50.0
    assert rows[1]["negative_latency_ms"] == 12.35


def test_end_to_end_yolo_decoder_returns_top_animal_score() -> None:
    benchmark = _load_script("benchmark_wildlife_detector")
    assert benchmark is not None
    onnx_benchmark = _load_script("benchmark_wildlife_onnx")
    output = np.array(
        [[[0, 0, 1, 1, 0.2, 0], [0, 0, 1, 1, 0.8, 1], [0, 0, 1, 1, 0.7, 0]]],
        dtype=np.float32,
    )

    assert onnx_benchmark._animal_score(output, animal_class_id=0) == pytest.approx(
        0.7
    )
