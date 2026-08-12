"""Shared fixtures-free helpers for tests that record human labels.

Kept out of a ``test_`` module so pytest does not collect it, and separate
from any one contract file so several label-related suites can seed the
same real-SQLite shape without importing each other.
"""

from __future__ import annotations

from tests.conftest import reset_config_in_place
from utils.db import insert_classification, insert_detection, insert_image


def _reset_test_config(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    ingest_dir = tmp_path / "ingest"
    output_dir.mkdir()
    ingest_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("INGEST_DIR", str(ingest_dir))
    monkeypatch.setenv("EDIT_PASSWORD", "test-password")
    reset_config_in_place()
    return output_dir


def _seed(
    conn,
    *,
    filename: str,
    timestamp: str,
    species: str = "Parus_major",
    review_status: str = "untagged",
) -> int:
    existing = conn.execute(
        "SELECT 1 FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    if not existing:
        insert_image(
            conn,
            {
                "filename": filename,
                "timestamp": timestamp,
                "source_id": 1,
                "content_hash": f"hash-{filename}",
            },
        )
        conn.execute(
            "UPDATE images SET review_status = ? WHERE filename = ?",
            (review_status, filename),
        )

    detection_id = insert_detection(
        conn,
        {
            "image_filename": filename,
            "bbox_x": 0.2,
            "bbox_y": 0.2,
            "bbox_w": 0.3,
            "bbox_h": 0.3,
            "od_class_name": "bird",
            "od_confidence": 0.9,
            "od_model_id": "yolo-test",
            "created_at": timestamp,
            "score": 0.95,
            "raw_species_name": species,
            "thumbnail_path": filename.replace(".jpg", "_crop_1.webp"),
        },
    )
    conn.execute(
        "UPDATE detections SET status = 'active' WHERE detection_id = ?",
        (detection_id,),
    )
    insert_classification(
        conn,
        {
            "detection_id": detection_id,
            "cls_class_name": species,
            "cls_confidence": 0.95,
            "cls_model_id": "cls-test",
            "rank": 1,
            "created_at": timestamp,
        },
    )
    conn.commit()
    return detection_id


def post(client, url: str, payload: dict):
    return client.post(url, json=payload, headers={"X-CSRF-Token": "test-csrf-token"})
