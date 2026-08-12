"""Contracts for the versioned, read-only canonical label bundle."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from config import get_config
from core.canonical_dataset import build_canonical_dataset
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from utils.db import connection as db_connection
from utils.path_manager import PathManager
from web.services.canonical_dataset_service import (
    render_canonical_bundle,
    write_canonical_bundle,
)


@pytest.fixture
def canonical_case(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
    db_connection._schema_initialized_paths.clear()
    conn = db_connection.get_connection()
    pm = PathManager(str(tmp_path))
    provenance = LabelProvenance(
        installation_id="0123456789abcdef0123456789abcdef",
        app_version="0.6.0",
        context="normal_correction",
        source_kind="watchmybirds_ui",
        created_at="2026-08-10T10:00:00+00:00",
    )

    def add_image(filename: str, detections: int) -> list[int]:
        conn.execute(
            "INSERT INTO images(filename, timestamp, content_hash) VALUES (?, ?, ?)",
            (filename, "2026-08-10T09:00:00+00:00", f"hash-{filename}"),
        )
        ids = []
        for index in range(detections):
            cursor = conn.execute(
                """
                INSERT INTO detections(
                    image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
                    raw_species_name, detector_model_version,
                    classifier_model_version, frame_width, frame_height,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    0.1 + index * 0.4,
                    0.2,
                    0.25,
                    0.3,
                    "Parus_major",
                    "det-v1",
                    "cls-v2",
                    1000,
                    800,
                    "2026-08-10T09:00:01+00:00",
                ),
            )
            ids.append(int(cursor.lastrowid))
        path = pm.get_original_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("image:" + filename).encode())
        return ids

    partial_ids = add_image("20260810_090000_partial.jpg", 2)
    negative_ids = add_image("20260810_090100_negative.jpg", 1)
    cls_ids = add_image("20260810_090200_cls.jpg", 1)

    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="20260810_090000_partial.jpg",
            detection_id=partial_ids[0],
            object_bird_presence="present",
            bbox_quality="suitable",
            species_identity="unknown",
        ),
        provenance,
    )
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="20260810_090100_negative.jpg",
            image_bird_presence="absent",
        ),
        provenance,
    )
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="20260810_090200_cls.jpg",
            detection_id=cls_ids[0],
            object_bird_presence="present",
            bbox_quality="unsuitable",
            species_identity="corrected",
            species_key="Cyanistes_caeruleus",
        ),
        provenance,
    )
    conn.commit()
    yield conn, pm, provenance, partial_ids, negative_ids, cls_ids
    conn.close()


def _jsonl(archive: zipfile.ZipFile, name: str) -> list[dict]:
    return [json.loads(line) for line in archive.read(name).decode().splitlines()]


def test_bundle_is_deterministic_and_export_is_read_only(canonical_case) -> None:
    conn, pm, *_ = canonical_case
    before = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
    first = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())
    second = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())

    first_bytes = render_canonical_bundle(first, path_resolver=pm.get_original_path)
    second_bytes = render_canonical_bundle(second, path_resolver=pm.get_original_path)

    assert first.bundle_id == second.bundle_id
    assert first_bytes == second_bytes
    assert conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == before


def test_bundle_keeps_partial_facts_and_gives_every_decision_reasons(
    canonical_case,
) -> None:
    conn, pm, _, partial_ids, *_ = canonical_case
    bundle = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())
    archive = zipfile.ZipFile(BytesIO(render_canonical_bundle(bundle, path_resolver=pm.get_original_path)))

    facts = _jsonl(archive, "facts.jsonl")
    manifest = _jsonl(archive, "manifest.jsonl")
    partial_od = next(
        row
        for row in manifest
        if row["view"] == "od_positive" and row["detection_id"] == partial_ids[0]
    )

    assert {row["fact_type"] for row in facts if row["detection_id"] == partial_ids[0]} == {
        "bird_presence",
        "bbox_quality",
        "species_identity",
    }
    assert partial_od["decision"] == "excluded"
    assert "frame_has_unresolved_objects" in partial_od["reasons"]
    assert all(row["reasons"] for row in manifest)


def test_od_cls_negative_and_unresolved_views_are_independent(canonical_case) -> None:
    conn, pm, _, partial_ids, negative_ids, cls_ids = canonical_case
    bundle = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())
    by_key = {(row["view"], row.get("detection_id"), row["image_filename"]): row for row in bundle.manifest}

    assert by_key[("od_positive", partial_ids[0], "20260810_090000_partial.jpg")]["decision"] == "excluded"
    assert by_key[("cls_positive", cls_ids[0], "20260810_090200_cls.jpg")]["decision"] == "included"
    assert by_key[("od_positive", cls_ids[0], "20260810_090200_cls.jpg")]["decision"] == "excluded"
    assert by_key[("od_negative", None, "20260810_090100_negative.jpg")]["decision"] == "included"
    assert negative_ids[0] in bundle.unresolved_detection_ids


def test_resolving_sibling_makes_complete_od_frame_eligible(canonical_case) -> None:
    conn, pm, provenance, partial_ids, *_ = canonical_case
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="20260810_090000_partial.jpg",
            detection_id=partial_ids[1],
            object_bird_presence="absent",
        ),
        provenance,
    )
    conn.commit()

    bundle = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())
    decision = next(
        row
        for row in bundle.manifest
        if row["view"] == "od_positive" and row["detection_id"] == partial_ids[0]
    )

    assert decision["decision"] == "included"
    assert decision["reasons"] == ["explicit_object_bird_and_suitable_bbox"]


def test_explicit_transfer_writes_the_same_bundle_bytes(canonical_case, tmp_path) -> None:
    conn, pm, *_ = canonical_case
    bundle = build_canonical_dataset(conn, media_exists=lambda name: pm.get_original_path(name).is_file())
    expected = render_canonical_bundle(bundle, path_resolver=pm.get_original_path)
    destination = tmp_path / "transfer" / "labels.zip"

    written = write_canonical_bundle(
        bundle,
        path_resolver=pm.get_original_path,
        destination=destination,
    )

    assert written == destination
    assert destination.read_bytes() == expected
