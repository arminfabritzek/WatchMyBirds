"""Contracts for the versioned, read-only canonical label bundle."""

from __future__ import annotations

import json
import os
import zipfile

import pytest

from config import get_config
from core.canonical_dataset import build_canonical_dataset
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from utils.db import connection as db_connection
from utils.path_manager import PathManager
from web.services.canonical_dataset_service import (
    render_canonical_bundle_to_path,
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


def test_bundle_is_deterministic_and_export_is_read_only(
    canonical_case, tmp_path
) -> None:
    conn, pm, *_ = canonical_case
    before = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
    first = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    second = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )

    first_path = render_canonical_bundle_to_path(
        first, path_resolver=pm.get_original_path, destination=tmp_path / "first.zip"
    )
    second_path = render_canonical_bundle_to_path(
        second, path_resolver=pm.get_original_path, destination=tmp_path / "second.zip"
    )

    assert first.bundle_id == second.bundle_id
    assert first_path.read_bytes() == second_path.read_bytes()
    assert (
        conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0] == before
    )


def test_bundle_keeps_partial_facts_and_gives_every_decision_reasons(
    canonical_case, tmp_path
) -> None:
    conn, pm, _, partial_ids, *_ = canonical_case
    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    archive_path = render_canonical_bundle_to_path(
        bundle, path_resolver=pm.get_original_path, destination=tmp_path / "bundle.zip"
    )
    archive = zipfile.ZipFile(archive_path)

    facts = _jsonl(archive, "facts.jsonl")
    manifest = _jsonl(archive, "manifest.jsonl")
    partial_od = next(
        row
        for row in manifest
        if row["view"] == "od_positive" and row["detection_id"] == partial_ids[0]
    )

    assert {
        row["fact_type"] for row in facts if row["detection_id"] == partial_ids[0]
    } == {
        "bird_presence",
        "bbox_quality",
        "species_identity",
    }
    assert partial_od["decision"] == "excluded"
    assert "frame_has_unresolved_objects" in partial_od["reasons"]
    assert all(row["reasons"] for row in manifest)


def test_od_cls_negative_and_unresolved_views_are_independent(canonical_case) -> None:
    conn, pm, _, partial_ids, negative_ids, cls_ids = canonical_case
    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    by_key = {
        (row["view"], row.get("detection_id"), row["image_filename"]): row
        for row in bundle.manifest
    }

    assert (
        by_key[("od_positive", partial_ids[0], "20260810_090000_partial.jpg")][
            "decision"
        ]
        == "excluded"
    )
    assert (
        by_key[("cls_positive", cls_ids[0], "20260810_090200_cls.jpg")]["decision"]
        == "included"
    )
    assert (
        by_key[("od_positive", cls_ids[0], "20260810_090200_cls.jpg")]["decision"]
        == "excluded"
    )
    assert (
        by_key[("od_negative", None, "20260810_090100_negative.jpg")]["decision"]
        == "included"
    )
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

    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    decision = next(
        row
        for row in bundle.manifest
        if row["view"] == "od_positive" and row["detection_id"] == partial_ids[0]
    )

    assert decision["decision"] == "included"
    assert decision["reasons"] == ["explicit_object_bird_and_suitable_bbox"]


def test_explicit_transfer_writes_the_same_bundle_bytes(
    canonical_case, tmp_path
) -> None:
    conn, pm, *_ = canonical_case
    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    expected = render_canonical_bundle_to_path(
        bundle,
        path_resolver=pm.get_original_path,
        destination=tmp_path / "expected.zip",
    ).read_bytes()
    destination = tmp_path / "transfer" / "labels.zip"

    written = write_canonical_bundle(
        bundle,
        path_resolver=pm.get_original_path,
        destination=destination,
    )

    assert written == destination
    assert destination.read_bytes() == expected


def test_render_memory_stays_bounded_as_media_volume_grows(
    canonical_case, tmp_path
) -> None:
    """Rendering must not hold every selected image's bytes in memory at once.

    Regression test for GitHub issue #139: the old implementation built a
    ``dict[str, bytes]`` of every entry (including all media) before
    zipping, so peak memory scaled with total media volume. This writes a
    handful of large media files and asserts that Python-level tracked
    memory stays a small, roughly constant fraction of the total media
    size -- not a precise multiplier (that depends on zlib/OS buffering),
    but well under "all files resident at once."
    """
    import tracemalloc

    conn, pm, provenance, *_ = canonical_case
    large_bytes = 8 * 1024 * 1024  # 8 MiB per synthetic file
    file_count = 6  # ~48 MiB total media, one file per candidate image

    for index in range(file_count):
        filename = f"20260811_0{index}0000_large.jpg"
        conn.execute(
            "INSERT INTO images(filename, timestamp, content_hash) VALUES (?, ?, ?)",
            (filename, "2026-08-11T09:00:00+00:00", f"hash-large-{index}"),
        )
        path = pm.get_original_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic, non-zero content so zlib can't shortcut compression.
        path.write_bytes(os.urandom(large_bytes))
        record_human_answer(
            conn,
            HumanAnswer(image_filename=filename, image_bird_presence="absent"),
            provenance,
        )
    conn.commit()

    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    assert len(bundle.media_filenames) >= file_count

    tracemalloc.start()
    try:
        render_canonical_bundle_to_path(
            bundle,
            path_resolver=pm.get_original_path,
            destination=tmp_path / "large.zip",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    total_media_bytes = file_count * large_bytes
    # Streaming through a ~1 MiB copy buffer per file should stay far
    # below the total media volume; give generous headroom for zlib
    # internal state and Python object overhead.
    assert peak < total_media_bytes / 4, (
        f"peak traced memory {peak} bytes is too close to total media "
        f"volume {total_media_bytes} bytes -- media is likely being "
        f"held fully in memory instead of streamed"
    )


def test_failed_render_removes_temp_directory(canonical_case, tmp_path, monkeypatch):
    from web.services import canonical_dataset_service as service

    directory = tmp_path / "temporary"
    directory.mkdir()
    monkeypatch.setattr(service.tempfile, "mkdtemp", lambda **kwargs: str(directory))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service, "_write_bundle_zip", fail)
    with pytest.raises(OSError, match="disk full"):
        service.render_canonical_bundle_to_tempdir(None, path_resolver=lambda _: None)
    assert not directory.exists()


def test_disk_archive_preserves_legacy_compression_and_bytes(canonical_case, tmp_path):
    from io import BytesIO

    from web.services import canonical_dataset_service as service

    conn, pm, *_ = canonical_case
    bundle = build_canonical_dataset(
        conn, media_exists=lambda name: pm.get_original_path(name).is_file()
    )
    entries = service._non_media_entries(bundle)
    for filename in bundle.media_filenames:
        path = pm.get_original_path(filename)
        path.write_bytes((b"repetitive image-like fixture " * 4096) + bytes(range(256)))
        entries[service._media_archive_path(filename)] = path.read_bytes()
    legacy = BytesIO()
    with zipfile.ZipFile(
        legacy, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(entries):
            archive.writestr(service._zip_info(name), entries[name], compresslevel=9)
    result = service.write_canonical_bundle(
        bundle,
        path_resolver=pm.get_original_path,
        destination=tmp_path / "streamed.zip",
    )
    assert result.read_bytes() == legacy.getvalue()
