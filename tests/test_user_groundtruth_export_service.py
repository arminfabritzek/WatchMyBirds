"""
Tests for the user-groundtruth export service.

End-to-end coverage of the build → stream → record cycle:

- Batch construction with each bucket combination
- ZIP structure (file names, count, contents)
- COCO validity (images/annotations/categories cross-references)
- Manifest JSONL format and per-bucket provenance
- Idempotent record_batch_exported
- last_batch_until book-keeping for chained windows
- Deterministic builds: same inputs → same payload (modulo batch_id timestamp)

The tests use the same in-memory SQLite pattern as
``test_user_groundtruth_queries.py``, plus a tiny tmp directory of
fake image bytes so the streaming path actually opens files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from web.services.user_groundtruth_export_service import (
    EXPORTER_VERSION,
    build_batch,
    build_batch_id,
    estimate_batch_bytes,
    exclude_image_from_export,
    last_batch_until,
    list_recorded_batches,
    preview_counts,
    record_batch_exported,
    stream_batch_zip,
)

# ---------------------------------------------------------------------------
# Fixtures: in-memory DB with the full schema slice we need
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            review_updated_at TEXT
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            status TEXT DEFAULT 'active',
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_confidence REAL,
            od_class_name TEXT,
            detector_model_version TEXT,
            classifier_model_version TEXT,
            frame_width INTEGER, frame_height INTEGER,
            decision_level TEXT,
            raw_species_name TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            species_updated_at TEXT,
            is_favorite INTEGER DEFAULT 0,
            rating_source TEXT DEFAULT 'auto'
        );
        CREATE TABLE export_batches (
            batch_id TEXT PRIMARY KEY,
            built_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            since_at TEXT,
            until_at TEXT NOT NULL,
            hard_negatives_count INTEGER NOT NULL DEFAULT 0,
            confirmed_positives_count INTEGER NOT NULL DEFAULT 0,
            species_relabels_count INTEGER NOT NULL DEFAULT 0,
            favorites_count INTEGER NOT NULL DEFAULT 0,
            frame_integrity_dropped_count INTEGER NOT NULL DEFAULT 0,
            exporter_version TEXT NOT NULL,
            wmb_app_version TEXT,
            notes TEXT
        );
        CREATE TABLE groundtruth_export_exclusions (
            exclusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            detection_id INTEGER,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT NOT NULL DEFAULT 'dry_run',
            released_at TEXT
        );
        """
    )
    return conn


def _add_image(conn, filename, review_status=None, review_updated_at=None):
    conn.execute(
        "INSERT INTO images (filename, review_status, review_updated_at) "
        "VALUES (?, ?, ?)",
        (filename, review_status, review_updated_at),
    )


def _add_detection(
    conn,
    *,
    image_filename,
    decision_level=None,
    raw_species_name=None,
    manual_species_override=None,
    # Default species_source='manual' so any test that simply marks a
    # row decision_level='species' implicitly satisfies the explicit-
    # user-action gate added to fetch_confirmed_positives. Tests that
    # need to assert pipeline-only behaviour (e.g. "model_top1 rows
    # must NOT export") MUST override this to 'model_top1' or similar.
    species_source="manual",
    species_updated_at=None,
    od_confidence=0.75,
    od_class_name="bird",
    detector_model_version="yolox_v11",
    classifier_model_version="cls_v3",
    bbox=(0.1, 0.1, 0.2, 0.2),
    frame_wh=(1920, 1080),
    status="active",
    is_favorite=0,
    rating_source="auto",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename, status,
            bbox_x, bbox_y, bbox_w, bbox_h,
            od_confidence, od_class_name,
            detector_model_version, classifier_model_version,
            frame_width, frame_height,
            decision_level, raw_species_name,
            manual_species_override, species_source, species_updated_at,
            is_favorite, rating_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_filename,
            status,
            *bbox,
            od_confidence,
            od_class_name,
            detector_model_version,
            classifier_model_version,
            *frame_wh,
            decision_level,
            raw_species_name,
            manual_species_override,
            species_source,
            species_updated_at,
            is_favorite,
            rating_source,
        ),
    )
    new_id = cur.lastrowid
    assert new_id is not None
    return new_id


@pytest.fixture
def tmp_image_dir(tmp_path: Path):
    """Make a tmp dir with date-sharded fake JPGs and return a resolver."""
    base = tmp_path / "originals"

    def write_image(filename: str, content: bytes = b"FAKEJPGBYTES"):
        date = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
        d = base / date
        d.mkdir(parents=True, exist_ok=True)
        (d / filename).write_bytes(content)

    def resolver(filename: str) -> Path:
        date = f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}"
        return base / date / filename

    return resolver, write_image


# ---------------------------------------------------------------------------
# build_batch_id
# ---------------------------------------------------------------------------


def test_build_batch_id_format():
    bid = build_batch_id(datetime(2026, 5, 22, 20, 0, 0, tzinfo=UTC))
    # ISO week 21 of 2026
    assert bid.startswith("2026W21_")
    assert len(bid) == len("2026W21_") + 4
    # Suffix is hex
    int(bid.split("_")[1], 16)


def test_build_batch_id_distinct_in_same_week():
    """Two builds in the same week, different seconds → different IDs."""
    bid1 = build_batch_id(datetime(2026, 5, 22, 20, 0, 0, tzinfo=UTC))
    bid2 = build_batch_id(datetime(2026, 5, 22, 20, 0, 1, tzinfo=UTC))
    assert bid1 != bid2


# ---------------------------------------------------------------------------
# build_batch
# ---------------------------------------------------------------------------


def test_build_batch_empty_db(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    assert batch.is_empty
    assert batch.total_rows == 0
    assert batch.counts == {
        "hard_negatives": 0,
        "confirmed_positives": 0,
        "species_relabels": 0,
        "favorites": 0,
    }
    assert batch.until  # always populated when until=None means "now"


def test_build_batch_picks_up_all_three_buckets(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()

    # HN
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    write("20260522_100000_hn.jpg")

    # CP
    _add_image(conn, "20260522_110000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )
    write("20260522_110000_cp.jpg")

    # RL
    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_rl.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        wmb_app_version="0.42.1",
        now=datetime(2026, 5, 22, 13, 0, 0, tzinfo=UTC),
        include_confirmed_positives=True,
    )

    assert batch.counts == {
        "hard_negatives": 1,
        "confirmed_positives": 2,  # the RL row also counts as a positive
        "species_relabels": 1,
        "favorites": 0,
    }
    assert batch.wmb_app_version == "0.42.1"


def test_build_batch_default_excludes_confirmed_positives(tmp_image_dir):
    """The confirmed_positives bucket is off by default to avoid the
    confirmation-bias loop: Review-Confirm stamps species_source=
    'manual' whether the user authored the label or merely accepted
    the model's prediction. Same fixture as the all-three-buckets
    test, but no opt-in flag — confirmed_positives must be 0."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    write("20260522_100000_hn.jpg")
    _add_image(conn, "20260522_110000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )
    write("20260522_110000_cp.jpg")
    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_rl.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 13, 0, 0, tzinfo=UTC),
    )

    assert batch.counts == {
        "hard_negatives": 1,
        "confirmed_positives": 0,  # opt-out default
        "species_relabels": 1,
        "favorites": 0,
    }
    assert batch.confirmed_positives == []


def test_build_batch_caches_image_paths_across_buckets(tmp_image_dir):
    """The image-paths map is computed once. Even when a frame appears
    in two buckets (RL is also a CP), the path lookup happens once."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_120000_dual.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_dual.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_dual.jpg")

    calls: list[str] = []

    def counting_resolver(fn):
        calls.append(fn)
        return resolver(fn)

    build_batch(
        conn,
        path_resolver=counting_resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    # Exactly one resolver call per unique filename — the dual-bucket
    # row must not trigger two lookups.
    assert calls == ["20260522_120000_dual.jpg"]


# ---------------------------------------------------------------------------
# stream_batch_zip — structure
# ---------------------------------------------------------------------------


def test_stream_batch_zip_has_all_required_files(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_x.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_x.jpg")
    write("20260522_100000_x.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())

    assert "coco_annotations.json" in names
    assert "batch_metadata.json" in names
    assert "README.md" in names
    assert "manifests/hard_negatives.jsonl" in names
    assert "manifests/confirmed_positives.jsonl" in names
    assert "manifests/species_relabels.jsonl" in names
    assert "images/2026-05-22/20260522_100000_x.jpg" in names


def test_stream_batch_zip_empty_batch_still_valid_structure(tmp_image_dir):
    """A zero-row batch still produces a structurally valid ZIP —
    the operator's preview-page can hit Build on an empty window and
    get a sensible empty ZIP rather than an exception."""
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        # Manifests exist but are empty
        assert zf.read("manifests/hard_negatives.jsonl") == b""
        coco = json.loads(zf.read("coco_annotations.json"))
        assert coco["images"] == []
        assert coco["annotations"] == []
        assert coco["categories"] == []


def test_stream_batch_zip_skips_images_when_disabled(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_x.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_x.jpg")
    write("20260522_100000_x.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch, include_images=False)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    # No image files but manifests + COCO still there
    assert not any(n.startswith("images/") for n in names)
    assert "coco_annotations.json" in names


def test_stream_batch_zip_missing_image_recorded_in_metadata(tmp_image_dir):
    """If a DB row references a file that no longer exists on disk
    (e.g. operator deleted the source frame after labeling), the
    export must not crash AND must surface the issue in metadata."""
    resolver, _ = tmp_image_dir  # NB: not calling write() — file missing
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_missing.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_missing.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))
    assert "20260522_100000_missing.jpg" in meta["missing_images_on_disk"]


# ---------------------------------------------------------------------------
# COCO validity
# ---------------------------------------------------------------------------


def test_coco_has_species_categories_from_positives_and_relabels(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    # CP with one species
    _add_image(conn, "20260522_110000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )
    write("20260522_110000_cp.jpg")
    # RL with corrected species
    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        decision_level="species_review",
        raw_species_name="Parus_major",
        manual_species_override="Erithacus_rubecula",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_rl.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        coco = json.loads(zf.read("coco_annotations.json"))

    names = {c["name"] for c in coco["categories"]}
    assert names == {"Parus_major", "Erithacus_rubecula"}


def test_coco_annotation_bbox_is_pixels_not_normalized(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_110000_x.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_x.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
        bbox=(0.25, 0.5, 0.1, 0.2),
        frame_wh=(1920, 1080),
    )
    write("20260522_110000_x.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        coco = json.loads(zf.read("coco_annotations.json"))

    ann = coco["annotations"][0]
    # 0.25 * 1920 = 480; 0.5 * 1080 = 540; 0.1 * 1920 = 192; 0.2 * 1080 = 216
    assert ann["bbox"] == [480, 540, 192, 216]
    assert ann["area"] == 192 * 216


def test_coco_hard_negative_appears_as_image_with_zero_annotations(tmp_image_dir):
    """Hard-negative crops have NO species label — they ship as COCO
    images with zero corresponding annotations. That is the COCO-standard
    way to express 'this frame has no objects of the target classes'."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(
        conn,
        image_filename="20260522_100000_hn.jpg",
        decision_level="species_review",
        raw_species_name="Parus_major",
    )
    write("20260522_100000_hn.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        coco = json.loads(zf.read("coco_annotations.json"))
        # And the manifest still has the row, with full provenance
        hn_manifest = zf.read("manifests/hard_negatives.jsonl").decode().strip()

    assert len(coco["images"]) == 1
    assert coco["annotations"] == []
    assert hn_manifest, "hard-negative manifest must list the row"


def test_coco_image_ids_match_annotation_image_ids(tmp_image_dir):
    """A COCO consumer must be able to join annotations.image_id ->
    images.id. Bad-faith test: throw a mix of positives + HN + RL
    and assert every annotation points at an existing image."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    for i, sp in enumerate(
        ("Parus_major", "Cyanistes_caeruleus", "Erithacus_rubecula")
    ):
        fn = f"2026052{i}_120000_cp.jpg"
        _add_image(conn, fn)
        _add_detection(
            conn,
            image_filename=fn,
            decision_level="species",
            raw_species_name=sp,
            species_updated_at=f"2026-05-2{i}T12:00:00Z",
        )
        write(fn)

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        coco = json.loads(zf.read("coco_annotations.json"))

    img_ids = {img["id"] for img in coco["images"]}
    assert all(ann["image_id"] in img_ids for ann in coco["annotations"])
    cat_ids = {c["id"] for c in coco["categories"]}
    assert all(ann["category_id"] in cat_ids for ann in coco["annotations"])


# ---------------------------------------------------------------------------
# Manifest JSONL provenance
# ---------------------------------------------------------------------------


def test_manifest_jsonl_has_required_provenance_fields(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(
        conn,
        image_filename="20260522_100000_hn.jpg",
        od_confidence=0.687,
        detector_model_version="yolox_v11",
        classifier_model_version="cls_20260522",
    )
    write("20260522_100000_hn.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        line = zf.read("manifests/hard_negatives.jsonl").decode().strip()

    row = json.loads(line)
    assert row["bucket"] == "hard_negatives"
    assert row["user_action_at"] == "2026-05-22T10:00:00Z"
    assert row["batch_id"] == batch.batch_id
    assert row["model_at_inference"]["detector"] == "yolox_v11"
    assert row["model_at_inference"]["classifier"] == "cls_20260522"
    assert row["raw_pipeline_output"]["od_confidence"] == 0.687
    assert row["image_path_in_zip"] == "images/2026-05-22/20260522_100000_hn.jpg"


def test_manifest_relabel_carries_both_species(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_source="manual_relabel",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_rl.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        line = zf.read("manifests/species_relabels.jsonl").decode().strip()

    row = json.loads(line)
    assert row["model_predicted_species"] == "Parus_major"
    assert row["user_corrected_species"] == "Cyanistes_caeruleus"
    assert row["species_source"] == "manual_relabel"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_builds_with_identical_inputs_produce_identical_payloads(tmp_image_dir):
    """Re-running build with identical inputs at the same simulated
    `now` must produce identical COCO and manifest payloads. The
    batch_id is also stable when `now` is fixed."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_110000_x.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_x.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )
    write("20260522_110000_x.jpg")

    fixed_now = datetime(2026, 5, 22, 20, 0, 0, tzinfo=UTC)
    b1 = build_batch(
        conn,
        path_resolver=resolver,
        now=fixed_now,
        include_confirmed_positives=True,
    )
    b2 = build_batch(
        conn,
        path_resolver=resolver,
        now=fixed_now,
        include_confirmed_positives=True,
    )

    assert b1.batch_id == b2.batch_id

    buf1 = stream_batch_zip(b1, include_images=False)
    buf2 = stream_batch_zip(b2, include_images=False)
    with zipfile.ZipFile(buf1) as zf1, zipfile.ZipFile(buf2) as zf2:
        assert zf1.read("coco_annotations.json") == zf2.read("coco_annotations.json")
        assert zf1.read("manifests/confirmed_positives.jsonl") == zf2.read(
            "manifests/confirmed_positives.jsonl"
        )


# ---------------------------------------------------------------------------
# record_batch_exported + last_batch_until
# ---------------------------------------------------------------------------


def test_record_batch_exported_writes_row(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn,
        path_resolver=resolver,
        wmb_app_version="0.99.0",
        now=datetime(2026, 5, 22, 20, 0, 0, tzinfo=UTC),
    )
    record_batch_exported(conn, batch, notes="test run")
    rows = conn.execute(
        "SELECT batch_id, until_at, exporter_version, wmb_app_version, notes "
        "FROM export_batches"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["batch_id"] == batch.batch_id
    assert rows[0]["exporter_version"] == EXPORTER_VERSION
    assert rows[0]["wmb_app_version"] == "0.99.0"
    assert rows[0]["notes"] == "test run"


def test_record_batch_exported_is_idempotent(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    record_batch_exported(conn, batch)
    record_batch_exported(conn, batch)  # second call same batch_id
    n = conn.execute("SELECT COUNT(*) FROM export_batches").fetchone()[0]
    assert n == 1


def test_last_batch_until_returns_most_recent(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    b1 = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 15, 23, 59, 59, tzinfo=UTC)
    )
    record_batch_exported(conn, b1)
    b2 = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    record_batch_exported(conn, b2)

    last = last_batch_until(conn)
    assert last == b2.until


def test_last_batch_until_empty_db_returns_none():
    conn = _make_conn()
    assert last_batch_until(conn) is None


def test_chained_batches_dont_double_count(tmp_image_dir):
    """The next batch's `since` is the previous batch's `until`. A
    detection labeled at time T appears in EXACTLY one batch — never
    in both adjacent windows."""
    resolver, write = tmp_image_dir
    conn = _make_conn()

    # First HN at T0
    _add_image(
        conn,
        "20260520_100000_a.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-20T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260520_100000_a.jpg")
    write("20260520_100000_a.jpg")

    # Build batch 1 with until=T1
    t1 = "2026-05-21T00:00:00+00:00"
    b1 = build_batch(
        conn,
        path_resolver=resolver,
        since=None,
        until=t1,
        now=datetime(2026, 5, 21, 23, 59, 59, tzinfo=UTC),
    )
    record_batch_exported(conn, b1)
    assert b1.counts["hard_negatives"] == 1

    # Second HN at T2 (between T1 and now)
    _add_image(
        conn,
        "20260521_100000_b.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-21T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260521_100000_b.jpg")
    write("20260521_100000_b.jpg")

    # Build batch 2: since = b1.until
    b2 = build_batch(
        conn,
        path_resolver=resolver,
        since=b1.until,
        until=None,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
    )
    assert b2.counts["hard_negatives"] == 1  # only the new one
    # And the old one is NOT in this batch
    hn_filenames = {r["image_filename"] for r in b2.hard_negatives}
    assert "20260520_100000_a.jpg" not in hn_filenames
    assert "20260521_100000_b.jpg" in hn_filenames


# ---------------------------------------------------------------------------
# preview_counts
# ---------------------------------------------------------------------------


def test_preview_counts_matches_what_build_would_produce(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    _add_image(conn, "20260522_110000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )

    preview = preview_counts(conn)
    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 0, 0, tzinfo=UTC)
    )
    assert preview == batch.counts
    # Default opt-out: a row that would qualify as a confirmed_positive
    # is invisible to both preview and build.
    assert preview["confirmed_positives"] == 0


def test_preview_and_build_agree_when_confirmed_positives_opted_in(tmp_image_dir):
    """Same fixture, opt-in path: preview and build both report the
    CP row. The opt-in default flips in lockstep on both sides so
    the operator never sees a number that the ZIP would not match."""
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_110000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_110000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T11:00:00Z",
    )

    preview = preview_counts(conn, include_confirmed_positives=True)
    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 0, 0, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    assert preview == batch.counts
    assert preview["confirmed_positives"] == 1


def test_exclude_image_from_export_removes_frame_from_batch(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_100000_cp.jpg")
    _add_detection(
        conn,
        image_filename="20260522_100000_cp.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T10:00:00Z",
    )
    write("20260522_100000_cp.jpg")

    inserted = exclude_image_from_export(
        conn,
        image_filename="20260522_100000_cp.jpg",
        reason="bad dry-run row",
        now=datetime(2026, 5, 22, 10, 5, tzinfo=UTC),
    )
    inserted_again = exclude_image_from_export(
        conn,
        image_filename="20260522_100000_cp.jpg",
        reason="bad dry-run row",
        now=datetime(2026, 5, 22, 10, 6, tzinfo=UTC),
    )
    batch = build_batch(
        conn, path_resolver=resolver, include_confirmed_positives=True
    )

    assert inserted is True
    assert inserted_again is False
    assert batch.counts["confirmed_positives"] == 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM groundtruth_export_exclusions "
            "WHERE image_filename = ? AND released_at IS NULL",
            ("20260522_100000_cp.jpg",),
        ).fetchone()[0]
        == 1
    )


def test_build_batch_handles_local_naive_no_bird_timestamps(tmp_image_dir, monkeypatch):
    """Review no-bird writes used local naive timestamps for a while.

    The export builder uses a UTC ``until`` by default, so the time
    window must normalize both sides before comparing. Otherwise a
    Berlin-local ``21:41`` action is lexicographically after a
    same-instant-ish UTC ``19:45`` build cutoff and preview/build
    disagree.
    """
    resolver, write = tmp_image_dir
    previous_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Europe/Berlin")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        conn = _make_conn()
        _add_image(
            conn,
            "20260523_214156_hn.jpg",
            review_status="no_bird",
            review_updated_at="2026-05-23T21:41:56.329955",
        )
        _add_detection(conn, image_filename="20260523_214156_hn.jpg")
        write("20260523_214156_hn.jpg")

        batch = build_batch(
            conn,
            path_resolver=resolver,
            now=datetime(2026, 5, 23, 19, 45, 0, tzinfo=UTC),
        )
    finally:
        if previous_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", previous_tz)
        if hasattr(time, "tzset"):
            time.tzset()

    assert batch.counts["hard_negatives"] == 1


# ---------------------------------------------------------------------------
# Favorites bucket
# ---------------------------------------------------------------------------


def test_favorites_appear_in_batch_and_zip(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_120000_fav.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_fav.jpg",
        decision_level="species",
        raw_species_name="Erithacus_rubecula",
        is_favorite=1,
        rating_source="manual",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_fav.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    assert batch.counts["favorites"] == 1
    assert batch.counts["confirmed_positives"] == 1  # same row is both

    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
        assert "manifests/favorites.jsonl" in names
        fav_line = zf.read("manifests/favorites.jsonl").decode().strip()
        coco = json.loads(zf.read("coco_annotations.json"))

    fav = json.loads(fav_line)
    assert fav["bucket"] == "favorites"
    assert fav["is_gold_label"] is True
    assert fav["species"] == "Erithacus_rubecula"

    # Erithacus is a category in COCO
    assert any(c["name"] == "Erithacus_rubecula" for c in coco["categories"])
    # And the dedupe rule: exactly ONE annotation for this detection
    assert len(coco["annotations"]) == 1
    # tagged as the higher-priority bucket (favorites > confirmed_positives)
    assert coco["annotations"][0]["wmb_bucket"] == "favorites"


def test_favorites_with_auto_rating_source_excluded(tmp_image_dir):
    """Defense-in-depth: rating_source='auto' must not slip into the
    favorites bucket even with is_favorite=1."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_120000_x.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_x.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        is_favorite=1,
        rating_source="auto",  # legacy backfill
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_x.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )
    assert batch.counts["favorites"] == 0


# ---------------------------------------------------------------------------
# Frame-integrity filter
# ---------------------------------------------------------------------------


def test_frame_integrity_drops_frame_with_unconfirmed_sibling(tmp_image_dir):
    """A frame where one box is species-confirmed and a sibling is
    still species_review must be DROPPED entirely from the positive
    bucket — exporting a half-annotated frame would teach the trainer
    that the unannotated box is background."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_130000_mixed.jpg")
    # Confirmed sibling
    confirmed_id = _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T13:00:00Z",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    # Unconfirmed sibling (species_review — pipeline uncertain, no user touch)
    missing_id = _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species_review",
        raw_species_name="Cyanistes_caeruleus",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_mixed.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )

    # The confirmed-positive count drops to 0 because the only frame
    # with a CP signal has a mixed-status sibling.
    assert batch.counts["confirmed_positives"] == 0
    # And the audit list records the drop with a human-readable reason.
    assert len(batch.frame_integrity_dropped) == 1
    drop = batch.frame_integrity_dropped[0]
    assert drop["image_filename"] == "20260522_130000_mixed.jpg"
    assert "unconfirmed" in drop["reason"]
    assert drop["dropped_detection_ids"] == [confirmed_id]
    assert drop["missing_detection_ids"] == [missing_id]
    assert len(drop["active_siblings"]) == 2
    sibling_by_id = {
        sibling["detection_id"]: sibling for sibling in drop["active_siblings"]
    }
    assert sibling_by_id[confirmed_id]["is_positive_signal"] is True
    assert sibling_by_id[confirmed_id]["positive_buckets"] == ["confirmed_positives"]
    assert sibling_by_id[missing_id]["is_positive_signal"] is False
    assert sibling_by_id[missing_id]["bbox_x"] == 0.5


def test_frame_integrity_keeps_frame_when_all_siblings_confirmed(tmp_image_dir):
    """Two species-confirmed siblings on the same frame: both export,
    nothing dropped."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_130000_clean.jpg")
    _add_detection(
        conn,
        image_filename="20260522_130000_clean.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T13:00:00Z",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    _add_detection(
        conn,
        image_filename="20260522_130000_clean.jpg",
        decision_level="species",
        raw_species_name="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T13:00:01Z",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_clean.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )

    assert batch.counts["confirmed_positives"] == 2
    assert batch.frame_integrity_dropped == []


def test_frame_integrity_mixed_sources_count_as_confirmed(tmp_image_dir):
    """A frame where one sibling is species-confirmed and another is
    a relabel (also user-confirmed) — both are positive signals, frame
    qualifies, neither is dropped."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_130000_mixsrc.jpg")
    # Sibling 1: confirmed_positive
    _add_detection(
        conn,
        image_filename="20260522_130000_mixsrc.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T13:00:00Z",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    # Sibling 2: species relabel (user explicitly corrected)
    _add_detection(
        conn,
        image_filename="20260522_130000_mixsrc.jpg",
        decision_level="species_review",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T13:00:01Z",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_mixsrc.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )

    assert batch.counts["confirmed_positives"] == 2
    assert batch.counts["species_relabels"] == 1
    assert batch.frame_integrity_dropped == []


def test_frame_integrity_does_not_touch_hard_negatives(tmp_image_dir):
    """no_bird is a frame-level statement — every box is FP by
    definition, no integrity check needed even with multi-box frames."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_130000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T13:00:00Z",
    )
    _add_detection(
        conn,
        image_filename="20260522_130000_hn.jpg",
        decision_level="species_review",
        raw_species_name="Parus_major",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    _add_detection(
        conn,
        image_filename="20260522_130000_hn.jpg",
        decision_level="reject",
        raw_species_name="Cyanistes_caeruleus",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_hn.jpg")

    batch = build_batch(
        conn, path_resolver=resolver, now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC)
    )

    # Both boxes flow through as HN — no integrity drop.
    assert batch.counts["hard_negatives"] == 2
    assert batch.frame_integrity_dropped == []


def test_frame_integrity_audit_appears_in_metadata(tmp_image_dir):
    """The dropped-frames audit must be in batch_metadata.json so
    Pipeline-Dev sees why X manifest rows are missing from COCO."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_130000_mixed.jpg")
    _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T13:00:00Z",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species_review",
        raw_species_name="Cyanistes_caeruleus",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_mixed.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))

    assert len(meta["frame_integrity_dropped"]) == 1
    assert (
        meta["frame_integrity_dropped"][0]["image_filename"]
        == "20260522_130000_mixed.jpg"
    )


def test_frame_integrity_count_recorded_in_export_batches(tmp_image_dir):
    """record_batch_exported must persist the dropped-frames count so
    a future operator can audit "did this batch drop anything?"
    without re-reading the ZIP."""
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(conn, "20260522_130000_mixed.jpg")
    _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        species_updated_at="2026-05-22T13:00:00Z",
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    _add_detection(
        conn,
        image_filename="20260522_130000_mixed.jpg",
        decision_level="species_review",
        raw_species_name="Cyanistes_caeruleus",
        bbox=(0.5, 0.5, 0.2, 0.2),
    )
    write("20260522_130000_mixed.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        include_confirmed_positives=True,
    )
    record_batch_exported(conn, batch)

    row = conn.execute(
        "SELECT frame_integrity_dropped_count, favorites_count "
        "FROM export_batches WHERE batch_id = ?",
        (batch.batch_id,),
    ).fetchone()
    assert row["frame_integrity_dropped_count"] == 1
    assert row["favorites_count"] == 0


# ---------------------------------------------------------------------------
# list_recorded_batches
# ---------------------------------------------------------------------------


def test_list_recorded_batches_empty_db_returns_empty_list():
    conn = _make_conn()
    assert list_recorded_batches(conn) == []


def test_list_recorded_batches_returns_newest_first(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    older = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 15, 23, 59, 59, tzinfo=UTC),
    )
    record_batch_exported(conn, older, notes="older")
    newer = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
    )
    record_batch_exported(conn, newer, notes="newer")

    history = list_recorded_batches(conn)
    assert len(history) == 2
    assert history[0]["batch_id"] == newer.batch_id
    assert history[0]["notes"] == "newer"
    assert history[1]["batch_id"] == older.batch_id
    assert history[0]["counts"] == {
        "hard_negatives": 0,
        "confirmed_positives": 0,
        "species_relabels": 0,
        "favorites": 0,
    }
    assert history[0]["exporter_version"] == EXPORTER_VERSION


def test_list_recorded_batches_respects_limit(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    for i in range(5):
        batch = build_batch(
            conn,
            path_resolver=resolver,
            now=datetime(2026, 5, 10 + i, 23, 59, 59, tzinfo=UTC),
        )
        record_batch_exported(conn, batch)
    history = list_recorded_batches(conn, limit=3)
    assert len(history) == 3
    # Newest first → days 14, 13, 12
    days = [int(row["built_at"][8:10]) for row in history]
    assert days == [14, 13, 12]


def test_list_recorded_batches_zero_or_negative_limit_returns_empty(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
    )
    record_batch_exported(conn, batch)
    assert list_recorded_batches(conn, limit=0) == []
    assert list_recorded_batches(conn, limit=-1) == []


# ---------------------------------------------------------------------------
# estimate_batch_bytes
# ---------------------------------------------------------------------------


def test_estimate_batch_bytes_sums_file_sizes(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    write("20260522_100000_hn.jpg", content=b"X" * 1000)

    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T12:00:00Z",
    )
    write("20260522_120000_rl.jpg", content=b"Y" * 500)

    est = estimate_batch_bytes(conn, path_resolver=resolver)
    assert est["bytes"] == 1500
    assert est["image_count"] == 2
    assert est["missing_count"] == 0


def test_estimate_batch_bytes_counts_missing_files_separately(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    write("20260522_100000_hn.jpg", content=b"X" * 800)

    # Second row references a file that was never written to disk.
    _add_image(conn, "20260522_120000_rl.jpg")
    _add_detection(
        conn,
        image_filename="20260522_120000_rl.jpg",
        decision_level="species",
        raw_species_name="Parus_major",
        manual_species_override="Cyanistes_caeruleus",
        species_updated_at="2026-05-22T12:00:00Z",
    )

    est = estimate_batch_bytes(conn, path_resolver=resolver)
    assert est["bytes"] == 800
    assert est["image_count"] == 1
    assert est["missing_count"] == 1


def test_estimate_batch_bytes_empty_window_is_zero(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    est = estimate_batch_bytes(conn, path_resolver=resolver)
    assert est == {"bytes": 0, "image_count": 0, "missing_count": 0}


# ---------------------------------------------------------------------------
# Operator provenance (station_id / reviewer_id)
# ---------------------------------------------------------------------------


def test_build_batch_persists_station_and_reviewer_into_metadata(tmp_image_dir):
    resolver, write = tmp_image_dir
    conn = _make_conn()
    _add_image(
        conn,
        "20260522_100000_hn.jpg",
        review_status="no_bird",
        review_updated_at="2026-05-22T10:00:00Z",
    )
    _add_detection(conn, image_filename="20260522_100000_hn.jpg")
    write("20260522_100000_hn.jpg")

    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        station_id="station-garden-01",
        reviewer_id="armin",
    )
    assert batch.station_id == "station-garden-01"
    assert batch.reviewer_id == "armin"

    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))
    assert meta["station_id"] == "station-garden-01"
    assert meta["reviewer_id"] == "armin"


def test_build_batch_station_and_reviewer_default_to_empty_strings(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
    )
    assert batch.station_id == ""
    assert batch.reviewer_id == ""

    buf = stream_batch_zip(batch)
    with zipfile.ZipFile(buf) as zf:
        meta = json.loads(zf.read("batch_metadata.json"))
    assert meta["station_id"] == ""
    assert meta["reviewer_id"] == ""


def test_build_batch_strips_whitespace_from_identity_fields(tmp_image_dir):
    resolver, _ = tmp_image_dir
    conn = _make_conn()
    batch = build_batch(
        conn,
        path_resolver=resolver,
        now=datetime(2026, 5, 22, 23, 59, 59, tzinfo=UTC),
        station_id="  station-01  ",
        reviewer_id="  armin\n",
    )
    assert batch.station_id == "station-01"
    assert batch.reviewer_id == "armin"
