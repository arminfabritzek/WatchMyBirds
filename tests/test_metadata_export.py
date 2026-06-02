"""Tests for metadata burn-in into download copies (closes #19).

Covers the XMP packet builder, the Pillow copy-with-metadata writer, and
the core resolver that turns current DB state into an EventMetadata
envelope. The key invariants under test:

  - the copy carries the *effective* species (override beats model)
  - a relabel changes the next copy; the original on disk is byte-stable
  - existing EXIF (datetime/GPS) survives the copy
  - multi-bird frames yield multiple rdf:Bag entries
  - non-bird detections get no Aves/Animalia dwc block
"""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

import config
from core.event_metadata import EventMetadata, SpeciesEntry
from utils import path_manager
from utils.db import connection as db_connection
from utils.db import insert_classification, insert_detection, insert_image
from utils.image_ops import build_xmp_packet, save_jpeg_copy_with_metadata


def _read_xmp(jpeg_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(jpeg_bytes))
    raw = img.info.get("xmp")
    assert raw, "copy carries no XMP packet"
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


# --------------------------------------------------------------------------
# Pure packet/writer tests (no DB)
# --------------------------------------------------------------------------


def test_packet_multi_bird_bag_and_dwc_gating():
    meta = EventMetadata(
        species=[
            SpeciesEntry("Cyanistes caeruleus", "Blaumeise", True, 1, 0.93),
            SpeciesEntry("Parus major", "Kohlmeise", True, 2, 0.88),
            SpeciesEntry("squirrel", "Eichhörnchen", False, 3, 0.71),
        ],
    )
    xmp = build_xmp_packet(meta)
    # multi-bird → multiple bag entries (both names per species)
    assert "<rdf:Bag>" in xmp
    for term in ("Cyanistes caeruleus", "Blaumeise", "Parus major", "squirrel"):
        assert term in xmp
    # dwc class gated on birds only: two Aves, never three
    assert xmp.count("<dwc:class>Aves</dwc:class>") == 2
    assert xmp.count("<dwc:kingdom>Animalia</dwc:kingdom>") == 2
    # caption is "Common (Scientific)" from the first species
    assert "Blaumeise (Cyanistes caeruleus)" in xmp


def test_packet_xml_escaping():
    meta = EventMetadata(species=[SpeciesEntry("A & B <x>", 'q"u', False)])
    xmp = build_xmp_packet(meta)
    assert "&amp;" in xmp and "&lt;" in xmp


def test_packet_empty_has_no_content():
    assert EventMetadata().has_content is False
    # an empty envelope still produces a valid (content-free) packet
    xmp = build_xmp_packet(EventMetadata())
    assert "<dc:subject>" not in xmp


def _make_jpeg_with_exif(path) -> str:
    img = Image.new("RGB", (64, 48), (120, 60, 30))
    exif = img.getexif()
    exif[0x0110] = "TestCam"  # Model — stand-in for ingest-time EXIF
    img.save(path, format="JPEG", exif=exif.tobytes(), quality=90)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copy_preserves_exif_and_leaves_original_unchanged(tmp_path):
    src = tmp_path / "orig.jpg"
    orig_hash = _make_jpeg_with_exif(src)

    meta = EventMetadata(
        species=[SpeciesEntry("Cyanistes caeruleus", "Blaumeise", True, 1)]
    )
    copy_bytes = save_jpeg_copy_with_metadata(src, build_xmp_packet(meta))

    # original byte-stable
    assert hashlib.sha256(src.read_bytes()).hexdigest() == orig_hash
    # copy carries species + preserved EXIF
    assert "Cyanistes caeruleus" in _read_xmp(copy_bytes)
    assert Image.open(io.BytesIO(copy_bytes)).info.get("exif")


# --------------------------------------------------------------------------
# DB-backed core tests
# --------------------------------------------------------------------------


@pytest.fixture
def db_env(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    config._CONFIG = None
    db_connection._schema_initialized_paths.clear()
    path_manager._instance = None
    # make sure the in-memory config picks up the temp OUTPUT_DIR
    config.get_config()
    return output_dir


def _seed(conn, *, filename, timestamp, override=None, cls_species="Parus_major",
          od_class="bird", detection_status="active", is_favorite=0, rating=None,
          review_status="confirmed_bird"):
    insert_image(
        conn,
        {
            "filename": filename,
            "timestamp": timestamp,
            "source_id": 1,
            "content_hash": f"h-{filename}",
        },
    )
    conn.execute(
        "UPDATE images SET review_status = ? WHERE filename = ?",
        (review_status, filename),
    )
    det_id = insert_detection(
        conn,
        {
            "image_filename": filename,
            "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.2, "bbox_h": 0.2,
            "od_class_name": od_class,
            "od_confidence": 0.9,
            "od_model_id": "yolo-test",
            "created_at": timestamp,
            "score": 0.9,
        },
    )
    conn.execute(
        "UPDATE detections SET status = ?, manual_species_override = ?, "
        "is_favorite = ?, rating = ? WHERE detection_id = ?",
        (detection_status, override, is_favorite, rating, det_id),
    )
    if cls_species is not None:
        insert_classification(
            conn,
            {
                "detection_id": det_id,
                "cls_class_name": cls_species,
                "cls_confidence": 0.95,
                "cls_model_id": "cls-test",
                "rank": 1,
                "created_at": timestamp,
            },
        )
    conn.commit()
    return det_id


def _write_original(output_dir, filename):
    pm = path_manager.get_path_manager(str(output_dir))
    dest = pm.get_original_path(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), (10, 80, 160)).save(dest, format="JPEG", quality=90)
    return dest


def test_core_resolves_effective_species_and_relabel(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_143015_cam.jpg"
    ts = "20260602_143015"
    with db_connection.closing_connection() as conn:
        det_id = _seed(conn, filename=fn, timestamp=ts, cls_species="Parus_major")
    src = _write_original(output_dir, fn)
    orig_hash = hashlib.sha256(src.read_bytes()).hexdigest()

    # model species first
    copy1 = core.produce_copy_bytes(fn)
    assert "Parus major" in _read_xmp(copy1)

    # relabel via manual override → next copy reflects it
    with db_connection.closing_connection() as conn:
        conn.execute(
            "UPDATE detections SET manual_species_override = ? WHERE detection_id = ?",
            ("Cyanistes_caeruleus", det_id),
        )
        conn.commit()
    copy2 = core.produce_copy_bytes(fn)
    xmp2 = _read_xmp(copy2)
    assert "Cyanistes caeruleus" in xmp2
    assert "Parus major" not in xmp2

    # original byte-stable across both copies
    assert hashlib.sha256(src.read_bytes()).hexdigest() == orig_hash


def test_core_resolve_detection_and_filename(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_080000_cam.jpg"
    ts = "20260602_080000"
    with db_connection.closing_connection() as conn:
        det_id = _seed(conn, filename=fn, timestamp=ts, cls_species="Parus_major")
    _write_original(output_dir, fn)

    resolved = core.resolve_image_for_detection(det_id)
    assert resolved == (fn, ts)

    name = core.export_filename(fn, ts)
    assert name == f"Parus_major__2026-06-02T08-00-00__{det_id}.jpg"


def test_core_non_bird_gets_no_aves(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_090000_cam.jpg"
    ts = "20260602_090000"
    with db_connection.closing_connection() as conn:
        # squirrel: OD class IS the species, no classifier row
        _seed(conn, filename=fn, timestamp=ts, od_class="squirrel",
              cls_species=None)
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    assert len(meta.species) == 1
    assert meta.species[0].is_bird is False
    xmp = build_xmp_packet(meta)
    assert "Aves" not in xmp
    assert "squirrel" in xmp


def test_core_rejected_detection_excluded(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_100000_cam.jpg"
    ts = "20260602_100000"
    with db_connection.closing_connection() as conn:
        _seed(conn, filename=fn, timestamp=ts, cls_species="Parus_major",
              detection_status="rejected")
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    assert meta.species == []
    assert meta.has_content is False


def test_burn_in_toggle_reads_config(db_env, monkeypatch):
    from core import metadata_export_core as core

    monkeypatch.setitem(config.get_config(), "EXPORT_BURN_IN_METADATA", True)
    assert core.burn_in_enabled() is True
    monkeypatch.setitem(config.get_config(), "EXPORT_BURN_IN_METADATA", False)
    assert core.burn_in_enabled() is False


# --------------------------------------------------------------------------
# Rating / favorite / provenance (schema v2)
# --------------------------------------------------------------------------


def test_core_favorite_maps_to_rating_five_and_label(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_110000_cam.jpg"
    ts = "20260602_110000"
    with db_connection.closing_connection() as conn:
        # detection is a favorite AND has a 3-star rating; favorite must win
        _seed(conn, filename=fn, timestamp=ts, is_favorite=1, rating=3)
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    assert meta.is_favorite is True
    assert meta.xmp_rating() == 5
    xmp = build_xmp_packet(meta)
    assert "<xmp:Rating>5</xmp:Rating>" in xmp
    assert "<xmp:Label>Favorite</xmp:Label>" in xmp
    assert "<wmb:isFavorite>true</wmb:isFavorite>" in xmp


def test_core_rating_without_favorite(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_120000_cam.jpg"
    ts = "20260602_120000"
    with db_connection.closing_connection() as conn:
        _seed(conn, filename=fn, timestamp=ts, is_favorite=0, rating=4)
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    assert meta.is_favorite is False
    assert meta.xmp_rating() == 4
    xmp = build_xmp_packet(meta)
    assert "<xmp:Rating>4</xmp:Rating>" in xmp
    assert "xmp:Label" not in xmp


def test_core_no_rating_no_favorite_emits_no_rating_tag(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_130000_cam.jpg"
    ts = "20260602_130000"
    with db_connection.closing_connection() as conn:
        _seed(conn, filename=fn, timestamp=ts, is_favorite=0, rating=None)
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    assert meta.xmp_rating() is None
    xmp = build_xmp_packet(meta)
    assert "xmp:Rating" not in xmp


def test_core_provenance_creator_tool_and_review_status(db_env):
    from core import metadata_export_core as core

    output_dir = db_env
    fn = "20260602_140000_cam.jpg"
    ts = "20260602_140000"
    with db_connection.closing_connection() as conn:
        _seed(conn, filename=fn, timestamp=ts, review_status="confirmed_bird")
    _write_original(output_dir, fn)

    meta = core.build_event_metadata(fn)
    # Fixed app attribution, independent of DEVICE_NAME.
    assert meta.creator_tool == "WatchMyBirds"
    assert meta.creator == "WatchMyBirds"
    assert meta.review_status == "confirmed_bird"
    xmp = build_xmp_packet(meta)
    assert "<xmp:CreatorTool>WatchMyBirds</xmp:CreatorTool>" in xmp
    assert "<dc:creator><rdf:Seq><rdf:li>WatchMyBirds</rdf:li>" in xmp
    assert "<wmb:reviewStatus>confirmed_bird</wmb:reviewStatus>" in xmp
