"""Metadata-export core — burn current DB state into download copies.

Produces an in-memory JPEG copy of an original with species/location/
provenance injected as XMP, reflecting the *current* effective species at
download time. The on-disk original is never written (originals-immutable).

Used by both human-download surfaces (detail-modal single + edit-page batch
ZIP) through the thin ``web.services.metadata_export_service`` wrapper.

Lives in ``core/`` because it owns DB + image IO (``utils.db``,
``utils.image_ops``, ``utils.path_manager``, ``utils.species_names``); the
web service layer may not import ``utils.*`` (arch_hard import boundary), so
the data work belongs here and the service stays a pure delegator.
"""

from __future__ import annotations

import sqlite3

from config import get_config
from core.event_metadata import EventMetadata, SpeciesEntry
from logging_config import get_logger
from utils.db import closing_connection
from utils.image_ops import build_xmp_packet, save_jpeg_copy_with_metadata
from utils.path_manager import get_path_manager
from utils.species_names import (
    is_non_species_od_token,
    load_common_names,
    resolve_common_name,
    species_key_from_candidates,
)

logger = get_logger(__name__)


def burn_in_enabled() -> bool:
    """True when the runtime toggle says downloads should carry metadata."""
    return bool(get_config().get("EXPORT_BURN_IN_METADATA", True))


def _filename_safe_timestamp(ts: str | None) -> str:
    """Turn an ``images.timestamp`` (``YYYYMMDD_HHMMSS``) into an ISO-ish,
    filename-safe stamp (``YYYY-MM-DDTHH-MM-SS``). ``""`` if unparseable."""
    raw = str(ts or "").strip()
    if len(raw) < 15 or "_" not in raw:
        return ""
    date_part, _, time_part = raw.partition("_")
    if len(date_part) < 8 or len(time_part) < 6:
        return ""
    iso_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
    iso_time = f"{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}"
    return f"{iso_date}T{iso_time}"


def _rows_for_image(conn: sqlite3.Connection, image_filename: str) -> list:
    """All active detections on one image, with effective-species inputs.

    Returns the columns needed to resolve effective species and build the
    per-detection ``wmb:`` block. Only ``status = 'active'`` detections are
    included — a copy reflects what the operator considers real on that
    frame (mirrors the canonical gallery filter, not just "not rejected").
    """
    query = """
        SELECT
            d.detection_id,
            d.manual_species_override,
            d.od_class_name,
            c.cls_class_name,
            COALESCE(c.cls_confidence, d.od_confidence) AS confidence,
            COALESCE(d.detector_model_name, d.od_model_id) AS detector_model,
            d.classifier_model_name AS classifier_model,
            d.is_favorite,
            d.rating,
            i.review_status AS review_status
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c
            ON c.detection_id = d.detection_id
            AND c.rank = 1
            AND COALESCE(c.status, 'active') = 'active'
        WHERE d.image_filename = ?
          AND d.status = 'active'
        ORDER BY d.detection_id
    """
    try:
        return conn.execute(query, (image_filename,)).fetchall()
    except sqlite3.Error as exc:
        logger.warning("metadata-export: detection query failed: %s", exc)
        return []


def resolve_image_for_detection(detection_id: int) -> tuple[str, str] | None:
    """Map a ``detection_id`` to its ``(image_filename, timestamp)``.

    Returns ``None`` when the detection or its image row is missing. The
    timestamp is the raw ``images.timestamp`` (``YYYYMMDD_HHMMSS``), used to
    build both the originals path and the served-copy filename.
    """
    query = """
        SELECT i.filename AS image_filename, i.timestamp AS timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.detection_id = ?
        LIMIT 1
    """
    with closing_connection() as conn:
        try:
            row = conn.execute(query, (int(detection_id),)).fetchone()
        except sqlite3.Error as exc:
            logger.warning("metadata-export: image lookup failed: %s", exc)
            return None
    if not row or not row["image_filename"]:
        return None
    return row["image_filename"], row["timestamp"]


def build_event_metadata(image_filename: str) -> EventMetadata:
    """Build the metadata envelope for one image at current DB state.

    Resolves each detection's effective species (honoring
    ``manual_species_override``) and its localized common name. Bird vs.
    non-bird is decided from the OD class token so dwc class/kingdom are
    gated correctly.
    """
    locale = get_config().get("SPECIES_COMMON_NAME_LOCALE", "DE")
    # Caller owns a copy: load_common_names returns a shared lru_cache dict;
    # we only read it, never mutate (see issue #55 / lru_cache aliasing).
    common_names = load_common_names(locale)

    detector_model: str | None = None
    classifier_model: str | None = None
    review_status: str | None = None
    species: list[SpeciesEntry] = []

    with closing_connection() as conn:
        rows = _rows_for_image(conn, image_filename)

    for row in rows:
        od_class = row["od_class_name"]
        key = species_key_from_candidates(
            manual_override=row["manual_species_override"],
            cls_class_name=row["cls_class_name"],
            od_class_name=od_class,
        )
        if not key or key == "Unknown_species":
            continue

        # A bird is anything whose identity does NOT come from a non-bird OD
        # token. Garden mammals (squirrel/marten/…) come straight off the OD
        # class and must not receive Aves/Animalia.
        is_bird = is_non_species_od_token(od_class) or not od_class or (
            row["cls_class_name"] is not None
        )
        # The override can name a non-bird too; if the resolved key matches
        # the raw OD non-bird token, treat as non-bird.
        if od_class and key == od_class and not is_non_species_od_token(od_class):
            is_bird = False

        scientific = key.replace("_", " ")
        common = resolve_common_name(key, common_names)
        confidence = row["confidence"]
        rating = row["rating"]
        species.append(
            SpeciesEntry(
                scientific=scientific,
                common=common,
                is_bird=is_bird,
                detection_id=row["detection_id"],
                confidence=float(confidence) if confidence is not None else None,
                is_favorite=bool(row["is_favorite"]),
                rating=int(rating) if rating is not None else None,
            )
        )
        detector_model = detector_model or row["detector_model"]
        classifier_model = classifier_model or row["classifier_model"]
        review_status = review_status or row["review_status"]

    # Class-B provenance: a fixed app attribution — constant
    # "WatchMyBirds", not derived from DEVICE_NAME.
    return EventMetadata(
        species=species,
        detector_model=detector_model,
        classifier_model=classifier_model,
        review_status=review_status,
        creator_tool="WatchMyBirds",
        creator="WatchMyBirds",
    )


def export_filename(image_filename: str, timestamp: str | None) -> str:
    """Friendly served-copy name ``Genus_species__<iso-ts>__<id>.jpg``.

    Uses the image's *primary* (first) resolved species and the lowest
    detection id on the frame; falls back gracefully to the original name
    when no species resolves.
    """
    meta = build_event_metadata(image_filename)
    stamp = _filename_safe_timestamp(timestamp)
    if not meta.species:
        return image_filename
    first = meta.species[0]
    genus_species = (first.scientific or "image").replace(" ", "_")
    det_id = first.detection_id
    parts = [genus_species]
    if stamp:
        parts.append(stamp)
    if det_id is not None:
        parts.append(str(det_id))
    return "__".join(parts) + ".jpg"


def produce_copy_bytes(image_filename: str) -> bytes:
    """Return JPEG bytes of the original copy with XMP metadata injected.

    Raises ``FileNotFoundError`` if the original is missing (caller falls
    back to whatever it would have served). If burn-in is disabled the
    caller should not invoke this at all — but if it does, an envelope with
    no content yields a copy with an (essentially empty) XMP packet, never
    a crash.
    """
    path_mgr = get_path_manager()
    src_path = path_mgr.get_original_path(image_filename)
    meta = build_event_metadata(image_filename)
    xmp = build_xmp_packet(meta)
    return save_jpeg_copy_with_metadata(src_path, xmp)
