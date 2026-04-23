"""Training-data export service.

Packages human-reviewed detections from the local DB into a ZIP bundle
for handoff to the upstream training dev. Only **positive samples**
(Option A strict: ``manual_species_override IS NOT NULL`` AND
``manual_bbox_review = 'correct'``) are included in Phase 1 — hard
negatives and bbox-wrong rows stay out so the dev's downstream pipeline
has no extra branches to handle.

Lifecycle:
1. Every approved review event optionally lands in ``training_exports``
   with ``export_status='pending'`` (controlled by
   ``TRAINING_EXPORT_AUTO_OPT_IN``), OR the operator picks them
   explicitly via the Export modal.
2. The ZIP build path streams images + CSV, then flips the selected
   rows' ``export_status`` to ``'exported'`` and stamps
   ``exported_at`` + ``batch_id``.
3. Exported rows are EXCLUDED from future selections by default
   (prevents duplicate shipping to the dev). The UI can opt in to
   including them by passing ``include_already_exported=True``.

The CSV schema matches the dev's spec exactly:
    uuid, x1, y1, x2, y2, species, confidence_reviewer,
    station_id, timestamp_utc, reviewer_id, review_timestamp,
    source_detector_model, source_detector_confidence, uncertain
"""

from __future__ import annotations

import csv
import io
import os
import random
import sqlite3
import uuid as uuid_module
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core import path_core
from logging_config import get_logger

logger = get_logger(__name__)


# Default limits. Kept small so first batches stay quick to download
# and review. Operator can override via the UI.
DEFAULT_MAX_PER_SPECIES = 50
DEFAULT_MAX_TOTAL = 500


@dataclass
class SpeciesAvailability:
    """How many positive samples are available for a given species."""

    species: str
    available_count: int       # eligible, not yet exported
    pending_count: int          # auto-opt-in pool awaiting download
    already_exported_count: int  # shipped in a previous batch


@dataclass
class ExportSelection:
    """A concrete list of detection_ids chosen for a batch.

    The caller is expected to persist these via ``mark_exported`` once
    the ZIP stream completes successfully.
    """

    batch_id: str
    detection_ids: list[int]
    per_species_counts: dict[str, int] = field(default_factory=dict)
    # How many of the selected rows per species were user-favorites
    # (``is_favorite=1``). Surfaced in the manifest so the training
    # dev sees which batches carry more user-curated samples.
    favorites_per_species: dict[str, int] = field(default_factory=dict)


def _positive_sample_predicate(
    det_alias: str = "d",
    *,
    include_already_exported: bool = False,
    exported_alias: str = "te",
) -> str:
    """SQL fragment that identifies human-reviewed positive samples.

    Option A strict: species override AND bbox marked correct. The
    review queue's Approve-event flow stamps both in a single atomic
    transaction, so this predicate exactly matches the detections the
    operator explicitly approved.
    """
    base = f"""
        {det_alias}.status = 'active'
        AND {det_alias}.manual_species_override IS NOT NULL
        AND TRIM({det_alias}.manual_species_override) != ''
        AND {det_alias}.manual_bbox_review = 'correct'
    """
    if not include_already_exported:
        base += f"""
        AND (
            {exported_alias}.export_status IS NULL
            OR {exported_alias}.export_status != 'exported'
        )
        """
    return base


def list_species_availability(
    conn: sqlite3.Connection,
) -> list[SpeciesAvailability]:
    """Enumerate every reviewed species with its pool counts.

    Returns a list sorted descending by ``available_count`` so the UI
    can show the operator what is exportable right now. Species with
    zero availability (everything already exported) are still listed
    so the operator can see the total historical output.
    """
    query = f"""
        SELECT
            d.manual_species_override AS species,
            SUM(
                CASE
                    WHEN te.export_status IS NULL THEN 1
                    WHEN te.export_status = 'pending' THEN 1
                    ELSE 0
                END
            ) AS available_count,
            SUM(CASE WHEN te.export_status = 'pending' THEN 1 ELSE 0 END)
                AS pending_count,
            SUM(CASE WHEN te.export_status = 'exported' THEN 1 ELSE 0 END)
                AS already_exported_count
        FROM detections d
        LEFT JOIN training_exports te ON te.detection_id = d.detection_id
        WHERE {_positive_sample_predicate("d", include_already_exported=True)}
        GROUP BY d.manual_species_override
        ORDER BY available_count DESC, d.manual_species_override ASC
    """
    out: list[SpeciesAvailability] = []
    for row in conn.execute(query).fetchall():
        species = str(row["species"] or "").strip()
        if not species:
            continue
        out.append(
            SpeciesAvailability(
                species=species,
                available_count=int(row["available_count"] or 0),
                pending_count=int(row["pending_count"] or 0),
                already_exported_count=int(row["already_exported_count"] or 0),
            )
        )
    return out


def build_batch_id(suffix: str = "") -> str:
    """Stable, time-sorted batch id. The ``suffix`` is an optional
    short tag (e.g. "auto_approve" when the opt-in toggle fires).
    """
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"batch_{ts}_{suffix}" if suffix else f"batch_{ts}"


def select_export_batch(
    conn: sqlite3.Connection,
    *,
    species_limits: dict[str, int] | None = None,
    max_per_species: int = DEFAULT_MAX_PER_SPECIES,
    max_total: int | None = DEFAULT_MAX_TOTAL,
    include_already_exported: bool = False,
    rng_seed: int | None = None,
) -> ExportSelection:
    """Pick detection_ids for a batch according to the UI selection.

    Sampling policy — **favorites-first, then random fill**:

    1. Within each species bucket, ``is_favorite=1`` rows are picked
       first (shuffled among themselves when more favorites than cap).
    2. The remaining cap is filled with random non-favorite rows.
    3. When the overall ``max_total`` clips the combined pool,
       favorites are biased to survive the clip.

    Why favorites-first: a user who explicitly starred a detection
    has looked at the image a second time and stands behind the label
    — that is a strictly stronger quality signal than a one-shot
    Review-queue approval. Feeding these into the training pool
    first improves label quality at zero cost in diversity (favorites
    still shuffle among themselves for balance).

    The returned selection is DB-writeable but not yet persisted. The
    caller must call :func:`mark_exported` after the ZIP stream
    completes, or call :func:`mark_pending` for an opt-in flow.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    species_limits = species_limits or {}
    query = f"""
        SELECT d.detection_id, d.manual_species_override AS species,
               COALESCE(d.is_favorite, 0) AS is_favorite
        FROM detections d
        LEFT JOIN training_exports te ON te.detection_id = d.detection_id
        WHERE {_positive_sample_predicate(
            "d", include_already_exported=include_already_exported
        )}
            AND d.manual_species_override IN ({
                ",".join("?" for _ in species_limits)
            })
    """ if species_limits else f"""
        SELECT d.detection_id, d.manual_species_override AS species,
               COALESCE(d.is_favorite, 0) AS is_favorite
        FROM detections d
        LEFT JOIN training_exports te ON te.detection_id = d.detection_id
        WHERE {_positive_sample_predicate(
            "d", include_already_exported=include_already_exported
        )}
    """

    params = list(species_limits.keys()) if species_limits else []
    rows = conn.execute(query, params).fetchall()

    # Bucket by species, splitting favorites from the rest. Within
    # each sub-bucket we shuffle so repeated exports cover different
    # material.
    favorites_by_species: dict[str, list[int]] = {}
    others_by_species: dict[str, list[int]] = {}
    for row in rows:
        species = str(row["species"] or "").strip()
        if not species:
            continue
        det_id = int(row["detection_id"])
        if int(row["is_favorite"] or 0) == 1:
            favorites_by_species.setdefault(species, []).append(det_id)
        else:
            others_by_species.setdefault(species, []).append(det_id)

    selected: list[int] = []
    per_species_counts: dict[str, int] = {}
    favorites_per_species: dict[str, int] = {}
    favorite_set: set[int] = set()

    all_species = set(favorites_by_species) | set(others_by_species)
    for species in all_species:
        cap = species_limits.get(species, max_per_species)
        if cap <= 0:
            continue
        favs = favorites_by_species.get(species, [])
        others = others_by_species.get(species, [])
        rng.shuffle(favs)
        rng.shuffle(others)
        take_favs = favs[:cap]
        remaining = cap - len(take_favs)
        take_others = others[:remaining] if remaining > 0 else []
        bucket_take = [*take_favs, *take_others]
        selected.extend(bucket_take)
        per_species_counts[species] = len(bucket_take)
        favorites_per_species[species] = len(take_favs)
        favorite_set.update(take_favs)

    # Apply the overall cap if configured. Bias favorites to survive:
    # keep them all, then fill the remaining budget with a random
    # subset of the non-favorite picks.
    if max_total is not None and len(selected) > max_total:
        favs_in_selection = [i for i in selected if i in favorite_set]
        non_favs_in_selection = [i for i in selected if i not in favorite_set]
        if len(favs_in_selection) >= max_total:
            # Too many favorites — random-drop even among favorites.
            rng.shuffle(favs_in_selection)
            selected = favs_in_selection[:max_total]
        else:
            rng.shuffle(non_favs_in_selection)
            needed = max_total - len(favs_in_selection)
            selected = [*favs_in_selection, *non_favs_in_selection[:needed]]
        # Recompute per-species + per-species-favorites after the cap.
        per_species_counts = {}
        favorites_per_species = {}
        surviving = set(selected)
        for species in all_species:
            favs = favorites_by_species.get(species, [])
            others = others_by_species.get(species, [])
            f_count = sum(1 for i in favs if i in surviving)
            o_count = sum(1 for i in others if i in surviving)
            total_count = f_count + o_count
            if total_count:
                per_species_counts[species] = total_count
                favorites_per_species[species] = f_count

    return ExportSelection(
        batch_id=build_batch_id(),
        detection_ids=selected,
        per_species_counts=per_species_counts,
        favorites_per_species=favorites_per_species,
    )


def _resolve_frame_integrity(
    conn: sqlite3.Connection, detection_ids: list[int]
) -> tuple[list[int], list[int], list[int]]:
    """Expand the caller's detection_ids into a frame-integrity-safe set.

    For every frame that any selected detection touches, we pull ALL
    active detections on that frame. Then:

    - If EVERY detection on the frame is Option-A-strict (species
      override + bbox='correct'): all of them get shipped — the
      originally-selected ones plus any sibling we pulled in from
      the same frame. That keeps the dev's 'one uuid per frame,
      every bbox labelled' contract. Siblings joining the batch are
      reported separately so the caller can mark them as exported.

    - If ANY detection on the frame is ambiguous (bbox NULL / wrong
      / missing species / trashed): the whole frame is dropped,
      including the originally-selected detection. Shipping a
      partially-labelled frame would teach the OD trainer that the
      unlabelled vogel is "background", which is worse than shipping
      nothing from that frame.

    Returns three lists:
      - ``final_detection_ids``: to be written to the ZIP (original
        selection + pulled-in eligible siblings, minus frame-dropped)
      - ``pulled_in_siblings``: detection_ids added by frame-
        integrity that were NOT in the original selection. The
        caller must ``mark_exported`` these too, otherwise a future
        export would ship them again.
      - ``dropped_from_selection``: detection_ids that WERE in the
        original selection but got dropped because their frame had
        ambiguous siblings. The caller logs these so the operator
        understands why the batch shrank.
    """
    if not detection_ids:
        return [], [], []

    original_ids = set(int(i) for i in detection_ids)

    # Step 1: find every image touched by the selection.
    placeholders = ",".join("?" for _ in original_ids)
    frame_rows = conn.execute(
        f"""
        SELECT DISTINCT image_filename FROM detections
        WHERE detection_id IN ({placeholders}) AND image_filename IS NOT NULL
        """,
        list(original_ids),
    ).fetchall()
    touched_frames = [r["image_filename"] for r in frame_rows]
    if not touched_frames:
        return [], [], list(original_ids)

    # Step 2: pull every active detection on those frames.
    frame_placeholders = ",".join("?" for _ in touched_frames)
    all_dets = conn.execute(
        f"""
        SELECT
            detection_id,
            image_filename,
            manual_species_override,
            manual_bbox_review,
            status
        FROM detections
        WHERE image_filename IN ({frame_placeholders})
          AND COALESCE(status, 'active') = 'active'
        """,
        list(touched_frames),
    ).fetchall()

    # Step 3: classify each frame as clean (all dets Option-A-strict)
    # or ambiguous (any det not Option-A-strict). Clean frames pass;
    # ambiguous frames drop every one of their dets, including the
    # originally-selected one.
    dets_by_frame: dict[str, list[sqlite3.Row]] = {}
    for d in all_dets:
        dets_by_frame.setdefault(d["image_filename"], []).append(d)

    final_ids: list[int] = []
    pulled_siblings: list[int] = []
    dropped_from_selection: list[int] = []

    for filename, dets in dets_by_frame.items():
        frame_clean = True
        for d in dets:
            species_ok = bool(
                d["manual_species_override"]
                and str(d["manual_species_override"]).strip()
            )
            bbox_ok = d["manual_bbox_review"] == "correct"
            if not (species_ok and bbox_ok):
                frame_clean = False
                break
        if frame_clean:
            for d in dets:
                det_id = int(d["detection_id"])
                final_ids.append(det_id)
                if det_id not in original_ids:
                    pulled_siblings.append(det_id)
        else:
            for d in dets:
                det_id = int(d["detection_id"])
                if det_id in original_ids:
                    dropped_from_selection.append(det_id)

    # Finally, IDs from the original selection that are orphaned
    # (never showed up in any frame) — they have no image and can't
    # be exported anyway. Treat them as dropped so the caller can
    # audit them.
    returned_set = set(final_ids) | set(dropped_from_selection)
    for oid in original_ids:
        if oid not in returned_set:
            dropped_from_selection.append(oid)

    return final_ids, pulled_siblings, dropped_from_selection


def _fetch_export_rows(
    conn: sqlite3.Connection, detection_ids: list[int]
) -> list[sqlite3.Row]:
    """Fetch the full detection+image payload needed for the CSV."""
    if not detection_ids:
        return []
    placeholders = ",".join("?" for _ in detection_ids)
    query = f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.frame_width, d.frame_height,
            d.manual_species_override AS species,
            d.od_confidence AS source_detector_confidence,
            d.od_model_id AS source_detector_model,
            d.bbox_reviewed_at AS review_timestamp,
            d.species_updated_at,
            i.timestamp AS image_timestamp
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.detection_id IN ({placeholders})
    """
    return conn.execute(query, detection_ids).fetchall()


def _bbox_to_pixels(row: sqlite3.Row) -> tuple[int, int, int, int] | None:
    """Convert normalized [0,1] bbox + frame dims to integer pixel
    coords. Returns None for rows missing any required field (rare
    edge case for pre-migration data)."""
    try:
        bx = float(row["bbox_x"])
        by = float(row["bbox_y"])
        bw = float(row["bbox_w"])
        bh = float(row["bbox_h"])
        fw = int(row["frame_width"])
        fh = int(row["frame_height"])
    except (TypeError, ValueError):
        return None
    if fw <= 0 or fh <= 0:
        return None
    x1 = max(0, int(round(bx * fw)))
    y1 = max(0, int(round(by * fh)))
    x2 = min(fw, int(round((bx + bw) * fw)))
    y2 = min(fh, int(round((by + bh) * fh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def stream_export_zip(
    conn: sqlite3.Connection,
    selection: ExportSelection,
    *,
    output_dir: str,
    station_id: str = "",
    reviewer_id: str = "",
    app_version: str = "",
) -> tuple[io.BytesIO, dict[str, list[int]]]:
    """Build the complete ZIP in-memory and return the buffer plus a
    detection-id breakdown for the caller to persist.

    Returns a tuple ``(buffer, ids_for_persist)`` where
    ``ids_for_persist`` has three keys:

    - ``exported_ids``: detection_ids that ended up as annotations in
      the ZIP. The caller must ``mark_exported`` these. This is a
      superset of the originally-selected ids because frame-integrity
      may pull in sibling detections on the same frame (their uuid is
      shared, so shipping them together is mandatory).
    - ``pulled_in_siblings``: subset of ``exported_ids`` that were
      NOT in the original selection but got added via frame-integrity.
      Separate so the caller's log lines can reflect the broadened
      batch honestly.
    - ``dropped_ids``: detection_ids that WERE in the original
      selection but got dropped by frame-integrity (their frame had
      ambiguous siblings, so no part of the frame ships). These
      remain ``pending`` in the pool for a later batch once the
      operator cleans up the siblings in review.

    Frame-integrity rule: every frame in the ZIP must have ALL its
    active detections satisfy Option-A-strict. A frame with any
    ambiguous sibling (bbox NULL / wrong / missing species) is
    dropped whole. Otherwise we ship partially-labelled frames that
    would mislead the dev's OD trainer (unlabelled vogel =
    background in its loss).

    For Phase 1 we stay in-memory because the default 500-row cap
    keeps the ZIP small (a 1080p frame at ~300KB × 500 ≈ 150 MB
    worst case, well within a single HTTP response). If we ever lift
    the cap significantly, switch this to a streaming generator that
    yields ZIP chunks (``zipfile`` supports append-mode on a temp file).
    """
    final_ids, pulled_siblings, dropped_from_selection = _resolve_frame_integrity(
        conn, selection.detection_ids
    )
    if pulled_siblings:
        logger.info(
            "training_export frame-integrity: pulled in %d sibling "
            "detection(s) to keep frames whole: %s",
            len(pulled_siblings),
            pulled_siblings[:20],
        )
    if dropped_from_selection:
        logger.info(
            "training_export frame-integrity: dropped %d selected "
            "detection(s) whose frames had ambiguous siblings: %s",
            len(dropped_from_selection),
            dropped_from_selection[:20],
        )
    rows = _fetch_export_rows(conn, final_ids)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(
            [
                "uuid",
                "x1", "y1", "x2", "y2",
                "species",
                "confidence_reviewer",
                "station_id",
                "timestamp_utc",
                "reviewer_id",
                "review_timestamp",
                "source_detector_model",
                "source_detector_confidence",
                "uncertain",
            ]
        )

        # Per-frame UUID map: each source image gets ONE uuid, even
        # when it carries multiple approved detections. The dev's
        # training pipeline expects "one file in images/, multiple
        # rows in annotations.csv sharing that file's uuid" (see
        # dev handoff 2026-04-23). Generating a fresh uuid per
        # detection would:
        #   - bloat the ZIP (each frame written N times under N names)
        #   - break the dev's per-image multi-bbox assumption
        # So we dedupe by image_filename and reuse the uuid per row.
        uuid_by_filename: dict[str, str] = {}
        extension_by_filename: dict[str, str] = {}

        missing_images = 0
        skipped_bbox = 0
        written_rows = 0
        written_images = 0
        skipped_sibling_bbox = 0  # detections whose frame is present
        # but whose own bbox is invalid
        for row in rows:
            src_filename = row["image_filename"]
            if not src_filename:
                missing_images += 1
                continue

            # First occurrence of this filename: validate existence
            # and register a uuid. Subsequent detections on the same
            # frame skip the file-system check and reuse the uuid.
            if src_filename not in uuid_by_filename:
                src_path = path_core.get_original_path(
                    output_dir, src_filename
                )
                if not src_path.exists():
                    missing_images += 1
                    logger.debug(
                        f"training_export: original missing on disk for "
                        f"{src_filename}"
                    )
                    continue
                sample_uuid = uuid_module.uuid4().hex
                _ext = os.path.splitext(src_filename)[1].lower() or ".jpg"
                archive_name = f"images/{sample_uuid}{_ext}"
                try:
                    zf.write(src_path, archive_name)
                except OSError as exc:
                    missing_images += 1
                    logger.warning(
                        f"training_export: failed to add {src_path} to "
                        f"zip: {exc}"
                    )
                    continue
                uuid_by_filename[src_filename] = sample_uuid
                extension_by_filename[src_filename] = _ext
                written_images += 1

            bbox_px = _bbox_to_pixels(row)
            if bbox_px is None:
                skipped_sibling_bbox += 1
                logger.debug(
                    f"training_export: skipping detection_id="
                    f"{row['detection_id']} on {src_filename} — bbox/frame "
                    "dims missing or invalid (frame still included for "
                    "its other detections)"
                )
                continue

            sample_uuid = uuid_by_filename[src_filename]
            x1, y1, x2, y2 = bbox_px
            writer.writerow(
                [
                    sample_uuid,
                    x1, y1, x2, y2,
                    row["species"] or "",
                    "",  # confidence_reviewer — not tracked yet, leave blank
                    station_id,
                    row["image_timestamp"] or "",
                    reviewer_id,
                    row["review_timestamp"] or "",
                    row["source_detector_model"] or "",
                    row["source_detector_confidence"] or "",
                    "false",
                ]
            )
            written_rows += 1
        # ``skipped_bbox`` in the manifest keeps its historical name
        # (rows dropped due to invalid bbox geometry), unified across
        # "frame missing" and "sibling-bbox-bad" cases.
        skipped_bbox = skipped_sibling_bbox

        zf.writestr("annotations.csv", csv_buf.getvalue())

        manifest = {
            "batch_id": selection.batch_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            # annotation rows in CSV; matches detection rows that
            # survived the bbox sanity pass.
            "rows_written": written_rows,
            # distinct image files packed into the ZIP. Will be less
            # than rows_written whenever multiple detections share a
            # frame (the intended shape per dev handoff).
            "images_written": written_images,
            "rows_requested": len(selection.detection_ids),
            "skipped_missing_images": missing_images,
            "skipped_invalid_bbox": skipped_bbox,
            "per_species_counts": selection.per_species_counts,
            "favorites_per_species": selection.favorites_per_species,
            "station_id": station_id,
            "reviewer_id": reviewer_id,
            "app_version": app_version,
            "export_predicate": (
                "option_a_strict: manual_species_override IS NOT NULL "
                "AND manual_bbox_review='correct'"
            ),
            "schema_note": (
                "One uuid per source image. Multiple detections on the "
                "same frame share the uuid and appear as separate rows "
                "in annotations.csv."
            ),
        }
        import json as _json

        manifest["frame_integrity"] = {
            "pulled_in_siblings": len(pulled_siblings),
            "dropped_from_selection": len(dropped_from_selection),
            "note": (
                "pulled_in_siblings: detections added to this batch "
                "because they share a frame with a selected detection "
                "(frame must ship whole). dropped_from_selection: "
                "originally-selected detections excluded because their "
                "frame had an ambiguous sibling (unreviewed/wrong bbox)."
            ),
        }
        zf.writestr("manifest.json", _json.dumps(manifest, indent=2))

    buffer.seek(0)
    return buffer, {
        "exported_ids": final_ids,
        "pulled_in_siblings": pulled_siblings,
        "dropped_ids": dropped_from_selection,
    }


def mark_exported(
    conn: sqlite3.Connection,
    detection_ids: list[int],
    batch_id: str,
) -> int:
    """Persist that these detection_ids landed in a completed ZIP.

    Uses UPSERT semantics: rows already pending for this detection
    (e.g. from auto-opt-in) get promoted to 'exported' rather than
    duplicated. UNIQUE(detection_id) on the table enforces that.
    """
    if not detection_ids:
        return 0
    exported_at = datetime.now(UTC).isoformat()
    cur = conn.executemany(
        """
        INSERT INTO training_exports (batch_id, detection_id, export_status, exported_at)
        VALUES (?, ?, 'exported', ?)
        ON CONFLICT(detection_id) DO UPDATE SET
            batch_id = excluded.batch_id,
            export_status = 'exported',
            exported_at = excluded.exported_at
        """,
        [(batch_id, det_id, exported_at) for det_id in detection_ids],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount is not None else 0


def mark_pending(
    conn: sqlite3.Connection,
    detection_ids: list[int],
    batch_id: str,
) -> int:
    """Flag detections as 'pending' without writing a ZIP.

    Used by the auto-opt-in hook in the review Approve-event handler:
    every approved detection joins the export pool automatically.
    INSERT OR IGNORE so an auto-opt-in re-approval (rare) does not
    downgrade an already-exported row back to pending.
    """
    if not detection_ids:
        return 0
    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO training_exports
            (batch_id, detection_id, export_status, exported_at)
        VALUES (?, ?, 'pending', NULL)
        """,
        [(batch_id, det_id) for det_id in detection_ids],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount is not None else 0


def filter_eligible_for_pool(
    conn: sqlite3.Connection,
    detection_ids: list[int],
    *,
    auto_confirm_bbox: bool = False,
) -> dict[str, list[int]]:
    """Split a caller-supplied list of detection_ids into three buckets.

    Used by the gallery-edit "Confirm & Add to Training" batch button
    so the UI can report "7 added, 3 not eligible, 2 already in the
    pool" to the operator instead of a bare success count.

    Parameters:
        auto_confirm_bbox: When True, a detection is also eligible if
            it has a species_override + active status but
            ``manual_bbox_review`` is NULL (never reviewed). The
            caller's click in the gallery counts as the bbox
            confirmation itself — the UI layer (or the blueprint) is
            responsible for actually writing ``bbox_review='correct'``
            on those rows before persisting pool membership. Rows with
            ``bbox_review='wrong'`` stay ineligible regardless, because
            flipping a prior 'wrong' to 'correct' would contradict a
            previous human call.

    **Frame-level integrity guard**: even with ``auto_confirm_bbox=True``,
    a detection is only eligible if EVERY other active detection on
    the same image has ``bbox_review='correct'`` already. Mixing
    confirmed and unreviewed bboxes on one frame would ship a frame
    to the dev whose other bboxes are ambiguous, training the OD on
    partially-labelled images. Strict frame-level integrity prevents
    that.

    Buckets:
    - ``eligible``: passes all gates; safe to mark as pending.
    - ``ineligible``: fails the species / bbox / status / frame-
      integrity gate. UI shows the operator so they know to go
      through the review queue for those rows.
    - ``already_in_pool``: has any ``training_exports`` row already
      (pending OR exported). Skipped to avoid downgrading an exported
      row back to pending.

    Every submitted id lands in exactly one bucket. The sum of the
    three bucket sizes equals the deduplicated input length.
    """
    if not detection_ids:
        return {"eligible": [], "ineligible": [], "already_in_pool": []}

    unique_ids = list(dict.fromkeys(int(i) for i in detection_ids if int(i) > 0))
    if not unique_ids:
        return {"eligible": [], "ineligible": [], "already_in_pool": []}

    placeholders = ",".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT
            d.detection_id,
            d.image_filename,
            d.manual_species_override,
            d.manual_bbox_review,
            d.status,
            te.export_status
        FROM detections d
        LEFT JOIN training_exports te ON te.detection_id = d.detection_id
        WHERE d.detection_id IN ({placeholders})
        """,
        unique_ids,
    ).fetchall()

    # Per-row pre-classification — bucket each submitted detection
    # against the simple local predicates (species, bbox, status,
    # already-in-pool). Frame integrity is evaluated in the next pass
    # because it needs the sibling detections per image.
    per_row_state: dict[int, dict] = {}
    for row in rows:
        det_id = int(row["detection_id"])
        per_row_state[det_id] = {
            "image_filename": row["image_filename"],
            "species_set": bool(
                row["manual_species_override"]
                and str(row["manual_species_override"]).strip()
            ),
            "bbox_correct": row["manual_bbox_review"] == "correct",
            "bbox_wrong": row["manual_bbox_review"] == "wrong",
            "bbox_null": row["manual_bbox_review"] is None,
            "status_active": (row["status"] or "active") == "active",
            "in_pool": row["export_status"] is not None,
        }

    # Frame-integrity pre-check: for every image that a candidate
    # touches, count the active detections whose bbox_review is NOT
    # 'correct'. If the count is > 0, the frame is ambiguous and no
    # detection on it may enter the pool — regardless of how cleanly
    # individual rows score.
    #
    # Each candidate detection on the frame "passes through" this
    # check when it is itself eligible under auto_confirm_bbox (i.e.
    # species_override + status_active + bbox in {correct, null}),
    # because the caller's intent is to promote exactly those to
    # correct. A sibling with bbox='wrong' or missing species still
    # blocks the frame.
    candidate_images = {
        state["image_filename"]
        for state in per_row_state.values()
        if state["image_filename"]
    }
    frame_blocked: set[str] = set()
    if candidate_images:
        # Pull ALL active detections on the candidate frames, so we
        # can decide whether the non-candidate siblings are clean.
        img_placeholders = ",".join("?" for _ in candidate_images)
        sibling_rows = conn.execute(
            f"""
            SELECT
                image_filename,
                manual_species_override,
                manual_bbox_review,
                status
            FROM detections
            WHERE image_filename IN ({img_placeholders})
              AND COALESCE(status, 'active') = 'active'
            """,
            list(candidate_images),
        ).fetchall()
        for srow in sibling_rows:
            filename = srow["image_filename"]
            bbox_review = srow["manual_bbox_review"]
            # Any sibling with bbox 'wrong' or missing species makes
            # the whole frame ambiguous — we cannot ship it as a
            # clean training example.
            species_ok = bool(
                srow["manual_species_override"]
                and str(srow["manual_species_override"]).strip()
            )
            if bbox_review == "wrong":
                frame_blocked.add(filename)
                continue
            if not species_ok:
                frame_blocked.add(filename)
                continue
            # auto_confirm_bbox=True promotes NULL to 'correct' only
            # for the rows the caller explicitly submitted. Siblings
            # on the same frame that the caller did NOT submit must
            # already be 'correct' — their bbox is nobody's call to
            # make right now.
            if bbox_review != "correct":
                frame_blocked.add(filename)

    # Now classify each submitted id into a bucket. Siblings that
    # belong to the caller's submission can "clear" themselves: the
    # earlier loop flagged every non-correct sibling as blocking,
    # including the caller's own candidates. Re-allow those by
    # checking: if every blocking detection on the frame is itself a
    # submitted eligible candidate (after auto_confirm_bbox), the
    # frame becomes clean.
    if auto_confirm_bbox and frame_blocked:
        submitted_by_image: dict[str, set[int]] = {}
        for det_id, state in per_row_state.items():
            filename = state["image_filename"]
            if filename:
                submitted_by_image.setdefault(filename, set()).add(det_id)
        newly_ok: set[str] = set()
        img_placeholders = ",".join("?" for _ in frame_blocked)
        bad_siblings_rows = conn.execute(
            f"""
            SELECT
                detection_id,
                image_filename,
                manual_species_override,
                manual_bbox_review
            FROM detections
            WHERE image_filename IN ({img_placeholders})
              AND COALESCE(status, 'active') = 'active'
              AND COALESCE(manual_bbox_review, '') != 'correct'
            """,
            list(frame_blocked),
        ).fetchall()
        bad_by_image: dict[str, list[sqlite3.Row]] = {}
        for br in bad_siblings_rows:
            bad_by_image.setdefault(br["image_filename"], []).append(br)
        for filename, blockers in bad_by_image.items():
            submitted_here = submitted_by_image.get(filename, set())
            all_resolvable = True
            for br in blockers:
                bdid = int(br["detection_id"])
                bbox_review = br["manual_bbox_review"]
                species_ok = bool(
                    br["manual_species_override"]
                    and str(br["manual_species_override"]).strip()
                )
                # Non-submitted blockers cannot be auto-confirmed.
                if bdid not in submitted_here:
                    all_resolvable = False
                    break
                # Submitted blockers must be NULL-bbox with species
                # set. 'wrong' bbox stays a hard block.
                if bbox_review == "wrong" or not species_ok:
                    all_resolvable = False
                    break
            if all_resolvable:
                newly_ok.add(filename)
        frame_blocked -= newly_ok

    # Final bucketing pass.
    # Each decision is logged at DEBUG granularity so operators can
    # reconstruct why a specific submission landed where. The summary
    # line at INFO level (emitted by the blueprint) carries the
    # counts; this per-row stream carries the reasons.
    eligible: list[int] = []
    ineligible: list[int] = []
    already_in_pool: list[int] = []
    reasons: list[str] = []

    for det_id in unique_ids:
        state = per_row_state.get(det_id)
        if state is None:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: unknown_detection_id"
            )
            continue
        if state["in_pool"]:
            already_in_pool.append(det_id)
            reasons.append(
                f"det_{det_id} already_in_pool: row exists in training_exports"
            )
            continue
        if not state["status_active"]:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: detection_status_not_active"
            )
            continue
        if not state["species_set"]:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: manual_species_override is empty"
            )
            continue
        if state["bbox_wrong"]:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: manual_bbox_review='wrong' "
                "(operator previously rejected this bbox)"
            )
            continue
        if state["bbox_null"] and not auto_confirm_bbox:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: manual_bbox_review is NULL "
                "and auto_confirm_bbox is False (call came from the "
                "review-queue path, not the gallery-batch path)"
            )
            continue
        if state["image_filename"] in frame_blocked:
            ineligible.append(det_id)
            reasons.append(
                f"det_{det_id} ineligible: frame {state['image_filename']!r} "
                "has sibling detections whose bbox_review is not 'correct' "
                "and which were not part of this submission (frame-level "
                "integrity guard)"
            )
            continue
        eligible.append(det_id)
        reasons.append(f"det_{det_id} eligible")

    if reasons:
        logger.info(
            "training_export filter: submitted=%d eligible=%d ineligible=%d "
            "already_in_pool=%d auto_confirm_bbox=%s",
            len(unique_ids),
            len(eligible),
            len(ineligible),
            len(already_in_pool),
            auto_confirm_bbox,
        )
        for reason in reasons:
            logger.info("  %s", reason)

    return {
        "eligible": eligible,
        "ineligible": ineligible,
        "already_in_pool": already_in_pool,
    }


def confirm_bbox_and_mark_pending(
    conn: sqlite3.Connection,
    detection_ids: list[int],
    batch_id: str,
) -> int:
    """Set manual_bbox_review='correct' on NULL-bbox rows, then
    mark all given rows as pending in the pool.

    Invoked by the gallery-edit "Confirm & Add to Training" button,
    where the caller's click IS the bbox-review act. Rows whose
    bbox_review is already 'correct' are left as-is. Rows with
    'wrong' bbox are untouched (the caller should never have gotten
    them here — ``filter_eligible_for_pool`` filters them out).

    Returns the number of pool rows freshly inserted (delegates to
    :func:`mark_pending`'s rowcount).
    """
    if not detection_ids:
        return 0

    now_iso = datetime.now(UTC).isoformat()
    placeholders = ",".join("?" for _ in detection_ids)
    conn.execute(
        f"""
        UPDATE detections
        SET manual_bbox_review = 'correct',
            bbox_reviewed_at = COALESCE(bbox_reviewed_at, ?)
        WHERE detection_id IN ({placeholders})
          AND manual_bbox_review IS NULL
          AND COALESCE(status, 'active') = 'active'
        """,
        [now_iso, *detection_ids],
    )
    return mark_pending(conn, detection_ids, batch_id)


def auto_opt_in_if_enabled(
    conn: sqlite3.Connection,
    detection_ids: list[int],
    app_config: dict,
    source_tag: str = "auto_approve",
) -> int:
    """Single-call helper for review handlers that produce an
    Option-A-strict approval (species override + bbox=correct).

    Reads the ``TRAINING_EXPORT_AUTO_OPT_IN`` setting; when False,
    returns 0 without touching the DB. When True, inserts all
    ``detection_ids`` as 'pending' via :func:`mark_pending` under a
    freshly-minted batch id tagged with ``source_tag`` so the pool
    history shows which approval path created each pending row.

    Returning 0 silently (instead of raising) lets every approval
    handler wrap this in a single line without a feature-flag check
    of their own.
    """
    if not detection_ids:
        return 0
    if not app_config.get("TRAINING_EXPORT_AUTO_OPT_IN", False):
        return 0
    batch_id = build_batch_id(source_tag)
    return mark_pending(conn, detection_ids, batch_id)
