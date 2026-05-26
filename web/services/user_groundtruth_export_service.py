"""
User-Groundtruth Export Service.

Orchestrates the conversion of user-interaction signals (queried via
``utils.db.user_groundtruth``) into a COCO+bucket-manifest ZIP that
downstream training can consume directly.

Output shape (per ``build_batch`` + ``stream_batch_zip``):

    user_groundtruth_<batch_id>.zip
    ├── images/
    │   └── YYYY-MM-DD/<filename>.jpg          (original frames, date-sharded)
    ├── coco_annotations.json                   (all buckets merged, COCO-standard)
    ├── manifests/
    │   ├── hard_negatives.jsonl                (1 line per detection, with provenance)
    │   ├── confirmed_positives.jsonl
    │   └── species_relabels.jsonl
    ├── batch_metadata.json
    └── README.md                               (downstream training consumption hints)

The COCO file uses bbox=[x, y, w, h] in **pixel** coordinates, computed
from the normalized ``bbox_x/y/w/h`` columns and the detection's
``frame_width/height``. Detections with missing geometry are skipped
from COCO annotations but still appear in the bucket manifests, so
downstream training sees the existence of the row even if the geometry is
unusable for box-supervised training.

Frame-integrity policy: every frame that ships gets ALL its active
detections included in the COCO annotations, regardless of whether
the sibling detections individually qualified for a bucket. This
matches the existing ``training_export_service`` policy (avoids
training on a partial-box scene) and keeps the COCO file
self-consistent with the bird-detector's actual frame output.

Batch construction is read-only against the live ``detections``/
``images`` tables. Operator actions around the dry-run page can write
export bookkeeping/exclusions and, when explicitly requested, move a
frame to WMB Trash.
"""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.user_groundtruth_core import (
    count_pending_by_bucket,
    fetch_confirmed_positives,
    fetch_favorites,
    fetch_hard_negatives,
    fetch_species_relabels,
)

EXPORTER_VERSION = "1.2"  # bumped: confirmed_positives bucket opt-in (default off)

#: Default for the ``confirmed_positives`` opt-in. Off because the
#: Review-Confirm write path cannot distinguish "user authored this
#: label" from "user accepted the model's predicted label" — both
#: stamp ``species_source='manual'`` — so including the bucket
#: silently feeds model output back as ground-truth. Re-enable once
#: the pipeline-side "Mark as training data" path is wired through
#: to an unambiguous DB flag.
INCLUDE_CONFIRMED_POSITIVES_DEFAULT = False

#: Bucket names that carry a positive class label. A frame in any of
#: these must have ALL its active detections also user-confirmed
#: (i.e. present in some positive bucket), else the WHOLE frame is
#: dropped for training-correctness. Hard-negatives are NOT in this
#: tuple because every active detection on a ``no_bird`` frame is
#: by definition FP — no integrity check needed.
POSITIVE_BUCKETS: tuple[str, ...] = (
    "confirmed_positives",
    "species_relabels",
    "favorites",
)


@dataclass
class Batch:
    """All rows + metadata needed to materialize one export ZIP.

    Constructed by ``build_batch``; consumed by ``stream_batch_zip``
    and ``record_batch_exported``. Intentionally a plain dataclass so
    tests can mutate it freely between build and stream.

    The four buckets are populated by the per-detection queries; the
    ``frame_integrity_dropped`` list records frames that were filtered
    out because their active sibling detections were not all user-
    confirmed. Dropped frames + their detections do NOT appear in
    the positive buckets, the COCO file, or the ZIP — they live only
    in this audit list so the operator can investigate later.
    """

    batch_id: str
    since: str | None
    until: str
    built_at: str
    wmb_app_version: str
    hard_negatives: list[dict[str, Any]]
    confirmed_positives: list[dict[str, Any]]
    species_relabels: list[dict[str, Any]]
    favorites: list[dict[str, Any]]
    # Operator-supplied provenance — pseudonyms, browser-side persisted.
    # Surfaced in batch_metadata.json so downstream training can disambiguate
    # batches across stations and reviewers. Not persisted in the
    # export_batches DB table (browser localStorage is the home).
    station_id: str = ""
    reviewer_id: str = ""
    # Audit trail: frames dropped by frame-integrity. Each entry:
    #   {"image_filename": str, "reason": str,
    #    "dropped_detection_ids": list[int]}
    frame_integrity_dropped: list[dict[str, Any]] = field(default_factory=list)
    # filename -> absolute Path on disk. Built once during build_batch
    # so stream_batch_zip does not re-query PathManager per row.
    _image_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "hard_negatives": len(self.hard_negatives),
            "confirmed_positives": len(self.confirmed_positives),
            "species_relabels": len(self.species_relabels),
            "favorites": len(self.favorites),
        }

    @property
    def total_rows(self) -> int:
        return sum(self.counts.values())

    @property
    def is_empty(self) -> bool:
        return self.total_rows == 0


def build_batch_id(now: datetime | None = None) -> str:
    """Return an ISO-week batch identifier like ``2026W21_a3b1``.

    The suffix is a stable 4-char hash of the build timestamp so two
    batches built in the same week (e.g. re-export after operator
    correction) can coexist as separate rows in ``export_batches``.
    """
    now = now or datetime.now(UTC)
    iso_year, iso_week, _ = now.isocalendar()
    # 4-char hex of seconds-since-epoch is unique enough for one
    # operator's manual workflow; we are not racing builds at scale.
    suffix = format(int(now.timestamp()) & 0xFFFF, "04x")
    return f"{iso_year}W{iso_week:02d}_{suffix}"


def build_batch(
    conn: sqlite3.Connection,
    *,
    path_resolver,
    since: str | None = None,
    until: str | None = None,
    wmb_app_version: str = "",
    now: datetime | None = None,
    include_confirmed_positives: bool = INCLUDE_CONFIRMED_POSITIVES_DEFAULT,
    station_id: str = "",
    reviewer_id: str = "",
) -> Batch:
    """Materialize a Batch from the live DB.

    Args:
        conn: open DB connection with row factory.
        path_resolver: a callable ``filename -> Path`` (typically
            ``PathManager.get_original_path``). Decoupled from
            PathManager itself so unit tests can inject a tmp-dir
            resolver without instantiating PathManager.
        since: ISO timestamp lower bound (inclusive). ``None`` means
            "all history" — appropriate for the very first batch.
        until: ISO timestamp upper bound (exclusive). ``None`` means
            "up to now"; in that case we substitute ``now`` to make
            the batch reproducible (``since`` of the next batch will
            be exactly this ``until``).
        wmb_app_version: app version string for provenance.
        now: override clock for tests.
        include_confirmed_positives: opt-in for the confirmation-bias-
            prone bucket. See ``INCLUDE_CONFIRMED_POSITIVES_DEFAULT``.
    """
    now = now or datetime.now(UTC)
    built_at = now.isoformat()
    effective_until = until or built_at

    hn = fetch_hard_negatives(conn, since=since, until=effective_until)
    if include_confirmed_positives:
        cp = fetch_confirmed_positives(conn, since=since, until=effective_until)
    else:
        cp = []
    rl = fetch_species_relabels(conn, since=since, until=effective_until)
    fav = fetch_favorites(conn, since=since, until=effective_until)

    # Frame-integrity: drop any frame whose active sibling detections
    # are not all user-confirmed. Hard-negatives are exempt by
    # definition (no_bird applies frame-wide).
    cp, rl, fav, dropped = _apply_frame_integrity(
        conn,
        confirmed_positives=cp,
        species_relabels=rl,
        favorites=fav,
    )

    # Build the filename -> Path map for every distinct image referenced
    # across all four buckets. PathManager.get_original_path() is pure,
    # but caching avoids multiple lookups per shared-frame row.
    all_filenames = {
        r["image_filename"] for bucket in (hn, cp, rl, fav) for r in bucket
    }
    image_paths = {fn: path_resolver(fn) for fn in all_filenames}

    return Batch(
        batch_id=build_batch_id(now),
        since=since,
        until=effective_until,
        built_at=built_at,
        wmb_app_version=wmb_app_version,
        hard_negatives=hn,
        confirmed_positives=cp,
        species_relabels=rl,
        favorites=fav,
        frame_integrity_dropped=dropped,
        _image_paths=image_paths,
        station_id=station_id.strip(),
        reviewer_id=reviewer_id.strip(),
    )


def _apply_frame_integrity(
    conn: sqlite3.Connection,
    *,
    confirmed_positives: list[dict[str, Any]],
    species_relabels: list[dict[str, Any]],
    favorites: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Drop frames where active siblings are not all user-confirmed.

    A frame qualifies for the positive buckets only if **every** one
    of its active detections appears in at least one positive bucket
    (confirmed_positive, species_relabel, OR favorite). If even one
    active sibling is in ``species_review``/``reject``/unrated, the
    whole frame is dropped from all three positive buckets — exporting
    a partial-box frame would teach the trainer that the unannotated
    box positions are "background", actively poisoning the model.

    Hard-negatives are not passed in here: ``review_status='no_bird'``
    is a frame-level statement, so all detections in a no_bird frame
    are uniformly FP and the integrity check is structurally
    satisfied.

    Args:
        conn: open DB connection — used to look up sibling counts.
        confirmed_positives, species_relabels, favorites: bucket lists
            as returned by the per-detection queries.

    Returns:
        Tuple of (filtered_cp, filtered_rl, filtered_fav, dropped_audit).
        dropped_audit has one entry per dropped frame:
            {"image_filename": str,
             "reason": str (human-readable),
             "dropped_detection_ids": list[int] (ids that WERE in some
                positive bucket but got dropped because a sibling
                wasn't)}
    """
    # Set of (image_filename, detection_id) tuples currently in any
    # positive bucket. Used to decide whether every sibling on a
    # frame is "covered".
    positive_pairs: set[tuple[str, int]] = set()
    positive_buckets_by_pair: dict[tuple[str, int], set[str]] = {}
    positive_frames: set[str] = set()
    for bucket in (confirmed_positives, species_relabels, favorites):
        for r in bucket:
            key = (r["image_filename"], r["detection_id"])
            positive_pairs.add(key)
            positive_buckets_by_pair.setdefault(key, set()).add(r["bucket"])
            positive_frames.add(r["image_filename"])

    if not positive_frames:
        return confirmed_positives, species_relabels, favorites, []

    # Look up the active sibling detection_ids per frame. Single
    # parameterised query is faster than N round-trips even at the
    # ~30-frame scale we see today, and stays O(positive_frames)
    # rather than O(active_detections) at larger scales.
    placeholders = ",".join("?" for _ in positive_frames)
    rows = conn.execute(
        f"""
        SELECT
            image_filename,
            detection_id,
            bbox_x, bbox_y, bbox_w, bbox_h,
            frame_width, frame_height,
            decision_level,
            raw_species_name,
            manual_species_override,
            species_source
        FROM detections
        WHERE status = 'active'
          AND image_filename IN ({placeholders})
        """,
        list(positive_frames),
    ).fetchall()

    siblings_by_frame: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        try:
            fn = row["image_filename"]
            did = int(row["detection_id"])
        except (TypeError, KeyError):
            fn = row[0]
            did = int(row[1])
        pair = (fn, did)
        siblings_by_frame.setdefault(fn, {})[did] = {
            "detection_id": did,
            "bbox_x": _row_value(row, "bbox_x", 2),
            "bbox_y": _row_value(row, "bbox_y", 3),
            "bbox_w": _row_value(row, "bbox_w", 4),
            "bbox_h": _row_value(row, "bbox_h", 5),
            "frame_width": _row_value(row, "frame_width", 6),
            "frame_height": _row_value(row, "frame_height", 7),
            "decision_level": _row_value(row, "decision_level", 8),
            "raw_species_name": _row_value(row, "raw_species_name", 9),
            "manual_species_override": _row_value(row, "manual_species_override", 10),
            "species_source": _row_value(row, "species_source", 11),
            "is_positive_signal": pair in positive_pairs,
            "positive_buckets": sorted(positive_buckets_by_pair.get(pair, set())),
        }

    # Identify frames where at least one sibling is NOT in any
    # positive bucket — those frames must be dropped entirely.
    dropped_frames: set[str] = set()
    dropped_audit: list[dict[str, Any]] = []
    for fn, sibling_map in siblings_by_frame.items():
        sibling_ids = set(sibling_map)
        positive_ids_for_frame = {
            did for (filename, did) in positive_pairs if filename == fn
        }
        missing = sibling_ids - positive_ids_for_frame
        if missing:
            dropped_frames.add(fn)
            dropped_audit.append(
                {
                    "image_filename": fn,
                    "reason": (
                        f"frame has {len(sibling_ids)} active detection(s) but "
                        f"only {len(positive_ids_for_frame)} carry a user-"
                        f"confirmed signal; {len(missing)} sibling(s) "
                        f"unconfirmed (ids: {sorted(missing)})"
                    ),
                    "dropped_detection_ids": sorted(positive_ids_for_frame),
                    "missing_detection_ids": sorted(missing),
                    "active_siblings": [
                        sibling_map[did] for did in sorted(sibling_map)
                    ],
                }
            )

    if not dropped_frames:
        return confirmed_positives, species_relabels, favorites, []

    def _filter(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows_in if r["image_filename"] not in dropped_frames]

    return (
        _filter(confirmed_positives),
        _filter(species_relabels),
        _filter(favorites),
        dropped_audit,
    )


def _row_value(row: sqlite3.Row, key: str, index: int) -> Any:
    try:
        return row[key]
    except (TypeError, KeyError, IndexError):
        return row[index]


def stream_batch_zip(
    batch: Batch,
    *,
    include_images: bool = True,
) -> io.BytesIO:
    """Serialize a Batch into an in-memory ZIP and return the buffer.

    Args:
        batch: the materialized Batch from ``build_batch``.
        include_images: when False, skip embedding image bytes (the
            COCO + manifests still ship). Used in tests and in any
            future ``--manifest-only`` operator option.

    Returns:
        BytesIO positioned at 0, ready to be streamed via Flask
        ``send_file``.

    The ZIP is built in-memory; for the current dataset size (<1k
    detections, <100MB total) this is well under the RAM budget on
    the RPi. If batches ever exceed ~1GB this should be refactored
    to a streaming generator, but premature optimization would
    complicate testing.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Image files, date-sharded under images/YYYY-MM-DD/
        missing_images: list[str] = []
        if include_images:
            for filename, src_path in batch._image_paths.items():
                if not src_path.is_file():
                    missing_images.append(filename)
                    continue
                date_folder = _date_folder_from_filename(filename)
                arcname = f"images/{date_folder}/{filename}"
                zf.write(src_path, arcname=arcname)

        # 2. Four bucket manifest files (JSONL)
        for bucket_name, rows in [
            ("hard_negatives", batch.hard_negatives),
            ("confirmed_positives", batch.confirmed_positives),
            ("species_relabels", batch.species_relabels),
            ("favorites", batch.favorites),
        ]:
            lines = [_manifest_line(r, batch) for r in rows]
            zf.writestr(
                f"manifests/{bucket_name}.jsonl",
                "\n".join(lines) + ("\n" if lines else ""),
            )

        # 3. Merged COCO annotations
        coco = _build_coco(batch, missing_images=set(missing_images))
        zf.writestr(
            "coco_annotations.json",
            json.dumps(coco, indent=2, ensure_ascii=False),
        )

        # 4. Batch metadata
        meta = _build_metadata(batch, missing_images=missing_images)
        zf.writestr(
            "batch_metadata.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )

        # 5. README for downstream training
        zf.writestr("README.md", _build_readme(batch))

    buffer.seek(0)
    return buffer


def record_batch_exported(
    conn: sqlite3.Connection,
    batch: Batch,
    *,
    notes: str = "",
) -> None:
    """Insert a row into export_batches as a permanent record.

    Idempotent: if the same batch_id already exists, this is a no-op.
    Callers that re-run the same build (e.g. retry after a Flask
    response error) get exactly one DB row.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO export_batches (
            batch_id, built_at, since_at, until_at,
            hard_negatives_count, confirmed_positives_count,
            species_relabels_count, favorites_count,
            frame_integrity_dropped_count,
            exporter_version, wmb_app_version, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch.batch_id,
            batch.built_at,
            batch.since,
            batch.until,
            batch.counts["hard_negatives"],
            batch.counts["confirmed_positives"],
            batch.counts["species_relabels"],
            batch.counts["favorites"],
            len(batch.frame_integrity_dropped),
            EXPORTER_VERSION,
            batch.wmb_app_version,
            notes,
        ),
    )
    conn.commit()


def last_batch_until(conn: sqlite3.Connection) -> str | None:
    """Return the ``until_at`` of the most recent batch, or None.

    Used by the preview-page to default ``since`` for the next build.
    The next batch's ``since`` is the previous batch's ``until``,
    so user actions in the gap are picked up exactly once.
    """
    row = conn.execute(
        "SELECT until_at FROM export_batches ORDER BY built_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    try:
        return row["until_at"]
    except (TypeError, IndexError, KeyError):
        return row[0]


def list_recorded_batches(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most-recent ``limit`` rows from ``export_batches``.

    Newest first. Used by the export page's history table — operator
    sees what was shipped, when, with how many rows per bucket. Pure
    read; no side effects.
    """
    if limit <= 0:
        return []
    rows = conn.execute(
        """
        SELECT
            batch_id, built_at, since_at, until_at,
            hard_negatives_count, confirmed_positives_count,
            species_relabels_count, favorites_count,
            frame_integrity_dropped_count,
            exporter_version, wmb_app_version, notes
        FROM export_batches
        ORDER BY built_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "batch_id": r["batch_id"],
            "built_at": r["built_at"],
            "since_at": r["since_at"],
            "until_at": r["until_at"],
            "counts": {
                "hard_negatives": r["hard_negatives_count"],
                "confirmed_positives": r["confirmed_positives_count"],
                "species_relabels": r["species_relabels_count"],
                "favorites": r["favorites_count"],
            },
            "frame_integrity_dropped_count": r["frame_integrity_dropped_count"],
            "exporter_version": r["exporter_version"],
            "wmb_app_version": r["wmb_app_version"],
            "notes": r["notes"],
        }
        for r in rows
    ]


def estimate_batch_bytes(
    conn: sqlite3.Connection,
    *,
    path_resolver,
    since: str | None = None,
    until: str | None = None,
    include_confirmed_positives: bool = INCLUDE_CONFIRMED_POSITIVES_DEFAULT,
) -> dict[str, Any]:
    """Estimate the on-disk byte size of a batch BEFORE building it.

    Builds the batch in-memory (cheap — just SQL + path resolution,
    no ZIP construction) and sums the byte size of every distinct
    image file. The ZIP overhead (manifests, COCO JSON, README) is
    negligible vs. the image files at the dataset sizes we care
    about, so the estimate is intentionally just the image total.

    Returns dict with:
        bytes: int — sum of file sizes on disk
        image_count: int — distinct files counted
        missing_count: int — files referenced by the batch but not
            present on disk (excluded from ``bytes``)
    """
    now = datetime.now(UTC)
    effective_until = until or now.isoformat()
    batch = build_batch(
        conn,
        path_resolver=path_resolver,
        since=since,
        until=effective_until,
        include_confirmed_positives=include_confirmed_positives,
        now=now,
    )

    total = 0
    counted = 0
    missing = 0
    for path in batch._image_paths.values():
        try:
            if path.is_file():
                total += path.stat().st_size
                counted += 1
            else:
                missing += 1
        except OSError:
            missing += 1
    return {
        "bytes": total,
        "image_count": counted,
        "missing_count": missing,
    }


def preview_counts(
    conn: sqlite3.Connection,
    since: str | None = None,
    *,
    include_confirmed_positives: bool = INCLUDE_CONFIRMED_POSITIVES_DEFAULT,
) -> dict[str, int]:
    """Return per-bucket counts that WOULD be in a batch built now.

    Cheap enough to call on every page render — uses the same SQL as
    the fetch_* functions so the preview never lies about what the
    actual export will contain. ``confirmed_positives`` reports 0
    when the opt-in is off, mirroring ``build_batch``.
    """
    counts = count_pending_by_bucket(conn, since=since)
    if not include_confirmed_positives:
        counts["confirmed_positives"] = 0
    return counts


def ensure_export_exclusion_schema(conn: sqlite3.Connection) -> None:
    """Create the non-destructive export quarantine table if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groundtruth_export_exclusions (
            exclusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL CHECK(scope IN ('image', 'detection')),
            image_filename TEXT NOT NULL,
            detection_id INTEGER,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT NOT NULL DEFAULT 'dry_run',
            released_at TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_groundtruth_export_exclusions_image "
        "ON groundtruth_export_exclusions(image_filename, released_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_groundtruth_export_exclusions_detection "
        "ON groundtruth_export_exclusions(detection_id, released_at)"
    )


def exclude_image_from_export(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    reason: str = "",
    now: datetime | None = None,
) -> bool:
    """Quarantine a whole frame from future user-groundtruth exports.

    Returns ``True`` when a new active exclusion was inserted, ``False``
    when the image was already excluded. The source image and detection
    rows remain untouched.
    """
    ensure_export_exclusion_schema(conn)
    created_at = (now or datetime.now(UTC)).isoformat()
    cur = conn.execute(
        """
        INSERT INTO groundtruth_export_exclusions (
            scope, image_filename, detection_id, reason, created_at, created_by
        )
        SELECT 'image', ?, NULL, ?, ?, 'dry_run'
        WHERE NOT EXISTS (
            SELECT 1
            FROM groundtruth_export_exclusions
            WHERE scope = 'image'
              AND image_filename = ?
              AND released_at IS NULL
        )
        """,
        (image_filename, reason, created_at, image_filename),
    )
    return cur.rowcount > 0


def move_image_to_wmb_trash_and_exclude(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    reason: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Move one source frame to WMB Trash and suppress export rows.

    This is intentionally a reversible Trash move, not a hard delete:
    the original file can still be restored or purged from the Trash UI.
    The export exclusion is kept even though ``review_status='no_bird'``
    would remove positive rows, because active detections on no-bird
    frames otherwise become hard-negative export candidates.
    """
    updated_at = (now or datetime.now(UTC)).isoformat()
    detection_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM detections
        WHERE image_filename = ?
          AND COALESCE(status, 'active') = 'active'
        """,
        (image_filename,),
    ).fetchone()[0]
    image_cur = conn.execute(
        """
        UPDATE images
        SET review_status = 'no_bird',
            review_updated_at = ?
        WHERE filename = ?
        """,
        (updated_at, image_filename),
    )
    inserted = exclude_image_from_export(
        conn,
        image_filename=image_filename,
        reason=reason,
        now=now,
    )
    return {
        "updated_images": int(image_cur.rowcount),
        "active_detections": int(detection_count or 0),
        "excluded": inserted,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _date_folder_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from a filename like ``20260522_120512_abc.jpg``."""
    stem = Path(filename).stem
    if len(stem) >= 8 and stem[:8].isdigit():
        return f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
    # Fallback: shouldn't happen for production files, but the export
    # path must not crash on malformed filenames — bucket them under
    # an explicit "unknown_date" subfolder so the operator notices.
    return "unknown_date"


def _manifest_line(row: dict[str, Any], batch: Batch) -> str:
    """Build one JSON line for the per-bucket JSONL manifest.

    Adds the provenance fields that downstream training needs but that aren't
    in the raw DB row: the date-folder-prefixed image path inside the
    ZIP, the pixel-space bbox, and the originating batch_id.
    """
    filename = row["image_filename"]
    line: dict[str, Any] = {
        "detection_id": row["detection_id"],
        "image_filename": filename,
        "image_path_in_zip": f"images/{_date_folder_from_filename(filename)}/{filename}",
        "bbox_xywh_normalized": [
            row["bbox_x"],
            row["bbox_y"],
            row["bbox_w"],
            row["bbox_h"],
        ],
        "bbox_xywh_px": _bbox_to_pixels(row),
        "frame_wh": [row["frame_width"], row["frame_height"]],
        "bucket": row["bucket"],
        "user_action_at": row["user_action_at"],
        "batch_id": batch.batch_id,
        "model_at_inference": {
            "detector": row["detector_model_version"],
            "classifier": row["classifier_model_version"],
        },
        "raw_pipeline_output": {
            "od_confidence": row["od_confidence"],
            "od_class_name": row["od_class_name"],
        },
    }
    # Bucket-specific provenance
    if row["bucket"] == "confirmed_positives":
        line["species"] = row.get("species")
        line["species_source"] = row.get("species_source")
    elif row["bucket"] == "species_relabels":
        line["model_predicted_species"] = row.get("model_predicted_species")
        line["user_corrected_species"] = row.get("user_corrected_species")
        line["species_source"] = row.get("species_source")
    elif row["bucket"] == "favorites":
        line["species"] = row.get("species")
        line["species_source"] = row.get("species_source")
        line["is_gold_label"] = True  # explicit hint for downstream training
    return json.dumps(line, ensure_ascii=False)


def _bbox_to_pixels(row: dict[str, Any]) -> list[int] | None:
    """Convert normalized bbox to pixel-space [x, y, w, h].

    Returns None when any component is missing — downstream training
    coco-consumer should reject such rows for box-supervised training
    but can still use them for image-level classification.
    """
    fw = row.get("frame_width")
    fh = row.get("frame_height")
    bx = row.get("bbox_x")
    by = row.get("bbox_y")
    bw = row.get("bbox_w")
    bh = row.get("bbox_h")
    if fw is None or fh is None or bx is None or by is None or bw is None or bh is None:
        return None
    return [
        int(round(bx * fw)),
        int(round(by * fh)),
        int(round(bw * fw)),
        int(round(bh * fh)),
    ]


def _build_coco(
    batch: Batch,
    *,
    missing_images: set[str],
) -> dict[str, Any]:
    """Build a standard COCO dict from the batch.

    Categories: every distinct species in confirmed_positives or
    species_relabels gets a category. Hard-negatives become images
    with zero annotations (the standard COCO way to express "no
    object in this frame").

    Categories include the ``__non_bird__`` synthetic category for
    hard-negative crops where the operator's signal was "this is not
    a bird" — downstream training can include those as negative-class
    annotations or skip them depending on training regime.
    """
    # Collect distinct species across all three positive buckets.
    # Relabels carry both old and new species; we use the user-
    # corrected one as the authoritative label for COCO. Favorites
    # carry the resolved species (manual_species_override-wins-over-raw
    # already collapsed in the query).
    species_set: set[str] = set()
    for r in batch.confirmed_positives:
        sp = r.get("species")
        if sp:
            species_set.add(sp)
    for r in batch.species_relabels:
        sp = r.get("user_corrected_species")
        if sp:
            species_set.add(sp)
    for r in batch.favorites:
        sp = r.get("species")
        if sp:
            species_set.add(sp)

    # Stable category IDs (sorted, 1-indexed — COCO convention).
    sorted_species = sorted(species_set)
    species_to_cat_id: dict[str, int] = {
        sp: i + 1 for i, sp in enumerate(sorted_species)
    }
    categories = [
        {"id": cat_id, "name": sp, "supercategory": "bird"}
        for sp, cat_id in species_to_cat_id.items()
    ]

    # Filename -> image_id assignment. Only images present on disk
    # appear; missing-on-disk images skip both the image entry and
    # its annotations (but stay in the manifest for audit).
    sorted_filenames = sorted(
        fn for fn in batch._image_paths.keys() if fn not in missing_images
    )
    fn_to_img_id: dict[str, int] = {fn: i + 1 for i, fn in enumerate(sorted_filenames)}

    images: list[dict[str, Any]] = []
    # Frame width/height: take from any detection on that frame
    # (they all share the frame, so any row works). Build a quick
    # lookup so we don't iterate all detections per image.
    fn_to_geometry: dict[str, tuple[int | None, int | None]] = {}
    for bucket in (
        batch.hard_negatives,
        batch.confirmed_positives,
        batch.species_relabels,
        batch.favorites,
    ):
        for r in bucket:
            fn = r["image_filename"]
            if fn not in fn_to_geometry:
                fn_to_geometry[fn] = (r.get("frame_width"), r.get("frame_height"))

    for fn, img_id in fn_to_img_id.items():
        fw, fh = fn_to_geometry.get(fn, (None, None))
        images.append(
            {
                "id": img_id,
                "file_name": f"images/{_date_folder_from_filename(fn)}/{fn}",
                "width": fw,
                "height": fh,
            }
        )

    # Annotations: one per detection that has both an image and a bbox.
    # Hard-negatives without species → no annotation row (just the
    # image entry with zero anns, signaling "negative example").
    # A detection can appear in multiple positive buckets (e.g.
    # confirmed-and-favorited) — dedupe by detection_id so COCO has
    # exactly one annotation per detection, with the highest-priority
    # bucket tag for downstream sampling.
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    seen_detection_ids: set[int] = set()

    def _label_for(row: dict[str, Any]) -> str | None:
        bucket = row["bucket"]
        if bucket == "confirmed_positives":
            return row.get("species")
        if bucket == "species_relabels":
            return row.get("user_corrected_species")
        if bucket == "favorites":
            return row.get("species")
        return None  # hard_negatives have no positive label

    # Bucket iteration order = priority: a detection seen first under
    # ``favorites`` (gold-label) stays tagged as favorite even if it
    # also appears in confirmed_positives.
    for bucket in (
        batch.hard_negatives,
        batch.favorites,
        batch.species_relabels,
        batch.confirmed_positives,
    ):
        for r in bucket:
            fn = r["image_filename"]
            if fn in missing_images or fn not in fn_to_img_id:
                continue
            det_id = r["detection_id"]
            if det_id in seen_detection_ids:
                continue
            label = _label_for(r)
            if label is None:
                # Hard-negative crop — skip annotation row, frame still
                # ships as an image so the trainer sees the negative.
                # Mark as seen so a later positive-bucket pass on the
                # same id (shouldn't happen because HN/positives are
                # mutually exclusive by review_status, but defensive)
                # would not produce a competing annotation.
                seen_detection_ids.add(det_id)
                continue
            bbox_px = _bbox_to_pixels(r)
            if bbox_px is None:
                seen_detection_ids.add(det_id)
                continue  # geometry missing, can't supervise
            x, y, w, h = bbox_px
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": fn_to_img_id[fn],
                    "category_id": species_to_cat_id[label],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "wmb_detection_id": det_id,
                    "wmb_bucket": r["bucket"],
                }
            )
            ann_id += 1
            seen_detection_ids.add(det_id)

    return {
        "info": {
            "description": "WatchMyBirds user-groundtruth export",
            "batch_id": batch.batch_id,
            "exporter_version": EXPORTER_VERSION,
            "built_at": batch.built_at,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _build_metadata(
    batch: Batch,
    *,
    missing_images: list[str],
) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "exporter_version": EXPORTER_VERSION,
        "wmb_app_version": batch.wmb_app_version,
        "built_at": batch.built_at,
        "since": batch.since,
        "until": batch.until,
        "counts": batch.counts,
        "total_rows": batch.total_rows,
        "missing_images_on_disk": missing_images,
        # Operator provenance — empty string when the browser had
        # nothing stored; downstream training treats both keys as optional.
        "station_id": batch.station_id,
        "reviewer_id": batch.reviewer_id,
        # Frame-integrity audit: frames the export *dropped* because
        # their active siblings were not all user-confirmed. Each
        # entry names the frame, why it was dropped, and which
        # detection_ids had carried a positive signal before the
        # drop (so the operator can investigate). Empty list when
        # every positive frame was clean.
        "frame_integrity_dropped": batch.frame_integrity_dropped,
    }


def _build_readme(batch: Batch) -> str:
    counts = batch.counts
    dropped_n = len(batch.frame_integrity_dropped)
    integrity_note = (
        (
            f"- **{dropped_n} frame(s) dropped by frame-integrity** "
            f"— see `batch_metadata.json::frame_integrity_dropped`. "
            f"These frames had at least one active sibling detection without "
            f"a user-confirmed signal, so the WHOLE frame was excluded to "
            f"avoid teaching the trainer that the unannotated boxes are "
            f"background."
        )
        if dropped_n
        else (
            "- No frames dropped by frame-integrity (all positive frames had "
            "every sibling user-confirmed)."
        )
    )
    return f"""# WatchMyBirds — User-Groundtruth Batch {batch.batch_id}

Built at: {batch.built_at}
Time window: {batch.since or "(all history)"} → {batch.until}
Exporter version: {EXPORTER_VERSION}

## Contents

- **{counts["hard_negatives"]} hard-negatives** — user-verified FP crops
  (image-level "not a bird" signal). Use these to retrain the OD with
  higher emphasis on background-vs-bird discrimination on this station.
- **{counts["confirmed_positives"]} confirmed-positives** — species
  the user explicitly authored (via Review-Queue, Unclear "Confirm
  Day", or bulk-relabel). Rows are gated by
  ``species_source='manual'`` OR a non-empty
  ``manual_species_override`` — pure pipeline predictions are
  excluded to prevent a confirmation-bias loop.
- **{counts["species_relabels"]} species-relabels** — user explicitly
  corrected the species. Includes both the model's wrong prediction
  and the user's correction — highest-value rows for classifier
  calibration.
- **{counts["favorites"]} favorites** — user explicitly heart-clicked
  these detections in the gallery UI. The codebase treats them as
  the manual gold label that wins over every algorithmic ranking
  (`gallery_core.py:542`). Treat them as your highest-confidence
  positive anchors.

{integrity_note}

## Files

- `coco_annotations.json` — standard COCO with all positives merged.
  Each annotation carries `wmb_detection_id` and `wmb_bucket` so you
  can stratify training by source signal. A detection appearing in
  multiple positive buckets (e.g. confirmed AND favorited) gets ONE
  annotation tagged with the highest-priority bucket
  (favorites > species_relabels > confirmed_positives).
- `manifests/<bucket>.jsonl` — one JSON object per line, full
  provenance (which user action, when, which model version was active
  at inference). Use these to weight buckets in your sampler. Note
  that the same detection MAY appear in multiple JSONL files if it
  qualified for multiple buckets — the COCO file deduplicates, the
  manifests preserve every signal-source.
- `images/YYYY-MM-DD/...` — original frames, date-sharded. Hard-negative
  frames appear here with zero annotations in COCO — that is intentional
  and tells your trainer "this image has no birds".
- `batch_metadata.json` — machine-readable batch summary, including
  the `frame_integrity_dropped` audit list.

## Frame-integrity guarantee

Every positive frame in this ZIP has **all** its active detections
covered by a user-confirmed signal. Frames where one box was
confirmed but a sibling box was still `species_review` or `reject`
are NOT here — they would have taught your trainer that the
unannotated box positions are "background", actively poisoning the
detector. See the dropped-frames list in `batch_metadata.json` for
the audit trail.

## Suggested sampling weights for retraining

These are starting points, tune per validation results:

- Favorites: **5x** oversampled — gold-label, lowest-noise positives
- Hard-negatives: 2-3x oversampled (most underrepresented signal)
- Species-relabels: 1.5-2x oversampled (rare but high information)
- Confirmed-positives: 1.0x (the baseline)

## Provenance trust

Every row in this batch traces back to an explicit user action
recorded in the WMB DB. The batch deliberately excludes ambiguous
signals: no Trash-deletions, no star ratings (98% are auto-tagged).
Downstream training can treat each row as ground-truth from a single
operator.

## What is NOT in this batch

- Detections that the pipeline confirmed as species but the user
  never touched: those flow via the older `training_export` path
  if you still consume it.
- Bbox corrections (`manual_bbox_review`): not included yet — current
  batches use the original detector geometry.
- Detections from frames the user later deleted (Trash): excluded
  by policy (ambiguous signal — could be FP or "ugly but bird").
- Auto-tagged star ratings (`rating_source='auto'`): not a user
  statement. Only favorites with `rating_source='manual'` qualify.

"""
