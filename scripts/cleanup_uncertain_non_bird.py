"""Cleanup UNCERTAIN non-bird detections and their orphan image files.

Targets the storage / DB residue left over from the period when the
non-bird scoring track had no dedicated confidence floor — typically
static-bbox marten / cat / squirrel / hedgehog detections that the
operator never sees (UNCERTAIN is hidden from Stream / Gallery /
Telegram) but that still occupy disk and DB rows.

By default this is a **dry run**: it counts and writes a CSV listing the
deletion candidates, then exits with no mutation. Pass ``--apply`` to
execute the deletion.

Safety:

- ``manual_species_override`` rows are NEVER deleted, even if the OD
  class is non-bird (the operator manually re-labelled the row).
- Image files are deleted only when no surviving detection references
  them. A bird sibling on the same frame keeps the image and its
  derivatives.
- A short DB backup is taken in --apply mode before any DELETE runs.

Usage:
    python scripts/cleanup_uncertain_non_bird.py /path/to/images.db
        [--apply] [--output-csv path] [--data-root path]

On the RPi:
    sudo python3 scripts/cleanup_uncertain_non_bird.py \\
        /opt/app/data/output/images.db \\
        --data-root /opt/app/data/output \\
        --output-csv /tmp/cleanup_non_bird.csv
    # review the CSV, then re-run with --apply
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

NON_BIRD_CLASSES = ("marten_mustelid", "cat", "squirrel", "hedgehog")


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: {db_path} not found", file=sys.stderr)
        sys.exit(2)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _find_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return the detection rows that A2 would delete.

    Filter:
      - decision_state == 'uncertain'
      - od_class_name in NON_BIRD_CLASSES
      - manual_species_override is NULL or empty (never touch human edits)
    """
    placeholders = ",".join("?" for _ in NON_BIRD_CLASSES)
    return list(
        conn.execute(
            f"""
            SELECT
                detection_id,
                image_filename,
                od_class_name,
                od_confidence,
                decision_state,
                manual_species_override,
                created_at
            FROM detections
            WHERE decision_state = 'uncertain'
              AND od_class_name IN ({placeholders})
              AND (manual_species_override IS NULL OR manual_species_override = '')
            ORDER BY created_at
            """,
            NON_BIRD_CLASSES,
        ).fetchall()
    )


def _orphan_images(
    conn: sqlite3.Connection, candidate_filenames: set[str]
) -> set[str]:
    """Return image filenames whose only detections are A2 candidates.

    An image is orphaned iff every active detection on it is in the
    candidate set. A bird detection or a human-edited row on the same
    frame keeps the image alive.
    """
    if not candidate_filenames:
        return set()

    orphans: set[str] = set()
    for filename in candidate_filenames:
        # Count detections on this image that are NOT in the candidate set.
        # If zero survive, the image is orphan-deletable.
        survivors = conn.execute(
            """
            SELECT COUNT(*) FROM detections
            WHERE image_filename = ?
              AND NOT (
                  decision_state = 'uncertain'
                  AND od_class_name IN ('marten_mustelid','cat','squirrel','hedgehog')
                  AND (manual_species_override IS NULL OR manual_species_override = '')
              )
            """,
            (filename,),
        ).fetchone()[0]
        if survivors == 0:
            orphans.add(filename)
    return orphans


def _resolve_image_files(filename: str, data_root: Path) -> list[Path]:
    """Best-effort lookup of every file on disk that belongs to ``filename``.

    Filenames look like ``20260510_232238_071265.jpg``; the leading 8
    chars are the YYYYMMDD shard directory used under originals/ and
    derivatives/. Thumbnails carry a ``_crop_<N>.webp`` suffix.
    """
    stem = Path(filename).stem  # 20260510_232238_071265
    shard = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"  # 2026-05-10

    paths: list[Path] = []
    # Original JPG
    paths.append(data_root / "originals" / shard / filename)
    # Optimized webp (same stem, .webp extension)
    paths.append(data_root / "derivatives" / "optimized" / shard / f"{stem}.webp")
    # Crop thumbnails (any rank — collect by glob)
    thumb_dir = data_root / "derivatives" / "thumbs" / shard
    if thumb_dir.is_dir():
        paths.extend(sorted(thumb_dir.glob(f"{stem}_crop_*.webp")))
    return paths


def _write_csv(
    rows: list[sqlite3.Row],
    orphan_set: set[str],
    output_path: Path,
    data_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "detection_id",
                "image_filename",
                "od_class_name",
                "od_confidence",
                "decision_state",
                "manual_species_override",
                "created_at",
                "image_orphaned",
                "image_files_on_disk",
            ]
        )
        for r in rows:
            files_on_disk = ""
            if r["image_filename"] in orphan_set:
                paths = _resolve_image_files(r["image_filename"], data_root)
                existing = [str(p) for p in paths if p.exists()]
                files_on_disk = ";".join(existing)
            writer.writerow(
                [
                    r["detection_id"],
                    r["image_filename"],
                    r["od_class_name"],
                    r["od_confidence"],
                    r["decision_state"],
                    r["manual_species_override"] or "",
                    r["created_at"],
                    "yes" if r["image_filename"] in orphan_set else "no",
                    files_on_disk,
                ]
            )


def _backup_db(db_path: Path) -> Path:
    """Snapshot the DB via sqlite3 backup before any DELETE."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_suffix(f".pre-a2-cleanup-{timestamp}.db")
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return backup


def _apply_deletion(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    orphan_set: set[str],
    data_root: Path,
) -> dict[str, int]:
    stats = {"detections_deleted": 0, "images_db_deleted": 0, "files_deleted": 0}

    detection_ids = [r["detection_id"] for r in rows]
    cursor = conn.cursor()

    # 1. Delete detection rows in chunks (sqlite parameter limit).
    chunk = 500
    for i in range(0, len(detection_ids), chunk):
        ids = detection_ids[i : i + chunk]
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"DELETE FROM detections WHERE detection_id IN ({placeholders})",
            ids,
        )
        stats["detections_deleted"] += cursor.rowcount

    # 2. Delete orphan image rows (CASCADE would already have triggered for
    #    detections, but the images table itself needs explicit deletion).
    orphan_list = sorted(orphan_set)
    for i in range(0, len(orphan_list), chunk):
        names = orphan_list[i : i + chunk]
        placeholders = ",".join("?" for _ in names)
        cursor.execute(
            f"DELETE FROM images WHERE filename IN ({placeholders})",
            names,
        )
        stats["images_db_deleted"] += cursor.rowcount

    conn.commit()

    # 3. Delete files on disk for orphaned images.
    for filename in orphan_list:
        for path in _resolve_image_files(filename, data_root):
            if path.exists():
                try:
                    path.unlink()
                    stats["files_deleted"] += 1
                except OSError as e:
                    print(f"WARN: could not unlink {path}: {e}", file=sys.stderr)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="Path to images.db")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root of originals/ and derivatives/. Defaults to the DB's parent dir.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/tmp/cleanup_uncertain_non_bird.csv"),
        help="Where to write the candidate CSV.",
    )
    args = parser.parse_args()

    data_root = args.data_root or args.db_path.parent
    if not data_root.exists():
        print(f"ERROR: data-root {data_root} not found", file=sys.stderr)
        return 2

    conn = _connect(args.db_path)
    rows = _find_candidates(conn)
    if not rows:
        print("No UNCERTAIN non-bird detections to clean up.")
        return 0

    candidate_filenames = {r["image_filename"] for r in rows}
    orphan_set = _orphan_images(conn, candidate_filenames)

    # Summary
    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r["od_class_name"]] = by_class.get(r["od_class_name"], 0) + 1

    print(f"Found {len(rows)} UNCERTAIN non-bird detections:")
    for cls, n in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}")
    print(f"Images that would be orphaned (no surviving detection): {len(orphan_set)}")
    print(f"  → ~{len(orphan_set) * 0.6:.0f} MB of disk freed (estimate at 600 KB/image)")

    _write_csv(rows, orphan_set, args.output_csv, data_root)
    print(f"\nCandidate CSV: {args.output_csv}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    print("\nDB backup before deletion …")
    backup_path = _backup_db(args.db_path)
    print(f"  → {backup_path}")

    stats = _apply_deletion(conn, rows, orphan_set, data_root)
    print("\nDeletion summary:")
    print(f"  detections rows deleted: {stats['detections_deleted']}")
    print(f"  images rows deleted:     {stats['images_db_deleted']}")
    print(f"  files unlinked on disk:  {stats['files_deleted']}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
