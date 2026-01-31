"""
DELETE POLICY — WatchMyBirds
============================

Purpose
-------
Define clear and consistent semantics for deleting detections and their
associated files (database + filesystem) in order to prevent data loss
and uncontrolled storage growth.

Terminology
-----------
- Soft Delete  = Move to Trash (reversible)
- Hard Delete  = Delete Forever / Empty Trash (irreversible)

1) Soft Delete (Trash)
----------------------
Triggered by:
- "Move to Trash"
- "Reject"

Semantics:
- DATABASE ONLY operation
- Detection status is updated (e.g. active -> trashed / rejected)
- NO files may be deleted from disk

Rationale:
- Restore must always be possible
- Prevents accidental data loss

2) Restore from Trash
---------------------
Triggered by:
- "Restore"

Semantics:
- DATABASE ONLY operation
- Detection status is set back to active
- NO files are created or deleted

3) Hard Delete (Delete Forever / Empty Trash)
---------------------------------------------
Triggered by:
- "Delete Forever"
- "Empty Trash"

Semantics:
- DATABASE AND FILESYSTEM cleanup
- Operation is irreversible

MANDATORY ORDER OF OPERATIONS:
1. Resolve all affected file paths from the database.
   - detection-specific files: thumbnail_path, crop images, etc.
   - shared files: original_image, optimized_image
2. Attempt to delete files from disk (log WARNING for missing files, ERROR for failures or out-of-bounds).
3. Delete database records after file deletion attempts, regardless of individual file deletion outcomes, to ensure idempotent hard delete behavior.

4) Handling shared files
------------------------
- Multiple detections may reference the same original/optimized files
- Shared files MUST be deleted only if:
    - No other detection (active, trashed, or rejected)
      references the same file path
- Thumbnail files are detection-specific and may always be deleted immediately

5) Safety rules
---------------
- Never delete files outside OUTPUT_DIR
- Normalize and validate paths before deletion
- No silent failures during file deletion
- No database deletion without prior file cleanup
- file_gc operates exclusively on absolute paths resolved via PathManager.

6) Policy goals
---------------
- Prevent uncontrolled filesystem growth
- Clear separation between reversible and irreversible actions
- Predictable and auditable delete behavior
- Solid foundation for future features (retention, cloud sync, ground truth)

This policy is binding for all delete operations.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config import get_config
from utils.path_manager import get_path_manager


logger = logging.getLogger(__name__)


def _safe_delete(abs_path: Path, output_dir: Path) -> str:
    """
    Safely deletes a file at absolute path.
    Verifies path is within output_dir.
    """
    if not abs_path:
        return "skipped"

    try:
        # Resolve to ensure no symlink tricks, though PathManager usually handles this.
        abs_path = abs_path.resolve()
        output_dir = output_dir.resolve()

        # Verify safety: must be relative to output_dir
        if not str(abs_path).startswith(str(output_dir)):
            logger.error(f"Refusing to delete outside OUTPUT_DIR: {abs_path}")
            return "error"

    except Exception as e:
        logger.error(f"Path verification error for {abs_path}: {e}")
        return "error"

    try:
        if abs_path.exists():
            abs_path.unlink()
            return "deleted"
        else:
            logger.warning(f"File not found for deletion: {abs_path}")
            return "missing"
    except Exception as e:
        logger.error(f"Failed to delete file: {abs_path} ({e})")
        return "error"


def hard_delete_detections(
    conn,
    detection_ids: Iterable[int] = None,
    before_date: str = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes rejected detections and associated files.
    Order: resolve paths -> delete files -> delete DB rows.
    """
    if not detection_ids and not before_date:
        raise ValueError("Hard delete requires explicit detection_ids or filter")

    where_clauses = ["d.status = 'rejected'"]
    params: list[Any] = []

    if detection_ids:
        ids = list(detection_ids)
        placeholders = ",".join("?" for _ in ids)
        where_clauses.append(f"d.detection_id IN ({placeholders})")
        params.extend(ids)

    if before_date:
        date_prefix = before_date.replace("-", "")
        where_clauses.append("i.timestamp < ?")
        params.append(date_prefix)

    where_sql = " AND ".join(where_clauses)

    # 1. Fetch Candidates (Include timestamp for PathManager)
    # We select i.timestamp because PathManager needs it for folder structure if not in filename (mostly in filename)
    cur = conn.execute(
        f"""
        SELECT
            d.detection_id,
            d.thumbnail_path,
            i.filename as original_name,
            i.timestamp as image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {where_sql}
        """,
        params,
    )
    rows = cur.fetchall()

    affected_ids = [row["detection_id"] for row in rows]
    if dry_run:
        return {
            "purged": False,
            "would_purge": len(affected_ids),
            "detection_ids": affected_ids,
        }

    cfg = get_config()
    output_dir = Path(cfg["OUTPUT_DIR"]).resolve()
    pm = get_path_manager(output_dir)

    # 2. Identify Files to Delete
    target_originals: set[Path] = set()
    target_optimized: set[Path] = set()
    target_thumbnails: set[Path] = set()

    # Maps for reference counting (Path -> Count of candidates using it)
    candidate_orig_refs: dict[Path, int] = {}
    candidate_opt_refs: dict[Path, int] = {}
    # candidate_thumb_refs: Dict[Path, int] = {} # Thumbnails are generally unique per detection

    for row in rows:
        original_name = row["original_name"]
        thumb_name = row["thumbnail_path"]

        if original_name:
            orig_path = pm.get_original_path(original_name)
            opt_path = pm.get_derivative_path(original_name, "optimized")

            target_originals.add(orig_path)
            target_optimized.add(opt_path)

            candidate_orig_refs[orig_path] = candidate_orig_refs.get(orig_path, 0) + 1
            candidate_opt_refs[opt_path] = candidate_opt_refs.get(opt_path, 0) + 1

        # Resolve Thumbnail
        if not thumb_name and original_name:
            # Fallback to crop 1
            thumb_name = original_name.replace(".jpg", "_crop_1.webp")

        if thumb_name:
            # Type is "thumb" based on PathManager convention
            thumb_path = pm.get_derivative_path(thumb_name, "thumb")
            target_thumbnails.add(thumb_path)

    # 3. Global Reference Check (Are these files used by ACTIVE or NON-PURGED detections?)
    # We query the DB for GLOBAL usage counts of the candidate files.

    unique_orig_names = {p.name for p in target_originals}

    # Total references in the ENTIRE DB (active + rejected + trashed)
    total_orig_refs: dict[Path, int] = {}
    total_opt_refs: dict[Path, int] = {}

    if unique_orig_names:
        placeholders = ",".join("?" for _ in unique_orig_names)
        # We perform a GROUP BY to get count of all detections per image
        cur_refs = conn.execute(
            f"""
            SELECT i.filename, COUNT(*) as cnt
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            WHERE i.filename IN ({placeholders})
            GROUP BY i.filename
            """,
            list(unique_orig_names),
        )

        for r in cur_refs:
            fname = r["filename"]
            count = r["cnt"]

            op = pm.get_original_path(fname)
            opp = pm.get_derivative_path(fname, "optimized")

            total_orig_refs[op] = count
            total_opt_refs[opp] = count

    # 4. Filter for Deletion
    files_to_delete: set[Path] = set()

    # Shared Files
    for path, candidate_count in candidate_orig_refs.items():
        total_count = total_orig_refs.get(path, 0)
        # If the number of candidates we are deleting EQUALS the total number of references in DB,
        # then no one else is using it. Safe to delete.
        if total_count <= candidate_count:
            files_to_delete.add(path)

    for path, candidate_count in candidate_opt_refs.items():
        total_count = total_opt_refs.get(path, 0)
        if total_count <= candidate_count:
            files_to_delete.add(path)

    # Thumbnails - Always delete (Detection Specific)
    # Rationale: Detections own their thumbnails. One detection -> One crop.
    # Exception: If multiple detections somehow point to same `thumbnail_path`.
    # But clean slate architecture implies `thumbnail_path` is specific.
    files_to_delete.update(target_thumbnails)

    # 5. Execute Deletions
    files_deleted = 0
    files_missing = 0
    files_failed = 0

    # Sort for consistent log output
    for abs_path in sorted(list(files_to_delete)):
        result = _safe_delete(abs_path, output_dir)
        if result == "deleted":
            files_deleted += 1
        elif result == "missing":
            files_missing += 1
        elif result == "error":
            files_failed += 1

    # 6. Delete Database Records (Idempotent)
    if affected_ids:
        id_placeholders = ",".join("?" for _ in affected_ids)
        conn.execute(
            f"DELETE FROM detections WHERE detection_id IN ({id_placeholders})",
            affected_ids,
        )

        # Cleanup orphaned images (no remaining detections)
        for path in files_to_delete:
            # Check if this path corresponds to an original
            if path in target_originals:
                # Inefficient reverse lookup?
                # Better: Iterate target_originals, check if in files_to_delete
                pass

        # Better: We know which originals we deemed safe to delete.
        # `files_to_delete` contains paths.

        # Let's collect filenames for DB cleanup
        for orig_path in target_originals:
            if orig_path in files_to_delete:
                # Extract filename? We don't have mapping path->filename handy without re-looping or storing.
                # Let's rebuild mapping in loop 2.
                pass

        # Re-verify: `images` table usually stays or cleaned up?
        # Orphan management typically handles `images` without detections.
        # But if we hard delete, we want to clean up.

        # Let's just delete `images` rows where we deleted the file.
        # We can map Path -> Filename.
        path_to_filename = {pm.get_original_path(n): n for n in unique_orig_names}

        images_to_delete = []
        for p in files_to_delete:
            if p in path_to_filename:
                images_to_delete.append(path_to_filename[p])

        if images_to_delete:
            img_placeholders = ",".join("?" for _ in images_to_delete)
            conn.execute(
                f"DELETE FROM images WHERE filename IN ({img_placeholders})",
                images_to_delete,
            )

        conn.commit()

    return {
        "purged": True,
        "count": len(affected_ids),
        "detection_ids": affected_ids,
        "rows_deleted": len(affected_ids),
        "files_deleted": files_deleted,
        "files_missing": files_missing,
        "files_failed": files_failed,
    }


def hard_delete_images(
    conn,
    filenames: Iterable[str] = None,
    delete_all: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes no_bird images and associated files.
    Order: resolve paths -> delete files -> delete DB rows.

    Safeguard: Only operates on images where review_status = 'no_bird'.

    Args:
        conn: Database connection
        filenames: Specific filenames to delete (None for delete_all)
        delete_all: If True and filenames is None, deletes ALL no_bird images
        dry_run: If True, returns what would be deleted without actually deleting

    Returns:
        dict with keys: purged, rows_deleted, files_deleted, files_missing, files_failed
    """
    if not filenames and not delete_all:
        return {
            "purged": False,
            "rows_deleted": 0,
            "files_deleted": 0,
            "files_missing": 0,
            "files_failed": 0,
        }

    # 1. Fetch candidate images from DB (only no_bird status)
    if filenames:
        names = list(filenames)
        placeholders = ",".join("?" for _ in names)
        cur = conn.execute(
            f"""
            SELECT filename, timestamp
            FROM images
            WHERE filename IN ({placeholders}) AND review_status = 'no_bird'
            """,
            names,
        )
    else:
        cur = conn.execute(
            "SELECT filename, timestamp FROM images WHERE review_status = 'no_bird'"
        )

    rows = cur.fetchall()
    affected_filenames = [row["filename"] for row in rows]

    if dry_run:
        return {
            "purged": False,
            "would_purge": len(affected_filenames),
            "filenames": affected_filenames,
        }

    if not affected_filenames:
        return {
            "purged": True,
            "rows_deleted": 0,
            "files_deleted": 0,
            "files_missing": 0,
            "files_failed": 0,
        }

    cfg = get_config()
    output_dir = Path(cfg["OUTPUT_DIR"]).resolve()
    pm = get_path_manager(output_dir)

    # 2. Identify files to delete for each image
    files_to_delete: set[Path] = set()

    for filename in affected_filenames:
        # Original image
        files_to_delete.add(pm.get_original_path(filename))

        # Optimized derivative
        files_to_delete.add(pm.get_derivative_path(filename, "optimized"))

        # Preview thumbnail (used in Review Queue)
        files_to_delete.add(pm.get_preview_thumb_path(filename))

    # 3. Check for any detection thumbnails tied to these images
    # (In case a no_bird image had detections that were later removed/rejected)
    if affected_filenames:
        placeholders = ",".join("?" for _ in affected_filenames)
        det_cur = conn.execute(
            f"""
            SELECT thumbnail_path
            FROM detections
            WHERE image_filename IN ({placeholders}) AND thumbnail_path IS NOT NULL
            """,
            affected_filenames,
        )
        for det_row in det_cur:
            thumb_name = det_row["thumbnail_path"]
            if thumb_name:
                thumb_path = pm.get_derivative_path(thumb_name, "thumb")
                files_to_delete.add(thumb_path)

    # 4. Execute file deletions
    files_deleted = 0
    files_missing = 0
    files_failed = 0

    for abs_path in sorted(list(files_to_delete)):
        result = _safe_delete(abs_path, output_dir)
        if result == "deleted":
            files_deleted += 1
        elif result == "missing":
            files_missing += 1
        elif result == "error":
            files_failed += 1

    # 5. Delete database records
    # Note: CASCADE from images -> detections -> classifications should handle related records
    placeholders = ",".join("?" for _ in affected_filenames)
    conn.execute(
        f"DELETE FROM images WHERE filename IN ({placeholders})", affected_filenames
    )
    conn.commit()

    logger.info(
        f"hard_delete_images: {len(affected_filenames)} images deleted, "
        f"{files_deleted} files removed, {files_missing} missing, {files_failed} failed"
    )

    return {
        "purged": True,
        "rows_deleted": len(affected_filenames),
        "files_deleted": files_deleted,
        "files_missing": files_missing,
        "files_failed": files_failed,
    }
