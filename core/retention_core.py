"""Execute the shared photo deletion plan in bounded, rechecked transactions."""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.deletion_plan import (
    DeletionSettings,
    PhotoPlan,
    _time,
    deletion_preview,
    plan_deletions,
)
from utils.db.connection import human_answers_migrated
from utils.db.storage import image_batches
from utils.path_manager import PathManager
from utils.settings import load_settings_yaml

logger = logging.getLogger(__name__)

_ROW_KEYS = {
    "human_label_facts": "fact_id",
    "label_subjects": "subject_id",
    "classifications": "classification_id",
    "manual_object_revisions": "manual_revision_id",
    "manual_object_requests": "request_id",
    "station_event_review_members": "rowid",
    "station_event_reviews": "review_id",
    "rescan_proposals": "proposal_id",
    "bird_add_requests": "request_id",
    "detections": "detection_id",
    "manual_objects": "manual_object_id",
    "images": "filename",
}


def settings_for_plan(settings: dict[str, Any]) -> DeletionSettings:
    return DeletionSettings(
        old_bird_photos=settings.get("OLD_BIRD_PHOTOS", "keep_previews")
    )


def activation_date(settings: dict[str, Any]) -> datetime | None:
    if settings.get("AUTOMATIC_DELETION_ENABLED") is not True:
        return None
    return _time(settings.get("AUTOMATIC_DELETION_ENABLED_AT"))


def _settings_from_config() -> dict[str, Any]:
    from config import get_config

    return load_settings_yaml(str(get_config()["OUTPUT_DIR"]))


def preview(
    *,
    settings: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    from config import get_config
    from utils.db import closing_connection

    paths = PathManager(str(get_config()["OUTPUT_DIR"]))
    with closing_connection() as conn:
        settings = settings if settings is not None else _settings_from_config()
        conn.execute("BEGIN")
        if progress_callback:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            progress_callback(dict(images_total=total))
        result = deletion_preview(
            conn,
            settings_for_plan(settings),
            datetime.now(UTC),
            paths=paths,
            automatic_deletion_enabled_at=activation_date(settings),
            progress_callback=progress_callback,
        )
        result["migration_ready"] = human_answers_migrated(conn)
    result["enabled"] = activation_date(settings) is not None
    result["enabled_at"] = (
        settings.get("AUTOMATIC_DELETION_ENABLED_AT") if result["enabled"] else None
    )
    result["last_run"] = load_last_run_status()
    return result


def _checked_media(paths: PathManager, reference: str, category: str) -> Path:
    root = paths.originals_dir if category == "original" else paths.derivatives_dir
    candidate = paths.base_dir / reference
    checked = paths.contained_path(candidate, root)
    if (
        checked is None
        or checked == root.resolve()
        or not checked.is_relative_to(paths.base_dir.resolve())
        or checked != paths.base_dir.resolve() / reference
    ):
        raise ValueError("Unsafe media path")
    return checked


def _unlink_media(paths: PathManager, reference: str, category: str) -> int:
    _checked_media(paths, reference, category)
    descriptor = os.open(paths.base_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = Path(reference).parts
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        info = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Not a regular media file")
        os.unlink(parts[-1], dir_fd=descriptor)
        return info.st_size
    finally:
        os.close(descriptor)


def _regenerate_retained_previews(
    conn: sqlite3.Connection, paths: PathManager, photo: PhotoPlan, now: datetime
) -> None:
    original = next((f for f in photo.files if f.category == "original"), None)
    if original is None or not original.present or not original.decision.due(now):
        return
    retained = [
        f for f in photo.files if f.category != "original" and not f.decision.due(now)
    ]
    if not retained:
        return
    from core.crop_refresh import refresh_detection_thumbnail
    from core.gallery_core import regenerate_derivative
    from utils.db.detections import effective_bbox_projection_sql
    from utils.image_ops import generate_preview_thumbnail

    thumbnails = {
        row[0] or f"{Path(photo.filename).stem}_crop_1.webp"
        for row in conn.execute(
            "SELECT thumbnail_path FROM detections WHERE image_filename=?",
            (photo.filename,),
        )
    }
    for item in retained:
        if item.present:
            continue
        path = _checked_media(paths, item.reference, item.category)
        if item.category == "optimized":
            ok = regenerate_derivative(str(paths.base_dir), path.name, "optimized")
        elif path.name in thumbnails:
            bbox_sql, bbox_join = effective_bbox_projection_sql(conn)
            row = conn.execute(
                f"SELECT {', '.join(bbox_sql)} FROM detections d {bbox_join} WHERE d.image_filename=? AND (d.thumbnail_path=? OR d.thumbnail_path IS NULL) ORDER BY d.detection_id LIMIT 1",
                (photo.filename, path.name),
            ).fetchone()
            ok = row is not None and refresh_detection_thumbnail(
                original_path=_checked_media(paths, original.reference, "original"),
                thumbnail_path=path,
                bbox_norm=tuple(row),
            )
        elif item.category == "preview" and not thumbnails:
            ok = generate_preview_thumbnail(
                str(_checked_media(paths, original.reference, "original")), str(path)
            )
        else:
            continue
        if not ok or not path.is_file():
            raise OSError("Could not regenerate retained preview")


def _remove_rows(conn: sqlite3.Connection, photo: PhotoPlan, now: datetime) -> int:
    due = {(row.table, row.key) for row in photo.rows if row.decision.due(now)}
    deleted = 0
    for table, column in _ROW_KEYS.items():
        for row in sorted(
            photo.rows, key=lambda item: str(item.key).zfill(24), reverse=True
        ):
            if row.table != table or (table, row.key) not in due:
                continue
            if (
                table == "images"
                and conn.execute(
                    """SELECT 1 WHERE EXISTS(SELECT 1 FROM detections WHERE image_filename=?)
                OR EXISTS(SELECT 1 FROM manual_objects WHERE image_filename=?)
                OR EXISTS(SELECT 1 FROM label_subjects WHERE image_filename=?)""",
                    (photo.filename,) * 3,
                ).fetchone()
            ):
                continue
            deleted += conn.execute(
                f"DELETE FROM {table} WHERE {column}=?", (row.key,)
            ).rowcount
    return deleted


def execute_plan(
    conn: sqlite3.Connection,
    output_dir: str,
    settings: dict[str, Any],
    now: datetime | None = None,
    *,
    batch_size: int = 100,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    settings_provider: Callable[[], dict[str, Any]] | None = None,
    pressure_stage: int = 0,
) -> dict[str, Any]:
    if not human_answers_migrated(conn):
        raise RuntimeError("Human-answer migration is incomplete; deletion is disabled")
    if conn.in_transaction:
        raise ValueError("Deletion requires its own transactions")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    paths = PathManager(str(Path(output_dir).resolve()))
    result = dict(
        deleted=0,
        deleted_rows=0,
        freed_bytes=0,
        missing=0,
        errors=0,
        processed=0,
        total=0,
        stopped=False,
        disabled=False,
    )
    current_settings = settings_provider or (lambda: settings)
    if activation_date(current_settings()) is None:
        result["disabled"] = True
        return result
    result["total"] = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    for batch in image_batches(conn, batch_size):
        if stop_requested and stop_requested():
            result["stopped"] = True
            break
        current = current_settings()
        enabled_at = activation_date(current)
        if enabled_at is None:
            result["disabled"] = True
            break
        original_markers = {
            image["filename"]: image["original_present"] for image in batch
        }
        # Only actionable photos need a second inventory and a write lock. Every
        # deletion still uses a fresh plan under that lock, including new answers.
        candidates = {
            photo.filename
            for photo in plan_deletions(
                conn,
                settings_for_plan(current),
                now,
                paths=paths,
                filenames=list(original_markers),
                automatic_deletion_enabled_at=enabled_at,
                pressure_stage=pressure_stage,
            )
            if any(item.error for item in photo.files)
            or any(row.decision.due(now) for row in photo.rows)
            or any(
                item.decision.due(now)
                and (
                    item.present
                    or item.category == "original"
                    and original_markers[photo.filename]
                )
                for item in photo.files
            )
        }
        for image in batch:
            if stop_requested and stop_requested():
                result["stopped"] = True
                break
            if image["filename"] not in candidates:
                result["processed"] += 1
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = current_settings()
                enabled_at = activation_date(current)
                if enabled_at is None:
                    result["disabled"] = True
                    conn.rollback()
                    break
                if not human_answers_migrated(conn):
                    raise RuntimeError("Human-answer migration is incomplete")
                photo = next(
                    plan_deletions(
                        conn,
                        settings_for_plan(current),
                        now,
                        paths=paths,
                        filenames=[image["filename"]],
                        automatic_deletion_enabled_at=enabled_at,
                        pressure_stage=pressure_stage,
                    ),
                    None,
                )
                if photo is None:
                    conn.rollback()
                    continue
                if any(item.error for item in photo.files):
                    raise ValueError(
                        "Photo contains an unsafe or unavailable media path"
                    )
                _regenerate_retained_previews(conn, paths, photo, now)
                # The write lock keeps answers, favorites, trash and settings stable
                # from this fresh plan through file removal and row cleanup.
                for item in photo.files:
                    if not item.decision.due(now):
                        continue
                    try:
                        size = _unlink_media(paths, item.reference, item.category)
                        result["deleted"] += 1
                        result["freed_bytes"] += size
                    except FileNotFoundError:
                        result["missing"] += 1
                    if item.category == "original":
                        conn.execute(
                            "UPDATE images SET original_present=0, original_deleted_at=COALESCE(original_deleted_at,?) WHERE filename=?",
                            (now.isoformat(), photo.filename),
                        )
                deleted_rows = _remove_rows(conn, photo, now)
                conn.commit()
                result["deleted_rows"] += deleted_rows
            except (OSError, ValueError, sqlite3.Error) as exc:
                logger.warning(
                    "Photo cleanup failed for %s: %s", image["filename"], exc
                )
                conn.rollback()
                result["errors"] += 1
            except Exception:
                conn.rollback()
                raise
            result["processed"] += 1
        if progress_callback:
            progress_callback(dict(result))
        if result["stopped"] or result["disabled"]:
            break
    return result


def reclaim_space(
    conn: sqlite3.Connection,
    output_dir: str,
    settings: dict[str, Any],
    *,
    free_bytes: Callable[[], int],
    target_bytes: int,
    now: datetime | None = None,
    settings_provider: Callable[[], dict[str, Any]] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(deleted=0, deleted_rows=0, freed_bytes=0, errors=0)
    result["stages"] = []
    for stage in (1, 2, 3):
        current = settings_provider() if settings_provider else settings
        if (
            stage == 3
            and current.get("OLD_BIRD_PHOTOS", "keep_previews") != "delete_all"
        ):
            break
        if free_bytes() >= target_bytes or stop_requested and stop_requested():
            break
        step = execute_plan(
            conn,
            output_dir,
            settings,
            now,
            pressure_stage=stage,
            settings_provider=settings_provider,
            stop_requested=lambda: (
                free_bytes() >= target_bytes
                or bool(stop_requested and stop_requested())
            ),
        )
        result["stages"].append({"stage": stage, **step})
        for key in ("deleted", "deleted_rows", "freed_bytes", "errors"):
            result[key] += step[key]
        if step["disabled"]:
            break
    result["free_bytes"] = free_bytes()
    return result


def run(
    *,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from config import get_config
    from utils.db import closing_connection

    with closing_connection() as conn:
        return execute_plan(
            conn,
            str(get_config()["OUTPUT_DIR"]),
            _settings_from_config(),
            stop_requested=stop_requested,
            progress_callback=progress_callback,
            settings_provider=_settings_from_config,
        )


def load_last_run_status() -> dict[str, Any]:
    """Load the last completed deletion result from the background-job record."""
    from core.nightly_job_state import load_status

    return load_status("retention")


def is_original_retention_deleted(filename: str) -> bool:
    """True iff the image row exists and its original was removed by retention.

    Reflects the permanent ``original_present = 0`` marker, independent of
    whether retention is currently enabled (enable -> delete -> disable must
    still report 410). The serving layer keeps this off the hot path by only
    calling it when the original file is actually missing on disk.
    """
    from utils.db import closing_connection

    with closing_connection() as conn:
        row = conn.execute(
            "SELECT original_present FROM images WHERE filename = ?",
            (filename,),
        ).fetchone()
    return row is not None and not row["original_present"]
