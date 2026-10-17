"""Protect originals waiting for training or held by an unconfirmed export."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from config import get_config
from core.training_export import original_resolver, waiting_revisions
from utils.path_manager import PathManager


def protection_sources_any(
    conn: sqlite3.Connection,
    filenames: list[str],
    *,
    media_exists: Callable[[str], bool] | None = None,
) -> dict[str, set[str]]:
    if media_exists is None:
        resolve = original_resolver(PathManager(str(get_config()["OUTPUT_DIR"])))
        media_exists = lambda name: resolve(name).is_file()
    relevant: dict[str, set[str]] = {}
    for row in waiting_revisions(conn, media_exists=media_exists, filenames=filenames):
        relevant.setdefault(str(row["image_filename"]), set()).add(
            "waiting_for_training"
        )
    batch_size = max(1, min(400, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)))
    for start in range(0, len(filenames), batch_size):
        batch = filenames[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        for row in conn.execute(
            f"""SELECT DISTINCT m.image_filename FROM labeling_export_members m
                JOIN labeling_exports e USING(export_id)
                WHERE e.received_at IS NULL AND m.image_filename IN ({placeholders})""",
            batch,
        ):
            relevant.setdefault(row["image_filename"], set()).add("unconfirmed_export")
    return relevant


def is_export_relevant_any(
    conn: sqlite3.Connection,
    filenames: list[str],
    *,
    media_exists: Callable[[str], bool] | None = None,
) -> set[str]:
    return set(protection_sources_any(conn, filenames, media_exists=media_exists))
