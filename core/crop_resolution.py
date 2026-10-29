"""Re-render stored bird crops that are smaller than the current crop size.

Crops are written once at capture time. When ``THUMBNAIL_SIZE`` grows, older
crops stay small until they are rendered again from their original. Only the
image header is read for crops that already match, so repeated runs stay cheap.
Crops whose original has expired keep their current size.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from core.crop_refresh import THUMBNAIL_SIZE, refresh_detection_thumbnail
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


def _crop_side(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            return min(image.size)
    except (OSError, ValueError):
        return None


def upgrade_crops(
    conn: sqlite3.Connection,
    paths: PathManager,
    *,
    size: int = THUMBNAIL_SIZE,
    stop_requested: Callable[[], bool] = lambda: False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    from utils.db.detections import effective_bbox_projection_sql

    if conn.in_transaction:
        raise ValueError("Crop upgrades require their own transactions")
    geometry, join = effective_bbox_projection_sql(conn)
    total = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE thumbnail_path IS NOT NULL"
    ).fetchone()[0]
    result: dict[str, Any] = {
        "done": 0,
        "upgraded": 0,
        "without_original": 0,
        "errors": 0,
        "total": total,
        "stopped": False,
    }
    last_id = 0
    while True:
        rows = conn.execute(
            f"""SELECT d.detection_id, d.image_filename, d.thumbnail_path,
                       {", ".join(geometry)}
                FROM detections d {join}
                WHERE d.thumbnail_path IS NOT NULL AND d.detection_id > ?
                ORDER BY d.detection_id LIMIT ?""",
            (last_id, batch_size),
        ).fetchall()
        if not rows:
            break
        for detection_id, image_filename, thumbnail, *bbox in rows:
            if stop_requested():
                result["stopped"] = True
                break
            last_id = detection_id
            result["done"] += 1
            thumb = paths.contained_path(
                paths.get_derivative_path(thumbnail, "thumb"), paths.thumbs_dir
            )
            if thumb is None or not thumb.is_file() or None in bbox:
                continue
            side = _crop_side(thumb)
            if side is None or side >= size:
                continue
            original = paths.contained_path(
                paths.get_original_path(image_filename), paths.originals_dir
            )
            if original is None or not original.is_file():
                result["without_original"] += 1
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix=".crop-upgrade-", dir=thumb.parent
                ) as directory:
                    prepared = Path(directory) / thumb.name
                    if not refresh_detection_thumbnail(
                        original_path=original,
                        thumbnail_path=prepared,
                        bbox_norm=tuple(bbox),
                        size=size,
                    ):
                        raise OSError("Could not render crop")
                    # Encoding stays outside the write lock; a newer correction
                    # or deletion must win before the derivative is published.
                    with conn:
                        conn.execute("BEGIN IMMEDIATE")
                        current = conn.execute(
                            f"""SELECT d.image_filename, d.thumbnail_path,
                                       {", ".join(geometry)}
                                FROM detections d {join}
                                WHERE d.detection_id=?""",
                            (detection_id,),
                        ).fetchone()
                        if (
                            current is None
                            or tuple(current) != (image_filename, thumbnail, *bbox)
                            or not thumb.is_file()
                        ):
                            continue
                        prepared.replace(thumb)
                        result["upgraded"] += 1
            except OSError:
                result["errors"] += 1
                logger.warning("crop upgrade failed for detection %s", detection_id)
        if progress_callback is not None:
            progress_callback(dict(result))
        if result["stopped"]:
            break
    return result


def run(
    *,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from config import get_config
    from utils.db import closing_connection

    paths = PathManager(str(get_config()["OUTPUT_DIR"]))
    with closing_connection() as conn:
        return upgrade_crops(
            conn,
            paths,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
        )
