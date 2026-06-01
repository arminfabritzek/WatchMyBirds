"""Nightly crop sharpness scoring job.

Reads every detection with a non-null ``thumbnail_path`` and a null
``sharpness_score``, opens the crop file, computes
``laplacian_sharpness`` + ``crop_brightness`` from
``utils.image_ops``, and writes both values back to the row.

Purely additive aux signal. Never filters, never gates, never
changes export. Crops without a resolvable file are logged once and
skipped — the column stays NULL so the next run can pick them up if
the file reappears (e.g. after restore).

Scope semantics
---------------
- ``today``: scan rows ``created_at >= today_local_midnight_utc``.
- ``backlog``: scan rows ``sharpness_score IS NULL`` regardless of
  age. Used by the backfill CLI.
- The job picks ``backlog`` if invoked from the daily loop too — the
  daily-loop nightly run is the steady-state catch-up after a day
  of new detections. There is no separate "yesterday only" mode;
  one query handles both new and old.

Lifecycle
---------
- Batch size: ``SHARPNESS_BATCH_SIZE`` (default 100). Between
  batches, the job checks ``stop_event.is_set()`` and returns rc=0
  if stopped. Each batch is one transaction.
- Errors per crop are logged at WARNING, the row is skipped, the
  column stays NULL. A crashing job releases its lock via the hub
  worker's finally clause.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import cv2

from utils.image_ops import crop_brightness, laplacian_sharpness
from web.services.nightly_job_hub import JobBase, update_progress

logger = logging.getLogger(__name__)


class SharpnessJob(JobBase):
    @property
    def name(self) -> str:
        return "sharpness"

    @property
    def display_name(self) -> str:
        return "Crop Sharpness Scoring"

    def run(self, stop_event: threading.Event, reason: str) -> int:
        """Scan and score every unscored detection.

        Reads ``WMB_DB_PATH`` and ``WMB_CROPS_ROOT`` from the
        environment (with the same defaults the aesthetic-tagger
        uses). Each batch is one transaction; a stop in the middle
        leaves partial progress on disk and the rest still NULL.
        """
        db_path = _resolve_db_path()
        crops_root = _resolve_crops_root()
        batch_size = int(os.environ.get("WMB_SHARPNESS_BATCH_SIZE", 100))

        if not db_path.exists():
            logger.error("SharpnessJob: DB not found: %s", db_path)
            return 1

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=15000;")
        try:
            return self._run_inner(
                conn, crops_root, batch_size, stop_event, reason
            )
        finally:
            conn.close()

    def _run_inner(
        self,
        conn: sqlite3.Connection,
        crops_root: Path,
        batch_size: int,
        stop_event: threading.Event,
        reason: str,
    ) -> int:
        started = time.monotonic()
        # Count up-front for the UI progress meter.
        # Schema note: `detections.image_filename` is the FK to
        # `images.filename` (text PK), not an integer image_id.
        total = conn.execute(
            """
            SELECT COUNT(*)
              FROM detections d
              JOIN images i ON i.filename = d.image_filename
             WHERE d.sharpness_score IS NULL
               AND d.thumbnail_path IS NOT NULL
               AND d.status = 'active'
            """
        ).fetchone()[0]
        logger.info(
            "SharpnessJob: %d crops to score (reason=%s, batch=%d)",
            total,
            reason,
            batch_size,
        )

        update_progress(
            self.name,
            {"total": total, "done": 0, "skipped": 0, "errored": 0},
        )

        done = 0
        skipped = 0
        errored = 0
        # Cursor pagination via detection_id. We page DESCENDING, so
        # `cursor` is the *largest* detection_id we have NOT yet
        # processed (i.e. the next batch's upper bound, exclusive).
        # Starting at None means "no upper bound yet". Without this,
        # detections whose crop file is missing stay NULL forever and
        # the LIMIT query would return them on every iteration —
        # producing an infinite loop.
        cursor: int | None = None

        while True:
            if stop_event.is_set():
                logger.info(
                    "SharpnessJob: stop requested after %d/%d crops",
                    done,
                    total,
                )
                break

            if cursor is None:
                rows = conn.execute(
                    """
                    SELECT d.detection_id, d.thumbnail_path, d.image_filename
                      FROM detections d
                     WHERE d.sharpness_score IS NULL
                       AND d.thumbnail_path IS NOT NULL
                       AND d.status = 'active'
                     ORDER BY d.detection_id DESC
                     LIMIT ?
                    """,
                    (batch_size,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT d.detection_id, d.thumbnail_path, d.image_filename
                      FROM detections d
                     WHERE d.sharpness_score IS NULL
                       AND d.thumbnail_path IS NOT NULL
                       AND d.status = 'active'
                       AND d.detection_id < ?
                     ORDER BY d.detection_id DESC
                     LIMIT ?
                    """,
                    (cursor, batch_size),
                ).fetchall()
            if not rows:
                break
            # Advance the cursor to the smallest detection_id in this
            # batch, regardless of whether each row was scored or
            # skipped. Skipped rows (missing file) thus don't trap
            # the loop.
            cursor = min(int(r[0]) for r in rows)

            batch_updates: list[tuple[float, float, int]] = []
            for detection_id, thumb_name, image_filename in rows:
                if stop_event.is_set():
                    break
                crop_path = _resolve_crop_path(
                    crops_root, thumb_name, image_filename
                )
                if crop_path is None or not crop_path.exists():
                    skipped += 1
                    continue
                try:
                    img = cv2.imread(str(crop_path))
                    if img is None:
                        skipped += 1
                        continue
                    s = laplacian_sharpness(img)
                    b = crop_brightness(img)
                    batch_updates.append((s, b, int(detection_id)))
                    done += 1
                except Exception:
                    logger.exception(
                        "SharpnessJob: failed to score detection %d "
                        "(crop=%s)",
                        detection_id,
                        crop_path,
                    )
                    errored += 1
                    continue

            if batch_updates:
                conn.executemany(
                    "UPDATE detections SET sharpness_score = ?, "
                    "crop_brightness = ? WHERE detection_id = ?",
                    batch_updates,
                )
                conn.commit()

            update_progress(
                self.name,
                {
                    "total": total,
                    "done": done,
                    "skipped": skipped,
                    "errored": errored,
                    "elapsed_s": round(time.monotonic() - started, 1),
                },
            )

            if not batch_updates and not rows:
                break  # nothing left

        # After scoring, recompute the station-adaptive gallery quality
        # floor from the FULL current distribution. Runs even if this
        # batch scored nothing new (a config change to the percentile
        # should still re-apply against existing scores). A stop request
        # skips it — the next run recomputes from scratch anyway.
        if not stop_event.is_set():
            self._recompute_gallery_eligibility(conn)

        elapsed = time.monotonic() - started
        logger.info(
            "SharpnessJob: finished — done=%d skipped=%d errored=%d "
            "elapsed=%.1fs",
            done,
            skipped,
            errored,
            elapsed,
        )
        return 0

    def _recompute_gallery_eligibility(self, conn: sqlite3.Connection) -> None:
        """Set ``quality_gallery_ok`` from this station's own sharpness
        distribution. Station-adaptive: the cut is a percentile of the
        local scores, never a fixed pixel threshold.

        Rules (mirrors the plan's scope guarantees):
        - Only scored, active crops participate. NULL-score rows keep
          ``quality_gallery_ok`` NULL (the reader's COALESCE shows them).
        - Below ``GALLERY_QUALITY_MIN_SCORED`` scored crops, the cut is a
          no-op: everything scored is flagged ``1`` (show). A tiny or
          fresh station never hides birds on a thin distribution.
        - ``GALLERY_QUALITY_BOTTOM_PCT == 0`` disables the floor — all
          scored crops become ``1``.
        - ``is_gallery_eligible`` (the aesthetic-tagger AI-pick axis) is
          never read or written here. The two axes stay orthogonal.
        """
        bottom_pct = _gallery_quality_bottom_pct()
        min_scored = _gallery_quality_min_scored()

        scored = conn.execute(
            """
            SELECT COUNT(*)
              FROM detections
             WHERE sharpness_score IS NOT NULL
               AND status = 'active'
            """
        ).fetchone()[0]

        if bottom_pct <= 0 or scored < min_scored:
            # No cut: flag every scored crop as visible. Leaves NULL-score
            # rows untouched (they stay NULL → shown via COALESCE).
            conn.execute(
                """
                UPDATE detections
                   SET quality_gallery_ok = 1
                 WHERE sharpness_score IS NOT NULL
                   AND status = 'active'
                """
            )
            conn.commit()
            logger.info(
                "SharpnessJob: gallery quality floor skipped "
                "(scored=%d, min=%d, bottom_pct=%d) — all %d scored "
                "crops flagged visible",
                scored,
                min_scored,
                bottom_pct,
                scored,
            )
            return

        # Station-relative cutoff: the sharpness_score at the bottom
        # percentile. Pulled into Python so the UPDATE is a single plain
        # comparison (SQLite's UPDATE-FROM-CTE is awkward across versions).
        offset = int(scored * bottom_pct / 100)
        cutoff_row = conn.execute(
            """
            SELECT sharpness_score
              FROM detections
             WHERE sharpness_score IS NOT NULL
               AND status = 'active'
             ORDER BY sharpness_score ASC
             LIMIT 1 OFFSET ?
            """,
            (offset,),
        ).fetchone()
        if cutoff_row is None:
            return
        cutoff = float(cutoff_row[0])

        conn.execute(
            """
            UPDATE detections
               SET quality_gallery_ok = CASE
                     WHEN sharpness_score < ? THEN 0 ELSE 1 END
             WHERE sharpness_score IS NOT NULL
               AND status = 'active'
            """,
            (cutoff,),
        )
        conn.commit()

        hidden = conn.execute(
            """
            SELECT COUNT(*)
              FROM detections
             WHERE quality_gallery_ok = 0
               AND status = 'active'
            """
        ).fetchone()[0]
        logger.info(
            "SharpnessJob: gallery quality floor applied — bottom %d%% "
            "(cutoff sharpness=%.1f over %d scored crops) → %d hidden",
            bottom_pct,
            cutoff,
            scored,
            hidden,
        )


# ---------------------------------------------------------------------------
# Path helpers (mirrored from aesthetic_tag_nightly so the hub job does not
# depend on importing a script file).
# ---------------------------------------------------------------------------


def _gallery_quality_bottom_pct() -> int:
    """Percent of the station's own sharpness distribution to hide from
    gallery thumbnails. 0 disables. Read live so a settings.yaml change
    takes effect on the next nightly run without a restart."""
    try:
        from config import get_config

        return int(get_config().get("GALLERY_QUALITY_BOTTOM_PCT", 15))
    except Exception:
        return 15


def _gallery_quality_min_scored() -> int:
    """Minimum scored-crop count before the floor applies at all. Below
    this, every scored crop is flagged visible — a small/fresh station
    never hides birds on a thin distribution."""
    try:
        from config import get_config

        return int(get_config().get("GALLERY_QUALITY_MIN_SCORED", 200))
    except Exception:
        return 200


def _resolve_db_path() -> Path:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/opt/app/data/output"))
    return Path(os.environ.get("WMB_DB_PATH", str(output_dir / "images.db")))


def _resolve_crops_root() -> Path:
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/opt/app/data/output"))
    return Path(
        os.environ.get(
            "WMB_CROPS_ROOT", str(output_dir / "derivatives" / "thumbs")
        )
    )


def _resolve_crop_path(
    crops_root: Path, thumbnail_path: str | None, image_filename: str | None
) -> Path | None:
    if not thumbnail_path or not image_filename:
        return None
    if len(image_filename) < 8:
        return None
    yyyymmdd = image_filename[:8]
    day_dir = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return crops_root / day_dir / thumbnail_path
