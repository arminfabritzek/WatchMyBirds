"""Use-case service for canonical human label answers."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

from config import get_config
from core.crop_refresh import refresh_detection_thumbnail
from core.human_label_core import (
    HumanAnswer,
    LabelProvenance,
    get_or_create_labeling_installation_id,
    object_training_readiness,
    record_human_answer,
)
from core.human_label_core import (
    retract_bbox_quality as retract_bbox_quality_core,
)


def _app_version(explicit_version: str = "") -> str:
    version = explicit_version.strip() or os.environ.get("APP_VERSION", "").strip()
    if version:
        return version
    version_file = Path(__file__).resolve().parents[2] / "APP_VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def record_answer(
    conn: sqlite3.Connection,
    answer: HumanAnswer,
    *,
    source_kind: str = "watchmybirds_ui",
    source_ref: str | None = None,
    context: str = "normal_correction",
    app_version: str = "",
) -> list[int]:
    """Record one human action with local, non-telemetry provenance."""
    cfg = get_config()
    provenance = LabelProvenance(
        installation_id=get_or_create_labeling_installation_id(
            str(cfg["OUTPUT_DIR"])
        ),
        app_version=_app_version(app_version),
        context=context,
        source_kind=source_kind,
        source_ref=source_ref,
    )
    return record_human_answer(conn, answer, provenance)

logger = logging.getLogger(__name__)


def refresh_thumbnail_for_corrected_box(
    conn: sqlite3.Connection,
    *,
    detection_id: int,
    bbox: tuple[float, float, float, float],
    original_resolver: Callable[[str], Path],
    thumb_resolver: Callable[[str], Path],
) -> bool:
    """Rewrite the detection crop so the person can see their own edit.

    Best-effort by design: the label fact is already committed, and a stale
    derivative must never turn a successful correction into an error.
    """
    row = conn.execute(
        """
        SELECT image_filename, thumbnail_path
        FROM detections WHERE detection_id = ?
        """,
        (detection_id,),
    ).fetchone()
    if row is None or not row["thumbnail_path"]:
        return False

    try:
        return refresh_detection_thumbnail(
            original_path=original_resolver(row["image_filename"]),
            thumbnail_path=thumb_resolver(row["thumbnail_path"]),
            bbox_norm=bbox,
        )
    except Exception:
        logger.warning(
            "crop refresh failed for detection %s", detection_id, exc_info=True
        )
        return False


def fetch_current_facts(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int | None = None,
) -> list[dict[str, object]]:
    """Return current independent facts for one image and optional object."""
    clauses = ["image_filename = ?"]
    params: list[object] = [image_filename]
    if detection_id is not None:
        clauses.append("(scope = 'image' OR detection_id = ?)")
        params.append(detection_id)
    rows = conn.execute(
        f"""
        SELECT *
        FROM current_human_label_facts
        WHERE {' AND '.join(clauses)}
        ORDER BY scope, fact_type
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def summarize_object_state(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int,
    facts: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Return shared readiness plus honest per-image object progress."""
    rows = conn.execute(
        """
        SELECT d.detection_id, COUNT(current.fact_id) AS fact_count
        FROM detections d
        LEFT JOIN label_subjects subject
          ON subject.scope = 'object'
         AND subject.detection_id = d.detection_id
        LEFT JOIN current_human_label_facts current
          ON current.subject_id = subject.subject_id
        WHERE d.image_filename = ?
        GROUP BY d.detection_id
        ORDER BY d.detection_id
        """,
        (image_filename,),
    ).fetchall()
    counts = {int(row["detection_id"]): int(row["fact_count"] or 0) for row in rows}
    answered = sum(fact_count > 0 for fact_count in counts.values())
    progress = {
        "total": len(counts),
        "answered": answered,
        "unanswered": len(counts) - answered,
        "active_detection_id": detection_id,
        "active_fact_count": counts.get(detection_id, 0),
    }
    return object_training_readiness(facts), progress


def retract_bbox_quality(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int,
    source_ref: str | None = None,
    app_version: str = "",
) -> int | None:
    """Retract one explicit bbox-quality answer through the canonical path."""
    cfg = get_config()
    provenance = LabelProvenance(
        installation_id=get_or_create_labeling_installation_id(
            str(cfg["OUTPUT_DIR"])
        ),
        app_version=_app_version(app_version),
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref=source_ref,
    )
    return retract_bbox_quality_core(
        conn,
        image_filename=image_filename,
        detection_id=detection_id,
        provenance=provenance,
    )
