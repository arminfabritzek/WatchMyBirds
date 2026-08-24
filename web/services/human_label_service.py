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
from core.station_report import record_station_event_review


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


def record_event_review(
    conn: sqlite3.Connection,
    *,
    event_key: str,
    detection_ids: list[int],
    anchor_detection_id: int,
    outcome: str,
    evidence_quality: str,
    event_start: str,
    event_end: str,
    candidate_species_key: str | None = None,
    species_key: str | None = None,
    source_ref: str | None = None,
    app_version: str = "",
) -> int:
    """Append the event-level diagnostic-evidence statement."""
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
    return record_station_event_review(
        conn,
        event_key=event_key,
        detection_ids=detection_ids,
        anchor_detection_id=anchor_detection_id,
        outcome=outcome,
        evidence_quality=evidence_quality,
        event_start=event_start,
        event_end=event_end,
        provenance=provenance,
        candidate_species_key=candidate_species_key,
        species_key=species_key,
    )

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


def fetch_detection_review_states(
    conn: sqlite3.Connection,
    detection_ids: list[int],
) -> dict[int, dict[str, object]]:
    """Return canonical human species-review state for offered detections.

    AI decision state and legacy review columns are intentionally absent from
    this projection. An explicit image-level no-bird answer is a reviewed
    negative for every offered detection on that image, but it remains an
    image-scoped fact in storage.
    """
    ids = sorted({int(detection_id) for detection_id in detection_ids if detection_id})
    states: dict[int, dict[str, object]] = {
        detection_id: {
            "state": "unreviewed",
            "reviewed": False,
            "species_key": None,
        }
        for detection_id in ids
    }
    if not ids:
        return states

    priority = {
        "unreviewed": 0,
        "reviewed_unknown": 1,
        "confirmed": 2,
        "corrected": 3,
        "reviewed_negative": 4,
    }

    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        object_rows = conn.execute(
            f"""
            SELECT current.detection_id, current.scope, current.fact_type,
                   current.assertion_state, current.answer_value,
                   current.species_key
            FROM current_human_label_facts current
            WHERE current.scope = 'object'
              AND current.detection_id IN ({placeholders})
              AND current.fact_type IN ('bird_presence', 'species_identity')
            """,
            chunk,
        ).fetchall()
        image_rows = conn.execute(
            f"""
            SELECT d.detection_id, current.scope, current.fact_type,
                   current.assertion_state, current.answer_value,
                   current.species_key
            FROM detections d
            JOIN current_human_label_facts current
              ON current.scope = 'image'
             AND current.image_filename = d.image_filename
             AND current.fact_type = 'bird_presence'
            WHERE d.detection_id IN ({placeholders})
            """,
            chunk,
        ).fetchall()

        for row in (*object_rows, *image_rows):
            if row["assertion_state"] != "asserted":
                continue
            next_state: str | None = None
            if (
                row["fact_type"] == "bird_presence"
                and row["answer_value"] == "absent"
            ):
                next_state = "reviewed_negative"
            elif row["fact_type"] == "species_identity":
                if row["answer_value"] in {"confirmed", "corrected"}:
                    next_state = str(row["answer_value"])
                elif row["answer_value"] == "unknown":
                    next_state = "reviewed_unknown"

            if next_state is None:
                continue
            detection_id = int(row["detection_id"])
            current_state = str(states[detection_id]["state"])
            if priority[next_state] < priority[current_state]:
                continue
            states[detection_id] = {
                "state": next_state,
                "reviewed": True,
                "species_key": (
                    row["species_key"]
                    if next_state in {"confirmed", "corrected"}
                    else None
                ),
            }

    return states


def summarize_detection_review_progress(
    review_states: dict[int, dict[str, object]],
    detection_ids: list[int],
) -> dict[str, int | bool]:
    """Count reviewed offered detections without implying wider scope."""
    ids = {int(detection_id) for detection_id in detection_ids if detection_id}
    reviewed = sum(bool(review_states.get(detection_id, {}).get("reviewed")) for detection_id in ids)
    total = len(ids)
    return {
        "reviewed": reviewed,
        "total": total,
        "complete": total > 0 and reviewed == total,
    }


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
