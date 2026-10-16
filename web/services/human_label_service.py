"""Use-case service for canonical human label answers."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

from config import get_config
from core.bird_identity import add_or_correct_bird
from core.crop_refresh import refresh_detection_thumbnail
from core.human_label_core import (
    BBox,
    HumanAnswer,
    LabelProvenance,
    get_or_create_labeling_installation_id,
    object_training_readiness,
    record_human_answer,
    restore_rejected_detection,
    species_review_state,
)
from core.human_label_core import original_model_proposal as original_model_proposal
from core.human_label_core import (
    retract_bbox_quality as retract_bbox_quality_core,
)
from core.human_label_core import (
    retract_species_identity as retract_species_identity_core,
)
from core.manual_object_core import (
    ManualObjectDraft,
)
from core.manual_object_core import (
    retract_manual_object as retract_manual_object_core,
)
from core.manual_object_core import (
    update_manual_object as update_manual_object_core,
)
from core.review_state import refresh_review_image_visibility


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
        installation_id=get_or_create_labeling_installation_id(str(cfg["OUTPUT_DIR"])),
        app_version=_app_version(app_version),
        context=context,
        source_kind=source_kind,
        source_ref=source_ref,
    )
    facts = record_human_answer(conn, answer, provenance)
    if answer.image_bird_presence is None:
        refresh_review_image_visibility(
            conn, answer.image_filename, float(cfg["GALLERY_DISPLAY_THRESHOLD"])
        )
    return facts


def _ui_provenance(*, source_ref: str | None, app_version: str) -> LabelProvenance:
    cfg = get_config()
    return LabelProvenance(
        installation_id=get_or_create_labeling_installation_id(str(cfg["OUTPUT_DIR"])),
        app_version=_app_version(app_version),
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref=source_ref,
    )


def restore_detection(
    conn: sqlite3.Connection,
    *,
    detection_id: int,
    image_filename: str,
    app_version: str = "",
) -> list[int]:
    facts = restore_rejected_detection(
        conn,
        detection_id=detection_id,
        image_filename=image_filename,
        provenance=_ui_provenance(
            source_ref=f"image-correction:restore:{detection_id}",
            app_version=app_version,
        ),
    )
    refresh_review_image_visibility(
        conn, image_filename, float(get_config()["GALLERY_DISPLAY_THRESHOLD"])
    )
    return facts


def create_manual_object(
    conn: sqlite3.Connection,
    draft: ManualObjectDraft,
    *,
    original_path: Path,
    locale: str,
    app_version: str = "",
    target_key: str | None = None,
    match_token: str | None = None,
) -> tuple[dict[str, object], bool]:
    """Persist one missed bird without fabricating a detector proposal."""
    return add_or_correct_bird(
        conn,
        draft,
        _ui_provenance(
            source_ref=f"manual-object:create:{draft.image_filename}",
            app_version=app_version,
        ),
        original_path=original_path,
        locale=locale,
        gallery_threshold=float(get_config()["GALLERY_DISPLAY_THRESHOLD"]),
        target_key=target_key,
        match_token=match_token,
    )


def update_manual_object(
    conn: sqlite3.Connection,
    *,
    manual_object_id: int,
    image_filename: str,
    expected_revision: int,
    locale: str,
    app_version: str = "",
    bbox: BBox | None = None,
    species_supplied: bool = False,
    species_key: str | None = None,
    original_path: Path | None = None,
) -> tuple[dict[str, object], bool]:
    """Persist supplied manual-object axes with optimistic concurrency."""
    return update_manual_object_core(
        conn,
        manual_object_id=manual_object_id,
        image_filename=image_filename,
        expected_revision=expected_revision,
        provenance=_ui_provenance(
            source_ref=f"manual-object:update:{manual_object_id}",
            app_version=app_version,
        ),
        locale=locale,
        bbox=bbox,
        species_supplied=species_supplied,
        species_key=species_key,
        original_path=original_path,
    )


def retract_manual_object(
    conn: sqlite3.Connection,
    *,
    manual_object_id: int,
    image_filename: str,
    expected_revision: int,
    app_version: str = "",
) -> dict[str, object]:
    """Retract one manually added object without deleting its audit history."""
    return retract_manual_object_core(
        conn,
        manual_object_id=manual_object_id,
        image_filename=image_filename,
        expected_revision=expected_revision,
        provenance=_ui_provenance(
            source_ref=f"manual-object:retract:{manual_object_id}",
            app_version=app_version,
        ),
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
        WHERE {" AND ".join(clauses)}
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
            "bbox_verdict": None,
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
                   current.species_key, current.proposal_species_key
            FROM current_human_label_facts current
            WHERE current.scope = 'object'
              AND current.detection_id IN ({placeholders})
              AND current.fact_type IN ('bird_presence', 'species_identity', 'bbox_quality')
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
            if row["fact_type"] == "bbox_quality":
                states[int(row["detection_id"])]["bbox_verdict"] = row["answer_value"]
                continue
            next_state: str | None = None
            if row["fact_type"] == "bird_presence" and row["answer_value"] == "absent":
                next_state = "reviewed_negative"
            elif row["fact_type"] == "species_identity":
                if row["answer_value"] in {"confirmed", "corrected"}:
                    next_state = species_review_state(
                        str(row["answer_value"]),
                        row["species_key"],
                        row["proposal_species_key"],
                    )
                elif row["answer_value"] == "unknown":
                    next_state = "reviewed_unknown"

            if next_state is None:
                continue
            detection_id = int(row["detection_id"])
            current_state = str(states[detection_id]["state"])
            if priority[next_state] < priority[current_state]:
                continue
            states[detection_id] = {
                **states[detection_id],
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
    reviewed = sum(
        bool(review_states.get(detection_id, {}).get("reviewed"))
        for detection_id in ids
    )
    total = len(ids)
    return {
        "reviewed": reviewed,
        "total": total,
        "boxes_reviewed": sum(
            review_states.get(detection_id, {}).get("bbox_verdict")
            in {"suitable", "unsuitable"}
            for detection_id in ids
        ),
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


def retract_species_identity(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int,
    source_ref: str | None = None,
    app_version: str = "",
) -> int | None:
    """Withdraw one species answer through the canonical path."""
    cfg = get_config()
    provenance = LabelProvenance(
        installation_id=get_or_create_labeling_installation_id(str(cfg["OUTPUT_DIR"])),
        app_version=_app_version(app_version),
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref=source_ref,
    )
    fact_id = retract_species_identity_core(
        conn,
        image_filename=image_filename,
        detection_id=detection_id,
        provenance=provenance,
    )
    refresh_review_image_visibility(
        conn, image_filename, float(cfg["GALLERY_DISPLAY_THRESHOLD"])
    )
    return fact_id


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
        installation_id=get_or_create_labeling_installation_id(str(cfg["OUTPUT_DIR"])),
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


def record_review_answers(
    conn: sqlite3.Connection, detection_ids: list[int], action: str, species: str | None
) -> int:
    from core.station_observations import answer_review_selection

    return answer_review_selection(
        conn,
        detection_ids,
        action,
        species,
        _ui_provenance(source_ref="review-event-selection", app_version=""),
        float(get_config()["GALLERY_DISPLAY_THRESHOLD"]),
    )
