"""Use-case service for canonical human label answers."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from config import get_config
from core.bird_identity import add_or_correct_bird
from core.crop_refresh import refresh_detection_thumbnail
from core.human_label_core import (
    BBox,
    HumanAnswer,
    HumanLabelError,
    LabelProvenance,
    get_or_create_labeling_installation_id,
    object_training_readiness,
    record_human_answer,
    require_no_manual_birds,
    retract_answer,
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
        source_ref=source_ref or "answer:" + uuid4().hex,
    )
    facts = record_human_answer(conn, answer, provenance)
    return facts


def _ui_provenance(*, source_ref: str | None, app_version: str) -> LabelProvenance:
    cfg = get_config()
    return LabelProvenance(
        installation_id=get_or_create_labeling_installation_id(str(cfg["OUTPUT_DIR"])),
        app_version=_app_version(app_version),
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref=source_ref or "answer:" + uuid4().hex,
    )


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


def full_image_absence_blocker(conn: sqlite3.Connection, filename: str) -> str | None:
    try:
        require_no_manual_birds(conn, filename)
    except HumanLabelError as exc:
        return str(exc)
    return None


def apply_answer_request(
    conn: sqlite3.Connection, data: dict, *, app_version: str = ""
) -> dict:
    """Validate the whole selection and write one atomic human action."""
    conn.execute("BEGIN IMMEDIATE")
    ids = data.get("detection_ids")
    if ids is None and data.get("detection_id") is not None:
        ids = [data["detection_id"]]
    filename = data.get("filename")
    targets: list[tuple[str, int | None]] = []
    if ids is not None:
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 1000
            or any(type(did) is not int or did <= 0 for did in ids)
            or len(set(ids)) != len(ids)
        ):
            raise HumanLabelError(
                "detection_ids must contain 1 to 1000 unique object IDs"
            )
        for did in ids:
            row = conn.execute(
                "SELECT image_filename FROM detections WHERE detection_id=?", (did,)
            ).fetchone()
            if row is None or (filename is not None and filename != row[0]):
                raise HumanLabelError("detection does not belong to image")
            targets.append((row[0], did))
    elif isinstance(filename, str) and filename.strip():
        filename = filename.strip()
        if not conn.execute(
            "SELECT 1 FROM images WHERE filename=?", (filename,)
        ).fetchone():
            raise HumanLabelError("image does not exist")
        targets.append((filename, None))
    else:
        raise HumanLabelError("answer requires a filename or detection_ids")
    if data.get("review_only"):
        from core.station_observations import validate_review_targets

        validate_review_targets(conn, [did for _, did in targets if did is not None])
    for key in ("retract_bbox_quality", "detector_miss", "review_only"):
        if key in data and not isinstance(data[key], bool):
            raise HumanLabelError("answer flags must be boolean")
    for key in (
        "species_identity",
        "object_bird_presence",
        "image_bird_presence",
        "bbox_quality",
        "retract",
    ):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise HumanLabelError("answer values must be text")
    bbox_data = data.get("bbox_correction")
    try:
        bbox = BBox(**bbox_data).validated() if isinstance(bbox_data, dict) else None
    except HumanLabelError:
        raise
    except (TypeError, ValueError):
        raise HumanLabelError("bbox is invalid") from None
    if bbox_data is not None and bbox is None:
        raise HumanLabelError("bbox is invalid")
    if data.get("species_key") is not None and not isinstance(data["species_key"], str):
        raise HumanLabelError("answer species is invalid")
    provenance = _ui_provenance(
        source_ref="answer:" + uuid4().hex, app_version=app_version
    )
    fact_ids = []
    for image_filename, did in targets:
        if data.get("retract"):
            if any(
                data.get(key) is not None
                for key in (
                    "species_identity",
                    "object_bird_presence",
                    "image_bird_presence",
                    "bbox_quality",
                    "bbox_correction",
                    "detector_miss",
                )
            ):
                raise HumanLabelError("answer cannot be set and withdrawn together")
            fact_ids.extend(
                retract_answer(
                    conn,
                    image_filename=image_filename,
                    detection_id=did,
                    fact_type=data["retract"],
                    provenance=provenance,
                )
            )
        else:
            fact_ids.extend(
                record_human_answer(
                    conn,
                    HumanAnswer(
                        image_filename=image_filename,
                        detection_id=did,
                        image_bird_presence=data.get("image_bird_presence"),
                        object_bird_presence=data.get("object_bird_presence"),
                        bbox_quality=data.get("bbox_quality"),
                        bbox_correction=bbox,
                        species_identity=data.get("species_identity"),
                        species_key=data.get("species_key"),
                        detector_miss=data.get("detector_miss") is True,
                        retract_bbox_quality=data.get("retract_bbox_quality") is True,
                    ),
                    provenance,
                )
            )
    thumbnail_targets: set[tuple[str, int]] = set()
    for start in range(0, len(fact_ids), 400):
        batch = fact_ids[start : start + 400]
        placeholders = ",".join("?" for _ in batch)
        thumbnail_targets.update(
            (row[0], row[1])
            for row in conn.execute(
                f"SELECT s.image_filename,s.detection_id FROM human_label_facts f JOIN label_subjects s USING(subject_id) WHERE f.fact_id IN ({placeholders}) AND f.fact_type='bbox_correction' AND s.detection_id IS NOT NULL",
                batch,
            )
        )
    states = fetch_detection_review_states(
        conn, [did for _, did in targets if did is not None]
    )
    return {
        "fact_ids": fact_ids,
        "answered": len(targets),
        "targets": sorted(thumbnail_targets),
        "facts": fetch_current_facts(
            conn, image_filename=targets[0][0], detection_id=targets[0][1]
        ),
        "review_state": states.get(targets[0][1]),
        "review_states": states,
    }


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


def refresh_answer_thumbnails(
    conn: sqlite3.Connection, targets: list[tuple[str, int | None]], paths: object
) -> None:
    from core.crop_refresh import current_detection_bbox

    for _, did in targets:
        if did is None:
            continue
        row = current_detection_bbox(conn, did)
        if row:
            refresh_thumbnail_for_corrected_box(
                conn,
                detection_id=did,
                bbox=tuple(row),
                original_resolver=paths.get_original_path,
                thumb_resolver=lambda name: paths.get_derivative_path(name, "thumb"),
            )


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

    Image statements are training evidence; object facts determine review state.
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
        for row in object_rows:
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
        source_ref=source_ref or "answer:" + uuid4().hex,
    )
    fact_id = retract_species_identity_core(
        conn,
        image_filename=image_filename,
        detection_id=detection_id,
        provenance=provenance,
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
        source_ref=source_ref or "answer:" + uuid4().hex,
    )
    return retract_bbox_quality_core(
        conn,
        image_filename=image_filename,
        detection_id=detection_id,
        provenance=provenance,
    )
