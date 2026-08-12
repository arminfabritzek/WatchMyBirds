"""Canonical, append-only human label facts.

This core module owns semantic validation and transaction-neutral writes. It
does not commit: callers can combine one explicit human action with temporary
legacy compatibility projections in a single transaction.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime


class HumanLabelError(ValueError):
    """Raised when a requested human fact would violate label semantics."""


LABEL_SOURCE_KINDS_V1 = frozenset(
    {
        "watchmybirds_ui",
        "watchmybirds_bulk_moderation",
    }
)


@dataclass(frozen=True)
class BBox:
    """Normalized top-left x/y/width/height geometry."""

    x: float
    y: float
    w: float
    h: float

    def validated(self) -> BBox:
        values = (self.x, self.y, self.w, self.h)
        if not all(math.isfinite(value) for value in values):
            raise HumanLabelError("bbox coordinates must be finite")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise HumanLabelError("bbox origin must be within the frame")
        if not (0.0 < self.w <= 1.0 and 0.0 < self.h <= 1.0):
            raise HumanLabelError("bbox size must be positive and normalized")
        if self.x + self.w > 1.0 or self.y + self.h > 1.0:
            raise HumanLabelError("bbox must not extend beyond the frame")
        return self


@dataclass(frozen=True)
class LabelProvenance:
    """Local provenance captured on every human answer."""

    installation_id: str
    app_version: str
    context: str
    source_kind: str
    source_ref: str | None = None
    created_at: str | None = None

    def values(self) -> tuple[str, str, str, str, str | None, str]:
        installation_id = self.installation_id.strip()
        app_version = self.app_version.strip() or "unknown"
        source_kind = self.source_kind.strip()
        if not installation_id:
            raise HumanLabelError("installation_id is required")
        if self.context not in {"normal_correction", "targeted_training"}:
            raise HumanLabelError("invalid label context")
        if source_kind not in LABEL_SOURCE_KINDS_V1:
            raise HumanLabelError("invalid source_kind")
        created_at = self.created_at or datetime.now(UTC).isoformat()
        return (
            installation_id,
            app_version,
            self.context,
            source_kind,
            self.source_ref,
            created_at,
        )


@dataclass(frozen=True)
class HumanAnswer:
    """Independent facts explicitly asserted in one user action."""

    image_filename: str
    detection_id: int | None = None
    image_bird_presence: str | None = None
    object_bird_presence: str | None = None
    bbox_quality: str | None = None
    bbox_correction: BBox | None = None
    species_identity: str | None = None
    species_key: str | None = None
    detector_miss: bool = False


def object_training_readiness(
    facts: Iterable[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Derive independent OD/CLS readiness from current object facts.

    These reason codes are the shared semantic contract for the optional UI
    details and the canonical bundle builder. Callers must not infer readiness
    from legacy review columns.
    """
    by_type = {
        str(fact.get("fact_type")): fact
        for fact in facts
        if fact.get("scope") == "object"
    }
    bird_fact = by_type.get("bird_presence")
    bird_value = bird_fact.get("answer_value") if bird_fact else None

    od_reasons: list[str] = []
    cls_reasons: list[str] = []
    if bird_value is None:
        od_reasons.append("object_bird_presence_unknown")
        cls_reasons.append("object_bird_presence_unknown")
    elif bird_value != "present":
        od_reasons.append("object_bird_absent")
        cls_reasons.append("object_bird_absent")

    bbox_fact = by_type.get("bbox_quality")
    bbox_value = bbox_fact.get("answer_value") if bbox_fact else None
    if bbox_value is None:
        od_reasons.append("bbox_quality_unknown")
    elif bbox_value != "suitable":
        od_reasons.append("bbox_unsuitable")

    species_fact = by_type.get("species_identity")
    species_value = species_fact.get("answer_value") if species_fact else None
    species_key = species_fact.get("species_key") if species_fact else None
    if species_value is None:
        cls_reasons.append("species_identity_unknown")
    elif species_value == "unknown":
        cls_reasons.append("species_unknown")
    elif species_value == "wrong":
        cls_reasons.append("species_wrong")
    elif species_value not in {"confirmed", "corrected"}:
        cls_reasons.append("species_identity_unsupported")
    elif not str(species_key or "").strip():
        cls_reasons.append("species_key_missing")

    return {
        "od": {"ready": not od_reasons, "reasons": od_reasons},
        "cls": {"ready": not cls_reasons, "reasons": cls_reasons},
    }


def get_or_create_labeling_installation_id(output_dir: str) -> str:
    """Return the stable local labeling ID without touching telemetry state."""
    from utils.settings import load_settings_yaml, save_settings_yaml

    settings = load_settings_yaml(output_dir)
    existing = str(settings.get("labeling_installation_id") or "").strip().lower()
    if len(existing) == 32 and all(char in "0123456789abcdef" for char in existing):
        return existing

    installation_id = uuid.uuid4().hex
    settings["labeling_installation_id"] = installation_id
    save_settings_yaml(settings, output_dir)
    return installation_id


def record_human_answer(
    conn: sqlite3.Connection,
    answer: HumanAnswer,
    provenance: LabelProvenance,
) -> list[int]:
    """Persist one explicit action and its legacy projections atomically.

    The caller owns commit/rollback. Each non-``None`` answer is stored as an
    independent fact; omitted axes stay unknown.
    """
    filename = answer.image_filename.strip()
    if not filename:
        raise HumanLabelError("image filename is required")

    fact_ids: list[int] = []
    image_subject_id: int | None = None
    object_subject_id: int | None = None
    action_at = provenance.values()[-1]

    if answer.image_bird_presence is not None or answer.detector_miss:
        image_subject_id = ensure_image_subject(conn, filename)

    object_axes_present = any(
        value is not None
        for value in (
            answer.object_bird_presence,
            answer.bbox_quality,
            answer.bbox_correction,
            answer.species_identity,
        )
    )
    if object_axes_present:
        if answer.detection_id is None:
            raise HumanLabelError("object facts require detection_id")
        object_subject_id = ensure_object_subject(conn, answer.detection_id)
        subject_filename = conn.execute(
            "SELECT image_filename FROM label_subjects WHERE subject_id = ?",
            (object_subject_id,),
        ).fetchone()[0]
        if subject_filename != filename:
            raise HumanLabelError("detection does not belong to image")

    if answer.image_bird_presence is not None:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(image_subject_id),
                fact_type="bird_presence",
                answer_value=answer.image_bird_presence,
                provenance=provenance,
            )
        )
        review_status = (
            "confirmed_bird"
            if answer.image_bird_presence == "present"
            else "no_bird"
        )
        conn.execute(
            """
            UPDATE images
            SET review_status = ?, review_updated_at = ?
            WHERE filename = ?
            """,
            (review_status, action_at, filename),
        )

    if answer.detector_miss:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(image_subject_id),
                fact_type="detector_miss",
                answer_value="reported",
                provenance=provenance,
            )
        )

    if answer.object_bird_presence is not None:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(object_subject_id),
                fact_type="bird_presence",
                answer_value=answer.object_bird_presence,
                provenance=provenance,
            )
        )
        status = "active" if answer.object_bird_presence == "present" else "rejected"
        conn.execute(
            "UPDATE detections SET status = ? WHERE detection_id = ?",
            (status, answer.detection_id),
        )
        conn.execute(
            "UPDATE classifications SET status = ? WHERE detection_id = ?",
            (status, answer.detection_id),
        )

    if answer.bbox_quality is not None:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(object_subject_id),
                fact_type="bbox_quality",
                answer_value=answer.bbox_quality,
                provenance=provenance,
            )
        )
        legacy_bbox_state = (
            "correct" if answer.bbox_quality == "suitable" else "wrong"
        )
        conn.execute(
            """
            UPDATE detections
            SET manual_bbox_review = ?, bbox_reviewed_at = ?
            WHERE detection_id = ?
            """,
            (legacy_bbox_state, action_at, answer.detection_id),
        )

    if answer.bbox_correction is not None:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(object_subject_id),
                fact_type="bbox_correction",
                answer_value="provided",
                bbox=answer.bbox_correction,
                provenance=provenance,
            )
        )

    if answer.species_identity is not None:
        fact_ids.append(
            append_fact(
                conn,
                subject_id=_required_subject(object_subject_id),
                fact_type="species_identity",
                answer_value=answer.species_identity,
                species_key=answer.species_key,
                provenance=provenance,
            )
        )
        if answer.species_identity in {"confirmed", "corrected"}:
            conn.execute(
                """
                UPDATE detections
                SET manual_species_override = ?,
                    species_source = 'manual',
                    species_updated_at = ?,
                    decision_state = 'confirmed',
                    decision_level = 'species'
                WHERE detection_id = ?
                """,
                (answer.species_key, action_at, answer.detection_id),
            )
        else:
            conn.execute(
                """
                UPDATE detections
                SET manual_species_override = NULL,
                    species_source = ?,
                    species_updated_at = ?,
                    decision_state = 'unknown',
                    decision_level = NULL
                WHERE detection_id = ?
                """,
                (
                    f"manual_{answer.species_identity}",
                    action_at,
                    answer.detection_id,
                ),
            )

    if not fact_ids:
        raise HumanLabelError("answer contains no facts")
    return fact_ids


def retract_bbox_quality(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int,
    provenance: LabelProvenance,
) -> int | None:
    """Retract a canonical bbox-quality answer and clear its legacy projection.

    Legacy-only rows have no fact to retract. They are cleared without
    synthesizing history from ambiguous pre-migration state.
    """
    row = conn.execute(
        """
        SELECT d.image_filename, s.subject_id
        FROM detections d
        LEFT JOIN label_subjects s
          ON s.scope = 'object' AND s.detection_id = d.detection_id
        WHERE d.detection_id = ?
        """,
        (int(detection_id),),
    ).fetchone()
    if row is None:
        raise HumanLabelError("detection does not exist")
    if str(row[0]) != image_filename.strip():
        raise HumanLabelError("detection does not belong to image")

    fact_id: int | None = None
    if row[1] is not None:
        current = conn.execute(
            """
            SELECT fact_id
            FROM current_human_label_facts
            WHERE subject_id = ? AND fact_type = 'bbox_quality'
            """,
            (int(row[1]),),
        ).fetchone()
        if current is not None:
            fact_id = append_fact(
                conn,
                subject_id=int(row[1]),
                fact_type="bbox_quality",
                answer_value=None,
                assertion_state="retracted",
                provenance=provenance,
            )

    conn.execute(
        """
        UPDATE detections
        SET manual_bbox_review = NULL, bbox_reviewed_at = NULL
        WHERE detection_id = ?
        """,
        (int(detection_id),),
    )
    return fact_id


def _required_subject(subject_id: int | None) -> int:
    if subject_id is None:  # pragma: no cover - guarded by answer routing
        raise HumanLabelError("required label subject is missing")
    return subject_id


def ensure_image_subject(conn: sqlite3.Connection, image_filename: str) -> int:
    """Return the immutable image subject, creating it when first answered."""
    filename = image_filename.strip()
    if not filename:
        raise HumanLabelError("image filename is required")

    image = conn.execute(
        """
        SELECT filename, detector_model_id, classifier_model_id
        FROM images
        WHERE filename = ?
        """,
        (filename,),
    ).fetchone()
    if image is None:
        raise HumanLabelError("image does not exist")

    conn.execute(
        """
        INSERT OR IGNORE INTO label_subjects (
            scope, image_filename, detector_model_version,
            classifier_model_version, created_at
        ) VALUES ('image', ?, ?, ?, ?)
        """,
        (
            filename,
            image[1],
            image[2],
            datetime.now(UTC).isoformat(),
        ),
    )
    row = conn.execute(
        """
        SELECT subject_id
        FROM label_subjects
        WHERE scope = 'image' AND image_filename = ?
        """,
        (filename,),
    ).fetchone()
    if row is None:  # pragma: no cover - guarded by schema constraints
        raise HumanLabelError("could not create image label subject")
    return int(row[0])


def ensure_object_subject(conn: sqlite3.Connection, detection_id: int) -> int:
    """Return a proposal snapshot for an existing detection."""
    detection = conn.execute(
        """
        SELECT
            d.detection_id,
            d.image_filename,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            COALESCE(
                NULLIF(d.raw_species_name, ''),
                (
                    SELECT NULLIF(c.cls_class_name, '')
                    FROM classifications c
                    WHERE c.detection_id = d.detection_id
                    ORDER BY c.rank, c.classification_id
                    LIMIT 1
                ),
                NULLIF(d.od_class_name, '')
            ) AS proposal_species_key,
            COALESCE(
                NULLIF(d.detector_model_version, ''),
                NULLIF(d.od_model_id, ''),
                NULLIF(i.detector_model_id, '')
            ) AS detector_model_version,
            COALESCE(
                NULLIF(d.classifier_model_version, ''),
                NULLIF(i.classifier_model_id, '')
            ) AS classifier_model_version
        FROM detections d
        JOIN images i ON i.filename = d.image_filename
        WHERE d.detection_id = ?
        """,
        (int(detection_id),),
    ).fetchone()
    if detection is None:
        raise HumanLabelError("detection does not exist")

    bbox_values = detection[2:6]
    if any(value is None for value in bbox_values):
        raise HumanLabelError("object subject requires a proposal bbox")
    bbox = BBox(*(float(value) for value in bbox_values)).validated()

    conn.execute(
        """
        INSERT OR IGNORE INTO label_subjects (
            scope,
            image_filename,
            detection_id,
            proposal_bbox_x,
            proposal_bbox_y,
            proposal_bbox_w,
            proposal_bbox_h,
            proposal_coordinate_space,
            proposal_species_key,
            detector_model_version,
            classifier_model_version,
            created_at
        ) VALUES ('object', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            detection[1],
            detection[0],
            bbox.x,
            bbox.y,
            bbox.w,
            bbox.h,
            "frame_fraction_xywh_v1",
            detection[6],
            detection[7],
            detection[8],
            datetime.now(UTC).isoformat(),
        ),
    )
    row = conn.execute(
        """
        SELECT subject_id
        FROM label_subjects
        WHERE scope = 'object' AND detection_id = ?
        """,
        (int(detection_id),),
    ).fetchone()
    if row is None:  # pragma: no cover - guarded by schema constraints
        raise HumanLabelError("could not create object label subject")
    return int(row[0])


def append_fact(
    conn: sqlite3.Connection,
    *,
    subject_id: int,
    fact_type: str,
    answer_value: str | None,
    provenance: LabelProvenance,
    assertion_state: str = "asserted",
    species_key: str | None = None,
    bbox: BBox | None = None,
) -> int:
    """Append and supersede one fact without committing the transaction."""
    subject = conn.execute(
        """
        SELECT subject_id, scope, image_filename, detection_id
        FROM label_subjects
        WHERE subject_id = ?
        """,
        (int(subject_id),),
    ).fetchone()
    if subject is None:
        raise HumanLabelError("label subject does not exist")

    if assertion_state not in {"asserted", "retracted"}:
        raise HumanLabelError("invalid assertion state")
    if assertion_state == "retracted":
        if answer_value is not None or species_key is not None or bbox is not None:
            raise HumanLabelError("retraction must not carry an answer payload")
    elif answer_value is None:
        raise HumanLabelError("asserted fact requires an answer")

    if bbox is not None:
        bbox = bbox.validated()
    _reject_presence_conflict(
        conn,
        scope=str(subject[1]),
        image_filename=str(subject[2]),
        fact_type=fact_type,
        answer_value=answer_value,
        assertion_state=assertion_state,
    )

    head = conn.execute(
        """
        SELECT current.fact_id
        FROM human_label_facts current
        WHERE current.subject_id = ?
          AND current.fact_type = ?
          AND NOT EXISTS (
              SELECT 1
              FROM human_label_facts successor
              WHERE successor.supersedes_fact_id = current.fact_id
          )
        """,
        (int(subject_id), fact_type),
    ).fetchone()
    supersedes_fact_id = int(head[0]) if head is not None else None
    (
        installation_id,
        app_version,
        context,
        source_kind,
        source_ref,
        created_at,
    ) = provenance.values()

    cursor = conn.execute(
        """
        INSERT INTO human_label_facts (
            subject_id,
            fact_type,
            assertion_state,
            answer_value,
            species_key,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            coordinate_space,
            context,
            source_kind,
            source_ref,
            installation_id,
            app_version,
            created_at,
            supersedes_fact_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(subject_id),
            fact_type,
            assertion_state,
            answer_value,
            species_key.strip() if species_key is not None else None,
            bbox.x if bbox is not None else None,
            bbox.y if bbox is not None else None,
            bbox.w if bbox is not None else None,
            bbox.h if bbox is not None else None,
            "frame_fraction_xywh_v1" if bbox is not None else None,
            context,
            source_kind,
            source_ref,
            installation_id,
            app_version,
            created_at,
            supersedes_fact_id,
        ),
    )
    return int(cursor.lastrowid)


def _reject_presence_conflict(
    conn: sqlite3.Connection,
    *,
    scope: str,
    image_filename: str,
    fact_type: str,
    answer_value: str | None,
    assertion_state: str,
) -> None:
    if (
        fact_type != "bird_presence"
        or assertion_state != "asserted"
        or answer_value not in {"present", "absent"}
    ):
        return

    if scope == "image" and answer_value == "absent":
        conflict = conn.execute(
            """
            SELECT 1
            FROM current_human_label_facts
            WHERE image_filename = ?
              AND scope = 'object'
              AND fact_type = 'bird_presence'
              AND answer_value = 'present'
            LIMIT 1
            """,
            (image_filename,),
        ).fetchone()
    elif scope == "object" and answer_value == "present":
        conflict = conn.execute(
            """
            SELECT 1
            FROM current_human_label_facts
            WHERE image_filename = ?
              AND scope = 'image'
              AND fact_type = 'bird_presence'
              AND answer_value = 'absent'
            LIMIT 1
            """,
            (image_filename,),
        ).fetchone()
    else:
        conflict = None

    if conflict is not None:
        raise HumanLabelError("bird-presence conflict requires explicit resolution")
