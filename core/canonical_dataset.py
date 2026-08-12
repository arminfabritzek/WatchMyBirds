"""Versioned, read-only derivation of canonical training-data views."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from core.human_label_core import object_training_readiness

MANIFEST_SCHEMA_VERSION = "watchmybirds.canonical-label-bundle.v1"
RULE_VERSION = "training-readiness-v1"


@dataclass(frozen=True)
class CanonicalDataset:
    bundle_id: str
    subjects: list[dict[str, object]]
    facts: list[dict[str, object]]
    manifest: list[dict[str, object]]
    coco: dict[str, object]
    classifier_ready: list[dict[str, object]]
    unresolved_detection_ids: list[int]
    media_filenames: list[str]
    snapshot: dict[str, object]

    @property
    def counts(self) -> dict[str, int]:
        included = defaultdict(int)
        excluded = defaultdict(int)
        for row in self.manifest:
            target = included if row["decision"] == "included" else excluded
            target[str(row["view"])] += 1
        return {
            **{f"{view}_included": count for view, count in sorted(included.items())},
            **{f"{view}_excluded": count for view, count in sorted(excluded.items())},
            "subjects": len(self.subjects),
            "facts": len(self.facts),
        }


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, object]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _stable_sample_id(scope: str, filename: str, detection_id: int | None) -> str:
    value = f"{scope}:{filename}:{detection_id or 'image'}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _canonical_line_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _reasoned_decision(
    *,
    view: str,
    filename: str,
    detection_id: int | None,
    included: bool,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "sample_id": _stable_sample_id(
            "object" if detection_id is not None else "image",
            filename,
            detection_id,
        ),
        "view": view,
        "decision": "included" if included else "excluded",
        "reasons": reasons,
        "image_filename": filename,
        "detection_id": detection_id,
        "rule_version": RULE_VERSION,
    }


def _current_by_subject(
    current_facts: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for fact in current_facts:
        grouped[int(fact["subject_id"])].append(fact)
    return grouped


def _fact_value(facts: list[Mapping[str, object]], fact_type: str) -> object:
    for fact in facts:
        if fact.get("fact_type") == fact_type:
            return fact.get("answer_value")
    return None


def _effective_bbox(
    subject: Mapping[str, object] | None,
    facts: list[Mapping[str, object]],
) -> tuple[float, float, float, float] | None:
    for fact in facts:
        if fact.get("fact_type") == "bbox_correction":
            values = tuple(fact.get(f"bbox_{axis}") for axis in ("x", "y", "w", "h"))
            if all(value is not None for value in values):
                return tuple(float(value) for value in values)  # type: ignore[return-value]
    if subject is None:
        return None
    values = tuple(subject.get(f"proposal_bbox_{axis}") for axis in ("x", "y", "w", "h"))
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _object_resolved_for_od(facts: list[Mapping[str, object]]) -> bool:
    presence = _fact_value(facts, "bird_presence")
    if presence == "absent":
        return True
    return presence == "present" and _fact_value(facts, "bbox_quality") == "suitable"


def build_canonical_dataset(
    conn: sqlite3.Connection,
    *,
    media_exists: Callable[[str], bool],
) -> CanonicalDataset:
    """Derive all views from explicit facts without mutating the database."""
    subjects = _rows(
        conn,
        """
        SELECT s.subject_id, s.scope, s.image_filename, s.detection_id,
               s.proposal_bbox_x, s.proposal_bbox_y, s.proposal_bbox_w,
               s.proposal_bbox_h, s.proposal_coordinate_space,
               s.proposal_species_key, s.detector_model_version,
               s.classifier_model_version, s.created_at,
               i.timestamp AS image_timestamp, i.content_hash,
               d.frame_width, d.frame_height
        FROM label_subjects s
        JOIN images i ON i.filename = s.image_filename
        LEFT JOIN detections d ON d.detection_id = s.detection_id
        ORDER BY s.subject_id
        """,
    )
    facts = _rows(
        conn,
        """
        SELECT f.fact_id, f.schema_version, f.subject_id, s.scope,
               s.image_filename, s.detection_id, f.fact_type,
               f.assertion_state, f.answer_value, f.species_key,
               f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
               f.coordinate_space, f.context, f.source_kind, f.source_ref,
               f.installation_id, f.app_version, f.created_at,
               f.supersedes_fact_id
        FROM human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        ORDER BY f.fact_id
        """,
    )
    current_facts = _rows(
        conn,
        "SELECT * FROM current_human_label_facts ORDER BY subject_id, fact_type",
    )
    current_by_subject = _current_by_subject(current_facts)
    subject_by_detection = {
        int(subject["detection_id"]): subject
        for subject in subjects
        if subject["scope"] == "object" and subject["detection_id"] is not None
    }
    image_subjects = {
        str(subject["image_filename"]): subject
        for subject in subjects
        if subject["scope"] == "image"
    }
    candidate_filenames = sorted({str(subject["image_filename"]) for subject in subjects})
    detections: list[dict[str, object]] = []
    if candidate_filenames:
        placeholders = ",".join("?" for _ in candidate_filenames)
        detections = _rows(
            conn,
            f"""
            SELECT detection_id, image_filename, frame_width, frame_height
            FROM detections
            WHERE image_filename IN ({placeholders})
            ORDER BY image_filename, detection_id
            """,
            tuple(candidate_filenames),
        )
    detections_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for detection in detections:
        detections_by_image[str(detection["image_filename"])].append(detection)

    image_facts: dict[str, list[dict[str, object]]] = {}
    object_facts: dict[int, list[dict[str, object]]] = {}
    for filename, subject in image_subjects.items():
        image_facts[filename] = current_by_subject.get(int(subject["subject_id"]), [])
    for detection_id, subject in subject_by_detection.items():
        object_facts[detection_id] = current_by_subject.get(int(subject["subject_id"]), [])

    manifest: list[dict[str, object]] = []
    media_filenames: set[str] = set()
    complete_positive_frames: set[str] = set()
    included_negative_frames: set[str] = set()
    od_included_ids: set[int] = set()
    cls_included_ids: set[int] = set()
    unresolved_ids: set[int] = set()

    for filename in candidate_filenames:
        present_media = media_exists(filename)
        if present_media:
            media_filenames.add(filename)
        frame_facts = image_facts.get(filename, [])
        image_presence = _fact_value(frame_facts, "bird_presence")
        negative_reasons: list[str] = []
        if not present_media:
            negative_reasons.append("media_missing")
        if image_presence == "absent":
            if not negative_reasons:
                included_negative_frames.add(filename)
                media_filenames.add(filename)
                negative_reasons = ["explicit_full_image_no_bird"]
        elif image_presence == "present":
            negative_reasons.append("image_bird_present")
        else:
            negative_reasons.append("image_bird_absence_unknown")
        manifest.append(
            _reasoned_decision(
                view="od_negative",
                filename=filename,
                detection_id=None,
                included=filename in included_negative_frames,
                reasons=negative_reasons,
            )
        )

        frame_detection_rows = detections_by_image.get(filename, [])
        detector_miss = _fact_value(frame_facts, "detector_miss") == "reported"
        frame_complete = bool(frame_detection_rows) and not detector_miss and all(
            _object_resolved_for_od(object_facts.get(int(row["detection_id"]), []))
            for row in frame_detection_rows
        )
        frame_has_od_positive = any(
            object_training_readiness(
                object_facts.get(int(row["detection_id"]), [])
            )["od"]["ready"]
            for row in frame_detection_rows
        )
        if frame_complete and frame_has_od_positive and present_media:
            complete_positive_frames.add(filename)

        for detection in frame_detection_rows:
            detection_id = int(detection["detection_id"])
            subject = subject_by_detection.get(detection_id)
            active_facts = object_facts.get(detection_id, [])
            readiness = object_training_readiness(active_facts)
            bbox = _effective_bbox(subject, active_facts)

            od_reasons = list(readiness["od"]["reasons"])
            if detector_miss:
                od_reasons.append("detector_miss_reported")
            if not frame_complete:
                od_reasons.append("frame_has_unresolved_objects")
            if not present_media:
                od_reasons.append("media_missing")
            if bbox is None:
                od_reasons.append("bbox_geometry_missing")
            od_ready = not od_reasons
            if od_ready:
                od_included_ids.add(detection_id)
                media_filenames.add(filename)
                od_reasons = ["explicit_object_bird_and_suitable_bbox"]
            manifest.append(
                _reasoned_decision(
                    view="od_positive",
                    filename=filename,
                    detection_id=detection_id,
                    included=od_ready,
                    reasons=list(dict.fromkeys(od_reasons)),
                )
            )

            cls_reasons = list(readiness["cls"]["reasons"])
            if not present_media:
                cls_reasons.append("media_missing")
            if bbox is None:
                cls_reasons.append("bbox_geometry_missing")
            cls_ready = not cls_reasons
            if cls_ready:
                cls_included_ids.add(detection_id)
                media_filenames.add(filename)
                cls_reasons = ["explicit_object_bird_and_species"]
            manifest.append(
                _reasoned_decision(
                    view="cls_positive",
                    filename=filename,
                    detection_id=detection_id,
                    included=cls_ready,
                    reasons=list(dict.fromkeys(cls_reasons)),
                )
            )

            unresolved_reasons = list(
                dict.fromkeys(
                    list(readiness["od"]["reasons"])
                    + list(readiness["cls"]["reasons"])
                    + (["frame_has_unresolved_objects"] if not frame_complete else [])
                )
            )
            unresolved = bool(unresolved_reasons)
            if unresolved:
                unresolved_ids.add(detection_id)
            manifest.append(
                _reasoned_decision(
                    view="unresolved",
                    filename=filename,
                    detection_id=detection_id,
                    included=unresolved,
                    reasons=unresolved_reasons or ["all_requested_training_views_resolved"],
                )
            )

    manifest.sort(
        key=lambda row: (
            str(row["image_filename"]),
            str(row["view"]),
            int(row["detection_id"] or 0),
        )
    )

    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    annotation_id = 1
    coco_filenames = sorted(complete_positive_frames | included_negative_frames)
    image_id_by_name = {filename: index + 1 for index, filename in enumerate(coco_filenames)}
    for filename in coco_filenames:
        media_filenames.add(filename)
        first_detection = next(iter(detections_by_image.get(filename, [])), {})
        coco_images.append(
            {
                "id": image_id_by_name[filename],
                "file_name": f"media/{filename[:4]}-{filename[4:6]}-{filename[6:8]}/{filename}",
                "width": first_detection.get("frame_width"),
                "height": first_detection.get("frame_height"),
            }
        )
        if filename in included_negative_frames:
            continue
        for detection in detections_by_image.get(filename, []):
            detection_id = int(detection["detection_id"])
            if detection_id not in od_included_ids:
                continue
            bbox = _effective_bbox(subject_by_detection.get(detection_id), object_facts.get(detection_id, []))
            width = int(detection.get("frame_width") or 0)
            height = int(detection.get("frame_height") or 0)
            if bbox is None or width <= 0 or height <= 0:
                continue
            x, y, w, h = bbox
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id_by_name[filename],
                    "category_id": 1,
                    "bbox": [x * width, y * height, w * width, h * height],
                    "bbox_normalized": [x, y, w, h],
                    "area": w * width * h * height,
                    "iscrowd": 0,
                    "detection_id": detection_id,
                }
            )
            annotation_id += 1

    classifier_ready: list[dict[str, object]] = []
    for detection_id in sorted(cls_included_ids):
        subject = subject_by_detection[detection_id]
        filename = str(subject["image_filename"])
        active_facts = object_facts[detection_id]
        species_fact = next(fact for fact in active_facts if fact["fact_type"] == "species_identity")
        bbox = _effective_bbox(subject, active_facts)
        classifier_ready.append(
            {
                "sample_id": _stable_sample_id("object", filename, detection_id),
                "image_filename": filename,
                "media_path": f"media/{filename[:4]}-{filename[4:6]}-{filename[6:8]}/{filename}",
                "detection_id": detection_id,
                "species_key": species_fact["species_key"],
                "bbox_xywh_normalized": list(bbox) if bbox else None,
                "rule_version": RULE_VERSION,
            }
        )

    coco = {
        "info": {"schema_version": MANIFEST_SCHEMA_VERSION, "rule_version": RULE_VERSION},
        "licenses": [],
        "categories": [{"id": 1, "name": "bird", "supercategory": "animal"}],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    snapshot = {
        "max_subject_id": max((int(row["subject_id"]) for row in subjects), default=0),
        "max_fact_id": max((int(row["fact_id"]) for row in facts), default=0),
        "latest_fact_created_at": max((str(row["created_at"]) for row in facts), default=""),
    }
    identity_payload = b"".join(
        (
            MANIFEST_SCHEMA_VERSION.encode(),
            RULE_VERSION.encode(),
            _canonical_line_bytes(subjects),
            _canonical_line_bytes(facts),
            _canonical_line_bytes(manifest),
            json.dumps(coco, sort_keys=True, separators=(",", ":")).encode(),
            _canonical_line_bytes(classifier_ready),
        )
    )
    bundle_id = hashlib.sha256(identity_payload).hexdigest()
    return CanonicalDataset(
        bundle_id=bundle_id,
        subjects=subjects,
        facts=facts,
        manifest=manifest,
        coco=coco,
        classifier_ready=classifier_ready,
        unresolved_detection_ids=sorted(unresolved_ids),
        media_filenames=sorted(media_filenames),
        snapshot=snapshot,
    )
