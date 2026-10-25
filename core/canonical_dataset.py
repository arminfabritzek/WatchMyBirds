"""Versioned, read-only derivation of canonical training-data views."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from core.human_label_core import object_training_readiness

MANIFEST_SCHEMA_VERSION = "watchmybirds.canonical-label-bundle.v3"
RULE_VERSION = "training-readiness-v3"


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
    objects: list[dict[str, object]] = field(default_factory=list)
    manual_objects: list[dict[str, object]] = field(default_factory=list)
    manual_revisions: list[dict[str, object]] = field(default_factory=list)

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


def _rows(
    conn: sqlite3.Connection, sql: str, params: tuple = ()
) -> list[dict[str, object]]:
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
    values = tuple(
        subject.get(f"proposal_bbox_{axis}") for axis in ("x", "y", "w", "h")
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _object_resolved_for_od(facts: list[Mapping[str, object]]) -> bool:
    presence = _fact_value(facts, "bird_presence")
    if presence == "absent":
        return True
    return presence == "present" and _fact_value(facts, "bbox_quality") == "suitable"


def read_media_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError):
        return None


def build_canonical_dataset(
    conn: sqlite3.Connection,
    *,
    labeling_installation_id: str,
    filenames: set[str] | None = None,
    media_exists: Callable[[str], bool],
    media_size: Callable[[str], tuple[int, int] | None] | None = None,
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
    if filenames is not None:
        subjects = [row for row in subjects if row["image_filename"] in filenames]
        facts = [row for row in facts if row["image_filename"] in filenames]
        current_facts = [
            row for row in current_facts if row["image_filename"] in filenames
        ]
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
    manual_objects = _rows(
        conn, "SELECT * FROM manual_objects ORDER BY manual_object_id"
    )
    manual_revisions = _rows(
        conn, "SELECT * FROM manual_object_revisions ORDER BY manual_revision_id"
    )
    if filenames is not None:
        manual_objects = [
            row for row in manual_objects if row["image_filename"] in filenames
        ]
        manual_ids = {row["manual_object_id"] for row in manual_objects}
        manual_revisions = [
            row for row in manual_revisions if row["manual_object_id"] in manual_ids
        ]
    candidate_filenames = sorted(
        {str(subject["image_filename"]) for subject in subjects}
        | {
            str(obj["image_filename"])
            for obj in manual_objects
            if obj["status"] == "active"
        }
    )
    detections = _rows(
        conn,
        """
        SELECT detection_id, image_filename, frame_width, frame_height, status, bbox_x, bbox_y, bbox_w, bbox_h, raw_species_name, decision_state, decision_level FROM detections WHERE image_filename IN (
            SELECT image_filename FROM label_subjects
            UNION SELECT image_filename FROM manual_objects WHERE status = 'active'
        ) ORDER BY image_filename, detection_id
    """,
    )
    candidate_names = set(candidate_filenames)
    detections = [row for row in detections if row["image_filename"] in candidate_names]
    image_facts = {
        name: current_by_subject.get(int(subject["subject_id"]), [])
        for name, subject in image_subjects.items()
    }
    objects: list[dict[str, object]] = []
    dimensions: dict[str, tuple[int, int]] = {}
    for detection in detections:
        detection_id = int(detection["detection_id"])
        filename = str(detection["image_filename"])
        subject = subject_by_detection.get(detection_id)
        current = (
            current_by_subject.get(int(subject["subject_id"]), []) if subject else []
        )
        readiness = object_training_readiness(current)
        species = next(
            (
                fact.get("species_key")
                for fact in current
                if fact["fact_type"] == "species_identity"
            ),
            None,
        )
        width, height = (
            int(detection.get("frame_width") or 0),
            int(detection.get("frame_height") or 0),
        )
        if width > 0 and height > 0:
            dimensions[filename] = (width, height)
        objects.append(
            {
                "object_key": f"detection:{detection_id}",
                "detection_id": detection_id,
                "manual_object_id": None,
                "image_filename": filename,
                "bbox": _effective_bbox(subject, current)
                if subject
                else tuple(detection[f"bbox_{axis}"] for axis in ("x", "y", "w", "h")),
                "proposal_species_key": detection["raw_species_name"],
                "decision_state": detection["decision_state"],
                "decision_level": detection["decision_level"],
                "species_key": None
                if _fact_value(current, "bird_presence") == "absent"
                else species,
                "presence": _fact_value(current, "bird_presence"),
                "active": detection.get("status", "active") == "active",
                "resolved_od": _object_resolved_for_od(current),
                "od_reasons": list(readiness["od"]["reasons"]),
                "cls_reasons": list(readiness["cls"]["reasons"]),
            }
        )
    for manual in manual_objects:
        if manual["status"] != "active":
            continue
        manual_id = int(manual["manual_object_id"])
        objects.append(
            {
                "object_key": f"manual:{manual_id}",
                "detection_id": None,
                "manual_object_id": manual_id,
                "image_filename": manual["image_filename"],
                "bbox": tuple(
                    float(manual[f"bbox_{axis}"]) for axis in ("x", "y", "w", "h")
                ),
                "species_key": manual["species_key"],
                "presence": "present",
                "active": True,
                "resolved_od": True,
                "od_reasons": [],
                "cls_reasons": []
                if manual["species_state"] == "identified"
                else ["species_unknown"],
            }
        )
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        by_image[str(obj["image_filename"])].append(obj)

    manifest: list[dict[str, object]] = []
    media_filenames: set[str] = set()
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    classifier_ready: list[dict[str, object]] = []
    unresolved_ids: set[int] = set()
    for filename in candidate_filenames:
        present_media = media_exists(filename)
        if present_media:
            media_filenames.add(filename)
        if filename not in dimensions and present_media and media_size:
            size = media_size(filename)
            if size:
                dimensions[filename] = size
        width, height = dimensions.get(filename, (0, 0))
        frame_facts = image_facts.get(filename, [])
        image_presence = _fact_value(frame_facts, "bird_presence")
        frame_objects = by_image[filename]
        positive_objects = [
            obj
            for obj in frame_objects
            if obj["active"] and obj["presence"] == "present"
        ]
        negative_reasons = []
        if not present_media:
            negative_reasons.append("media_missing")
        if positive_objects:
            negative_reasons.append("image_has_positive_objects")
        if image_presence != "absent":
            negative_reasons.append(
                "image_bird_present"
                if image_presence == "present"
                else "image_bird_absence_unknown"
            )
        negative_ready = not negative_reasons
        manifest.append(
            _reasoned_decision(
                view="od_negative",
                filename=filename,
                detection_id=None,
                included=negative_ready,
                reasons=negative_reasons or ["explicit_full_image_no_bird"],
            )
        )
        detector_miss = _fact_value(frame_facts, "detector_miss") == "reported"
        frame_complete = (
            bool(frame_objects)
            and not detector_miss
            and image_presence != "absent"
            and all(
                obj["presence"] == "absent"
                or (obj["active"] and obj["resolved_od"] and obj["bbox"] is not None)
                for obj in frame_objects
            )
        )
        frame_od_ready = (
            frame_complete
            and bool(positive_objects)
            and present_media
            and width > 0
            and height > 0
        )
        image_id = len(coco_images) + 1
        if frame_od_ready or negative_ready:
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": f"media/{filename[:4]}-{filename[4:6]}-{filename[6:8]}/{filename}",
                    "width": width or None,
                    "height": height or None,
                }
            )
        for obj in frame_objects:
            detection_id = obj["detection_id"]
            key = obj["object_key"]
            sample_id = (
                _stable_sample_id("object", filename, detection_id)
                if detection_id is not None
                else hashlib.sha256(f"object:{filename}:{key}".encode()).hexdigest()[
                    :24
                ]
            )
            identity = {
                "sample_id": sample_id,
                "object_key": key,
                "detection_id": detection_id,
                "manual_object_id": obj["manual_object_id"],
            }
            shared_reasons = []
            if not present_media:
                shared_reasons.append("media_missing")
            if obj["bbox"] is None:
                shared_reasons.append("bbox_geometry_missing")
            if not obj["active"] and obj["presence"] != "absent":
                shared_reasons.append("object_inactive")
            if image_presence == "absent":
                shared_reasons.append("image_bird_absent")
            od_reasons = list(obj["od_reasons"]) + shared_reasons
            if detector_miss:
                od_reasons.append("detector_miss_reported")
            if not frame_complete:
                od_reasons.append("frame_has_unresolved_objects")
            if width <= 0 or height <= 0:
                od_reasons.append("frame_dimensions_missing")
            cls_reasons = list(obj["cls_reasons"]) + shared_reasons
            for view, reasons, success in (
                ("od_positive", od_reasons, "explicit_object_bird_and_suitable_bbox"),
                ("cls_positive", cls_reasons, "explicit_object_bird_and_species"),
            ):
                manifest.append(
                    {
                        **_reasoned_decision(
                            view=view,
                            filename=filename,
                            detection_id=detection_id,
                            included=not reasons,
                            reasons=list(dict.fromkeys(reasons)) or [success],
                        ),
                        **identity,
                    }
                )
            if not od_reasons:
                x, y, w, h = obj["bbox"]
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x * width, y * height, w * width, h * height],
                        "bbox_normalized": [x, y, w, h],
                        "area": w * width * h * height,
                        "iscrowd": 0,
                        **identity,
                    }
                )
            if not cls_reasons:
                classifier_ready.append(
                    {
                        **identity,
                        "image_filename": filename,
                        "media_path": f"media/{filename[:4]}-{filename[4:6]}-{filename[6:8]}/{filename}",
                        "species_key": obj["species_key"],
                        "bbox_xywh_normalized": list(obj["bbox"]),
                        "rule_version": RULE_VERSION,
                    }
                )
            unresolved = list(dict.fromkeys(od_reasons + cls_reasons))
            if unresolved and detection_id is not None:
                unresolved_ids.add(int(detection_id))
            manifest.append(
                {
                    **_reasoned_decision(
                        view="unresolved",
                        filename=filename,
                        detection_id=detection_id,
                        included=bool(unresolved),
                        reasons=unresolved or ["all_requested_training_views_resolved"],
                    ),
                    **identity,
                }
            )

    def station_sample_id(local_id: str) -> str:
        return f"{labeling_installation_id}:{local_id}"

    for rows in (manifest, annotations, classifier_ready):
        for row in rows:
            row["sample_id"] = station_sample_id(str(row["sample_id"]))
    sample_ids = {
        row["object_key"]: row["sample_id"] for row in manifest if "object_key" in row
    }
    for obj in objects:
        obj["sample_id"] = sample_ids[obj["object_key"]]
    manifest.sort(
        key=lambda row: (
            str(row["image_filename"]),
            str(row["view"]),
            str(row.get("object_key", "")),
        )
    )
    coco = {
        "info": {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "rule_version": RULE_VERSION,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "bird", "supercategory": "animal"}],
        "images": coco_images,
        "annotations": annotations,
    }
    snapshot = {
        "max_subject_id": max((int(row["subject_id"]) for row in subjects), default=0),
        "max_fact_id": max((int(row["fact_id"]) for row in facts), default=0),
        "max_manual_revision_id": max(
            (int(row["manual_revision_id"]) for row in manual_revisions), default=0
        ),
        "latest_fact_created_at": max(
            (str(row["created_at"]) for row in facts), default=""
        ),
    }
    identity_payload = b"".join(
        (
            MANIFEST_SCHEMA_VERSION.encode(),
            RULE_VERSION.encode(),
            _canonical_line_bytes(subjects),
            _canonical_line_bytes(facts),
            _canonical_line_bytes(manual_objects),
            _canonical_line_bytes(manual_revisions),
            _canonical_line_bytes(manifest),
            json.dumps(coco, sort_keys=True, separators=(",", ":")).encode(),
            _canonical_line_bytes(classifier_ready),
        )
    )
    return CanonicalDataset(
        bundle_id=hashlib.sha256(identity_payload).hexdigest(),
        subjects=subjects,
        facts=facts,
        manifest=manifest,
        coco=coco,
        classifier_ready=classifier_ready,
        unresolved_detection_ids=sorted(unresolved_ids),
        media_filenames=sorted(media_filenames),
        snapshot=snapshot,
        objects=objects,
        manual_objects=manual_objects,
        manual_revisions=manual_revisions,
    )
