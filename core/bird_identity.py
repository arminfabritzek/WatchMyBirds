"""Resolve a drawn bird against existing objects before creating another identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from core.human_label_core import (
    BBox,
    HumanAnswer,
    HumanLabelError,
    LabelProvenance,
    record_human_answer,
)
from core.manual_object_core import (
    ManualObjectDraft,
    create_manual_object,
    update_manual_object,
)
from core.review_state import refresh_review_image_visibility
from utils.db.detections import effective_bbox_projection_sql
from utils.species_names import canonical_species_key, is_known_bird_species


class BirdMatchRequired(HumanLabelError):
    def __init__(self, candidates: list[dict], token: str) -> None:
        super().__init__("Choose an existing box to correct, or a different bird.")
        self.candidates = candidates
        self.token = token


def matching_birds(
    conn: sqlite3.Connection, filename: str, bbox: BBox
) -> tuple[list[dict], str]:
    geometry, join = effective_bbox_projection_sql(conn)
    rows = conn.execute(
        f"""SELECT d.detection_id, d.status,
            COALESCE(d.manual_species_override, d.raw_species_name) AS species_key,
            {geometry[0]} AS x, {geometry[1]} AS y, {geometry[2]} AS w, {geometry[3]} AS h,
            (SELECT COALESCE(MAX(f.fact_id), 0) FROM current_human_label_facts f
             WHERE f.detection_id = d.detection_id) AS revision
            FROM detections d {join}
            WHERE d.image_filename = ? AND d.status IN ('active', 'rejected')
            ORDER BY d.detection_id""",
        (filename,),
    ).fetchall()
    objects = [
        dict(row, object_key=f"detection:{row['detection_id']}", manual_object_id=None)
        for row in rows
    ]
    rows = conn.execute(
        """SELECT manual_object_id, status, species_key, revision,
            bbox_x AS x, bbox_y AS y, bbox_w AS w, bbox_h AS h
            FROM manual_objects WHERE image_filename = ? AND status = 'active'
            ORDER BY manual_object_id""",
        (filename,),
    ).fetchall()
    objects.extend(
        dict(row, object_key=f"manual:{row['manual_object_id']}", detection_id=None)
        for row in rows
    )
    matches = []
    for obj in objects:
        try:
            other = BBox(*(float(obj[a]) for a in ("x", "y", "w", "h"))).validated()
        except (TypeError, ValueError):
            continue
        intersection = max(
            0, min(bbox.x + bbox.w, other.x + other.w) - max(bbox.x, other.x)
        ) * max(0, min(bbox.y + bbox.h, other.y + other.h) - max(bbox.y, other.y))
        smaller = min(bbox.w * bbox.h, other.w * other.h)
        union = bbox.w * bbox.h + other.w * other.h - intersection
        if intersection / union >= 0.35 or intersection / smaller >= 0.75:
            matches.append({**obj, "bbox": {a: obj[a] for a in ("x", "y", "w", "h")}})
    image = conn.execute(
        "SELECT review_status FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    token = hashlib.sha256(
        json.dumps([objects, list(image) if image else None], sort_keys=True).encode()
    ).hexdigest()
    return matches, token


def add_or_correct_bird(
    conn: sqlite3.Connection,
    draft: ManualObjectDraft,
    provenance: LabelProvenance,
    *,
    original_path: Path,
    locale: str,
    gallery_threshold: float,
    target_key: str | None = None,
    match_token: str | None = None,
) -> tuple[dict, bool]:
    bbox = draft.bbox.validated()
    filename = draft.image_filename.strip()
    species = canonical_species_key(draft.species_key) or None
    if species and not is_known_bird_species(species, locale):
        raise HumanLabelError("species is not in the bird species catalog")
    if not 8 <= len(draft.request_id) <= 128 or any(
        c.isspace() for c in draft.request_id
    ):
        raise HumanLabelError("request_id is invalid")
    if not original_path.is_file():
        raise HumanLabelError("image original is unavailable")
    if not conn.execute(
        "SELECT 1 FROM images WHERE filename = ?", (filename,)
    ).fetchone():
        raise HumanLabelError("image does not exist")
    signature = json.dumps(
        [filename, bbox.x, bbox.y, bbox.w, bbox.h, species, target_key],
        separators=(",", ":"),
    )
    prior = conn.execute(
        "SELECT payload_json, result_json FROM bird_add_requests WHERE request_id = ?",
        (draft.request_id,),
    ).fetchone()
    if prior:
        if prior["payload_json"] != signature:
            raise HumanLabelError("request_id already belongs to another object")
        return json.loads(prior["result_json"]), False
    candidates, token = matching_birds(conn, filename, bbox)
    if candidates and (match_token != token or not target_key):
        raise BirdMatchRequired(candidates, token)
    target = next((obj for obj in candidates if obj["object_key"] == target_key), None)
    if target_key and target_key != "new" and target is None:
        raise BirdMatchRequired(candidates, token)
    # An explicit bird correction supersedes an earlier whole-frame absence.
    absent = conn.execute(
        "SELECT 1 FROM current_human_label_facts WHERE image_filename=? AND scope='image' AND fact_type='bird_presence' AND answer_value='absent'",
        (filename,),
    ).fetchone()
    if absent:
        record_human_answer(
            conn, HumanAnswer(filename, image_bird_presence="present"), provenance
        )
    if target and target["detection_id"] is not None:
        detection_id = target["detection_id"]
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename=filename,
                detection_id=detection_id,
                object_bird_presence="present",
                bbox_correction=bbox,
                bbox_quality="suitable",
                species_identity="corrected" if species else "unknown",
                species_key=species,
            ),
            provenance,
        )
        result = {
            "detection_id": detection_id,
            "manual_object_id": None,
            "object_key": target_key,
            "object_kind": "detection",
            "image_filename": filename,
            "bbox": dict(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h),
            "species_key": species,
            "needs_review": species is None,
            "human_bbox_verdict": "suitable",
            "provenance": "manually_identified",
        }
        created = False
    elif target:
        result, _ = update_manual_object(
            conn,
            manual_object_id=target["manual_object_id"],
            image_filename=filename,
            expected_revision=target["revision"],
            provenance=provenance,
            locale=locale,
            bbox=bbox,
            species_supplied=True,
            species_key=species,
            original_path=original_path,
        )
        created = False
    else:
        result, created = create_manual_object(
            conn, draft, provenance, original_path=original_path, locale=locale
        )
    refresh_review_image_visibility(conn, filename, gallery_threshold)
    conn.execute(
        "INSERT INTO bird_add_requests(request_id,image_filename,payload_json,result_json) VALUES (?,?,?,?)",
        (draft.request_id, filename, signature, json.dumps(result)),
    )
    return result, created
