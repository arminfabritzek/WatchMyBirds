"""Persistence for birds added manually to an existing stored image.

Manual objects deliberately do not masquerade as detector proposals. Their
current state lives in ``manual_objects`` and every explicit save appends an
auditable revision describing only the facts changed by that action.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.human_label_core import BBox, HumanLabelError, LabelProvenance
from utils.species_names import canonical_species_key, is_known_bird_species


@dataclass(frozen=True)
class ManualObjectDraft:
    """Explicit facts supplied when a missed bird is saved."""

    image_filename: str
    bbox: BBox
    species_key: str | None
    request_id: str


def _validated_species(species_key: str | None, locale: str) -> str | None:
    key = canonical_species_key(species_key)
    if not key:
        return None
    if not is_known_bird_species(key, locale):
        raise HumanLabelError("species is not in the bird species catalog")
    return key


def _validate_request_id(request_id: str) -> str:
    value = request_id.strip()
    if not 8 <= len(value) <= 128 or any(char.isspace() for char in value):
        raise HumanLabelError("request_id is invalid")
    return value


def _row_payload(row: sqlite3.Row) -> dict[str, object]:
    return {
        "manual_object_id": int(row["manual_object_id"]),
        "image_filename": str(row["image_filename"]),
        "bbox": {
            "x": float(row["bbox_x"]),
            "y": float(row["bbox_y"]),
            "w": float(row["bbox_w"]),
            "h": float(row["bbox_h"]),
        },
        "species_key": row["species_key"],
        "species_state": str(row["species_state"]),
        "revision": int(row["revision"]),
        "provenance": "manually_added",
    }


def _fetch_active(conn: sqlite3.Connection, manual_object_id: int) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM manual_objects
        WHERE manual_object_id = ? AND status = 'active'
        """,
        (int(manual_object_id),),
    ).fetchone()
    if row is None:
        raise HumanLabelError("manual object does not exist")
    return row


def _append_revision(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    operation: str,
    asserted_facts: list[str],
    provenance: LabelProvenance,
) -> None:
    (
        installation_id,
        app_version,
        context,
        source_kind,
        source_ref,
        created_at,
    ) = provenance.values()
    conn.execute(
        """
        INSERT INTO manual_object_revisions (
            manual_object_id, revision, operation, asserted_facts,
            bbox_x, bbox_y, bbox_w, bbox_h, species_key, species_state,
            context, source_kind, source_ref, installation_id, app_version,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(row["manual_object_id"]),
            int(row["revision"]),
            operation,
            json.dumps(asserted_facts, separators=(",", ":")),
            row["bbox_x"],
            row["bbox_y"],
            row["bbox_w"],
            row["bbox_h"],
            row["species_key"],
            row["species_state"],
            context,
            source_kind,
            source_ref,
            installation_id,
            app_version,
            created_at,
        ),
    )


def create_manual_object(
    conn: sqlite3.Connection,
    draft: ManualObjectDraft,
    provenance: LabelProvenance,
    *,
    original_path: Path,
    locale: str,
) -> tuple[dict[str, object], bool]:
    """Create one manual object, or return the idempotent prior result."""
    filename = draft.image_filename.strip()
    if not filename:
        raise HumanLabelError("image filename is required")
    request_id = _validate_request_id(draft.request_id)
    bbox = draft.bbox.validated()
    species_key = _validated_species(draft.species_key, locale)

    image = conn.execute(
        "SELECT filename FROM images WHERE filename = ?", (filename,)
    ).fetchone()
    if image is None:
        raise HumanLabelError("image does not exist")
    if not original_path.is_file():
        raise HumanLabelError("image original is unavailable")

    prior = conn.execute(
        """
        SELECT object.*
        FROM manual_object_requests request
        JOIN manual_objects object
          ON object.manual_object_id = request.manual_object_id
        WHERE request.request_id = ?
        """,
        (request_id,),
    ).fetchone()
    if prior is not None:
        expected = (bbox.x, bbox.y, bbox.w, bbox.h, species_key)
        actual = (
            prior["bbox_x"],
            prior["bbox_y"],
            prior["bbox_w"],
            prior["bbox_h"],
            prior["species_key"],
        )
        if actual != expected or str(prior["image_filename"]) != filename:
            raise HumanLabelError("request_id already belongs to another object")
        return _row_payload(prior), False

    created_at = provenance.values()[-1]
    cursor = conn.execute(
        """
        INSERT INTO manual_objects (
            image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
            species_key, species_state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            bbox.x,
            bbox.y,
            bbox.w,
            bbox.h,
            species_key,
            "identified" if species_key else "unknown",
            created_at,
            created_at,
        ),
    )
    manual_object_id = int(cursor.lastrowid)
    row = _fetch_active(conn, manual_object_id)
    _append_revision(
        conn,
        row=row,
        operation="create",
        asserted_facts=["bird_presence", "bbox_geometry", "species_identity"],
        provenance=provenance,
    )
    conn.execute(
        """
        INSERT INTO manual_object_requests (
            request_id, manual_object_id, revision, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (request_id, manual_object_id, 1, created_at),
    )
    return _row_payload(row), True


def update_manual_object(
    conn: sqlite3.Connection,
    *,
    manual_object_id: int,
    image_filename: str,
    expected_revision: int,
    provenance: LabelProvenance,
    locale: str,
    bbox: BBox | None = None,
    species_supplied: bool = False,
    species_key: str | None = None,
    original_path: Path | None = None,
) -> tuple[dict[str, object], bool]:
    """Update only supplied axes and append one immutable revision."""
    row = _fetch_active(conn, manual_object_id)
    if str(row["image_filename"]) != image_filename.strip():
        raise HumanLabelError("manual object does not belong to image")
    if bbox is not None and (original_path is None or not original_path.is_file()):
        raise HumanLabelError("image original is unavailable")

    next_bbox = (
        bbox.validated()
        if bbox is not None
        else BBox(
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        )
    )
    next_species = (
        _validated_species(species_key, locale)
        if species_supplied
        else row["species_key"]
    )
    changed_facts: list[str] = []
    if bbox is not None and (
        next_bbox.x,
        next_bbox.y,
        next_bbox.w,
        next_bbox.h,
    ) != (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]):
        changed_facts.append("bbox_geometry")
    if species_supplied and next_species != row["species_key"]:
        changed_facts.append("species_identity")

    if not changed_facts:
        return _row_payload(row), False
    if int(row["revision"]) != int(expected_revision):
        raise HumanLabelError("manual object was changed in another editor")

    next_revision = int(row["revision"]) + 1
    updated_at = provenance.values()[-1]
    conn.execute(
        """
        UPDATE manual_objects
        SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?,
            species_key = ?, species_state = ?, revision = ?, updated_at = ?
        WHERE manual_object_id = ? AND revision = ? AND status = 'active'
        """,
        (
            next_bbox.x,
            next_bbox.y,
            next_bbox.w,
            next_bbox.h,
            next_species,
            "identified" if next_species else "unknown",
            next_revision,
            updated_at,
            int(manual_object_id),
            int(expected_revision),
        ),
    )
    next_row = _fetch_active(conn, manual_object_id)
    _append_revision(
        conn,
        row=next_row,
        operation="update",
        asserted_facts=changed_facts,
        provenance=provenance,
    )
    return _row_payload(next_row), True


def retract_manual_object(
    conn: sqlite3.Connection,
    *,
    manual_object_id: int,
    image_filename: str,
    expected_revision: int,
    provenance: LabelProvenance,
) -> dict[str, object]:
    """Retract one manually added object and preserve an immutable audit row."""
    row = _fetch_active(conn, manual_object_id)
    if str(row["image_filename"]) != image_filename.strip():
        raise HumanLabelError("manual object does not belong to image")
    if int(row["revision"]) != int(expected_revision):
        raise HumanLabelError("manual object was changed in another editor")

    next_revision = int(row["revision"]) + 1
    updated_at = provenance.values()[-1]
    cursor = conn.execute(
        """
        UPDATE manual_objects
        SET status = 'retracted', revision = ?, updated_at = ?
        WHERE manual_object_id = ? AND revision = ? AND status = 'active'
        """,
        (
            next_revision,
            updated_at,
            int(manual_object_id),
            int(expected_revision),
        ),
    )
    if cursor.rowcount != 1:
        raise HumanLabelError("manual object was changed in another editor")
    retracted = conn.execute(
        "SELECT * FROM manual_objects WHERE manual_object_id = ?",
        (int(manual_object_id),),
    ).fetchone()
    _append_revision(
        conn,
        row=retracted,
        operation="update",
        asserted_facts=["bird_presence"],
        provenance=provenance,
    )
    payload = _row_payload(retracted)
    payload["status"] = "retracted"
    return payload


def fetch_manual_objects(
    conn: sqlite3.Connection, image_filenames: list[str]
) -> dict[str, list[dict[str, object]]]:
    """Return active manual companions without promoting them to events."""
    names = sorted({name for name in image_filenames if name})
    if not names:
        return {}
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT * FROM manual_objects
        WHERE image_filename IN ({placeholders}) AND status = 'active'
        ORDER BY image_filename, manual_object_id
        """,
        names,
    ).fetchall()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_filename"]), []).append(_row_payload(row))
    return grouped
