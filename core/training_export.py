"""Recorded training snapshots and revision-specific receipt protection."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from core.canonical_dataset import (
    CanonicalDataset,
    build_canonical_dataset,
    read_media_size,
)
from utils.path_manager import PathManager


def original_resolver(paths: PathManager) -> Callable[[str], Path]:
    def resolve(filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("Invalid original filename")
        path = paths.contained_path(
            paths.get_original_path(filename), paths.originals_dir
        )
        if path is None:
            raise ValueError("Original is outside the media directory")
        return path

    return resolve


def answer_revisions(
    conn: sqlite3.Connection, filenames: list[str] | None = None
) -> list[dict[str, object]]:
    if filenames == []:
        return []
    batch_size = max(1, min(400, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)))
    if filenames is not None and len(filenames) > batch_size:
        return [
            row
            for start in range(0, len(filenames), batch_size)
            for row in answer_revisions(conn, filenames[start : start + batch_size])
        ]
    where = ""
    params = filenames or []
    if filenames is not None:
        where = " WHERE image_filename IN (" + ",".join("?" for _ in filenames) + ")"
    grouped: dict[int, list[int]] = defaultdict(list)
    names: dict[int, str] = {}
    selection_keys: dict[int, str] = {}
    for row in conn.execute(
        "SELECT subject_id, image_filename, fact_id, detection_id FROM current_human_label_facts"
        + where
        + " ORDER BY subject_id, fact_id",
        params,
    ):
        grouped[row["subject_id"]].append(row["fact_id"])
        names[row["subject_id"]] = row["image_filename"]
        selection_keys[row["subject_id"]] = (
            f"detection:{row['detection_id']}"
            if row["detection_id"] is not None
            else f"image:{row['subject_id']}"
        )
    revisions = [
        dict(
            kind="subject",
            selection_key=selection_keys[key],
            object_id=key,
            image_filename=names[key],
            revision_json=json.dumps(ids, separators=(",", ":")),
        )
        for key, ids in grouped.items()
    ]
    manual_where = where + (" AND" if where else " WHERE") + " status='active'"
    for row in conn.execute("SELECT * FROM manual_objects" + manual_where, params):
        revisions.append(
            dict(
                kind="manual",
                object_id=row["manual_object_id"],
                image_filename=row["image_filename"],
                selection_key=f"manual:{row['manual_object_id']}",
                revision_json=json.dumps(
                    [
                        row["revision"],
                        hashlib.sha256(
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "bbox_x",
                                        "bbox_y",
                                        "bbox_w",
                                        "bbox_h",
                                        "species_key",
                                        "species_state",
                                    )
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                    ],
                    separators=(",", ":"),
                ),
            )
        )
    return revisions


def waiting_revisions(
    conn: sqlite3.Connection,
    *,
    media_exists: Callable[[str], bool],
    filenames: list[str] | None = None,
) -> list[dict[str, object]]:
    waiting = []
    for row in answer_revisions(conn, filenames):
        if not media_exists(str(row["image_filename"])):
            continue
        exported = conn.execute(
            """SELECT 1 FROM labeling_export_members
               WHERE kind=? AND object_id=? AND revision_json=? LIMIT 1""",
            (row["kind"], row["object_id"], row["revision_json"]),
        ).fetchone()
        if exported is None:
            waiting.append(row)
    return waiting


def list_exports(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in conn.execute(
            """SELECT e.export_id, e.created_at, e.received_at,
                  COUNT(DISTINCT m.image_filename) AS photo_count
           FROM labeling_exports e LEFT JOIN labeling_export_members m USING(export_id)
           GROUP BY e.export_id ORDER BY e.created_at DESC, e.export_id"""
        )
    ]


def _selection_filenames(conn: sqlite3.Connection, selection: set[str]) -> list[str]:
    queries = {
        "detection": "SELECT image_filename FROM detections WHERE detection_id=?",
        "manual": "SELECT image_filename FROM manual_objects WHERE manual_object_id=? AND status='active'",
        "image": "SELECT image_filename FROM label_subjects WHERE subject_id=? AND scope='image'",
    }
    names = set()
    for key in selection:
        kind, _, identifier = key.partition(":")
        if (
            kind not in queries
            or not identifier.isascii()
            or not identifier.isdecimal()
            or len(identifier) > 19
            or int(identifier) > 2**63 - 1
        ):
            raise ValueError("Select current answers waiting for training")
        row = conn.execute(queries[kind], (int(identifier),)).fetchone()
        if row is None:
            raise ValueError("Select current answers waiting for training")
        names.add(str(row["image_filename"]))
    return sorted(names)


def create_export(
    conn: sqlite3.Connection,
    *,
    paths: PathManager,
    labeling_installation_id: str,
    selection: set[str] | None = None,
) -> tuple[str, CanonicalDataset]:
    resolve = original_resolver(paths)
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        waiting = waiting_revisions(
            conn,
            media_exists=lambda name: resolve(name).is_file(),
            filenames=(
                _selection_filenames(conn, selection) if selection is not None else None
            ),
        )
        selected = (
            waiting
            if selection is None
            else [row for row in waiting if row["selection_key"] in selection]
        )
        if selection is not None and len(selected) != len(selection):
            raise ValueError("Select current answers waiting for training")
        names = {str(row["image_filename"]) for row in selected}
        if not names:
            raise ValueError("No photos are waiting for training")
        bundle = build_canonical_dataset(
            conn,
            labeling_installation_id=labeling_installation_id,
            filenames=names,
            media_exists=lambda name: resolve(name).is_file(),
            media_size=lambda name: read_media_size(resolve(name)),
        )
        export_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO labeling_exports(export_id, created_at, bundle_json) VALUES (?, ?, ?)",
            (
                export_id,
                datetime.now(UTC).isoformat(),
                json.dumps(asdict(bundle), sort_keys=True, separators=(",", ":")),
            ),
        )
        for member in answer_revisions(conn, sorted(names)):
            conn.execute(
                """INSERT INTO labeling_export_members
                   (export_id, image_filename, kind, object_id, revision_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    export_id,
                    member["image_filename"],
                    member["kind"],
                    member["object_id"],
                    member["revision_json"],
                ),
            )
    return export_id, bundle


def load_export(
    conn: sqlite3.Connection,
    export_id: str,
) -> tuple[CanonicalDataset, dict[str, str]]:
    row = conn.execute(
        "SELECT * FROM labeling_exports WHERE export_id=?", (export_id,)
    ).fetchone()
    if row is None:
        raise LookupError("Export not found")
    return CanonicalDataset(**json.loads(row["bundle_json"])), json.loads(
        row["media_map_json"]
    )


def receive_export(conn: sqlite3.Connection, export_id: str) -> None:
    with conn:
        result = conn.execute(
            "UPDATE labeling_exports SET received_at=COALESCE(received_at, ?) WHERE export_id=?",
            (datetime.now(UTC).isoformat(), export_id),
        )
        if result.rowcount == 0:
            raise LookupError("Export not found")
