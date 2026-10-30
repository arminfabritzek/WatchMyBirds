"""Deterministic serialization and explicit transports for canonical bundles."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from config import get_config
from core.canonical_archive import (
    render_canonical_bundle_to_path,
    stream_canonical_bundle,
    write_canonical_bundle,
)
from core.canonical_dataset import (
    CanonicalDataset,
    build_canonical_dataset,
    read_media_size,
)
from core.human_label_core import get_or_create_labeling_installation_id
from core.training_export import (
    create_export,
    list_exports,
    load_export,
    original_resolver,
    preview_filenames,
    receive_export,
    waiting_revisions,
)
from core.training_export_jobs import TrainingExportJobs

__all__ = [
    "render_canonical_bundle_to_path",
    "stream_canonical_bundle",
    "write_canonical_bundle",
    "create_export",
    "list_exports",
    "load_export",
    "original_resolver",
    "receive_export",
    "waiting_revisions",
    "build_bundle",
    "get_or_create_labeling_installation_id",
    "create_jobs",
]

PathResolver = Callable[[str], Path]


def build_bundle(
    conn: sqlite3.Connection,
    *,
    path_resolver: PathResolver,
    filenames: set[str] | None = None,
) -> CanonicalDataset:
    """Build one snapshot using the same media eligibility for every transport."""
    return build_canonical_dataset(
        conn,
        filenames=filenames,
        labeling_installation_id=get_or_create_labeling_installation_id(
            str(get_config()["OUTPUT_DIR"])
        ),
        media_exists=lambda filename: path_resolver(filename).is_file(),
        media_size=lambda filename: read_media_size(path_resolver(filename)),
    )


def create_jobs(output_dir: str) -> TrainingExportJobs:
    return TrainingExportJobs(output_dir)


def preview_page(
    conn: sqlite3.Connection, *, path_resolver: PathResolver, after: str
) -> tuple[CanonicalDataset, dict]:
    names = preview_filenames(conn, after)
    selected = names[:200]
    return build_bundle(
        conn, path_resolver=path_resolver, filenames=set(selected)
    ), dict(
        filenames=selected,
        has_more=len(names) > 200,
        after=after,
        next_after=selected[-1] if selected and len(names) > 200 else None,
    )


def box_walkthrough_candidate_ids(bundle: CanonicalDataset) -> list[int]:
    """Detections with an answered species but no box verdict yet.

    Derived from the same manifest the page already renders, not a second
    query: a detection qualifies when its species is answered (the
    ``cls_positive`` view includes it) and ``od_positive`` names a missing
    ``bbox_quality`` fact among its blockers — not an unsuitable one, which
    is already answered and must not reappear. A sibling detection on the
    same frame still waiting on its own answer also marks this row
    ``frame_has_unresolved_objects``; that reason does not disqualify the
    row, since answering its own verdict is still the useful action.
    """
    cls_ready_ids = {
        row["detection_id"]
        for row in bundle.manifest
        if row["view"] == "cls_positive" and row["decision"] == "included"
    }
    allowed_reasons = {
        "bbox_quality_unknown",
        "frame_has_unresolved_objects",
        "frame_dimensions_missing",
    }
    return [
        int(row["detection_id"])
        for row in bundle.manifest
        if row["view"] == "od_positive"
        and row["detection_id"] is not None
        and row["decision"] == "excluded"
        and "bbox_quality_unknown" in row["reasons"]
        and set(row["reasons"]) <= allowed_reasons
        and row["detection_id"] in cls_ready_ids
    ]


def needs_box_verdict_count(bundle: CanonicalDataset) -> int:
    """Count of :func:`box_walkthrough_candidate_ids`."""
    return len(box_walkthrough_candidate_ids(bundle))


def missing_original_bird_count(bundle: CanonicalDataset) -> int:
    """Count labeled birds with missing media once across both training views."""
    presence_blockers = {"object_bird_presence_unknown", "object_bird_absent"}
    return len(
        {
            row.get("object_key", row["detection_id"])
            for row in bundle.manifest
            if row["view"] in {"cls_positive", "od_positive"}
            and row["decision"] == "excluded"
            and "media_missing" in row["reasons"]
            and not presence_blockers.intersection(row["reasons"])
        }
    )
