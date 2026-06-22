"""Review-queue cleanup — Planner + Executor for "Move Review Queue to Trash".

A reversible, file-free bulk action for operators who don't work the Review
desk. It moves the *current review queue* into Trash via the existing
reversible primitives — nothing is hard-deleted, no file is touched, and no
retention/original state changes.

Semantics (V1):
  - orphan / untagged queue images -> review_status='no_bird'
  - active unresolved detections   -> status='rejected'
Both are restorable from the Trash surface.

Queue membership is sourced from ``fetch_review_queue_images`` — the SAME
predicate the Review page renders — so the dry-run preview can never drift
from what the operator sees.

Layering: like ``retention_core``, this is core/* and may import utils/*
(H-02). It must not import web/* or Flask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import user_groundtruth_core
from core.events import build_bird_events
from utils.db.review_queue import fetch_review_queue_images


@dataclass
class ReviewCleanupPlan:
    image_filenames: list[str] = field(default_factory=list)
    detection_ids: list[int] = field(default_factory=list)
    event_count: int = 0
    favorite_count: int = 0
    export_relevant_count: int = 0


def build_plan(conn, gallery_threshold: float = 0.7) -> ReviewCleanupPlan:
    """Partition the live review queue into the two reversible buckets.

    Read-only: reads the queue and the export-relevance union, performs no
    writes. ``image_filenames`` carries orphan/untagged queue images,
    ``detection_ids`` carries active unresolved detections — exactly the two
    payloads the reversible ``bulk_reject`` semantics accept.
    """
    rows = fetch_review_queue_images(conn, gallery_threshold=gallery_threshold)

    image_filenames: list[str] = []
    detection_ids: list[int] = []
    detection_rows: list[dict[str, Any]] = []
    favorite_count = 0
    touched_filenames: set[str] = set()

    for row in rows:
        filename = row["filename"]
        if filename:
            touched_filenames.add(filename)
        if row["item_kind"] == "image":
            image_filenames.append(row["item_id"])
        else:  # detection
            detection_ids.append(int(row["item_id"]))
            detection_rows.append(dict(row))
            if row["is_favorite"]:
                favorite_count += 1

    export_relevant = user_groundtruth_core.is_export_relevant_any(
        conn, sorted(touched_filenames)
    )

    plan = ReviewCleanupPlan(
        image_filenames=image_filenames,
        detection_ids=detection_ids,
        event_count=len(build_bird_events(detection_rows)) if detection_rows else 0,
        favorite_count=favorite_count,
        export_relevant_count=len(export_relevant),
    )
    return plan


# ---------------------------------------------------------------------------
# Executor — reversible move to Trash. No file IO.
# ---------------------------------------------------------------------------


def execute_plan(conn, gallery_threshold: float = 0.7) -> dict[str, int]:
    """Move the live review queue to Trash via the reversible primitives.

    ``reject_detections`` (status='rejected') + ``update_review_status``
    (review_status='no_bird'). No file is deleted; both states restore from
    the Trash surface. Returns {"images_moved", "detections_moved"}.
    """
    from utils.db.detections import reject_detections
    from utils.db.review_queue import update_review_status

    plan = build_plan(conn, gallery_threshold=gallery_threshold)

    detections_moved = 0
    if plan.detection_ids:
        reject_detections(conn, plan.detection_ids)
        detections_moved = len(plan.detection_ids)

    images_moved = 0
    if plan.image_filenames:
        images_moved = update_review_status(conn, plan.image_filenames, "no_bird")

    return {"images_moved": images_moved, "detections_moved": detections_moved}


# ---------------------------------------------------------------------------
# Conn-opening entry points — the web service delegates here so it never
# opens a connection or imports utils itself (H-01 enforcement-test rule).
# ---------------------------------------------------------------------------


def _gallery_threshold() -> float:
    from config import get_config

    cfg = get_config()
    return float(cfg.get("GALLERY_DISPLAY_THRESHOLD", 0.7))


def preview() -> dict[str, Any]:
    """Dry-run: queue counts + favorite/export-relevant disclosure. No writes."""
    from utils.db import closing_connection

    threshold = _gallery_threshold()
    with closing_connection() as conn:
        plan = build_plan(conn, gallery_threshold=threshold)

    return {
        "events": plan.event_count,
        "images": len(plan.image_filenames),
        "detections": len(plan.detection_ids),
        "favorites": plan.favorite_count,
        "export_relevant": plan.export_relevant_count,
    }


def run() -> dict[str, int]:
    """Execute the reversible move-to-Trash against the live DB."""
    from utils.db import closing_connection

    threshold = _gallery_threshold()
    with closing_connection() as conn:
        result = execute_plan(conn, gallery_threshold=threshold)
        conn.commit()
    return result
