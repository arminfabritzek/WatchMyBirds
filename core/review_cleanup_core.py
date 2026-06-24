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
from utils.db.review_queue import (
    fetch_review_queue_summary,
    fetch_review_queue_summary_rows,
)


@dataclass
class ReviewCleanupActionPlan:
    """The minimal payload the reversible move-to-Trash needs.

    Only the two id buckets — no event/favorite/export counts. This is what
    ``execute_plan`` builds fresh on every run, so the action always acts on
    the queue as it is *now*, never on a stale preview snapshot.
    """

    image_filenames: list[str] = field(default_factory=list)
    detection_ids: list[int] = field(default_factory=list)


@dataclass
class ReviewCleanupPlan:
    image_filenames: list[str] = field(default_factory=list)
    detection_ids: list[int] = field(default_factory=list)
    event_count: int = 0
    favorite_count: int = 0
    export_relevant_count: int = 0


def build_action_plan(conn, gallery_threshold: float = 0.7) -> ReviewCleanupActionPlan:
    """Partition the live queue into the two reversible id buckets — lean.

    Read-only, count-free. Uses the lean summary projection (shared queue
    predicate, no sibling-count / cls-confidence subqueries), so the run
    path pays only for the ids it is about to act on. ``image_filenames``
    carries orphan/untagged queue images, ``detection_ids`` carries active
    unresolved detections.
    """
    rows = fetch_review_queue_summary_rows(conn, gallery_threshold=gallery_threshold)

    image_filenames: list[str] = []
    detection_ids: list[int] = []
    for row in rows:
        if row["item_kind"] == "image":
            image_filenames.append(row["item_id"])
        else:
            detection_ids.append(int(row["item_id"]))

    return ReviewCleanupActionPlan(
        image_filenames=image_filenames,
        detection_ids=detection_ids,
    )


def summarize_queue(conn, gallery_threshold: float = 0.7) -> dict[str, int]:
    """Dry-run counts for the preview — lean, no per-row render work.

    Returns ``{"events", "images", "detections", "favorites",
    "export_relevant"}``. ``images``/``detections``/``favorites`` come from
    a pure SQL aggregate; ``events`` clusters the lean detection projection
    (timestamp + species + bbox is all ``build_bird_events`` reads for the
    count); ``export_relevant`` is checked only over the queue's own touched
    filenames, not the whole history.
    """
    counts = fetch_review_queue_summary(conn, gallery_threshold=gallery_threshold)
    rows = fetch_review_queue_summary_rows(conn, gallery_threshold=gallery_threshold)

    detection_rows: list[dict[str, Any]] = []
    touched_filenames: set[str] = set()
    for row in rows:
        filename = row["filename"]
        if filename:
            touched_filenames.add(filename)
        if row["item_kind"] != "image":
            detection_rows.append(dict(row))

    export_relevant = user_groundtruth_core.is_export_relevant_any(
        conn, sorted(touched_filenames)
    )

    return {
        "events": len(build_bird_events(detection_rows)) if detection_rows else 0,
        "images": counts["images"],
        "detections": counts["detections"],
        "favorites": counts["favorites"],
        "export_relevant": len(export_relevant),
    }


def build_plan(conn, gallery_threshold: float = 0.7) -> ReviewCleanupPlan:
    """Full dry-run plan: id buckets + the five preview counts.

    Read-only. Kept as the public composition of ``build_action_plan`` and
    ``summarize_queue`` so existing callers and the parity tests keep a
    single object to assert against. The run path uses ``build_action_plan``
    directly; this is for the preview disclosure.
    """
    action = build_action_plan(conn, gallery_threshold=gallery_threshold)
    summary = summarize_queue(conn, gallery_threshold=gallery_threshold)
    return ReviewCleanupPlan(
        image_filenames=action.image_filenames,
        detection_ids=action.detection_ids,
        event_count=summary["events"],
        favorite_count=summary["favorites"],
        export_relevant_count=summary["export_relevant"],
    )


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

    plan = build_action_plan(conn, gallery_threshold=gallery_threshold)

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
        return summarize_queue(conn, gallery_threshold=threshold)


def run() -> dict[str, int]:
    """Execute the reversible move-to-Trash against the live DB."""
    from utils.db import closing_connection

    threshold = _gallery_threshold()
    with closing_connection() as conn:
        result = execute_plan(conn, gallery_threshold=threshold)
        conn.commit()
    return result
