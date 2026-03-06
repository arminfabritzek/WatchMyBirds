"""
Filter Service — Unified server-side filter resolution for moderation.

Provides a single `resolve_filtered_ids()` function that reproduces
the exact filtered set for any moderation surface (gallery, species_overview,
edit, review_queue, trash). This ensures that `all_filtered` bulk actions
operate on the same data the user sees in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from config import get_config
from logging_config import get_logger
from web.services import db_service

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FilterContext — the single contract for what "filtered" means
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterContext:
    """Immutable filter specification sent by the client.

    Each surface has its own required/optional fields. The resolver
    checks the surface and applies the appropriate filter logic.
    """

    surface: Literal["gallery", "species_overview", "edit", "review_queue", "trash"]

    # Gallery / Edit surface
    date: str | None = None  # YYYY-MM-DD

    # Species Overview surface
    species_key: str | None = None  # e.g. "Parus_major"

    # Score / confidence thresholds
    min_score: float = 0.0
    min_conf: float = 0.0

    # Edit-page status filter
    status_filter: str = "all"  # "all" | "downloaded" | "not_downloaded"

    # Sort order (informational — does not affect ID resolution)
    sort: str = "time_desc"

    @classmethod
    def from_dict(cls, data: dict) -> FilterContext:
        """Build from a request payload dict, with safe defaults."""
        return cls(
            surface=data.get("surface", "gallery"),
            date=data.get("date"),
            species_key=data.get("species_key"),
            min_score=float(data.get("min_score", 0.0)),
            min_conf=float(data.get("min_conf", 0.0)),
            status_filter=data.get("status_filter", "all"),
            sort=data.get("sort", "time_desc"),
        )


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------


@dataclass
class ResolvedSelection:
    """The result of resolving a filter context into concrete targets."""

    detection_ids: list[int] = field(default_factory=list)
    image_filenames: list[str] = field(default_factory=list)
    total_count: int = 0


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


def resolve_filtered_ids(ctx: FilterContext) -> ResolvedSelection:
    """Resolve filter context into concrete detection IDs / filenames.

    This delegates to surface-specific resolvers that replicate
    the exact same logic used for rendering the UI pages.
    """
    resolvers = {
        "gallery": _resolve_gallery,
        "species_overview": _resolve_species_overview,
        "edit": _resolve_edit,
        "review_queue": _resolve_review_queue,
        "trash": _resolve_trash,
    }
    resolver = resolvers.get(ctx.surface)
    if resolver is None:
        logger.error(f"Unknown surface: {ctx.surface}")
        return ResolvedSelection()

    return resolver(ctx)


# ---------------------------------------------------------------------------
# Surface-specific resolvers
# ---------------------------------------------------------------------------


def _resolve_gallery(ctx: FilterContext) -> ResolvedSelection:
    """Gallery (daily subgallery) — observation-based, filtered by min_score."""
    if not ctx.date:
        return ResolvedSelection()

    config = get_config()
    threshold = (
        ctx.min_score if ctx.min_score > 0 else config["GALLERY_DISPLAY_THRESHOLD"]
    )

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(conn, ctx.date, order_by="time")

    ids = []
    for row in rows:
        row_dict = dict(row)
        score = row_dict.get("score") or 0.0
        if threshold > 0 and score < threshold:
            continue
        ids.append(row_dict["detection_id"])

    return ResolvedSelection(detection_ids=ids, total_count=len(ids))


def _resolve_species_overview(ctx: FilterContext) -> ResolvedSelection:
    """Species overview — all detections for one species, filtered by min_score."""
    if not ctx.species_key:
        return ResolvedSelection()

    config = get_config()
    threshold = (
        ctx.min_score if ctx.min_score > 0 else config["GALLERY_DISPLAY_THRESHOLD"]
    )

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(conn, order_by="time")

    ids = []
    for row in rows:
        row_dict = dict(row)
        det_species = (
            row_dict.get("cls_class_name")
            or row_dict.get("od_class_name")
            or "Unknown_species"
        )
        if det_species != ctx.species_key:
            continue
        score = row_dict.get("score") or 0.0
        if threshold > 0 and score < threshold:
            continue
        ids.append(row_dict["detection_id"])

    return ResolvedSelection(detection_ids=ids, total_count=len(ids))


def _resolve_edit(ctx: FilterContext) -> ResolvedSelection:
    """Edit page — date-filtered detections with status/species/conf filters."""
    if not ctx.date:
        return ResolvedSelection()

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(conn, ctx.date, order_by="time")

    ids = []
    for row in rows:
        row_dict = dict(row)

        # Status filter
        is_downloaded = bool(row_dict.get("downloaded_timestamp"))
        if ctx.status_filter == "downloaded" and not is_downloaded:
            continue
        if ctx.status_filter == "not_downloaded" and is_downloaded:
            continue

        # Species filter
        if ctx.species_key and ctx.species_key != "all":
            sp = (
                row_dict.get("cls_class_name")
                or row_dict.get("od_class_name")
                or "Unknown"
            )
            if sp != ctx.species_key:
                continue

        # Confidence filter
        conf = max(
            row_dict.get("od_confidence") or 0, row_dict.get("cls_confidence") or 0
        )
        if conf < ctx.min_conf:
            continue

        # Score filter
        score = row_dict.get("score") or 0.0
        if ctx.min_score > 0 and score < ctx.min_score:
            continue

        ids.append(row_dict["detection_id"])

    return ResolvedSelection(detection_ids=ids, total_count=len(ids))


def _resolve_review_queue(ctx: FilterContext) -> ResolvedSelection:
    """Review queue — orphans + low-confidence, returns filenames (no detection IDs)."""
    config = get_config()
    threshold = config.get("GALLERY_DISPLAY_THRESHOLD", 0.7)

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_review_queue_images(conn, gallery_threshold=threshold)

    filenames = [dict(r)["filename"] for r in rows]
    return ResolvedSelection(image_filenames=filenames, total_count=len(filenames))


def _resolve_trash(ctx: FilterContext) -> ResolvedSelection:
    """Trash surface — rejected detections + no_bird images."""
    with db_service.closing_connection() as conn:
        # Rejected detections
        det_rows = conn.execute(
            "SELECT detection_id FROM detections WHERE status = 'rejected'"
        ).fetchall()
        det_ids = [r["detection_id"] for r in det_rows]

        # no_bird images (filenames only)
        img_rows = conn.execute(
            "SELECT filename FROM images WHERE review_status = 'no_bird'"
        ).fetchall()
        filenames = [r["filename"] for r in img_rows]

    return ResolvedSelection(
        detection_ids=det_ids,
        image_filenames=filenames,
        total_count=len(det_ids) + len(filenames),
    )
