"""
Moderation Blueprint.

Handles all bulk moderation routes:
- POST /api/moderation/resolve-selection  — Resolve selection to concrete IDs
- POST /api/moderation/bulk/relabel       — Bulk relabel detections
- POST /api/moderation/bulk/reject        — Bulk reject detections / review items
- POST /api/moderation/bulk/rescan        — Queue async rescan jobs
- GET  /api/moderation/rescan-jobs/<job_id>/status — Rescan job progress
- POST /api/moderation/rescan-proposals/<id>/apply — Accept a rescan proposal
"""

from __future__ import annotations

import uuid
from datetime import datetime

from flask import Blueprint, jsonify, request

from config import get_config
from logging_config import get_logger
from utils.species_names import build_species_picker_entries
from web.blueprints.auth import login_required
from web.services import db_service, gallery_service
from web.services.filter_service import FilterContext, resolve_filtered_ids

logger = get_logger(__name__)

moderation_bp = Blueprint("moderation", __name__)


def _parse_iso_date(raw_value: str, field_name: str) -> str:
    """Validate a YYYY-MM-DD date string and return it unchanged."""
    if not raw_value:
        raise ValueError(f"{field_name} required")

    try:
        datetime.strptime(raw_value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {raw_value}") from exc

    return raw_value


# ---------------------------------------------------------------------------
# POST /api/moderation/resolve-selection
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/resolve-selection", methods=["POST"])
@login_required
def resolve_selection() -> tuple:
    """Resolve a selection to concrete detection IDs / filenames.

    Accepts:
        { mode: "explicit" | "all_filtered" | "date_range" | "logical_filter",
          ids: [int, ...],              // only for mode=explicit
          filter_context: { ... },      // only for mode=all_filtered
          from_date: "YYYY-MM-DD",      // only for mode=date_range
          to_date: "YYYY-MM-DD",        // only for mode=date_range
          source_type: "folder_upload"  // only for mode=logical_filter
        }

    Returns:
        { status, detection_ids, image_filenames, total_count }
    """
    data = request.get_json() or {}
    mode = data.get("mode", "explicit")

    if mode == "explicit":
        ids = data.get("ids", [])
        filenames = data.get("filenames", [])
        return jsonify(
            {
                "status": "success",
                "detection_ids": ids,
                "image_filenames": filenames,
                "total_count": len(ids) + len(filenames),
            }
        )

    if mode == "all_filtered":
        raw_ctx = data.get("filter_context")
        if not raw_ctx or not isinstance(raw_ctx, dict):
            return jsonify(
                {"status": "error", "message": "filter_context required"}
            ), 400

        try:
            ctx = FilterContext.from_dict(raw_ctx)
        except (ValueError, KeyError) as exc:
            return jsonify(
                {"status": "error", "message": f"Invalid filter_context: {exc}"}
            ), 400

        result = resolve_filtered_ids(ctx)
        return jsonify(
            {
                "status": "success",
                "detection_ids": result.detection_ids,
                "image_filenames": result.image_filenames,
                "total_count": result.total_count,
            }
        )

    if mode == "date_range":
        try:
            from_date = _parse_iso_date(data.get("from_date"), "from_date")
            to_date = _parse_iso_date(data.get("to_date"), "to_date")
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        if from_date > to_date:
            return jsonify(
                {
                    "status": "error",
                    "message": "from_date must be on or before to_date",
                }
            ), 400

        with db_service.closing_connection() as conn:
            detection_ids = db_service.fetch_active_detection_ids_in_date_range(
                conn, from_date, to_date
            )

        return jsonify(
            {
                "status": "success",
                "detection_ids": detection_ids,
                "image_filenames": [],
                "total_count": len(detection_ids),
            }
        )

    if mode == "logical_filter":
        source_type = data.get("source_type")
        if not source_type:
            return jsonify({"status": "error", "message": "source_type required"}), 400

        with db_service.closing_connection() as conn:
            selection = db_service.fetch_active_detection_selection_by_source_type(
                conn, source_type
            )

        detection_ids = selection.get("detection_ids", [])
        image_count = selection.get("image_count", 0)
        return jsonify(
            {
                "status": "success",
                "detection_ids": detection_ids,
                "image_filenames": [],
                "total_count": len(detection_ids),
                "image_count": image_count,
            }
        )

    return jsonify({"status": "error", "message": f"Unknown mode: {mode}"}), 400


# ---------------------------------------------------------------------------
# POST /api/moderation/bulk/relabel
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/bulk/relabel", methods=["POST"])
@login_required
def bulk_relabel() -> tuple:
    """Bulk relabel detections to a new species.

    Accepts:
        { detection_ids: [int, ...], species: "Scientific_name" }
    """
    data = request.get_json() or {}
    detection_ids = data.get("detection_ids", [])
    new_species = data.get("species")

    if not detection_ids:
        return jsonify({"status": "error", "message": "detection_ids required"}), 400
    if not new_species:
        return jsonify({"status": "error", "message": "species required"}), 400

    with db_service.closing_connection() as conn:
        cfg = get_config()
        locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
        allowed_species = {
            entry["scientific"]
            for entry in build_species_picker_entries(conn, locale=locale)
        }
        if new_species not in allowed_species:
            return jsonify({"status": "error", "message": "unknown species"}), 400

        relabeled = db_service.apply_species_override_many(
            conn, detection_ids, new_species, "manual"
        )
        if not isinstance(relabeled, int):
            relabeled = len(detection_ids)

    # Invalidate gallery cache
    gallery_service.invalidate_cache()

    logger.info(
        f"Bulk relabel: {relabeled} succeeded, 0 failed → {new_species}"
    )
    return jsonify(
        {
            "status": "success",
            "relabeled": relabeled,
            "failed": [],
            "new_species": new_species,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/moderation/bulk/reject
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/bulk/reject", methods=["POST"])
@login_required
def bulk_reject() -> tuple:
    """Bulk reject detections and/or review queue items.

    Accepts:
        { detection_ids: [int, ...], image_filenames: [str, ...] }
    Either or both may be provided.
    """
    data = request.get_json() or {}
    detection_ids = data.get("detection_ids", [])
    image_filenames = data.get("image_filenames", [])

    if not detection_ids and not image_filenames:
        return jsonify({"status": "error", "message": "No targets provided"}), 400

    rejected_detections = 0
    rejected_images = 0

    with db_service.closing_connection() as conn:
        if detection_ids:
            db_service.reject_detections(conn, detection_ids)
            rejected_detections = len(detection_ids)

        if image_filenames:
            rejected_images = db_service.update_review_status(
                conn, image_filenames, "no_bird"
            )

    gallery_service.invalidate_cache()

    logger.info(
        f"Bulk reject: {rejected_detections} detections, {rejected_images} images"
    )
    return jsonify(
        {
            "status": "success",
            "rejected_detections": rejected_detections,
            "rejected_images": rejected_images,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/moderation/bulk/rescan
# ---------------------------------------------------------------------------

# In-memory job tracker (simple first — persists per-process only)
_rescan_jobs: dict[str, dict] = {}


@moderation_bp.route("/api/moderation/bulk/rescan", methods=["POST"])
@login_required
def bulk_rescan() -> tuple:
    """Queue async rescan jobs for given detections/images.

    Accepts:
        { detection_ids: [int, ...] }

    Resolves detection_ids to image filenames and queues each
    for deep analysis via the existing AnalysisQueue.
    Returns a job_id for status polling.
    """
    data = request.get_json() or {}
    detection_ids = data.get("detection_ids", [])

    if not detection_ids:
        return jsonify({"status": "error", "message": "detection_ids required"}), 400

    # Resolve detection_ids → image filenames
    with db_service.closing_connection() as conn:
        placeholders = ",".join("?" for _ in detection_ids)
        rows = conn.execute(
            f"SELECT DISTINCT image_filename FROM detections WHERE detection_id IN ({placeholders})",
            detection_ids,
        ).fetchall()

    filenames = [r["image_filename"] for r in rows]

    if not filenames:
        return jsonify(
            {"status": "error", "message": "No images found for given IDs"}
        ), 404

    job_id = str(uuid.uuid4())[:8]

    # Track the job
    _rescan_jobs[job_id] = {
        "status": "queued",
        "total": len(filenames),
        "queued": 0,
        "skipped": 0,
        "filenames": filenames,
        "detection_ids": detection_ids,
    }

    # Enqueue each filename via the moderation rescan path
    # (writes to rescan_proposals, never to detections)
    from web.services.analysis_service import submit_moderation_rescan

    # Build per-filename detection ID mapping for proposal linking
    fn_to_det_ids: dict[str, list[int]] = {}
    with db_service.closing_connection() as conn:
        id_rows = conn.execute(
            f"SELECT detection_id, image_filename FROM detections WHERE detection_id IN ({placeholders})",
            detection_ids,
        ).fetchall()
        for r in id_rows:
            fn_to_det_ids.setdefault(r["image_filename"], []).append(r["detection_id"])

    queued = 0
    skipped = 0
    for fn in filenames:
        target_ids = fn_to_det_ids.get(fn, [])
        if submit_moderation_rescan(fn, job_id=job_id, target_detection_ids=target_ids):
            queued += 1
        else:
            skipped += 1

    _rescan_jobs[job_id]["queued"] = queued
    _rescan_jobs[job_id]["skipped"] = skipped
    _rescan_jobs[job_id]["status"] = "running" if queued > 0 else "done"

    logger.info(
        f"Bulk rescan job {job_id}: {queued} queued, {skipped} skipped out of {len(filenames)}"
    )

    return jsonify(
        {
            "status": "success",
            "job_id": job_id,
            "queued": queued,
            "skipped": skipped,
            "total": len(filenames),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/moderation/rescan-jobs/<job_id>/status
# ---------------------------------------------------------------------------


@moderation_bp.route("/api/moderation/rescan-jobs/<job_id>/status", methods=["GET"])
@login_required
def rescan_job_status(job_id: str) -> tuple:
    """Returns progress and proposals for a rescan job."""
    job = _rescan_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404

    # Check how many filenames are still pending in the analysis queue
    from core.analysis_queue import analysis_queue

    still_pending = 0
    for fn in job.get("filenames", []):
        with analysis_queue._pending_lock:
            if fn in analysis_queue._pending_filenames:
                still_pending += 1

    if still_pending == 0 and job["status"] == "running":
        job["status"] = "done"

    # Fetch proposals from DB (if rescan_proposals table exists)
    proposals = []
    try:
        with db_service.closing_connection() as conn:
            placeholders = ",".join("?" for _ in job.get("filenames", []))
            if placeholders:
                rows = conn.execute(
                    f"""
                    SELECT proposal_id, image_filename, target_detection_id,
                           suggested_species, suggested_confidence, status
                    FROM rescan_proposals
                    WHERE image_filename IN ({placeholders}) AND job_id = ?
                    """,
                    [*job["filenames"], job_id],
                ).fetchall()
                proposals = [dict(r) for r in rows]
    except Exception:
        # Table may not exist yet — that's OK
        pass

    return jsonify(
        {
            "status": "success",
            "job_status": job["status"],
            "total": job["total"],
            "queued": job["queued"],
            "skipped": job["skipped"],
            "still_pending": still_pending,
            "proposals": proposals,
        }
    )


# ---------------------------------------------------------------------------
# POST /api/moderation/rescan-proposals/<proposal_id>/apply
# ---------------------------------------------------------------------------


@moderation_bp.route(
    "/api/moderation/rescan-proposals/<int:proposal_id>/apply", methods=["POST"]
)
@login_required
def apply_rescan_proposal(proposal_id: int) -> tuple:
    """Accept a rescan proposal — writes the suggested species to the detection."""
    with db_service.closing_connection() as conn:
        row = conn.execute(
            "SELECT * FROM rescan_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()

        if not row:
            return jsonify({"status": "error", "message": "Proposal not found"}), 404

        proposal = dict(row)
        if proposal["status"] == "applied":
            # Idempotent: already applied → success
            return jsonify(
                {
                    "status": "success",
                    "proposal_id": proposal_id,
                    "applied_species": proposal.get("suggested_species"),
                    "already_applied": True,
                }
            )
        if proposal["status"] not in ("ready",):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Proposal is '{proposal['status']}', expected 'ready'",
                    }
                ),
                409,
            )

        target_id = proposal.get("target_detection_id")
        new_species = proposal.get("suggested_species")

        if target_id and new_species:
            db_service.apply_species_override(
                conn, target_id, new_species, "proposal_applied"
            )

        # Mark proposal as applied
        conn.execute(
            "UPDATE rescan_proposals SET status = 'applied', applied_at = datetime('now') WHERE proposal_id = ?",
            (proposal_id,),
        )

    gallery_service.invalidate_cache()

    logger.info(
        f"Rescan proposal {proposal_id} applied: detection {target_id} → {new_species}"
    )
    return jsonify(
        {
            "status": "success",
            "proposal_id": proposal_id,
            "applied_species": new_species,
        }
    )
