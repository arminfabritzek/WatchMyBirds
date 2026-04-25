"""Training-data export blueprint.

Three endpoints:

- ``GET  /api/training-export/available`` — returns the species pool
  for the Export modal (per-species counts of available / pending /
  already-exported rows).
- ``POST /api/training-export/create`` — build a ZIP from the
  selected species_limits, stream it back as a file download. On
  successful stream, rows get marked as 'exported' in the DB.
- ``GET  /admin/export`` — renders the standalone export page
  (linked from the review queue's Export-button).

Auth: all endpoints require ``login_required``.  The export path
is not destructive to bird data but it exposes raw review metadata
(reviewer_id, station_id) so it must not be public.
"""

from __future__ import annotations

from datetime import UTC, datetime

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

from logging_config import get_logger
from utils.db import connection as db_conn
from utils.species_names import build_species_picker_entries
from web.blueprints.auth import login_required
from web.services.training_export_service import (
    DEFAULT_MAX_PER_SPECIES,
    DEFAULT_MAX_TOTAL,
    SpeciesAvailability,
    build_batch_id,
    confirm_bbox_and_mark_pending,
    filter_eligible_for_pool,
    list_species_availability,
    mark_exported,
    select_export_batch,
    stream_export_zip,
)

logger = get_logger(__name__)

training_export_bp = Blueprint("training_export", __name__)

# Populated at register time so the blueprint has access to shared
# app services (PathManager, app config, app version string).
_shared: dict[str, object] = {}


def init_training_export_bp(
    output_dir: str,
    app_config: dict,
    app_version: str = "",
) -> None:
    """Inject shared dependencies at registration time. Keeps the
    blueprint free of the module-level globals the older blueprints
    use, which helps the tests patch things without monkey-patching
    the import graph."""
    _shared["output_dir"] = output_dir
    _shared["config"] = app_config
    _shared["app_version"] = app_version


@training_export_bp.route("/admin/export", methods=["GET"])
@login_required
def export_page() -> Response:
    """Standalone export page. Linked from the review queue's
    Export-button so the modal can pre-fill species counts without a
    full page reload."""
    return render_template("training_export.html")


@training_export_bp.route("/api/training-export/available", methods=["GET"])
@login_required
def available() -> Response:
    """Returns the per-species availability breakdown.

    Shape:
        {
          "species": [
             {"species": "Parus_major", "available": 512, "pending": 0, "already_exported": 0},
             ...
          ],
          "defaults": {
            "max_per_species": 50,
            "max_total": 500
          }
        }
    """
    with db_conn.closing_connection() as conn:
        pool = list_species_availability(conn)

    def _serialize(s: SpeciesAvailability) -> dict:
        return {
            "species": s.species,
            "available": s.available_count,
            "pending": s.pending_count,
            "already_exported": s.already_exported_count,
        }

    return jsonify(
        {
            "species": [_serialize(s) for s in pool],
            "defaults": {
                "max_per_species": DEFAULT_MAX_PER_SPECIES,
                "max_total": DEFAULT_MAX_TOTAL,
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
    )


@training_export_bp.route("/api/training-export/create", methods=["POST"])
@login_required
def create_export() -> Response:
    """Build a training-export ZIP and stream it back.

    Request body:
      {
        "species_limits": {"Parus_major": 50, "Cyanistes_caeruleus": 30},
        "max_total": 500,
        "station_id": "station-01",    // optional; defaults to empty
        "reviewer_id": "user"          // optional
      }

    Response: application/zip, attachment filename per batch id.
    """
    data = request.get_json(silent=True) or {}
    raw_limits = data.get("species_limits") or {}
    if not isinstance(raw_limits, dict):
        return jsonify({"status": "error", "message": "species_limits must be a mapping"}), 400

    # Sanitize species_limits: coerce values to ints, drop non-positive.
    species_limits: dict[str, int] = {}
    for species, cap in raw_limits.items():
        try:
            cap_int = int(cap)
        except (TypeError, ValueError):
            continue
        if cap_int > 0 and isinstance(species, str) and species.strip():
            species_limits[species.strip()] = cap_int

    if not species_limits:
        return jsonify({"status": "error", "message": "no species selected"}), 400

    raw_total = data.get("max_total")
    try:
        max_total: int | None = (
            int(raw_total) if raw_total not in (None, "", 0) else None
        )
    except (TypeError, ValueError):
        max_total = None
    if max_total is not None and max_total <= 0:
        max_total = None

    station_id = str(data.get("station_id") or "").strip()
    reviewer_id = str(data.get("reviewer_id") or "").strip()

    output_dir = _shared.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        logger.error("training_export blueprint not initialized with an output_dir")
        return jsonify({"status": "error", "message": "server misconfigured"}), 500
    app_version = str(_shared.get("app_version") or "")

    with db_conn.closing_connection() as conn:
        selection = select_export_batch(
            conn,
            species_limits=species_limits,
            max_total=max_total,
        )
        if not selection.detection_ids:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "no eligible detections for the given selection",
                    }
                ),
                409,
            )
        zip_buffer, ids_for_persist = stream_export_zip(
            conn,
            selection,
            output_dir=output_dir,
            station_id=station_id,
            reviewer_id=reviewer_id,
            app_version=app_version,
        )
        # Persist exactly the detection_ids that actually made it
        # into the ZIP — frame-integrity may have broadened (pulled
        # in siblings) or narrowed (dropped dropped_ids) the set. The
        # pulled-in siblings MUST be marked exported too, otherwise
        # a future run would ship them again under a new uuid.
        # Frame-dropped ids stay in 'pending' so they can join a
        # later batch once the operator fixes their sibling bboxes.
        exported_ids = ids_for_persist.get("exported_ids") or []
        marked = mark_exported(conn, exported_ids, selection.batch_id)
        logger.info(
            f"training_export: batch={selection.batch_id} "
            f"selected={len(selection.detection_ids)} "
            f"exported={len(exported_ids)} marked={marked} "
            f"pulled_siblings={len(ids_for_persist.get('pulled_in_siblings') or [])} "
            f"dropped={len(ids_for_persist.get('dropped_ids') or [])}"
        )

    download_name = f"{selection.batch_id}.zip"
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@training_export_bp.route("/api/training-export/add", methods=["POST"])
@login_required
def add_to_pool() -> Response:
    """Manually add detection_ids to the training pool.

    Source: the batch-edit UI in gallery (Edit Page). The user
    multi-selects tiles and clicks "Add to training" — those
    detections get promoted to ``pending`` in the pool without
    having to go through the review queue.

    Guardrails:
    - Only Option-A-strict-eligible rows are added. Ineligible rows
      (missing species_override or bbox_review!='correct') are
      silently skipped and reported back to the UI so the operator
      sees "7 of 10 added — 3 not eligible" instead of a bare count.
    - Already-exported rows stay exported (UNIQUE+INSERT-OR-IGNORE
      in mark_pending). Already-pending rows stay pending.

    Request body:
      { "detection_ids": [123, 456, ...] }

    Response:
      { "status": "success",
        "added": 7,          # rows that were freshly marked pending
        "eligible": 7,       # rows passing Option-A-strict
        "ineligible": 3,     # rows filtered out (shown to user)
        "already_in_pool": 0 # rows that were already pending/exported
      }
    """
    data = request.get_json(silent=True) or {}
    raw_ids = data.get("detection_ids") or []
    if not isinstance(raw_ids, list):
        return jsonify({"status": "error", "message": "detection_ids must be a list"}), 400

    ids: list[int] = []
    for value in raw_ids:
        try:
            iid = int(value)
        except (TypeError, ValueError):
            continue
        if iid > 0:
            ids.append(iid)
    if not ids:
        return jsonify({"status": "error", "message": "no valid detection_ids"}), 400

    confirm_current_species = bool(data.get("confirm_current_species"))
    current_species = str(data.get("current_species") or "").strip()

    # auto_confirm_bbox=True treats the caller's click as the
    # bbox-review act. The frame-integrity guard inside
    # filter_eligible_for_pool still prevents mixed-review frames
    # from leaking into the pool.
    with db_conn.closing_connection() as conn:
        if confirm_current_species and current_species and len(ids) == 1:
            app_config = _shared.get("config")
            locale = (
                app_config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
                if isinstance(app_config, dict)
                else "DE"
            )
            allowed_species = {
                entry["scientific"]
                for entry in build_species_picker_entries(conn, locale=locale)
            }
            if current_species not in allowed_species:
                return jsonify({"status": "error", "message": "unknown species"}), 400
            conn.execute(
                """
                UPDATE detections
                SET manual_species_override = ?,
                    species_source = 'manual',
                    species_updated_at = ?
                WHERE detection_id = ?
                  AND COALESCE(status, 'active') = 'active'
                  AND (
                    manual_species_override IS NULL
                    OR TRIM(manual_species_override) = ''
                  )
                """,
                (current_species, datetime.now(UTC).isoformat(), ids[0]),
            )
            conn.commit()
        breakdown = filter_eligible_for_pool(
            conn, ids, auto_confirm_bbox=True
        )
        batch_id = build_batch_id("gallery_batch_add")
        added = confirm_bbox_and_mark_pending(
            conn, breakdown["eligible"], batch_id
        )

    logger.info(
        f"training_export: gallery_batch_add batch={batch_id} "
        f"submitted={len(ids)} eligible={len(breakdown['eligible'])} "
        f"ineligible={len(breakdown['ineligible'])} "
        f"already_in_pool={len(breakdown['already_in_pool'])} added={added}"
    )
    return jsonify(
        {
            "status": "success",
            "added": added,
            "eligible": len(breakdown["eligible"]),
            "ineligible": len(breakdown["ineligible"]),
            "already_in_pool": len(breakdown["already_in_pool"]),
            "batch_id": batch_id,
        }
    )
