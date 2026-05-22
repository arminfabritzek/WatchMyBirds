"""User-groundtruth export blueprint.

Three endpoints (all login-required):

- ``GET  /admin/groundtruth-export`` — standalone page that shows the
  current pending counts per bucket, the last batch's ``until_at``,
  and a Build button.
- ``GET  /api/groundtruth-export/preview`` — JSON: pending counts and
  last-batch metadata. Cheap; called on page render and after every
  build.
- ``POST /api/groundtruth-export/build`` — build the batch, stream the
  ZIP, then on the way out write a row into ``export_batches`` so the
  next batch's default ``since`` advances exactly to this one's
  ``until``.

Distinct from ``training_export``: that one is per-detection
approve-driven (operator picks species and counts in the modal).
This one is whole-window driven (everything user-labelled since the
last batch). They can coexist; this plan does not retire the older
endpoint.

Plan reference:
``agent_handoff/workflow/plans/2026-05-22_FEATURE_user-groundtruth-export-for-pipeline-dev.md``
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
from utils.path_manager import PathManager
from web.blueprints.auth import login_required
from web.services.user_groundtruth_export_service import (
    build_batch,
    last_batch_until,
    preview_counts,
    record_batch_exported,
    stream_batch_zip,
)

logger = get_logger(__name__)

user_groundtruth_export_bp = Blueprint("user_groundtruth_export", __name__)

# Populated at register time so the blueprint has access to shared
# app services. Mirrors the training_export pattern so tests can
# inject mock paths without monkey-patching imports.
_shared: dict[str, object] = {}


def init_user_groundtruth_export_bp(
    output_dir: str,
    app_version: str = "",
) -> None:
    """Inject shared dependencies at registration time."""
    _shared["output_dir"] = output_dir
    _shared["app_version"] = app_version


def _resolve_image_path_factory():
    """Return a ``filename -> Path`` resolver for the configured output_dir.

    Constructing PathManager per request is cheap (just stores
    base_dir) and avoids shared mutable state across requests.
    """
    output_dir = _shared.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise RuntimeError(
            "user_groundtruth_export blueprint not initialized with output_dir"
        )
    pm = PathManager(output_dir)
    return pm.get_original_path


@user_groundtruth_export_bp.route("/admin/groundtruth-export", methods=["GET"])
@login_required
def export_page() -> Response:
    """Render the standalone groundtruth-export page.

    The page is mostly empty HTML scaffold; the live data (counts,
    last batch) is fetched via /api/groundtruth-export/preview after
    the page mounts. Keeps the SSR path fast and avoids the
    blueprint's GET path touching the DB.
    """
    return render_template("user_groundtruth_export.html")


@user_groundtruth_export_bp.route(
    "/api/groundtruth-export/preview", methods=["GET"]
)
@login_required
def preview() -> Response:
    """JSON: pending counts per bucket + last-batch metadata.

    Query params:
        since (optional, ISO timestamp): override the default
            ``since`` (which is the most-recent batch's ``until``).
            Useful for "what if I export everything?" preview.

    Response shape:
        {
          "since": "2026-05-15T00:00:00+00:00" | null,
          "last_batch": {
            "batch_id": "...", "until_at": "...", "built_at": "..."
          } | null,
          "pending": {
            "hard_negatives": int,
            "confirmed_positives": int,
            "species_relabels": int
          },
          "total_pending": int,
          "generated_at_utc": "..."
        }
    """
    override_since = request.args.get("since") or None

    with db_conn.closing_connection() as conn:
        last_until = last_batch_until(conn)
        since = override_since if override_since is not None else last_until
        counts = preview_counts(conn, since=since)
        # Pull the full last-batch row for display, not just the until.
        last_row = conn.execute(
            "SELECT batch_id, built_at, until_at, "
            "hard_negatives_count, confirmed_positives_count, "
            "species_relabels_count, favorites_count, "
            "frame_integrity_dropped_count "
            "FROM export_batches ORDER BY built_at DESC LIMIT 1"
        ).fetchone()

    last_batch_dict = None
    if last_row is not None:
        last_batch_dict = {
            "batch_id": last_row["batch_id"],
            "built_at": last_row["built_at"],
            "until_at": last_row["until_at"],
            "counts": {
                "hard_negatives": last_row["hard_negatives_count"],
                "confirmed_positives": last_row["confirmed_positives_count"],
                "species_relabels": last_row["species_relabels_count"],
                "favorites": last_row["favorites_count"],
            },
            "frame_integrity_dropped_count": (
                last_row["frame_integrity_dropped_count"]
            ),
        }

    return jsonify(
        {
            "since": since,
            "last_batch": last_batch_dict,
            "pending": counts,
            "total_pending": sum(counts.values()),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
    )


@user_groundtruth_export_bp.route(
    "/api/groundtruth-export/build", methods=["POST"]
)
@login_required
def build_and_stream() -> Response:
    """Build the batch and stream the ZIP back. Records the batch
    on the way out so the next preview reflects the new ``since``.

    Request body (all optional):
        {
          "since": "2026-05-15T00:00:00+00:00",  // override
          "until": "2026-05-22T20:00:00+00:00",  // override (default now)
          "notes": "weekly batch W21"
        }

    Empty-batch behavior: builds and streams a structurally valid
    ZIP with empty manifests + empty COCO. The operator can decide
    whether to ship it or not. The empty batch IS recorded in
    ``export_batches`` so the time window advances.
    """
    try:
        resolver = _resolve_image_path_factory()
    except RuntimeError as e:
        logger.error("groundtruth-export not initialized: %s", e)
        return jsonify({"status": "error", "message": "server misconfigured"}), 500

    data = request.get_json(silent=True) or {}
    override_since = data.get("since") or None
    override_until = data.get("until") or None
    notes = str(data.get("notes") or "").strip()
    app_version = str(_shared.get("app_version") or "")

    with db_conn.closing_connection() as conn:
        # Default `since` is the previous batch's `until` — chained
        # half-open windows so no row appears in two batches.
        since = (
            override_since
            if override_since is not None
            else last_batch_until(conn)
        )
        batch = build_batch(
            conn,
            path_resolver=resolver,
            since=since,
            until=override_until,
            wmb_app_version=app_version,
        )
        buf = stream_batch_zip(batch)
        record_batch_exported(conn, batch, notes=notes)

    download_name = f"user_groundtruth_{batch.batch_id}.zip"
    logger.info(
        "groundtruth-export built batch=%s rows=%d hn=%d cp=%d rl=%d fav=%d "
        "frame_integrity_dropped=%d",
        batch.batch_id, batch.total_rows,
        batch.counts["hard_negatives"],
        batch.counts["confirmed_positives"],
        batch.counts["species_relabels"],
        batch.counts["favorites"],
        len(batch.frame_integrity_dropped),
    )
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )
