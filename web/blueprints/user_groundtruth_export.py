"""User-groundtruth export blueprint.

Three endpoints (all login-required):

- ``GET  /admin/groundtruth-export`` — standalone page that shows the
  current pending counts per bucket, the last batch's ``until_at``,
  and a Build button.
- ``GET  /admin/groundtruth-export/dry-run`` — HTML dry-run of the
  exact next batch contents, grouped by bucket, without recording a
  batch row.
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
from pathlib import Path
from typing import Any

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
    exclude_image_from_export,
    last_batch_until,
    move_image_to_wmb_trash_and_exclude,
    preview_counts,
    record_batch_exported,
    stream_batch_zip,
)
from web.species_thumbnails import resolve_static_species_asset_url

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
    return _path_manager().get_original_path


def _path_manager() -> PathManager:
    output_dir = _shared.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise RuntimeError(
            "user_groundtruth_export blueprint not initialized with output_dir"
        )
    return PathManager(output_dir)


def _image_path_in_zip(filename: str, pm: PathManager) -> str:
    date_folder = pm.extract_date_from_filename(filename)
    return f"images/{date_folder}/{filename}"


def _original_url(filename: str, pm: PathManager) -> str:
    date_folder = pm.extract_date_from_filename(filename)
    return f"/uploads/originals/{date_folder}/{filename}"


def _bbox_text(row: dict[str, Any]) -> str:
    parts = [row.get("bbox_x"), row.get("bbox_y"), row.get("bbox_w"), row.get("bbox_h")]
    if any(part is None for part in parts):
        return "missing"
    return ", ".join(f"{float(part):.3f}" for part in parts)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dry_run_bbox_view_fields(row: dict[str, Any]) -> dict[str, str]:
    x = _float_or_none(row.get("bbox_x"))
    y = _float_or_none(row.get("bbox_y"))
    w = _float_or_none(row.get("bbox_w"))
    h = _float_or_none(row.get("bbox_h"))
    if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
        return {}

    x = _clamp(x, 0.0, 1.0)
    y = _clamp(y, 0.0, 1.0)
    w = _clamp(w, 0.0, 1.0 - x)
    h = _clamp(h, 0.0, 1.0 - y)
    if w <= 0 or h <= 0:
        return {}

    span = _clamp(max(w, h) * 4.0, 0.16, 0.58)
    span = max(span, w, h)
    left = _clamp(x + (w / 2) - (span / 2), 0.0, max(0.0, 1.0 - span))
    top = _clamp(y + (h / 2) - (span / 2), 0.0, max(0.0, 1.0 - span))
    bg_x = 0.0 if span >= 1.0 else (left / (1.0 - span)) * 100.0
    bg_y = 0.0 if span >= 1.0 else (top / (1.0 - span)) * 100.0

    full_bbox_style = (
        f"left: {x * 100:.4f}%; top: {y * 100:.4f}%; "
        f"width: {w * 100:.4f}%; height: {h * 100:.4f}%;"
    )
    crop_bbox_style = (
        f"left: {(x - left) / span * 100:.4f}%; "
        f"top: {(y - top) / span * 100:.4f}%; "
        f"width: {w / span * 100:.4f}%; "
        f"height: {h / span * 100:.4f}%;"
    )
    return {
        "bbox_style": full_bbox_style,
        "crop_bg_size": f"{100.0 / span:.4f}% {100.0 / span:.4f}%",
        "crop_bg_position": f"{bg_x:.4f}% {bg_y:.4f}%",
        "crop_bbox_style": crop_bbox_style,
    }


def _display_label(row: dict[str, Any]) -> str:
    bucket = row.get("bucket")
    if bucket == "hard_negatives":
        return "No bird / false-positive"
    if bucket == "species_relabels":
        corrected = row.get("user_corrected_species") or "Unknown species"
        predicted = row.get("model_predicted_species") or "unknown"
        return f"{corrected} (was {predicted})"
    if bucket == "favorites":
        species = row.get("species") or "Unknown species"
        return f"{species} (favorite)"
    return row.get("species") or "Unknown species"


def _species_key_for_row(row: dict[str, Any]) -> str:
    bucket = row.get("bucket")
    if bucket == "species_relabels":
        return str(row.get("user_corrected_species") or "").strip()
    if bucket in ("confirmed_positives", "favorites"):
        return str(row.get("species") or "").strip()
    return ""


def _dry_run_avatar_fields(row: dict[str, Any]) -> dict[str, str]:
    species_key = _species_key_for_row(row)
    if species_key:
        return {
            "avatar_kind": "species",
            "avatar_species": species_key,
            "avatar_url": resolve_static_species_asset_url(species_key),
            "avatar_initial": species_key[:1].upper(),
        }
    if row.get("bucket") == "hard_negatives":
        return {
            "avatar_kind": "no_bird",
            "avatar_species": "",
            "avatar_url": "",
            "avatar_initial": "",
        }
    return {
        "avatar_kind": "unknown",
        "avatar_species": "",
        "avatar_url": "",
        "avatar_initial": "?",
    }


def _sibling_display_label(sibling: dict[str, Any]) -> str:
    return str(
        sibling.get("manual_species_override")
        or sibling.get("raw_species_name")
        or "Unknown species"
    )


def _decorate_dry_run_rows(
    rows: list[dict[str, Any]],
    *,
    pm: PathManager,
    image_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for row in rows:
        filename = str(row.get("image_filename") or "")
        image_path = image_paths.get(filename)
        item = dict(row)
        item["display_label"] = _display_label(row)
        item["bbox_text"] = _bbox_text(row)
        item["thumb_url"] = f"/api/review-thumb/{filename}"
        item["original_url"] = _original_url(filename, pm)
        item["image_path_in_zip"] = _image_path_in_zip(filename, pm)
        item["image_exists"] = bool(image_path and image_path.is_file())
        item.update(_dry_run_bbox_view_fields(row))
        item.update(_dry_run_avatar_fields(row))
        decorated.append(item)
    return decorated


def _decorate_frame_integrity_drops(
    drops: list[dict[str, Any]],
    *,
    pm: PathManager,
    image_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for drop in drops:
        filename = str(drop.get("image_filename") or "")
        image_path = image_paths.get(filename) or pm.get_original_path(filename)
        item = dict(drop)
        item["display_label"] = "Frame blocked by integrity check"
        item["original_url"] = _original_url(filename, pm)
        item["image_path_in_zip"] = _image_path_in_zip(filename, pm)
        item["image_exists"] = bool(image_path and image_path.is_file())

        siblings: list[dict[str, Any]] = []
        for sibling in item.get("active_siblings") or []:
            sibling_item = dict(sibling)
            sibling_item["display_label"] = _sibling_display_label(sibling_item)
            sibling_item.update(_dry_run_bbox_view_fields(sibling_item))
            siblings.append(sibling_item)

        if siblings:
            item["frame_width"] = siblings[0].get("frame_width")
            item["frame_height"] = siblings[0].get("frame_height")
        item["active_siblings"] = siblings
        decorated.append(item)
    return decorated


def _dry_run_buckets(batch, pm: PathManager) -> list[dict[str, Any]]:
    specs = [
        (
            "hard_negatives",
            "Kein Vogel (Training Hard-negatives)",
            "Frames marked No Bird. These ship as negative images with no COCO annotation.",
            batch.hard_negatives,
        ),
        (
            "confirmed_positives",
            "Confirmed positives",
            "Species labels explicitly confirmed by the operator.",
            batch.confirmed_positives,
        ),
        (
            "species_relabels",
            "Species re-labels",
            "Classifier species corrections. The corrected species is exported.",
            batch.species_relabels,
        ),
        (
            "favorites",
            "Favorites",
            "Heart-clicked gold labels. These are the strongest positive anchors.",
            batch.favorites,
        ),
    ]
    return [
        {
            "key": key,
            "label": label,
            "description": description,
            "rows": _decorate_dry_run_rows(
                rows,
                pm=pm,
                image_paths=batch._image_paths,
            ),
        }
        for key, label, description, rows in specs
    ]


def _dry_run_tab_groups(
    buckets: list[dict[str, Any]],
    *,
    frame_integrity_dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets_by_key = {bucket["key"]: bucket for bucket in buckets}
    specs = [
        {
            "source_key": "confirmed_positives",
            "key": "confirmed_positives",
            "label": "Confirmed positives",
            "title": "Confirmed positives",
        },
        {
            "source_key": "species_relabels",
            "key": "species_relabels",
            "label": "Species re-labels",
            "title": "Species re-labels",
        },
        {
            "source_key": "favorites",
            "key": "favorites",
            "label": "Favorites",
            "title": "Favorites",
        },
        {
            "source_key": "hard_negatives",
            "key": "hard_negatives",
            "label": "FPs / Kein Vogel",
            "title": "Kein Vogel (Training Hard-negatives)",
        },
    ]
    groups: list[dict[str, Any]] = []
    if frame_integrity_dropped:
        groups.append(
            {
                "key": "frame_integrity_dropped",
                "label": "Needs cleanup",
                "title": "Needs cleanup (not exported)",
                "kind": "frame_integrity_dropped",
                "rows": frame_integrity_dropped,
            }
        )
    for spec in specs:
        bucket = buckets_by_key[spec["source_key"]]
        rows = []
        for row in bucket["rows"]:
            item = dict(row)
            item["export_category"] = bucket["label"]
            rows.append(item)
        groups.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "title": spec["title"],
                "kind": "export_bucket",
                "rows": rows,
            }
        )
    return groups


def _valid_image_filename(image_filename: str) -> bool:
    return bool(
        image_filename
        and Path(image_filename).name == image_filename
        and "/" not in image_filename
        and "\\" not in image_filename
    )


def _request_reason(data: dict[str, Any], fallback: str) -> str:
    reason = str(data.get("reason") or fallback).strip()
    if len(reason) > 240:
        reason = reason[:240]
    return reason


@user_groundtruth_export_bp.route("/admin/groundtruth-export", methods=["GET"])
@login_required
def export_page() -> Response:
    """Render the standalone groundtruth-export page.

    The page is mostly empty HTML scaffold; the live data (counts,
    last batch) is fetched via /api/groundtruth-export/preview after
    the page mounts. Keeps the SSR path fast and avoids the
    blueprint's GET path touching the DB.
    """
    return render_template(
        "user_groundtruth_export.html",
        current_path="/admin/groundtruth-export",
    )


@user_groundtruth_export_bp.route("/admin/groundtruth-export/dry-run", methods=["GET"])
@login_required
def dry_run_page() -> Response:
    """Render the exact next export batch without recording it.

    Query params mirror the build endpoint:
        since (optional): override default last-batch window start.
        until (optional): freeze the dry-run window end. If absent,
            ``build_batch`` uses the current UTC timestamp and the
            rendered Build button reuses that timestamp so the ZIP
            matches what the operator reviewed.
    """
    try:
        pm = _path_manager()
    except RuntimeError as e:
        logger.error("groundtruth-export not initialized: %s", e)
        return jsonify({"status": "error", "message": "server misconfigured"}), 500

    override_since = request.args.get("since") or None
    override_until = request.args.get("until") or None
    app_version = str(_shared.get("app_version") or "")

    with db_conn.closing_connection() as conn:
        since = override_since if override_since is not None else last_batch_until(conn)
        batch = build_batch(
            conn,
            path_resolver=pm.get_original_path,
            since=since,
            until=override_until,
            wmb_app_version=app_version,
        )

    distinct_filenames = {
        row["image_filename"]
        for bucket in (
            batch.hard_negatives,
            batch.confirmed_positives,
            batch.species_relabels,
            batch.favorites,
        )
        for row in bucket
    }
    missing_images = sorted(
        filename for filename, path in batch._image_paths.items() if not path.is_file()
    )
    buckets = _dry_run_buckets(batch, pm)
    frame_integrity_dropped = _decorate_frame_integrity_drops(
        batch.frame_integrity_dropped,
        pm=pm,
        image_paths=batch._image_paths,
    )

    return render_template(
        "user_groundtruth_export_dry_run.html",
        current_path="/admin/groundtruth-export",
        batch=batch,
        tab_groups=_dry_run_tab_groups(
            buckets,
            frame_integrity_dropped=frame_integrity_dropped,
        ),
        distinct_image_count=len(distinct_filenames),
        missing_images=missing_images,
        has_dryrun_rows=bool(batch.total_rows or frame_integrity_dropped),
    )


@user_groundtruth_export_bp.route("/api/groundtruth-export/preview", methods=["GET"])
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
    "/api/groundtruth-export/exclusions", methods=["POST"]
)
@login_required
def exclude_from_export() -> Response:
    """Quarantine an unsafe image from user-groundtruth export."""
    data = request.get_json(silent=True) or {}
    scope = str(data.get("scope") or "image").strip()
    image_filename = str(data.get("image_filename") or "").strip()
    reason = _request_reason(data, "dry-run excluded by operator")

    if scope != "image":
        return jsonify({"status": "error", "message": "unsupported scope"}), 400
    if not _valid_image_filename(image_filename):
        return jsonify({"status": "error", "message": "invalid image filename"}), 400

    with db_conn.closing_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM images WHERE filename = ?",
            (image_filename,),
        ).fetchone()
        if row is None:
            return jsonify({"status": "error", "message": "image not found"}), 404
        inserted = exclude_image_from_export(
            conn,
            image_filename=image_filename,
            reason=reason,
        )

    return jsonify(
        {
            "status": "ok",
            "scope": "image",
            "image_filename": image_filename,
            "inserted": inserted,
        }
    )


@user_groundtruth_export_bp.route(
    "/api/groundtruth-export/trash-image", methods=["POST"]
)
@login_required
def trash_image_from_export() -> Response:
    """Move a dry-run image to WMB Trash and suppress it from export."""
    data = request.get_json(silent=True) or {}
    image_filename = str(data.get("image_filename") or "").strip()
    reason = _request_reason(data, "dry-run moved to WMB Trash by operator")

    if not _valid_image_filename(image_filename):
        return jsonify({"status": "error", "message": "invalid image filename"}), 400

    with db_conn.closing_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM images WHERE filename = ?",
            (image_filename,),
        ).fetchone()
        if row is None:
            return jsonify({"status": "error", "message": "image not found"}), 404
        result = move_image_to_wmb_trash_and_exclude(
            conn,
            image_filename=image_filename,
            reason=reason,
        )

    return jsonify(
        {
            "status": "ok",
            "scope": "image",
            "image_filename": image_filename,
            **result,
        }
    )


@user_groundtruth_export_bp.route("/api/groundtruth-export/build", methods=["POST"])
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
        since = override_since if override_since is not None else last_batch_until(conn)
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
        batch.batch_id,
        batch.total_rows,
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
