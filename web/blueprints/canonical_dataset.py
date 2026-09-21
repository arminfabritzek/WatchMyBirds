"""Read-only preview and manual download of the canonical label bundle."""

from __future__ import annotations

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
)

from utils.path_manager import PathManager
from web import view_helpers
from web.blueprints.auth import login_required
from web.services import canonical_dataset_service, db_service

canonical_dataset_bp = Blueprint("canonical_dataset", __name__)
_shared: dict[str, str] = {}


def init_canonical_dataset_bp(*, output_dir: str) -> None:
    _shared["output_dir"] = output_dir


def _path_manager() -> PathManager:
    output_dir = _shared.get("output_dir", "").strip()
    if not output_dir:
        raise RuntimeError("canonical dataset blueprint not initialized")
    return PathManager(output_dir)


def _build():
    path_manager = _path_manager()
    with db_service.closing_connection() as conn:
        bundle = canonical_dataset_service.build_bundle(
            conn,
            path_resolver=path_manager.get_original_path,
        )
    return bundle, path_manager


@canonical_dataset_bp.route("/admin/canonical-dataset")
@login_required
def canonical_dataset_page():
    bundle, _ = _build()
    return render_template(
        "canonical_dataset.html",
        bundle=bundle,
        included=[row for row in bundle.manifest if row["decision"] == "included"],
        excluded=[row for row in bundle.manifest if row["decision"] == "excluded"],
        cls_ready_count=len(bundle.classifier_ready),
        od_ready_count=len(bundle.coco["annotations"]),
        missing_original_bird_count=canonical_dataset_service.missing_original_bird_count(
            bundle
        ),
        needs_box_verdict_count=canonical_dataset_service.needs_box_verdict_count(
            bundle
        ),
    )


@canonical_dataset_bp.route("/admin/canonical-dataset/box-walkthrough")
@login_required
def box_walkthrough_page():
    """One detection at a time: confirm or correct its box, then advance.

    Reachable only from the dataset page's gap figure — never a queue with
    a claim on the operator's attention. ``skip`` names detection IDs to
    pass over without writing anything, so a skipped row can reappear on
    a later visit but not immediately in the same pass.
    """
    bundle, _ = _build()
    candidate_ids = canonical_dataset_service.box_walkthrough_candidate_ids(bundle)
    skipped = {
        int(value) for value in request.args.getlist("skip") if value.strip().isdigit()
    }
    remaining = [
        detection_id for detection_id in candidate_ids if detection_id not in skipped
    ]

    if not remaining:
        return render_template(
            "canonical_dataset_box_walkthrough.html",
            det=None,
            remaining_count=0,
        )

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(
            conn, detection_ids=[remaining[0]], order_by="time"
        )
    if not rows:
        # The candidate vanished between listing and fetch (deleted,
        # trashed) — treat it like a skip and try the next one.
        skip_args = [str(value) for value in sorted(skipped | {remaining[0]})]
        return redirect(
            "/admin/canonical-dataset/box-walkthrough?"
            + "&".join(f"skip={value}" for value in skip_args)
        )

    det = view_helpers.build_detection_view_from_gallery_row(dict(rows[0]))
    skip_query = "&".join(f"skip={value}" for value in sorted(skipped))
    return render_template(
        "canonical_dataset_box_walkthrough.html",
        det=det,
        can_moderate=bool(session.get("authenticated")),
        remaining_count=len(remaining),
        skip_query=skip_query,
    )


@canonical_dataset_bp.route("/api/canonical-dataset/preview")
@login_required
def canonical_dataset_preview():
    bundle, _ = _build()
    return jsonify(
        {
            "status": "success",
            "bundle_id": bundle.bundle_id,
            "counts": bundle.counts,
            "snapshot": bundle.snapshot,
            "manifest": bundle.manifest,
        }
    )


@canonical_dataset_bp.route("/api/canonical-dataset/download")
@login_required
def canonical_dataset_download():
    bundle, path_manager = _build()
    tmp_dir, archive_path = (
        canonical_dataset_service.render_canonical_bundle_to_tempdir(
            bundle,
            path_resolver=path_manager.get_original_path,
            temporary_root=path_manager.backup_dir,
        )
    )

    def _cleanup() -> None:
        archive_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    try:
        response = send_file(
            archive_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"watchmybirds-labels-{bundle.bundle_id[:12]}.zip",
            max_age=0,
        )
    except BaseException:
        _cleanup()
        raise
    response.direct_passthrough = False
    response.call_on_close(_cleanup)
    return response
