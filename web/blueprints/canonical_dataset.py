"""Training preview, recorded exports, streaming downloads and receipts."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask.blueprints import BlueprintSetupState

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


@canonical_dataset_bp.record_once
def init_export_jobs(state: BlueprintSetupState) -> None:
    state.app.extensions["training_export_jobs"] = (
        canonical_dataset_service.create_jobs(_shared["output_dir"])
    )


def _jobs() -> canonical_dataset_service.TrainingExportJobs:
    return current_app.extensions["training_export_jobs"]


def _build():
    path_manager = _path_manager()
    with db_service.closing_connection() as conn:
        after = request.args.get("after", "")
        if len(after) > 255 or (after and Path(after).name != after):
            abort(400, "Invalid preview cursor")
        bundle, page = canonical_dataset_service.preview_page(
            conn,
            path_resolver=canonical_dataset_service.original_resolver(path_manager),
            after=after,
        )
    return bundle, path_manager, page


@canonical_dataset_bp.route("/admin/canonical-dataset")
@login_required
def canonical_dataset_page():
    bundle, paths, page = _build()
    before = request.args.get("export_before", "")
    download_id = request.args.get("download", "")
    for value in (before, download_id):
        if value and (
            len(value) != 32 or any(c not in "0123456789abcdef" for c in value)
        ):
            abort(400, "Invalid export identifier")
    with db_service.closing_connection() as conn:
        waiting = canonical_dataset_service.waiting_revisions(
            conn,
            filenames=page["filenames"],
            media_exists=lambda name: canonical_dataset_service.original_resolver(
                paths
            )(name).is_file(),
        )
        export_rows = canonical_dataset_service.list_exports(
            conn, limit=21, before=before
        )
        exports = export_rows[:20]
    return render_template(
        "canonical_dataset.html",
        bundle=bundle,
        waiting=waiting,
        waiting_photos=len({row["image_filename"] for row in waiting}),
        exports=exports,
        export_before=before,
        next_export_before=exports[-1]["export_id"] if len(export_rows) > 20 else None,
        preview_page=page,
        jobs=_jobs().recent(),
        auto_job=request.args.get("job", ""),
        download_export_id=download_id or None,
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
    bundle, _, page = _build()
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
            preview_page=page,
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
            url_for(
                "canonical_dataset.box_walkthrough_page",
                after=page["after"],
                skip=skip_args,
            )
        )

    det = view_helpers.build_detection_view_from_gallery_row(dict(rows[0]))
    skip_query = "&".join(f"skip={value}" for value in sorted(skipped))
    if page["after"]:
        skip_query = urlencode({"after": page["after"]}) + (
            "&" + skip_query if skip_query else ""
        )
    return render_template(
        "canonical_dataset_box_walkthrough.html",
        det=det,
        can_moderate=bool(session.get("authenticated")),
        remaining_count=len(remaining),
        skip_query=skip_query,
        preview_page=page,
    )


@canonical_dataset_bp.route("/api/canonical-dataset/preview")
@login_required
def canonical_dataset_preview():
    bundle, _, page = _build()
    return jsonify(
        {
            "status": "success",
            "bundle_id": bundle.bundle_id,
            "counts": bundle.counts,
            "snapshot": bundle.snapshot,
            "manifest": bundle.manifest,
            "page": page,
        }
    )


def _download_response(
    bundle: canonical_dataset_service.CanonicalDataset,
    paths: PathManager,
    media_map: dict[str, str] | None = None,
) -> Response:
    resolve = canonical_dataset_service.original_resolver(paths)
    media_map = media_map or {}
    for filename in bundle.media_filenames:
        if not resolve(media_map.get(filename, filename)).is_file():
            abort(409, "An original required by this export is missing")
    return Response(
        canonical_dataset_service.stream_canonical_bundle(
            bundle, path_resolver=lambda name: resolve(media_map.get(name, name))
        ),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="watchmybirds-labels-{bundle.bundle_id[:12]}.zip"',
            "Cache-Control": "no-store",
        },
    )


def _queue_export(selection: set[str] | None = None) -> str:
    try:
        return _jobs().start(selection)
    except ValueError as error:
        abort(409, str(error))


@canonical_dataset_bp.route("/api/canonical-dataset/download", methods=["POST"])
@login_required
def canonical_dataset_download() -> tuple[Response, int]:
    selection = None
    if request.is_json:
        payload = request.get_json(silent=True)
        values = payload.get("selection") if isinstance(payload, dict) else None
        if (
            not isinstance(values, list)
            or len(values) > 10000
            or not all(isinstance(value, str) and len(value) <= 50 for value in values)
        ):
            abort(400, "selection must be a list of answer keys")
        selection = set(values)
    job_id = _queue_export(selection)
    status_url = url_for("canonical_dataset.export_job_status", job_id=job_id)
    response = jsonify(job_id=job_id, status_url=status_url)
    response.headers["Location"] = status_url
    return response, 202


@canonical_dataset_bp.route("/api/canonical-dataset/exports", methods=["POST"])
@login_required
def canonical_dataset_prepare() -> Response:
    return redirect(
        url_for("canonical_dataset.canonical_dataset_page", job=_queue_export()),
        code=303,
    )


@canonical_dataset_bp.get("/api/canonical-dataset/jobs/<job_id>")
@login_required
def export_job_status(job_id: str) -> Response:
    try:
        return jsonify(_jobs().status(job_id))
    except LookupError:
        abort(404)


@canonical_dataset_bp.post("/api/canonical-dataset/jobs/<job_id>/<action>")
@login_required
def export_job_action(job_id: str, action: str) -> Response:
    try:
        if action == "cancel":
            _jobs().cancel(job_id)
        elif action == "resume":
            _jobs().retry(job_id)
        else:
            abort(404)
    except (ValueError, LookupError) as error:
        abort(409, str(error))
    return redirect(
        url_for("canonical_dataset.canonical_dataset_page", job=job_id), code=303
    )


@canonical_dataset_bp.post("/api/canonical-dataset/exports/<export_id>/remove-archive")
@login_required
def export_remove_archive(export_id: str) -> Response:
    try:
        _jobs().remove_archive(export_id)
    except (ValueError, LookupError) as error:
        abort(409, str(error))
    return redirect(url_for("canonical_dataset.canonical_dataset_page"), code=303)


@canonical_dataset_bp.post("/api/canonical-dataset/exports/<export_id>/rebuild")
@login_required
def export_rebuild(export_id: str) -> Response:
    try:
        job_id = _jobs().start(rebuild_export_id=export_id)
    except ValueError as error:
        abort(409, str(error))
    return redirect(
        url_for("canonical_dataset.canonical_dataset_page", job=job_id), code=303
    )


@canonical_dataset_bp.route("/api/canonical-dataset/exports/<export_id>/download")
@login_required
def canonical_dataset_download_again(export_id: str) -> Response:
    try:
        artifact = _jobs().open_download(export_id)
    except FileNotFoundError:
        abort(410, "The prepared ZIP is no longer available")
    if artifact is not None:
        path, lease = artifact
        try:
            info = os.fstat(lease.fileno())
            response = send_file(
                lease,
                as_attachment=True,
                download_name=f"watchmybirds-labels-{export_id}.zip",
                mimetype="application/zip",
                last_modified=info.st_mtime,
            )
            response.content_length = info.st_size
            response.set_etag(f"{export_id}-{info.st_size}-{info.st_mtime_ns}")
            response.make_conditional(
                request.environ, accept_ranges=True, complete_length=info.st_size
            )
            response.direct_passthrough = False
            response.call_on_close(lease.close)
            return response
        except BaseException:
            lease.close()
            raise
    with db_service.closing_connection() as conn:
        try:
            bundle, media_map = canonical_dataset_service.load_export(conn, export_id)
        except LookupError:
            abort(404)
        except ValueError as error:
            abort(409, str(error))
    return _download_response(bundle, _path_manager(), media_map)


@canonical_dataset_bp.route(
    "/api/canonical-dataset/exports/<export_id>/received", methods=["POST"]
)
@login_required
def canonical_dataset_received(export_id: str) -> Response:
    with db_service.closing_connection() as conn:
        try:
            canonical_dataset_service.receive_export(conn, export_id)
        except LookupError:
            abort(404)
    return redirect(url_for("canonical_dataset.canonical_dataset_page"), code=303)
