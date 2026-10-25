"""Training preview, recorded exports, streaming downloads and receipts."""

from __future__ import annotations

from flask import (
    Blueprint,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
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
            path_resolver=canonical_dataset_service.original_resolver(path_manager),
        )
    return bundle, path_manager


@canonical_dataset_bp.route("/admin/canonical-dataset")
@login_required
def canonical_dataset_page():
    bundle, paths = _build()
    with db_service.closing_connection() as conn:
        waiting = canonical_dataset_service.waiting_revisions(
            conn,
            media_exists=lambda name: canonical_dataset_service.original_resolver(
                paths
            )(name).is_file(),
        )
        exports = canonical_dataset_service.list_exports(conn)
    return render_template(
        "canonical_dataset.html",
        bundle=bundle,
        waiting=waiting,
        waiting_photos=len({row["image_filename"] for row in waiting}),
        exports=exports,
        download_export_id=next(
            (
                item["export_id"]
                for item in exports
                if item["export_id"] == request.args.get("download")
            ),
            None,
        ),
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


def _create_export(
    selection: set[str] | None = None,
) -> tuple[str, canonical_dataset_service.CanonicalDataset]:
    paths = _path_manager()
    try:
        with db_service.closing_connection() as conn:
            return canonical_dataset_service.create_export(
                conn,
                paths=paths,
                labeling_installation_id=canonical_dataset_service.get_or_create_labeling_installation_id(
                    str(paths.base_dir)
                ),
                selection=selection,
            )
    except ValueError as error:
        abort(409, str(error))


@canonical_dataset_bp.route("/api/canonical-dataset/download", methods=["POST"])
@login_required
def canonical_dataset_download() -> Response:
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
    _, bundle = _create_export(selection)
    return _download_response(bundle, _path_manager())


@canonical_dataset_bp.route("/api/canonical-dataset/exports", methods=["POST"])
@login_required
def canonical_dataset_prepare() -> Response:
    export_id, _ = _create_export()
    return redirect(
        url_for("canonical_dataset.canonical_dataset_page", download=export_id),
        code=303,
    )


@canonical_dataset_bp.route("/api/canonical-dataset/exports/<export_id>/download")
@login_required
def canonical_dataset_download_again(export_id: str) -> Response:
    with db_service.closing_connection() as conn:
        try:
            bundle, media_map = canonical_dataset_service.load_export(conn, export_id)
        except LookupError:
            abort(404)
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
