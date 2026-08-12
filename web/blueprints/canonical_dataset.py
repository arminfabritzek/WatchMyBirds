"""Read-only preview and manual download of the canonical label bundle."""

from __future__ import annotations

import io

from flask import Blueprint, jsonify, render_template, send_file

from utils.path_manager import PathManager
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
    payload = canonical_dataset_service.render_canonical_bundle(
        bundle,
        path_resolver=path_manager.get_original_path,
    )
    return send_file(
        io.BytesIO(payload),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"watchmybirds-labels-{bundle.bundle_id[:12]}.zip",
        max_age=0,
    )
