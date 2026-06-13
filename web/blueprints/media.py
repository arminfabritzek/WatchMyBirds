import io
import re

from flask import Blueprint, abort, send_file

from config import get_config
from logging_config import get_logger
from web import view_helpers
from web.blueprints.auth import login_required
from web.services import gallery_service, path_service

logger = get_logger(__name__)
config = get_config()

media_bp = Blueprint("media", __name__)


def _output_dir() -> str:
    return config["OUTPUT_DIR"]


def _path_mgr():
    return path_service.get_path_manager(_output_dir())


@media_bp.route("/uploads/originals/<path:filename>")
def serve_original(filename):
    return view_helpers.send_contained_upload(
        _path_mgr().originals_dir,
        filename,
        max_age=view_helpers.IMAGE_CACHE_SECONDS,
        private=True,
        immutable=True,
    )


@media_bp.route("/api/image/download/<int:detection_id>")
def download_image_with_metadata(detection_id):
    from web.services import metadata_export_service as mx

    resolved = mx.resolve_image_for_detection(detection_id)
    if resolved is None:
        abort(404)
    image_filename, timestamp = resolved

    if mx.burn_in_enabled():
        try:
            copy_bytes = mx.produce_copy_bytes(image_filename)
            return send_file(
                io.BytesIO(copy_bytes),
                mimetype="image/jpeg",
                as_attachment=True,
                download_name=mx.export_filename(image_filename, timestamp),
            )
        except FileNotFoundError:
            abort(404)
        except Exception:
            logger.exception(
                "metadata burn-in failed for detection %s; serving raw original",
                detection_id,
            )

    original_path = _path_mgr().get_original_path(image_filename)
    if not original_path.is_file():
        abort(404)
    return send_file(
        str(original_path),
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=image_filename,
    )


@media_bp.route("/uploads/derivatives/thumbs/<path:filename>")
def serve_thumb(filename):
    path_mgr = _path_mgr()
    full_path = path_mgr.thumbs_dir / filename
    if not full_path.exists():
        if gallery_service.regenerate_derivative(_output_dir(), filename, "thumb"):
            if not full_path.exists():
                return "Regeneration failed", 500
        else:
            preview_name = re.sub(r"_crop_\d+\.webp$", "_preview.webp", filename)
            if preview_name != filename:
                return view_helpers.send_contained_upload(
                    path_mgr.thumbs_dir,
                    preview_name,
                    max_age=view_helpers.IMAGE_CACHE_SECONDS,
                    private=True,
                    immutable=True,
                )
            return "Not found and could not regenerate", 404
    return view_helpers.send_contained_upload(
        path_mgr.thumbs_dir,
        filename,
        max_age=view_helpers.IMAGE_CACHE_SECONDS,
        private=True,
        immutable=True,
    )


@media_bp.route("/uploads/derivatives/optimized/<path:filename>")
def serve_optimized(filename):
    path_mgr = _path_mgr()
    full_path = path_mgr.optimized_dir / filename
    if not full_path.exists():
        if gallery_service.regenerate_derivative(_output_dir(), filename, "optimized"):
            if not full_path.exists():
                return "Regeneration failed", 500
        else:
            return "Not found and could not regenerate", 404
    return view_helpers.send_contained_upload(
        path_mgr.optimized_dir,
        filename,
        max_age=view_helpers.IMAGE_CACHE_SECONDS,
        private=True,
        immutable=True,
    )


@media_bp.route("/uploads/derivatives/ptz_snapshots/<path:filename>")
@login_required
def serve_ptz_snapshot(filename):
    return view_helpers.send_contained_upload(
        _path_mgr().ptz_snapshots_dir,
        filename,
        max_age=60,
        private=True,
        immutable=False,
    )
