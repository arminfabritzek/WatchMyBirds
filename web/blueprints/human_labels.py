"""Canonical human-label answer API."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, jsonify, request

from config import get_config
from core.human_label_core import BBox, HumanAnswer, HumanLabelError
from core.manual_object_core import ManualObjectDraft
from logging_config import get_logger
from utils.path_manager import PathManager
from web.blueprints.auth import login_required
from web.security import safe_validation_message
from web.services import db_service, gallery_service, human_label_service

logger = get_logger(__name__)

human_labels_bp = Blueprint("human_labels", __name__)
_shared: dict[str, str] = {}


def init_human_labels_bp(*, output_dir: str, app_version: str = "") -> None:
    _shared["output_dir"] = output_dir
    _shared["app_version"] = app_version


def _request_bbox(value: object) -> BBox | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HumanLabelError("bbox is invalid")
    try:
        bbox = BBox(
            x=float(value["x"]),
            y=float(value["y"]),
            w=float(value["w"]),
            h=float(value["h"]),
        )
    except (KeyError, TypeError, ValueError):
        raise HumanLabelError("bbox is invalid") from None
    return bbox.validated()


def _manual_object_error(exc: Exception):
    message = safe_validation_message(
        exc,
        allowed_prefixes=(
            "bbox ",
            "image ",
            "manual object ",
            "request_id ",
            "species ",
        ),
        fallback="Manual object rejected",
    )
    return jsonify({"status": "error", "message": message}), 409


@human_labels_bp.route("/api/manual-objects", methods=["POST"])
@login_required
def manual_object_create():
    """Create a missed-bird object with explicit human provenance."""
    data = request.get_json(silent=True) or {}
    try:
        filename = str(data.get("filename") or "").strip()
        bbox = _request_bbox(data.get("bbox"))
        if bbox is None:
            raise HumanLabelError("bbox is required")
        request_id = str(data.get("request_id") or "")
        species_key = data.get("species_key")
        if species_key is not None and not isinstance(species_key, str):
            raise HumanLabelError("species is invalid")
        path_manager = PathManager(str(_shared["output_dir"]))
        with db_service.closing_connection() as conn:
            obj, created = human_label_service.create_manual_object(
                conn,
                ManualObjectDraft(
                    image_filename=filename,
                    bbox=bbox,
                    species_key=species_key,
                    request_id=request_id,
                ),
                original_path=path_manager.get_original_path(filename),
                locale=str(get_config().get("SPECIES_COMMON_NAME_LOCALE", "DE")),
                app_version=_shared.get("app_version", ""),
            )
        gallery_service.invalidate_cache()
    except (HumanLabelError, sqlite3.IntegrityError) as exc:
        logger.info("Manual object create rejected: %s", type(exc).__name__)
        return _manual_object_error(exc)
    return jsonify({"status": "success", "created": created, "object": obj})


@human_labels_bp.route("/api/manual-objects/<int:manual_object_id>", methods=["PATCH"])
@login_required
def manual_object_update(manual_object_id: int):
    """Update only the supplied axes of an existing manual object."""
    data = request.get_json(silent=True) or {}
    try:
        filename = str(data.get("filename") or "").strip()
        expected_revision = int(data.get("expected_revision"))
        bbox = _request_bbox(data.get("bbox"))
        species_supplied = "species_key" in data
        species_key = data.get("species_key")
        if species_key is not None and not isinstance(species_key, str):
            raise HumanLabelError("species is invalid")
        if bbox is None and not species_supplied:
            raise HumanLabelError("manual object update contains no facts")
        with db_service.closing_connection() as conn:
            path_manager = PathManager(str(_shared["output_dir"]))
            obj, changed = human_label_service.update_manual_object(
                conn,
                manual_object_id=manual_object_id,
                image_filename=filename,
                expected_revision=expected_revision,
                locale=str(get_config().get("SPECIES_COMMON_NAME_LOCALE", "DE")),
                app_version=_shared.get("app_version", ""),
                bbox=bbox,
                species_supplied=species_supplied,
                species_key=species_key,
                original_path=(
                    path_manager.get_original_path(filename)
                    if bbox is not None
                    else None
                ),
            )
        gallery_service.invalidate_cache()
    except (HumanLabelError, sqlite3.IntegrityError, TypeError, ValueError) as exc:
        logger.info("Manual object update rejected: %s", type(exc).__name__)
        return _manual_object_error(exc)
    return jsonify({"status": "success", "changed": changed, "object": obj})


@human_labels_bp.route(
    "/api/manual-objects/<int:manual_object_id>/retract", methods=["POST"]
)
@login_required
def manual_object_retract(manual_object_id: int):
    """Retract a manually added bird while retaining its revision history."""
    data = request.get_json(silent=True) or {}
    try:
        filename = str(data.get("filename") or "").strip()
        expected_revision = int(data.get("expected_revision"))
        with db_service.closing_connection() as conn:
            obj = human_label_service.retract_manual_object(
                conn,
                manual_object_id=manual_object_id,
                image_filename=filename,
                expected_revision=expected_revision,
                app_version=_shared.get("app_version", ""),
            )
        gallery_service.invalidate_cache()
    except (HumanLabelError, sqlite3.IntegrityError, TypeError, ValueError) as exc:
        logger.info("Manual object retract rejected: %s", type(exc).__name__)
        return _manual_object_error(exc)
    return jsonify({"status": "success", "object": obj})


@human_labels_bp.route("/api/labels/state", methods=["GET"])
@login_required
def label_state():
    filename = request.args.get("filename", "").strip()
    raw_detection_id = request.args.get("detection_id", "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400
    try:
        detection_id = int(raw_detection_id) if raw_detection_id else None
    except ValueError:
        return jsonify({"status": "error", "message": "invalid detection_id"}), 400

    with db_service.closing_connection() as conn:
        facts = human_label_service.fetch_current_facts(
            conn,
            image_filename=filename,
            detection_id=detection_id,
        )
        readiness = None
        object_progress = None
        if detection_id is not None:
            readiness, object_progress = human_label_service.summarize_object_state(
                conn,
                image_filename=filename,
                detection_id=detection_id,
                facts=facts,
            )
    return jsonify(
        {
            "status": "success",
            "facts": facts,
            "readiness": readiness,
            "object_progress": object_progress,
        }
    )


@human_labels_bp.route("/api/labels/bbox-quality/retract", methods=["POST"])
@login_required
def label_bbox_quality_retract():
    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    try:
        detection_id = int(data.get("detection_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid detection_id"}), 400

    try:
        with db_service.closing_connection() as conn:
            fact_id = human_label_service.retract_bbox_quality(
                conn,
                image_filename=filename,
                detection_id=detection_id,
                source_ref=f"image-correction:{filename}:{detection_id}",
                app_version=_shared.get("app_version", ""),
            )
            conn.commit()
    except HumanLabelError as exc:
        logger.info("Bbox-quality retraction rejected: %s", type(exc).__name__)
        return jsonify(
            {"status": "error", "message": "unable to clear the box verdict"}
        ), 400

    return jsonify({"status": "success", "fact_id": fact_id})


@human_labels_bp.route("/api/labels/species/retract", methods=["POST"])
@login_required
def label_species_retract():
    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    try:
        detection_id = int(data.get("detection_id"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid detection_id"}), 400

    try:
        with db_service.closing_connection() as conn:
            fact_id = human_label_service.retract_species_identity(
                conn,
                image_filename=filename,
                detection_id=detection_id,
                source_ref=f"image-correction:{filename}:{detection_id}",
                app_version=_shared.get("app_version", ""),
            )
            conn.commit()
    except HumanLabelError as exc:
        logger.info("Species retraction rejected: %s", type(exc).__name__)
        return jsonify(
            {"status": "error", "message": "unable to take the species back"}
        ), 400

    return jsonify({"status": "success", "fact_id": fact_id})


@human_labels_bp.route("/api/labels/answer", methods=["POST"])
@login_required
def label_answer():
    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "filename required"}), 400

    raw_detection_id = data.get("detection_id")
    try:
        detection_id = int(raw_detection_id) if raw_detection_id is not None else None
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid detection_id"}), 400

    bbox_data = data.get("bbox_correction")
    try:
        bbox = (
            BBox(
                x=float(bbox_data["x"]),
                y=float(bbox_data["y"]),
                w=float(bbox_data["w"]),
                h=float(bbox_data["h"]),
            )
            if isinstance(bbox_data, dict)
            else None
        )
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid bbox"}), 400

    answer = HumanAnswer(
        image_filename=filename,
        detection_id=detection_id,
        image_bird_presence=data.get("image_bird_presence"),
        object_bird_presence=data.get("object_bird_presence"),
        bbox_quality=data.get("bbox_quality"),
        bbox_correction=bbox,
        species_identity=data.get("species_identity"),
        species_key=data.get("species_key"),
        detector_miss=data.get("detector_miss") is True,
    )

    try:
        with db_service.closing_connection() as conn:
            fact_ids = human_label_service.record_answer(
                conn,
                answer,
                source_ref=f"image-correction:{filename}:{detection_id or 'image'}",
                app_version=_shared.get("app_version", ""),
            )
            if bbox is not None and detection_id is not None:
                path_manager = PathManager(str(_shared["output_dir"]))
                human_label_service.refresh_thumbnail_for_corrected_box(
                    conn,
                    detection_id=detection_id,
                    bbox=(bbox.x, bbox.y, bbox.w, bbox.h),
                    original_resolver=path_manager.get_original_path,
                    thumb_resolver=lambda name: path_manager.get_derivative_path(
                        name, "thumb"
                    ),
                )
            facts = human_label_service.fetch_current_facts(
                conn,
                image_filename=filename,
                detection_id=detection_id,
            )
        gallery_service.invalidate_cache()
    except HumanLabelError as exc:
        logger.info("Label answer rejected: %s", type(exc).__name__)
        message = safe_validation_message(
            exc,
            allowed_prefixes=(
                "bbox ",
                "object ",
                "detection ",
                "image ",
                "answer ",
                "bird-presence ",
            ),
            fallback="Label answer rejected",
        )
        return jsonify({"status": "error", "message": message}), 409
    except sqlite3.IntegrityError:
        logger.info("Label answer rejected: database constraint")
        return jsonify({"status": "error", "message": "Label answer rejected"}), 409

    return jsonify(
        {
            "status": "success",
            "fact_ids": fact_ids,
            "facts": facts,
        }
    )
