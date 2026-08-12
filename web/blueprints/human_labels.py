"""Canonical human-label answer API."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, jsonify, request

from core.human_label_core import BBox, HumanAnswer, HumanLabelError
from logging_config import get_logger
from utils.path_manager import PathManager
from web.blueprints.auth import login_required
from web.security import safe_validation_message
from web.services import db_service, human_label_service

logger = get_logger(__name__)

human_labels_bp = Blueprint("human_labels", __name__)
_shared: dict[str, str] = {}


def init_human_labels_bp(*, output_dir: str, app_version: str = "") -> None:
    _shared["output_dir"] = output_dir
    _shared["app_version"] = app_version


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
