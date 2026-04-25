"""
API v1 Blueprint.

This blueprint provides versioned API endpoints under /api/v1/*.
It is a 1:1 mirror of the existing /api/* routes - no changes to behavior or response format.

Purpose: Enable API versioning without breaking existing clients.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from config import get_config
from logging_config import get_logger
from utils.settings import mask_rtsp_url, unmask_rtsp_url
from web.blueprints.auth import login_required
from web.power_actions import (
    POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
    get_power_action_success_message,
    is_power_management_available,
    schedule_power_action,
)
from web.security import error_response as _error_response
from web.security import safe_log_value as _safe_log_value
from web.services import (
    backup_restore_service,
    db_service,
    onvif_service,
)
from web.species_thumbnails import get_species_thumbnail_map

logger = get_logger(__name__)
config = get_config()

# Create Blueprint
api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def _read_file_tail(path: Path, max_lines: int = 200) -> dict:
    """Read the last lines of a text file safely for diagnostics endpoints."""
    result = {
        "path": str(path),
        "exists": path.exists(),
        "tail_text": "",
        "line_count": 0,
        "error": "",
    }
    if not path.exists():
        return result

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            tail_lines = list(deque(f, maxlen=max_lines))
        text = "".join(tail_lines)
        result["tail_text"] = text
        result["line_count"] = len(text.splitlines())
    except Exception as e:
        result["error"] = str(e)

    return result


def _detect_runtime_environment() -> str:
    """Detect whether the app runs on host or in a container runtime."""
    if Path("/.dockerenv").exists():
        return "docker"

    try:
        cgroup_text = Path("/proc/1/cgroup").read_text(
            encoding="utf-8", errors="ignore"
        )
        lowered = cgroup_text.lower()
        if "kubepods" in lowered:
            return "kubernetes"
        if "docker" in lowered:
            return "docker"
        if "containerd" in lowered:
            return "containerd"
    except Exception:
        pass

    return "host"


def _run_command_safe(
    cmd: list[str],
    timeout_sec: float = 2.5,
    max_output_chars: int = 12000,
    expected_permission_error: bool = False,
) -> dict:
    """Run a diagnostic command with availability checks and strict timeout."""
    binary = cmd[0] if cmd else ""
    if not binary:
        return {
            "available": False,
            "ok": False,
            "returncode": -1,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": "empty command",
        }

    if shutil.which(binary) is None:
        return {
            "available": False,
            "ok": False,
            "returncode": 127,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": f"{binary} not available",
        }

    permission_error_markers = (
        "insufficient permissions",
        "not seeing messages from other users",
        "no journal files were opened due to insufficient permissions",
        "permission denied",
    )

    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
        combined_output = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        if stderr_text and stderr_text not in combined_output:
            combined_output = (
                f"{combined_output}\n{stderr_text}".strip()
                if combined_output
                else stderr_text
            )

        truncated = False
        if len(combined_output) > max_output_chars:
            combined_output = combined_output[:max_output_chars] + "\n... (truncated)"
            truncated = True

        normalized = combined_output.lower()
        permission_limited = expected_permission_error and any(
            marker in normalized for marker in permission_error_markers
        )

        return {
            "available": True,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "timed_out": False,
            "truncated": truncated,
            "output": combined_output,
            "error": "" if completed.returncode == 0 else stderr_text,
            "expected_permission_error": permission_limited,
        }
    except subprocess.TimeoutExpired as e:
        timeout_output = ""
        if e.stdout:
            timeout_output += e.stdout
        if e.stderr:
            timeout_output += f"\n{e.stderr}" if timeout_output else e.stderr
        timeout_output = timeout_output.strip()

        return {
            "available": True,
            "ok": False,
            "returncode": -1,
            "timed_out": True,
            "truncated": False,
            "output": timeout_output,
            "error": f"timeout after {timeout_sec:.1f}s",
            "expected_permission_error": False,
        }
    except Exception as e:
        return {
            "available": True,
            "ok": False,
            "returncode": -1,
            "timed_out": False,
            "truncated": False,
            "output": "",
            "error": str(e),
            "expected_permission_error": False,
        }


# =============================================================================
# Status & Control
# =============================================================================


@api_v1.route("/status", methods=["GET"])
@login_required
def status():
    """
    Returns system status including detection state.
    Mirror of: GET /api/status
    """
    # Note: detection_manager is injected via init_api_v1()
    try:
        output_dir = config.get("OUTPUT_DIR", "./data/output")
        dm = api_v1.detection_manager

        return jsonify(
            {
                "detection_paused": dm.paused,
                "detection_running": not dm.paused,
                "restart_required": backup_restore_service.is_restart_required(
                    output_dir
                ),
            }
        )
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({"error": str(e)}), 500


@api_v1.route("/species/thumbnails", methods=["GET"])
@login_required
def get_species_thumbnails():
    """
    Returns a mapping of species names to their latest thumbnail URL.

    Uses gallery_service.get_captured_detections() following established patterns.
    Returns thumbnails keyed by: scientific name (both formats) and German name.
    """
    # Load common names for localized mapping
    from utils.species_names import load_common_names

    cfg = get_config()
    locale = cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    common_names = load_common_names(locale)

    try:
        mapping = get_species_thumbnail_map(
            common_names=common_names,
            cache_key=None,
        )
    except Exception as e:
        logger.error(f"Failed to fetch species thumbnails: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success", "thumbnails": mapping})


@api_v1.route("/detection/pause", methods=["POST"])
@login_required
def detection_pause():
    """
    Pauses the detection loop.
    Mirror of: POST /api/detection/pause
    """
    try:
        dm = api_v1.detection_manager

        if dm.paused:
            return jsonify(
                {
                    "status": "paused",
                    "message": "Detection was already paused",
                }
            )

        dm.paused = True
        logger.info("Detection paused via API v1")

        return jsonify(
            {
                "status": "success",
                "message": "Detection paused",
            }
        )
    except Exception as e:
        logger.error(f"Detection pause error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/detection/resume", methods=["POST"])
@login_required
def detection_resume():
    """
    Resumes the detection loop.
    Mirror of: POST /api/detection/resume
    """
    try:
        dm = api_v1.detection_manager

        if not dm.paused:
            return jsonify(
                {
                    "status": "running",
                    "message": "Detection was already running",
                }
            )

        dm.paused = False
        logger.info("Detection resumed via API v1")

        return jsonify(
            {
                "status": "success",
                "message": "Detection resumed",
            }
        )
    except Exception as e:
        logger.error(f"Detection resume error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Models — detector registry + variant switching
# =============================================================================


def _regenerate_metadata_for_variant(model_dir: str, model_id: str) -> str | None:
    """Rewrite ``<model_dir>/model_metadata.json`` for the given variant.

    Reads ``<model_id>_model_config.yaml`` from *model_dir* (if present)
    and feeds it through the shared generator so the active detector
    picks up the right thresholds on reload.

    Returns the absolute metadata path on success, or ``None`` when the
    variant's YAML is not present (the detector then falls back to its
    hard-coded defaults, which is still correct but loses the
    release-specific metrics / conf / iou values).
    """
    import os

    from utils.model_downloader import _safe_model_dir_join

    yaml_basename = os.path.basename(f"{model_id}_model_config.yaml")
    yaml_path = _safe_model_dir_join(model_dir, yaml_basename)
    if yaml_path is None or not os.path.exists(yaml_path):
        logger.warning(
            "models/detector/pin: no %s_model_config.yaml found in %s; "
            "model_metadata.json not regenerated. Detector will fall back "
            "to hard-coded threshold defaults.",
            _safe_log_value(model_id),
            _safe_log_value(model_dir),
        )
        return None

    try:
        import yaml as _yaml

        from utils.model_metadata_generator import config_to_metadata

        config = _yaml.safe_load(open(yaml_path).read())
        if not isinstance(config, dict):
            raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")
        metadata = config_to_metadata(
            config, source_yaml_name=os.path.basename(yaml_path)
        )
        output_path = _safe_model_dir_join(model_dir, "model_metadata.json")
        if output_path is None:
            raise ValueError(f"model_dir {model_dir!r} failed containment check")
        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            import json as _json

            file.write(_json.dumps(metadata, indent=2) + "\n")
        os.replace(tmp_path, output_path)
        logger.info(
            "model_metadata.json regenerated for %s (conf=%s, iou=%s)",
            model_id,
            metadata.get("inference_thresholds", {}).get("confidence"),
            metadata.get("inference_thresholds", {}).get("iou_nms"),
        )
        return output_path
    except Exception as exc:
        logger.warning(f"Failed to regenerate model_metadata.json: {exc}")
        return None


@api_v1.route("/models/detector", methods=["GET"])
@login_required
def models_detector_get():
    """
    Return the detector registry payload for the AI settings panel.

    Response shape (see web.services.model_registry_service):
      {
        "model_dir": "/opt/app/data/models/object_detection",
        "active": {"id", "source", "pin_file", "pin_value_effective",
                   "hf_latest_id", "runtime_matches_on_disk"},
        "runtime": {"model_id", "model_path", "output_format",
                    "input_size", "num_classes", "class_names",
                    "conf_threshold_default", "iou_threshold_default"},
        "metadata": {...contents of model_metadata.json...},
        "variants": [{"id", "weights_path", "labels_path",
                      "is_available_locally", "is_active",
                      "is_hf_latest", ...}, ...]
      }
    """
    from web.services.model_registry_service import build_detector_registry_payload

    try:
        dm = api_v1.detection_manager
        detection_service = getattr(dm, "detection_service", None)
        detector_obj = getattr(detection_service, "_detector", None)
        underlying = getattr(detector_obj, "model", None) if detector_obj else None
        payload = build_detector_registry_payload(underlying)
        return jsonify(payload)
    except Exception as exc:
        return _error_response("models/detector GET failed", exc)


@api_v1.route("/models/detector/precision", methods=["POST"])
@login_required
def models_detector_precision():
    """Switch the active detector precision for a given model variant.

    Body: ``{"model_id": "<id>", "precision": "fp32" | "int8_qdq"}``.

    Parallels :func:`models_detector_pin`: writes the choice into
    ``latest_models.json`` under ``pinned_models[model_id].active_precision``
    (and top-level ``active_precision`` when ``model_id`` is the current
    default), then clears DetectionService so the next inference cycle
    reloads the correct weights file.

    The loader performs a try-load cascade through
    ``weights_int8_qdq_fallback_paths``; if all QDQ candidates fail on the
    host's ORT build, the detector falls back to fp32 with a warning log.
    Requires authentication.
    """
    from utils.model_downloader import (
        PRECISION_VALUES,
        _resolve_pin_for_cache_dir,
        set_active_precision,
    )
    from web.services.model_registry_service import (
        _model_dir,
        build_detector_registry_payload,
        variant_is_known,
    )

    try:
        data = request.get_json(silent=True) or {}
        model_id = str(data.get("model_id", "")).strip()
        precision = str(data.get("precision", "")).strip()
        if not model_id:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "model_id is required (non-empty).",
                    }
                ),
                400,
            )
        if precision not in PRECISION_VALUES:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"precision must be one of {list(PRECISION_VALUES)}, "
                            f"got {precision!r}."
                        ),
                    }
                ),
                400,
            )

        dm = api_v1.detection_manager
        detection_service = getattr(dm, "detection_service", None)
        detector_obj = getattr(detection_service, "_detector", None)
        underlying = getattr(detector_obj, "model", None) if detector_obj else None

        payload = build_detector_registry_payload(underlying)
        if not variant_is_known(payload, model_id):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Model id {model_id!r} is not a known locally-"
                            f"available variant. Use GET /api/v1/models/detector "
                            f"to see what's installed."
                        ),
                    }
                ),
                400,
            )

        model_dir = _model_dir()
        latest_path = set_active_precision(model_dir, model_id, precision)

        env_pin = _resolve_pin_for_cache_dir(model_dir)

        # Trigger live reload so the next detection cycle picks up the
        # new precision weights (parallels /pin's reload flow).
        reload_triggered = False
        if detection_service is not None:
            try:
                detection_service._detector = None
                detection_service._initialized = False
                detection_service._model_id = ""
                dm.detector_model_id = ""
                reload_triggered = True
                logger.info(
                    "models/detector/precision: model_id=%r precision=%r "
                    "-> live reload triggered",
                    model_id,
                    precision,
                )
            except Exception as reload_exc:
                logger.warning(
                    f"Failed to clear DetectionService for precision reload: "
                    f"{reload_exc}"
                )

        return jsonify(
            {
                "status": "success",
                "model_id": model_id,
                "precision": precision,
                "latest_models_path": latest_path,
                "env_pin_overrides": bool(env_pin),
                "reload_triggered": reload_triggered,
            }
        )
    except ValueError as ve:
        # ValueError here carries a deliberate, user-facing message
        # raised by our own validation code — safe to surface.
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as exc:
        return _error_response("models/detector/precision POST failed", exc)


@api_v1.route("/models/detector/pin", methods=["POST"])
@login_required
def models_detector_pin():
    """
    Switch the active detector variant by rewriting latest_models.json.

    Body: {"model_id": "<id>"} — must match one of the locally-available
    variants returned by GET /api/v1/models/detector (i.e. a key under
    the ``pinned_models`` block, or the current ``latest`` itself).

    Behaviour parallels POST /api/v1/cameras/<id>/use: the runtime
    config on disk is updated, then the DetectionService is cleared so
    the next inference cycle lazy-loads the new variant (~1-2 s, no
    service restart).

    An operator-set env-var pin (systemd drop-in) still wins over this
    change — the response returns ``effective_source`` so the UI can
    tell the user when the change was accepted but superseded.

    Security: requires authentication. Writes happen as the app's user
    only; no sudo, no systemd drop-in edits.
    """
    from utils.model_downloader import (
        _resolve_pin_for_cache_dir,
        set_latest_model_id,
    )
    from web.services.model_registry_service import (
        _model_dir,
        build_detector_registry_payload,
        variant_is_known,
    )

    try:
        data = request.get_json(silent=True) or {}
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "model_id is required (non-empty).",
                    }
                ),
                400,
            )

        dm = api_v1.detection_manager
        detection_service = getattr(dm, "detection_service", None)
        detector_obj = getattr(detection_service, "_detector", None)
        underlying = getattr(detector_obj, "model", None) if detector_obj else None

        payload = build_detector_registry_payload(underlying)
        if not variant_is_known(payload, model_id):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Model id {model_id!r} is not a known locally-available "
                            f"variant. Use GET /api/v1/models/detector to see "
                            f"what's installed."
                        ),
                    }
                ),
                400,
            )

        model_dir = _model_dir()
        latest_path = set_latest_model_id(model_dir, model_id)

        # Regenerate model_metadata.json for the new active variant so
        # the detector picks up the right confidence/iou thresholds on
        # reload. Without this, a switch to a variant with different
        # thresholds (e.g. S's conf=0.30 vs Tiny's 0.15) would run the
        # new ONNX with the previous variant's thresholds.
        metadata_path = _regenerate_metadata_for_variant(model_dir, model_id)

        # The env-var pin (systemd) still wins; tell the UI so it can
        # explain why the click looked like it worked but didn't flip
        # the loaded ONNX.
        env_pin = _resolve_pin_for_cache_dir(model_dir)
        effective_id = env_pin or model_id
        effective_source = "env_var_pin" if env_pin else "latest_models"

        # Trigger a live reload on the next detection cycle.
        reload_triggered = False
        if detection_service is not None:
            try:
                detection_service._detector = None
                detection_service._initialized = False
                detection_service._model_id = ""
                dm.detector_model_id = ""
                reload_triggered = True
                logger.info(
                    "models/detector/pin: latest=%r effective=%r source=%s -> live reload triggered",
                    model_id,
                    effective_id,
                    effective_source,
                )
            except Exception as reload_exc:
                logger.warning(
                    f"Failed to clear DetectionService for live reload: {reload_exc}"
                )

        return jsonify(
            {
                "status": "success",
                "model_id": model_id,
                "latest_models_path": latest_path,
                "metadata_path": metadata_path,
                "effective_id": effective_id,
                "effective_source": effective_source,
                "env_pin_overrides": bool(env_pin),
                "reload_triggered": reload_triggered,
            }
        )
    except Exception as exc:
        return _error_response("models/detector/pin POST failed", exc)


@api_v1.route("/models/detector/install", methods=["POST"])
@login_required
def models_detector_install():
    """
    Fetch a known variant's weights + labels from HuggingFace into the
    local model cache. Does not switch the active detector — the UI
    chains this with POST /pin afterwards when the user clicks the
    Switch button on a Not-installed row.

    Body: {"model_id": "<id>"} — must be a key in the registry payload
    (either under ``pinned_models`` or the current ``latest``). Arbitrary
    request-body strings are rejected; the HF URL is built from the
    hard-coded HF_BASE_URL plus the registry-provided relative paths,
    so this endpoint cannot be used as an SSRF primitive.

    Blocking: the HTTP request returns only after the download finishes
    (typ. a few seconds for the small YOLOX ONNX). Failures are
    reported in the response body, not retried.

    Security: requires authentication. Writes happen under the app's
    MODEL_BASE_PATH only.
    """
    from detectors.detector import HF_BASE_URL
    from utils.model_downloader import (
        _download_file,
        _fetch_companion_files,
        _normalize_rel_path,
    )
    from web.services.model_registry_service import (
        _model_dir,
        build_detector_registry_payload,
        variant_exists_in_registry,
    )

    try:
        data = request.get_json(silent=True) or {}
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "model_id is required (non-empty).",
                    }
                ),
                400,
            )

        dm = api_v1.detection_manager
        detection_service = getattr(dm, "detection_service", None)
        detector_obj = getattr(detection_service, "_detector", None)
        underlying = getattr(detector_obj, "model", None) if detector_obj else None

        payload = build_detector_registry_payload(underlying)
        variant = variant_exists_in_registry(payload, model_id)
        if variant is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Model id {model_id!r} is not listed in the "
                            f"registry (pinned_models or latest). Install "
                            f"only works for ids shipped with the release."
                        ),
                    }
                ),
                400,
            )

        if variant.get("is_available_locally"):
            return jsonify(
                {
                    "status": "success",
                    "model_id": model_id,
                    "already_installed": True,
                    "weights_path": variant.get("weights_path"),
                    "labels_path": variant.get("labels_path"),
                }
            )

        model_dir = _model_dir()
        weights_rel = str(variant.get("weights_path", ""))
        labels_rel = str(variant.get("labels_path", ""))
        if not weights_rel or not labels_rel:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Registry entry is missing weights_path or "
                            "labels_path; cannot install."
                        ),
                    }
                ),
                400,
            )

        from utils.model_downloader import _safe_model_dir_join

        weights_rel_norm = _normalize_rel_path(HF_BASE_URL, weights_rel)
        labels_rel_norm = _normalize_rel_path(HF_BASE_URL, labels_rel)
        weights_abs = _safe_model_dir_join(
            model_dir, os.path.basename(weights_rel_norm)
        )
        labels_abs = _safe_model_dir_join(model_dir, os.path.basename(labels_rel_norm))
        if weights_abs is None or labels_abs is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Registry paths failed containment check.",
                    }
                ),
                400,
            )

        weights_url = f"{HF_BASE_URL}/{weights_rel_norm}"
        labels_url = f"{HF_BASE_URL}/{labels_rel_norm}"

        logger.info(
            "models/detector/install: fetching %s weights=%s labels=%s (+ companions)",
            _safe_log_value(model_id),
            _safe_log_value(weights_url),
            _safe_log_value(labels_url),
        )
        if not _download_file(weights_url, weights_abs, base_dir=model_dir):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to download weights from {weights_url}",
                    }
                ),
                502,
            )
        if not _download_file(labels_url, labels_abs, base_dir=model_dir):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to download labels from {labels_url}",
                    }
                ),
                502,
            )

        # Companion files: _model_config.yaml (required for correct
        # threshold regeneration) + _metrics.json (honest recall/precision
        # display in the AI panel). Both best-effort — older releases may
        # not ship them. See _fetch_companion_files for the rationale.
        _fetch_companion_files(HF_BASE_URL, model_dir, model_id)
        from utils.model_downloader import _safe_model_dir_join

        yaml_abs = _safe_model_dir_join(
            model_dir, os.path.basename(f"{model_id}_model_config.yaml")
        )
        metrics_abs = _safe_model_dir_join(
            model_dir, os.path.basename(f"{model_id}_metrics.json")
        )

        return jsonify(
            {
                "status": "success",
                "model_id": model_id,
                "already_installed": False,
                "weights_path": weights_abs,
                "labels_path": labels_abs,
                "model_config_path": (
                    yaml_abs if yaml_abs and os.path.exists(yaml_abs) else None
                ),
                "metrics_path": (
                    metrics_abs if metrics_abs and os.path.exists(metrics_abs) else None
                ),
            }
        )
    except Exception as exc:
        return _error_response("models/detector/install POST failed", exc)


# =============================================================================
# Classifier model management (parallel to Detector, simpler —
# no precision chips, no int8 QDQ fallback, classes.txt not labels.json).
# =============================================================================


@api_v1.route("/models/classifier", methods=["GET"])
@login_required
def models_classifier_get():
    """Return the classifier registry payload for the AI settings panel."""
    from web.services.model_registry_service import build_classifier_registry_payload

    try:
        dm = api_v1.detection_manager
        classifier = getattr(dm, "classifier", None)
        payload = build_classifier_registry_payload(classifier)
        return jsonify(payload)
    except Exception as exc:
        return _error_response("models/classifier GET failed", exc)


@api_v1.route("/models/classifier/pin", methods=["POST"])
@login_required
def models_classifier_pin():
    """Switch the active classifier by rewriting classifier/latest_models.json.

    Body: ``{"model_id": "<id>"}`` — must match a locally-available variant
    from GET /api/v1/models/classifier. Triggers a lazy reload on the
    next classification cycle; no service restart.
    """
    from utils.model_downloader import (
        _resolve_pin_for_cache_dir,
        set_latest_model_id,
    )
    from web.services.model_registry_service import (
        _classifier_model_dir,
        build_classifier_registry_payload,
    )

    try:
        data = request.get_json(silent=True) or {}
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            return (
                jsonify(
                    {"status": "error", "message": "model_id is required (non-empty)."}
                ),
                400,
            )

        dm = api_v1.detection_manager
        classifier = getattr(dm, "classifier", None)
        payload = build_classifier_registry_payload(classifier)

        # Whitelist: must be a known locally-available variant.
        known = next(
            (
                v
                for v in payload.get("variants", [])
                if v.get("id") == model_id and v.get("is_available_locally")
            ),
            None,
        )
        if not known:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Model id {model_id!r} is not a known locally-available "
                            f"classifier variant. Use GET /api/v1/models/classifier "
                            f"to see what's installed."
                        ),
                    }
                ),
                400,
            )

        model_dir = _classifier_model_dir()
        latest_path = set_latest_model_id(model_dir, model_id)

        env_pin = _resolve_pin_for_cache_dir(model_dir)
        effective_id = env_pin or model_id
        effective_source = "env_var_pin" if env_pin else "latest_models"

        # Classifier lazy-loads via ImageClassifier._ensure_initialized.
        # Clearing the instance forces a fresh load on the next classify().
        reload_triggered = False
        if classifier is not None:
            try:
                classifier._initialized = False
                classifier.ort_session = None
                classifier.model_path = None
                classifier.class_path = None
                classifier.model_id = ""
                dm.classifier_model_id = ""
                reload_triggered = True
                logger.info(
                    "models/classifier/pin: latest=%r effective=%r source=%s -> live reload triggered",
                    _safe_log_value(model_id),
                    _safe_log_value(effective_id),
                    _safe_log_value(effective_source),
                )
            except Exception as reload_exc:
                logger.warning(
                    f"Failed to clear classifier for live reload: {reload_exc}"
                )

        return jsonify(
            {
                "status": "success",
                "model_id": model_id,
                "latest_models_path": latest_path,
                "effective_id": effective_id,
                "effective_source": effective_source,
                "env_pin_overrides": bool(env_pin),
                "reload_triggered": reload_triggered,
            }
        )
    except Exception as exc:
        return _error_response("models/classifier/pin POST failed", exc)


@api_v1.route("/models/classifier/install", methods=["POST"])
@login_required
def models_classifier_install():
    """Fetch a classifier variant's weights + classes from HuggingFace.

    Does NOT switch the active classifier. The UI chains this with POST
    /pin afterwards on the Not-installed row.
    """
    from detectors.classifier import HF_BASE_URL as CLS_HF_BASE_URL
    from utils.model_downloader import (
        _download_file,
        _fetch_companion_files,
        _normalize_rel_path,
    )
    from web.services.model_registry_service import (
        _classifier_model_dir,
        build_classifier_registry_payload,
        classifier_variant_exists_in_registry,
    )

    try:
        data = request.get_json(silent=True) or {}
        model_id = str(data.get("model_id", "")).strip()
        if not model_id:
            return (
                jsonify(
                    {"status": "error", "message": "model_id is required (non-empty)."}
                ),
                400,
            )

        dm = api_v1.detection_manager
        classifier = getattr(dm, "classifier", None)

        payload = build_classifier_registry_payload(classifier)
        variant = classifier_variant_exists_in_registry(payload, model_id)
        if variant is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"Model id {model_id!r} is not listed in the "
                            f"classifier registry. Install only works for "
                            f"ids shipped with the release."
                        ),
                    }
                ),
                400,
            )

        if variant.get("is_available_locally"):
            return jsonify(
                {
                    "status": "success",
                    "model_id": model_id,
                    "already_installed": True,
                    "weights_path": variant.get("weights_path"),
                    "classes_path": variant.get("classes_path"),
                }
            )

        model_dir = _classifier_model_dir()
        weights_rel = str(variant.get("weights_path", ""))
        classes_rel = str(variant.get("classes_path", ""))
        if not weights_rel or not classes_rel:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Registry entry is missing weights_path or "
                            "classes_path; cannot install."
                        ),
                    }
                ),
                400,
            )

        from utils.model_downloader import _safe_model_dir_join

        weights_rel_norm = _normalize_rel_path(CLS_HF_BASE_URL, weights_rel)
        classes_rel_norm = _normalize_rel_path(CLS_HF_BASE_URL, classes_rel)
        weights_abs = _safe_model_dir_join(
            model_dir, os.path.basename(weights_rel_norm)
        )
        classes_abs = _safe_model_dir_join(
            model_dir, os.path.basename(classes_rel_norm)
        )
        if weights_abs is None or classes_abs is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Registry paths failed containment check.",
                    }
                ),
                400,
            )

        weights_url = f"{CLS_HF_BASE_URL}/{weights_rel_norm}"
        classes_url = f"{CLS_HF_BASE_URL}/{classes_rel_norm}"

        logger.info(
            "models/classifier/install: fetching %s weights=%s classes=%s",
            _safe_log_value(model_id),
            _safe_log_value(weights_url),
            _safe_log_value(classes_url),
        )
        if not _download_file(weights_url, weights_abs, base_dir=model_dir):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to download weights from {weights_url}",
                    }
                ),
                502,
            )
        if not _download_file(classes_url, classes_abs, base_dir=model_dir):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Failed to download classes from {classes_url}",
                    }
                ),
                502,
            )

        # Best-effort companion pull (model_config.yaml + metrics.json).
        _fetch_companion_files(CLS_HF_BASE_URL, model_dir, model_id)

        return jsonify(
            {
                "status": "success",
                "model_id": model_id,
                "already_installed": False,
                "weights_path": weights_abs,
                "classes_path": classes_abs,
            }
        )
    except Exception as exc:
        return _error_response("models/classifier/install POST failed", exc)


# =============================================================================
# Settings
# =============================================================================


@api_v1.route("/settings", methods=["GET"])
@login_required
def settings_get():
    """
    Returns current application settings.
    Mirror of: GET /api/settings
    """
    from config import get_settings_payload

    payload = get_settings_payload()
    if "VIDEO_SOURCE" in payload and isinstance(payload["VIDEO_SOURCE"], dict):
        payload["VIDEO_SOURCE"]["value"] = mask_rtsp_url(
            payload["VIDEO_SOURCE"]["value"]
        )
    if "CAMERA_URL" in payload and isinstance(payload["CAMERA_URL"], dict):
        payload["CAMERA_URL"]["value"] = mask_rtsp_url(payload["CAMERA_URL"]["value"])

    return jsonify(payload)


@api_v1.route("/settings", methods=["POST"])
@login_required
def settings_post():
    """
    Updates application settings.
    Mirror of: POST /api/settings
    """
    from config import (
        ensure_go2rtc_stream_synced,
        get_config,
        resolve_effective_sources,
        update_runtime_settings,
        validate_runtime_updates,
    )

    try:
        data = request.get_json() or {}

        # Security: Unmask RTSP password if placeholder is present
        current_config = get_config()
        if "VIDEO_SOURCE" in data:
            original_url = current_config.get("VIDEO_SOURCE")
            data["VIDEO_SOURCE"] = unmask_rtsp_url(data["VIDEO_SOURCE"], original_url)
        if "CAMERA_URL" in data:
            original_cam = current_config.get("CAMERA_URL", "")
            data["CAMERA_URL"] = unmask_rtsp_url(data["CAMERA_URL"], original_cam)

        valid, errors = validate_runtime_updates(data)

        if errors:
            return jsonify({"status": "error", "errors": errors}), 400

        if valid:
            update_runtime_settings(valid)

            # Notify host (web_interface) about runtime setting changes
            _cb = getattr(api_v1, "on_runtime_settings_applied", None)
            if callable(_cb):
                _cb(valid)

            # --- Pre-sync go2rtc before resolving stream sources ---
            cfg = get_config()
            ensure_go2rtc_stream_synced(cfg)

            # --- Resolve effective sources after settings change ---
            resolved = resolve_effective_sources(cfg)
            cfg["VIDEO_SOURCE"] = resolved["video_source"]

            logger.info(
                "STREAM_SOURCE stream_mode=%s video_source=%s reason=%s",
                resolved["effective_mode"],
                resolved["video_source"][:40]
                if resolved["video_source"]
                else "(empty)",
                resolved["reason"],
            )

            dm = api_v1.detection_manager
            dm.update_configuration({"VIDEO_SOURCE": resolved["video_source"]})

        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Telegram Report  (Job-Status Flow)
# =============================================================================


# In-memory job registry  {job_id: {status, message, created_at}}
# Auto-evicts entries older than _REPORT_JOB_TTL seconds.
_report_jobs: dict[str, dict] = {}
_report_jobs_lock = threading.Lock()
_REPORT_JOB_TTL = 600  # 10 min


def _evict_stale_report_jobs() -> None:
    """Remove jobs older than TTL.  Called under lock."""
    cutoff = time.time() - _REPORT_JOB_TTL
    stale = [jid for jid, j in _report_jobs.items() if j["created_at"] < cutoff]
    for jid in stale:
        del _report_jobs[jid]


@api_v1.route("/telegram/send-report", methods=["POST"])
@login_required
def telegram_send_report():
    """
    Starts an on-demand daily report as a background job.

    Returns ``job_id`` immediately.  Poll status via
    ``GET /api/v1/telegram/send-report/<job_id>/status``.
    """
    cfg = get_config()
    bot_token = str(cfg.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    chat_id = str(cfg.get("TELEGRAM_CHAT_ID", "") or "").strip()

    if not bot_token or not chat_id:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Telegram credentials missing. Set Bot Token and Chat ID first.",
                }
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]

    with _report_jobs_lock:
        _evict_stale_report_jobs()
        _report_jobs[job_id] = {
            "status": "pending",
            "message": "Job queued.",
            "created_at": time.time(),
        }

    def _run(jid: str) -> None:
        # Mark running
        with _report_jobs_lock:
            if jid in _report_jobs:
                _report_jobs[jid]["status"] = "running"
                _report_jobs[jid]["message"] = "Report is being generated…"
        logger.info("Telegram report job %s started.", jid)

        try:
            from utils.daily_report import main as run_report

            # Provide ingest health for truthful status rendering
            health_provider = None
            dm = getattr(api_v1, "detection_manager", None)
            if dm is not None:
                health_provider = getattr(dm, "get_ingest_health_snapshot", None)

            run_report(ingest_health_provider=health_provider)

            with _report_jobs_lock:
                if jid in _report_jobs:
                    _report_jobs[jid]["status"] = "success"
                    _report_jobs[jid]["message"] = "Report sent successfully."
            logger.info("Telegram report job %s completed.", jid)

        except Exception as exc:
            error_msg = str(exc) or "Unknown error"
            with _report_jobs_lock:
                if jid in _report_jobs:
                    _report_jobs[jid]["status"] = "error"
                    _report_jobs[jid]["message"] = error_msg
            logger.error("Telegram report job %s failed: %s", jid, exc, exc_info=True)

    t = threading.Thread(
        target=_run, args=(job_id,), name=f"TgReport-{job_id}", daemon=True
    )
    t.start()

    return jsonify(
        {
            "status": "accepted",
            "job_id": job_id,
            "message": "Report job started.",
        }
    )


@api_v1.route("/telegram/send-report/<job_id>/status", methods=["GET"])
@login_required
def telegram_report_status(job_id: str):
    """
    Poll the status of a report job.

    Response shape::

        {
            "job_id":  "abc123",
            "status":  "pending" | "running" | "success" | "error",
            "message": "…"
        }
    """
    with _report_jobs_lock:
        job = _report_jobs.get(job_id)

    if not job:
        return jsonify(
            {
                "job_id": job_id,
                "status": "error",
                "message": "Job not found (expired or invalid ID).",
            }
        ), 404

    return jsonify(
        {
            "job_id": job_id,
            "status": job["status"],
            "message": job["message"],
        }
    )


# =============================================================================
# ONVIF Camera Discovery
# =============================================================================


@api_v1.route("/onvif/discover", methods=["GET"])
@login_required
def onvif_discover():
    """
    Scans network for ONVIF cameras.
    Mirror of: GET /api/onvif/discover
    """
    try:
        cameras = onvif_service.discover_cameras(fast=False)

        if not cameras:
            return jsonify({"status": "success", "cameras": []})

        return jsonify({"status": "success", "cameras": cameras})
    except Exception as e:
        logger.error(f"ONVIF Discovery error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/onvif/get_stream_uri", methods=["POST"])
@login_required
def onvif_get_stream_uri():
    """
    Retrieves RTSP stream URI for a camera.
    Mirror of: POST /api/onvif/get_stream_uri
    """
    try:
        data = request.get_json() or {}
        ip = data.get("ip")
        port = int(data.get("port", 80))
        user = data.get("username", "")
        password = data.get("password", "")

        if not ip:
            return jsonify({"status": "error", "message": "IP is required"}), 400

        uri = onvif_service.get_stream_uri(ip, port, user, password)

        if uri:
            return jsonify({"status": "success", "uri": uri})
        else:
            return jsonify(
                {"status": "error", "message": "Could not retrieve URI"}
            ), 404
    except Exception as e:
        logger.error(f"ONVIF Stream URI error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Camera Management
# =============================================================================


@api_v1.route("/cameras", methods=["GET"])
@login_required
def cameras_list():
    """
    Lists all saved cameras.
    Mirror of: GET /api/cameras
    """
    try:
        cameras = onvif_service.get_saved_cameras()
        return jsonify({"status": "success", "cameras": cameras})
    except Exception as e:
        logger.error(f"Camera list error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras", methods=["POST"])
@login_required
def cameras_add():
    """
    Adds a new camera.
    Mirror of: POST /api/cameras
    """
    try:
        data = request.get_json() or {}
        ip = data.get("ip")
        port = int(data.get("port", 80))
        username = data.get("username", "")
        password = data.get("password", "")
        name = data.get("name", "")

        if not ip:
            return jsonify({"status": "error", "message": "IP is required"}), 400

        camera = onvif_service.save_camera(
            ip=ip, port=port, username=username, password=password, name=name
        )

        return jsonify({"status": "success", "camera": camera})
    except Exception as e:
        logger.error(f"Camera add error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>", methods=["PUT"])
@login_required
def cameras_update(camera_id: int):
    """
    Updates an existing camera.
    Mirror of: PUT /api/cameras/<camera_id>
    """
    try:
        data = request.get_json() or {}
        onvif_service.update_camera(
            camera_id=camera_id,
            ip=data.get("ip"),
            port=int(data["port"]) if data.get("port") else None,
            username=data.get("username"),
            password=data.get("password"),
            name=data.get("name"),
        )
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Camera update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>", methods=["DELETE"])
@login_required
def cameras_delete(camera_id: int):
    """
    Deletes a camera.
    Mirror of: DELETE /api/cameras/<camera_id>
    """
    try:
        onvif_service.delete_camera(camera_id)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Camera delete error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>/test", methods=["POST"])
@login_required
def cameras_test(camera_id: int):
    """
    Tests camera connection.
    Mirror of: POST /api/cameras/<camera_id>/test
    """
    try:
        success = onvif_service.test_camera(camera_id)
        if success:
            return jsonify(
                {"status": "success", "message": "Camera connection successful"}
            )
        else:
            return jsonify(
                {"status": "error", "message": "Camera connection failed"}
            ), 500
    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/cameras/<int:camera_id>/use", methods=["POST"])
@login_required
def cameras_use(camera_id: int):
    """
    Sets camera as active video source.
    Mirror of: POST /api/cameras/<camera_id>/use

    Updates CAMERA_URL (user-facing) and resolves effective VIDEO_SOURCE
    through the central resolver.
    """
    try:
        from config import (
            ensure_go2rtc_stream_synced,
            get_config,
            resolve_effective_sources,
            update_runtime_settings,
        )

        uri = onvif_service.get_camera_uri(camera_id)
        if not uri:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        # Set CAMERA_URL (not VIDEO_SOURCE directly)
        update_runtime_settings({"CAMERA_URL": uri})

        # --- Pre-sync go2rtc before resolving ---
        cfg = get_config()
        ensure_go2rtc_stream_synced(cfg)

        # Resolve and apply
        resolved = resolve_effective_sources(cfg)
        cfg["VIDEO_SOURCE"] = resolved["video_source"]

        logger.info(
            "cameras_use camera_id=%s stream_mode=%s video_source=%s",
            camera_id,
            resolved["effective_mode"],
            resolved["video_source"][:40] if resolved["video_source"] else "(empty)",
        )

        dm = api_v1.detection_manager
        dm.update_configuration({"VIDEO_SOURCE": resolved["video_source"]})

        return jsonify({"status": "success", "message": "Video source updated"})
    except Exception as e:
        logger.error(f"Camera use error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Analytics
# =============================================================================


@api_v1.route("/analytics/summary", methods=["GET"])
@login_required
def analytics_summary():
    """
    Returns detection analytics summary.
    Mirror of: GET /api/analytics/summary (via add_url_rule)
    """
    conn = db_service.get_connection()
    try:
        summary = db_service.fetch_analytics_summary(conn)
    finally:
        conn.close()
    return jsonify(summary)


@api_v1.route("/analytics/decisions", methods=["GET"])
@api_v1.route("/decision-metrics", methods=["GET"])
@login_required
def analytics_decisions():
    """
    Returns decision state distribution for active detections.

    Response::

        {
            "status": "success",
            "total": 1234,
            "states": {
                "confirmed": 900,
                "uncertain": 150,
                "unknown": 80,
                "rejected": 54,
                "null": 50
            },
            "review_queue_count": 230,
            "manual_confirmed_count": 42
        }
    """
    conn = db_service.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(d.decision_state, 'null') as state,
                COUNT(*) as cnt
            FROM detections d
            WHERE d.status = 'active'
            GROUP BY COALESCE(d.decision_state, 'null')
            """
        ).fetchall()

        states = {row["state"]: row["cnt"] for row in rows}
        total = sum(states.values())

        review_count = db_service.fetch_review_queue_count(
            conn, config["GALLERY_DISPLAY_THRESHOLD"]
        )
        manual_confirmed_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM images
            WHERE review_status = 'confirmed_bird'
            """
        ).fetchone()
        manual_confirmed_count = (
            int(manual_confirmed_row[0]) if manual_confirmed_row else 0
        )
    finally:
        conn.close()

    return jsonify(
        {
            "status": "success",
            "total": total,
            "states": states,
            "review_queue_count": review_count,
            "manual_confirmed_count": manual_confirmed_count,
        }
    )


@api_v1.route("/analytics/decisions/daily", methods=["GET"])
@login_required
def analytics_decisions_daily():
    """
    Returns per-day decision state distribution for the last N days.

    Query params:
        days (int): Number of days to look back (default: 14, max: 90)

    Response::

        {
            "status": "success",
            "days": [
                {
                    "date": "2026-03-04",
                    "confirmed": 45,
                    "uncertain": 5,
                    "unknown": 3,
                    "rejected": 1,
                    "total": 54
                },
                ...
            ]
        }
    """
    days_back = min(request.args.get("days", 14, type=int), 90)

    conn = db_service.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) AS day,
                COALESCE(d.decision_state, 'null') AS state,
                COUNT(*) AS cnt
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            WHERE d.status = 'active'
              AND i.timestamp >= strftime('%Y%m%d', 'now', ? || ' days') || '_000000'
            GROUP BY day, state
            ORDER BY day DESC, state
            """,
            (f"-{days_back}",),
        ).fetchall()
    finally:
        conn.close()

    # Pivot into per-day dicts
    day_map: dict[str, dict[str, int]] = {}
    for row in rows:
        day = row["day"]
        state = row["state"]
        cnt = row["cnt"]
        if day not in day_map:
            day_map[day] = {
                "date": day,
                "confirmed": 0,
                "uncertain": 0,
                "unknown": 0,
                "rejected": 0,
                "null": 0,
                "total": 0,
            }
        if state in day_map[day]:
            day_map[day][state] = cnt
        day_map[day]["total"] += cnt

    # Sort by date descending
    days_list = sorted(day_map.values(), key=lambda d: d["date"], reverse=True)

    return jsonify({"status": "success", "days": days_list})


# =============================================================================
# Weather
# =============================================================================


@api_v1.route("/weather/now", methods=["GET"])
def weather_now():
    """
    Returns the current cached weather data.
    No login required - weather is public information.
    """
    from web.services.weather_service import get_current_weather

    try:
        weather = get_current_weather()
        if weather.get("timestamp") is None:
            return jsonify(
                {
                    "status": "pending",
                    "message": "Weather data not yet available. First fetch in progress.",
                    "weather": weather,
                }
            )
        return jsonify({"status": "success", "weather": weather})
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/weather/history", methods=["GET"])
def weather_history():
    """
    Returns weather history for the last N hours (default 24).
    Query param: ?hours=24
    """
    from web.services.weather_service import get_weather_history

    try:
        hours = request.args.get("hours", 24, type=int)
        hours = max(1, min(168, hours))  # Clamp 1h - 7d
        history = get_weather_history(hours=hours)
        return jsonify({"status": "success", "hours": hours, "data": history})
    except Exception as e:
        logger.error(f"Weather history API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# System
# =============================================================================


@api_v1.route("/health", methods=["GET"])
@login_required
def system_health():
    """
    Returns comprehensive system health status.

    Includes:
    - Overall status (ok/error/warning)
    - Database connectivity and latency
    - Disk space usage
    - OS vital signs (CPU/RAM/Temp/Throttling)
    """
    from web.services import health_service

    try:
        health = health_service.get_system_health()
        status_code = 200
        if health.get("status") == "error":
            status_code = 503

        return jsonify(health), status_code
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/health/public", methods=["GET"])
def system_health_public():
    """
    Public subset of system health for the always-visible status bar.

    Exposes only the four scalars the UI already shows (CPU%, RAM%,
    CPU temp, free disk GB). Database latency, throttling flags, the
    absolute output path, and last-detection timestamps stay behind
    /api/v1/health (login-required) to limit information disclosure.
    """
    from web.services import health_service

    try:
        health = health_service.get_system_health()
        sys_block = health.get("system") or {}
        disk_block = health.get("disk") or {}
        return jsonify(
            {
                "system": {
                    "cpu_percent": sys_block.get("cpu_percent"),
                    "ram_percent": sys_block.get("ram_percent"),
                    "cpu_temp_c": sys_block.get("cpu_temp_c"),
                },
                "disk": {
                    "free_gb": disk_block.get("free_gb"),
                    "percent": disk_block.get("percent"),
                },
            }
        )
    except Exception as e:
        logger.error(f"Public health check error: {e}")
        return jsonify({"system": {}, "disk": {}}), 200


@api_v1.route("/system/stats", methods=["GET"])
@login_required
def system_stats():
    """
    Returns system resource statistics.
    Mirror of: GET /api/system/stats
    """
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        # Disk usage
        disk = None
        try:
            output_dir = config.get("OUTPUT_DIR", "./data/output")
            disk_usage = psutil.disk_usage(output_dir)
            disk = {
                "total_gb": round(disk_usage.total / (1024**3), 1),
                "used_gb": round(disk_usage.used / (1024**3), 1),
                "free_gb": round(disk_usage.free / (1024**3), 1),
                "percent": disk_usage.percent,
            }
        except Exception:
            pass

        # Temperature
        temp = None
        try:
            import subprocess

            result = subprocess.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace("temp=", "").replace("'C", ""))
        except Exception:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for _name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            break
            except Exception:
                pass

        response = {"status": "success", "cpu": cpu_percent, "ram": mem.percent}
        if temp is not None:
            response["temp"] = temp
        if disk is not None:
            response["disk"] = disk

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/vitals", methods=["GET"])
@login_required
def system_vitals():
    """
    Returns system vitals from SystemMonitor.

    Provides hardware metrics collected by SystemMonitor including:
    - ts: ISO timestamp
    - cpu_percent: CPU usage percentage
    - ram_percent: RAM usage percentage
    - cpu_temp_c: CPU temperature in Celsius
    - throttled: RPi throttling flags (if applicable)
    - core_voltage: RPi core voltage (if applicable)

    If SystemMonitor is not running, returns a fallback response.
    """
    try:
        # Get system_monitor from blueprint (injected via init_api_v1)
        system_monitor = getattr(api_v1, "system_monitor", None)

        if system_monitor is None:
            # Fallback: return basic stats without monitor
            from datetime import datetime

            import psutil

            return jsonify(
                {
                    "status": "success",
                    "monitor_active": False,
                    "vitals": {
                        "ts": datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "ram_percent": psutil.virtual_memory().percent,
                        "cpu_temp_c": None,
                        "throttled": None,
                    },
                }
            )

        vitals = system_monitor.get_current_vitals()

        return jsonify(
            {
                "status": "success",
                "monitor_active": True,
                "vitals": vitals,
            }
        )
    except Exception as e:
        logger.error(f"System vitals error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/diagnostics", methods=["GET"])
@login_required
def system_diagnostics():
    """
    Returns an extended diagnostics snapshot for admin log view.

    Includes:
    - Runtime/process metadata
    - Current monitor vitals (if available)
    - app.log / vital_signs.csv / fd_leak_dump tails
    - Safe command probes (systemctl/journalctl/docker) with timeout
    """
    try:
        import psutil

        output_dir = Path(config.get("OUTPUT_DIR", "./data/output"))
        log_dir = output_dir / "logs"

        app_lines = max(50, min(int(request.args.get("app_lines", 300)), 2000))
        vitals_lines = max(30, min(int(request.args.get("vitals_lines", 240)), 2000))
        fd_dump_lines = max(20, min(int(request.args.get("fd_lines", 300)), 4000))

        app_log_tail = _read_file_tail(log_dir / "app.log", max_lines=app_lines)
        vitals_tail = _read_file_tail(
            log_dir / "vital_signs.csv", max_lines=vitals_lines
        )
        fd_dump_tail = _read_file_tail(
            log_dir / "fd_leak_dump.txt", max_lines=fd_dump_lines
        )
        fd_dump_present = (
            fd_dump_tail["exists"]
            and bool(fd_dump_tail["tail_text"].strip())
            and "fd_leak_dump_not_present" not in fd_dump_tail["tail_text"].lower()
        )

        monitor = getattr(api_v1, "system_monitor", None)
        monitor_active = monitor is not None
        if monitor_active:
            try:
                vitals = monitor.get_current_vitals()
            except Exception:
                vitals = {}
        else:
            vitals = {}

        proc = psutil.Process()
        process_rss_mb = 0.0
        process_threads = 0
        process_fds = -1
        with proc.oneshot():
            process_rss_mb = proc.memory_info().rss / (1024 * 1024)
            process_threads = proc.num_threads()
            try:
                process_fds = proc.num_fds()
            except Exception:
                process_fds = -1

        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(output_dir))

        load_avg = None
        if hasattr(os, "getloadavg"):
            try:
                load_avg = os.getloadavg()
            except Exception:
                load_avg = None

        runtime = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "environment": _detect_runtime_environment(),
            "generated_at": datetime.now().isoformat(),
        }

        boot_time_iso = None
        try:
            boot_time_iso = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        except Exception:
            boot_time_iso = None

        system = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": vm.percent,
            "ram_total_mb": round(vm.total / (1024 * 1024), 1),
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "disk_used_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "load_avg": list(load_avg) if load_avg else None,
            "boot_time_iso": boot_time_iso,
        }

        process = {
            "rss_mb": round(process_rss_mb, 1),
            "threads": process_threads,
            "fds": process_fds,
        }

        commands = {
            "systemctl_app": _run_command_safe(
                [
                    "systemctl",
                    "show",
                    "app",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "NRestarts",
                ],
                timeout_sec=2.5,
            ),
            "journal_app_tail": _run_command_safe(
                ["journalctl", "-u", "app", "-n", "80", "--no-pager"],
                timeout_sec=2.5,
                expected_permission_error=True,
            ),
        }

        return jsonify(
            {
                "status": "success",
                "runtime": runtime,
                "monitor_active": monitor_active,
                "vitals": vitals,
                "system": system,
                "process": process,
                "files": {
                    "app_log": app_log_tail,
                    "vitals_csv": vitals_tail,
                    "fd_leak_dump": {
                        **fd_dump_tail,
                        "present": fd_dump_present,
                    },
                },
                "commands": commands,
            }
        )
    except Exception as e:
        logger.error(f"System diagnostics error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/versions", methods=["GET"])
@login_required
def system_versions():
    """
    Returns software version and build metadata.

    Shared metadata subset (same as legacy ``/api/system/versions``):
      ``app_version``, ``git_commit``, ``build_date``, ``deploy_type``,
      ``kernel``, ``os``, ``bootloader``.

    V1-only extras:
      ``status``, ``python_version``, ``opencv_version``.
    """
    try:
        import platform as _platform
        import sys

        import cv2

        from utils.deploy_info import read_build_metadata

        meta = read_build_metadata()

        # System info (kernel, os, bootloader) — same logic as legacy route
        kernel = "Unknown"
        os_name = "Unknown"
        bootloader = "Unknown"

        try:
            kernel = _platform.release()
        except Exception:
            pass

        try:
            os_release = Path("/etc/os-release")
            if os_release.is_file():
                for line in os_release.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    if line.startswith("PRETTY_NAME="):
                        os_name = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass

        try:
            import shutil

            if shutil.which("rpi-eeprom-update"):
                import subprocess

                res = subprocess.run(
                    ["rpi-eeprom-update"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if "CURRENT:" in line:
                            parts = line.split("CURRENT:", 1)
                            if len(parts) > 1:
                                bootloader = parts[1].strip()
                                break
        except Exception:
            pass

        return jsonify(
            {
                "status": "success",
                # Shared metadata subset
                "app_version": meta["app_version"],
                "git_commit": meta["git_commit"],
                "build_date": meta["build_date"],
                "deploy_type": meta["deploy_type"],
                "kernel": kernel,
                "os": os_name,
                "bootloader": bootloader,
                # V1-only extras
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "opencv_version": cv2.__version__,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/shutdown", methods=["POST"])
@login_required
def system_shutdown():
    """
    Initiates system shutdown.
    Mirror of: POST /api/system/shutdown
    """
    try:
        if not is_power_management_available():
            logger.warning(
                "Shutdown ignored: systemd not available (likely container)."
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                    }
                ),
                400,
            )

        schedule_power_action("shutdown", logger)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": get_power_action_success_message("shutdown"),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error initiating shutdown: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/restart", methods=["POST"])
@login_required
def system_restart():
    """
    Initiates system restart.
    Mirror of: POST /api/system/restart
    """
    try:
        if not is_power_management_available():
            logger.warning("Restart ignored: systemd not available (likely container).")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": POWER_MANAGEMENT_UNAVAILABLE_MESSAGE,
                    }
                ),
                400,
            )

        schedule_power_action("restart", logger)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": get_power_action_success_message("restart"),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error initiating restart: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/public/go2rtc/health", methods=["GET"])
def go2rtc_health_public():
    """
    Public same-origin health endpoint for frontend go2rtc checks.

    Avoids browser CORS issues when the app UI (port 8050) probes go2rtc
    directly on port 1984.

    Returns diagnostic ``detail`` when unhealthy so the root cause
    (timeout, DNS, connection refused …) is visible without shell access.
    """
    try:
        import urllib.request

        from config import get_config

        cfg = get_config()
        api_base = str(cfg.get("GO2RTC_API_BASE", "http://127.0.0.1:1984") or "")
        probe_url = f"{api_base.rstrip('/')}/api/streams"
        detail = None

        try:
            req = urllib.request.Request(probe_url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                healthy = resp.status == 200
        except Exception as exc:
            healthy = False
            detail = str(exc)

        result = {
            "status": "success",
            "healthy": healthy,
            "api_base": api_base,
        }
        if detail:
            result["detail"] = detail
        return jsonify(result)
    except Exception as e:
        logger.error(f"go2rtc health API error: {e}")
        return jsonify({"status": "error", "healthy": False, "message": str(e)}), 500


@api_v1.route("/public/bbox-heatmap", methods=["GET"])
def bbox_heatmap_public():
    """
    Disabled in the public backport build.
    """
    return (
        jsonify(
            {
                "status": "error",
                "message": "This feature is not available in the public backport build.",
            }
        ),
        404,
    )


# =============================================================================
# OTA Update Endpoints (RPi only)
# =============================================================================


@api_v1.route("/system/updates/check", methods=["GET"])
@login_required
def system_updates_check():
    """
    Check for available updates.

    Returns current version, latest release, and whether an update is available.
    Only queries GitHub — does not modify anything.
    """
    try:
        from utils.deploy_info import read_build_metadata
        from web.services.update_service import get_latest_release, is_update_supported

        meta = read_build_metadata()
        current = meta.get("app_version", "Unknown")
        latest = get_latest_release()

        update_available = False
        if latest and current not in ("Unknown", "") and latest["tag_name"]:
            tag = latest["tag_name"].lstrip("v")
            cur = current.lstrip("v")
            update_available = tag != cur

        return jsonify(
            {
                "status": "success",
                "current_version": current,
                "latest_release": latest,
                "update_available": update_available,
                "update_supported": is_update_supported(),
            }
        )
    except Exception as e:
        logger.error("Error checking for updates: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/updates/releases", methods=["GET"])
@login_required
def system_updates_releases():
    """
    List available GitHub releases.

    Query params:
      limit (int, default 10): how many releases to return.
    """
    try:
        from web.services.update_service import list_releases

        limit = min(int(request.args.get("limit", 10)), 50)
        releases = list_releases(limit=limit)
        return jsonify({"status": "success", "releases": releases})
    except Exception as e:
        logger.error("Error fetching releases: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/updates/status", methods=["GET"])
@login_required
def system_updates_status():
    """
    Return the current update status (idle / downloading / installing / …).

    The wmb-update.service writes progress to a JSON file; this endpoint reads it.
    """
    try:
        from web.services.update_service import get_update_status

        return jsonify({"status": "success", **get_update_status()})
    except Exception as e:
        logger.error("Error reading update status: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_v1.route("/system/updates/install", methods=["POST"])
@login_required
def system_updates_install():
    """
    Trigger installation of a specific version.

    JSON body:
      target (str): release tag (e.g. "v0.2.0") or "main" for latest main branch.

    Only works on RPi deployments with systemd.
    """
    try:
        from web.services.update_service import is_update_supported, request_update

        if not is_update_supported():
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "OTA updates are only available on RPi deployments.",
                    }
                ),
                400,
            )

        data = request.get_json(silent=True) or {}
        target = (data.get("target") or "").strip()
        if not target:
            return (
                jsonify({"status": "error", "message": "Missing 'target' field."}),
                400,
            )

        success, message = request_update(target)
        if success:
            return jsonify({"status": "success", "message": message})
        return jsonify({"status": "error", "message": message}), 500
    except Exception as e:
        logger.error("Error installing update: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================================
# Blueprint Initialization
# =============================================================================


def init_api_v1(
    app,
    detection_manager,
    system_monitor=None,
    on_runtime_settings_applied=None,
):
    """
    Initialize the API v1 blueprint and register it with the app.

    Args:
        app: Flask application instance
        detection_manager: DetectionManager instance for detection control
        system_monitor: Optional SystemMonitor instance for vitals API
        on_runtime_settings_applied: Optional callback(valid_updates: dict)
            invoked after runtime settings have been persisted.  Lets the
            host (web_interface) react to config changes (e.g. locale reload).
    """
    # Store detection_manager reference on blueprint for route access
    api_v1.detection_manager = detection_manager

    # Store system_monitor reference for vitals API (optional)
    api_v1.system_monitor = system_monitor

    # Store runtime-settings callback
    api_v1.on_runtime_settings_applied = on_runtime_settings_applied

    # Register blueprint
    app.register_blueprint(api_v1)

    logger.info("API v1 blueprint registered at /api/v1")
