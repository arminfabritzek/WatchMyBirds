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
    ptz_service,
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
    except OSError:
        # /proc/self/cgroup missing (non-Linux); treat as host deploy.
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
    except Exception as exc:
        return _error_response("Failed to fetch species thumbnails", exc)

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
    except Exception as exc:
        return _error_response("Detection pause error", exc)


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
    except Exception as exc:
        return _error_response("Detection resume error", exc)


# =============================================================================
# Models — detector registry + variant switching
# =============================================================================


def _regenerate_metadata_for_variant(
    model_dir: str, model_id: str, *, refresh_companions: bool = False
) -> str | None:
    """Rewrite ``<model_dir>/model_metadata.json`` for the given variant.

    Reads ``<model_id>_model_config.yaml`` from *model_dir* (if present)
    and feeds it through the shared generator so the active detector
    picks up the right thresholds on reload.

    When ``refresh_companions=True``, first force-refreshes the YAML +
    metrics from HF so an operator pin click guarantees fresh metadata
    (per-class thresholds, suppressed_classes, etc.) — the local cache
    is overwritten only on a successful download. This is the UI path.
    Cold-start callers keep ``refresh_companions=False`` so reboots
    don't re-hit HF on every restart.

    Returns the absolute metadata path on success, or ``None`` when the
    variant's YAML is not present (the detector then falls back to its
    hard-coded defaults, which is still correct but loses the
    release-specific metrics / conf / iou values).
    """
    import os

    from detectors.detector import HF_BASE_URL
    from utils.model_downloader import _fetch_companion_files, _safe_model_dir_join

    if refresh_companions:
        try:
            _fetch_companion_files(HF_BASE_URL, model_dir, model_id, force_refresh=True)
        except Exception as exc:
            # Force-refresh is best-effort: HF outage, network issue,
            # or older release without the YAML — log and fall through
            # to the existing local cache.
            logger.warning(
                "Force-refresh of companion files for %s failed: %s; "
                "falling back to local cache.",
                _safe_log_value(model_id),
                exc,
            )

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

        with open(yaml_path, encoding="utf-8") as file:
            config = _yaml.safe_load(file)
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
        logger.warning(
            "Failed to regenerate model_metadata.json [%s]",
            type(exc).__name__,
            exc_info=True,
        )
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
                    _safe_log_value(model_id),
                    _safe_log_value(precision),
                )
            except Exception as reload_exc:
                logger.warning(
                    "Failed to clear DetectionService for precision reload [%s]",
                    type(reload_exc).__name__,
                    exc_info=True,
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
        #
        # ``refresh_companions=True``: a UI-driven pin click means the
        # operator wants the freshest possible metadata. We force-refresh
        # the YAML + metrics from HF so model-owned changes
        # (per-class thresholds, suppressed_classes, ...) land instantly
        # without ssh access. Fail-soft: on network error the local
        # cache stays untouched.
        metadata_path = _regenerate_metadata_for_variant(
            model_dir, model_id, refresh_companions=True
        )

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
                    _safe_log_value(model_id),
                    _safe_log_value(effective_id),
                    _safe_log_value(effective_source),
                )
            except Exception as reload_exc:
                logger.warning(
                    "Failed to clear DetectionService for live reload [%s]",
                    type(reload_exc).__name__,
                    exc_info=True,
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
    except Exception as exc:
        return _error_response("Settings update error", exc)


# =============================================================================
# Telemetry — Anonymous Opt-In Usage Heartbeat
# =============================================================================
# Default OFF. The toggle in Settings -> Privacy is the ONLY enable surface.
# See web/services/telemetry_service.py and docs/PRIVACY.md.


@api_v1.route("/settings/telemetry/status", methods=["GET"])
@login_required
def telemetry_status():
    """Return current telemetry state for the Settings UI.

    No PII: the installation_id is returned only as its first 8 hex
    chars, enough for the user to verify "yes that matches what's in
    the cloud" without exposing the full identifier in the page DOM.
    """
    from web.services.telemetry_service import (
        _get_last_sent_path,
        _read_last_sent_date,
        is_enabled,
    )

    cfg = get_config()
    output_dir = str(cfg.get("OUTPUT_DIR", "./data/output"))
    full_id = str(cfg.get("telemetry_installation_id", "") or "")
    short_id = full_id[:8] if len(full_id) == 32 else ""

    return jsonify(
        {
            "enabled": is_enabled(cfg),
            "installation_id_short": short_id,
            "installation_id_set": bool(short_id),
            "last_sent_at": _read_last_sent_date(output_dir),
            "endpoint": str(cfg.get("telemetry_endpoint", "")),
            "last_sent_marker_path": str(_get_last_sent_path(output_dir)),
        }
    )


@api_v1.route("/settings/telemetry/preview", methods=["GET"])
@login_required
def telemetry_preview():
    """Return the exact payload that WOULD be sent if telemetry were on.

    Read-only. Does not generate or persist a UUID. Does not touch the
    last-sent marker. Does not actually send anything anywhere.

    Purpose: let the operator inspect the live payload BEFORE opting
    in, so consent is informed. Also useful as a transparency check
    after opt-in ("what is this thing actually sending right now?").

    The endpoint mirror to the Worker that would receive it is also
    returned, so the operator can verify the destination matches what
    the Privacy page describes.
    """
    from web.services.telemetry_service import USER_AGENT, build_payload_preview

    cfg = get_config()
    return jsonify(
        {
            "endpoint": str(cfg.get("telemetry_endpoint", "")),
            "method": "POST",
            "user_agent": USER_AGENT,
            "payload": build_payload_preview(cfg),
        }
    )


@api_v1.route("/settings/telemetry", methods=["POST"])
@login_required
def telemetry_toggle():
    """Toggle telemetry on or off.

    Body: {"enabled": true | false}

    Persists to settings.yaml. On enable, also lazily generates the
    `installation_id` (if not already present) so the Settings UI
    immediately shows a real UUID instead of the placeholder — and
    so the scheduler's next tick can find a UUID ready to send.

    Toggling off does NOT wipe the installation_id (per locked
    decision D5); use /rotate for that.
    """
    from utils.settings import load_settings_yaml, save_settings_yaml
    from web.services.telemetry_service import _ensure_installation_id, wake_now

    try:
        data = request.get_json(silent=True) or {}
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            return jsonify(
                {"status": "error", "message": "'enabled' must be a boolean"}
            ), 400

        cfg = get_config()
        output_dir = str(cfg.get("OUTPUT_DIR", "./data/output"))

        yaml_settings = load_settings_yaml(output_dir)
        yaml_settings["telemetry_enabled"] = enabled
        save_settings_yaml(yaml_settings, output_dir)

        # Mirror into in-memory config so other workers see it without
        # needing to reload settings.yaml from disk.
        cfg["telemetry_enabled"] = enabled

        # On enable: generate a real installation_id NOW (idempotent —
        # existing IDs are preserved, only missing/malformed trigger
        # generation). This way the UI shows a real ID immediately
        # after the toggle flips, instead of the preview placeholder
        # that confused operators expecting "toggle on = ID exists".
        if enabled:
            new_id = _ensure_installation_id(cfg)
            cfg["telemetry_installation_id"] = new_id

            # Poke the scheduler thread out of its current sleep so
            # the first heartbeat goes within ~10ms instead of waiting
            # up to one full check_interval (300s default).
            wake_now()

        logger.info("Telemetry toggle changed to %s by user.", _safe_log_value(enabled))
        return jsonify({"status": "success", "enabled": enabled})
    except Exception as exc:
        return _error_response("Telemetry toggle error", exc)


@api_v1.route("/settings/telemetry/rotate", methods=["POST"])
@login_required
def telemetry_rotate():
    """Rotate the installation_id.

    The user explicitly says "treat me as a new install from now on."
    Old rows in the cloud DB stay (and TTL out via the daily 90d
    cleanup); the next ping uses the new UUID.

    Body: {} — no parameters; this is a destructive intent button.
    """
    try:
        from web.services.telemetry_service import rotate_installation_id

        cfg = get_config()
        output_dir = str(cfg.get("OUTPUT_DIR", "./data/output"))
        new_id = rotate_installation_id(output_dir)

        # Mirror into in-memory config.
        cfg["telemetry_installation_id"] = new_id

        return jsonify(
            {
                "status": "success",
                "installation_id_short": new_id[:8],
            }
        )
    except Exception as exc:
        return _error_response("Telemetry rotate error", exc)


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
            # Bridge: tag today's unscored detections before composing
            # the manual report so the operator sees aesthetic-ranked
            # photos even when triggering ad-hoc. Best-effort —
            # see web/services/report_scheduler.py for the same pattern
            # on the scheduled path.
            try:
                from web.services.aesthetic_tag_scheduler import run_now as tag_now

                tag_now(f"pre-telegram bridge (manual job {jid})", today_only=True)
            except Exception as exc:
                logger.warning(
                    "Aesthetic pre-run bridge failed for job %s: %s", jid, exc
                )

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


# --- Aesthetic re-score endpoint -----------------------------------------
#
# Re-runs the CLIP aesthetic scorer on detections that *already* have a
# score. The nightly tagger and the pre-Telegram bridge both skip
# already-scored detections (idempotency). When prompts or the score
# threshold change, the operator wants to bring existing data up to
# date — that's what this endpoint exists for.
#
# Pattern mirrors /telegram/send-report: POST starts a background job,
# GET /aesthetic/rescore/<job_id>/status polls progress.

_rescore_jobs: dict[str, dict] = {}
_rescore_jobs_lock = threading.Lock()
_RESCORE_JOB_TTL = 1800  # 30 min — re-scoring can take a while on the Pi


def _evict_stale_rescore_jobs() -> None:
    """Remove jobs older than TTL. Called under lock."""
    cutoff = time.time() - _RESCORE_JOB_TTL
    stale = [jid for jid, j in _rescore_jobs.items() if j["created_at"] < cutoff]
    for jid in stale:
        del _rescore_jobs[jid]


@api_v1.route("/aesthetic/rescore", methods=["POST"])
@login_required
def aesthetic_rescore():
    """Re-run the aesthetic scorer on already-scored detections.

    Body / query params (all optional):
      since:   ISO date "YYYY-MM-DD" — defaults to today (local).
      species: CLS class name like "Parus_major" — restrict to one species.
      limit:   Cap detections processed (debug / quick smoke).

    Returns a job_id; poll /aesthetic/rescore/<job_id>/status for progress.
    """
    payload = request.get_json(silent=True) or {}
    since = (
        payload.get("since")
        or request.args.get("since")
        or datetime.now().date().isoformat()
    )
    species = payload.get("species") or request.args.get("species")
    raw_limit = payload.get("limit") or request.args.get("limit")
    limit: int | None
    try:
        limit = int(raw_limit) if raw_limit is not None else None
    except (TypeError, ValueError):
        limit = None

    job_id = uuid.uuid4().hex[:12]
    with _rescore_jobs_lock:
        _evict_stale_rescore_jobs()
        _rescore_jobs[job_id] = {
            "status": "pending",
            "message": "Re-score job queued.",
            "created_at": time.time(),
            "since": since,
            "species": species,
            "limit": limit,
        }

    def _run(jid: str) -> None:
        with _rescore_jobs_lock:
            if jid in _rescore_jobs:
                _rescore_jobs[jid]["status"] = "running"
                _rescore_jobs[jid]["message"] = (
                    f"Re-scoring detections since={since}"
                    + (f", species={species}" if species else "")
                    + (f", limit={limit}" if limit else "")
                )
        logger.info("Aesthetic rescore job %s started.", jid)

        try:
            from scripts.aesthetic_tag_nightly import main_with_args
        except ImportError as exc:
            with _rescore_jobs_lock:
                if jid in _rescore_jobs:
                    _rescore_jobs[jid]["status"] = "error"
                    _rescore_jobs[jid]["message"] = (
                        f"Aesthetic worker not importable: {exc}"
                    )
            logger.error("Aesthetic rescore job %s: worker missing (%s)", jid, exc)
            return

        argv: list[str] = ["--rescore", "--since", since]
        if species:
            argv.extend(["--species", species])
        if limit:
            argv.extend(["--limit", str(limit)])

        try:
            rc = main_with_args(argv)
            with _rescore_jobs_lock:
                if jid in _rescore_jobs:
                    if rc == 0:
                        _rescore_jobs[jid]["status"] = "success"
                        _rescore_jobs[jid]["message"] = "Re-score completed."
                    else:
                        _rescore_jobs[jid]["status"] = "error"
                        _rescore_jobs[jid]["message"] = (
                            f"Worker returned non-zero exit code {rc}."
                        )
            logger.info("Aesthetic rescore job %s finished (rc=%d).", jid, rc)
        except Exception as exc:
            error_msg = str(exc) or "Unknown error"
            with _rescore_jobs_lock:
                if jid in _rescore_jobs:
                    _rescore_jobs[jid]["status"] = "error"
                    _rescore_jobs[jid]["message"] = error_msg
            logger.error("Aesthetic rescore job %s failed: %s", jid, exc, exc_info=True)

    t = threading.Thread(
        target=_run, args=(job_id,), name=f"AestheticRescore-{job_id}", daemon=True
    )
    t.start()

    return jsonify(
        {
            "status": "accepted",
            "job_id": job_id,
            "message": "Re-score job started.",
            "since": since,
            "species": species,
            "limit": limit,
        }
    )


@api_v1.route("/aesthetic/rescore/<job_id>/status", methods=["GET"])
@login_required
def aesthetic_rescore_status(job_id: str):
    """Poll the status of an aesthetic re-score job.

    Response shape mirrors /telegram/send-report status::

        { "job_id": "...", "status": "pending|running|success|error",
          "message": "...", "since": "...", "species": "...", "limit": ... }
    """
    with _rescore_jobs_lock:
        job = _rescore_jobs.get(job_id)

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
            "since": job.get("since"),
            "species": job.get("species"),
            "limit": job.get("limit"),
        }
    )


@api_v1.route("/telegram/seen-species", methods=["GET"])
@login_required
def telegram_seen_species_list():
    """Return the list of species the new_species_only mode has already
    alerted on (and is therefore now suppressing). Used by the Settings
    panel to show the operator what's currently silenced and offer a
    reset.
    """
    try:
        from utils.db.seen_species import list_seen_species

        rows = list_seen_species()
        return jsonify({"status": "success", "species": rows, "count": len(rows)})
    except Exception as exc:
        logger.error("seen_species list failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


@api_v1.route("/telegram/seen-species", methods=["DELETE"])
@login_required
def telegram_seen_species_reset():
    """Wipe the seen-species log so new_species_only mode re-fires alerts
    for every species again. Useful after a model swap or when the
    operator wants to retest the rarity-alert flow.
    """
    try:
        from utils.db.seen_species import reset_seen_species

        deleted = reset_seen_species()
        logger.info("seen_species log reset; %d row(s) removed.", deleted)
        return jsonify({"status": "success", "deleted": deleted})
    except Exception as exc:
        logger.error("seen_species reset failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


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
    except Exception as exc:
        return _error_response("ONVIF Discovery error", exc)


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
    except Exception as exc:
        return _error_response("ONVIF Stream URI error", exc)


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
    except Exception as exc:
        return _error_response("Camera list error", exc)


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
    except Exception as exc:
        return _error_response("Camera add error", exc)


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
    except Exception as exc:
        return _error_response("Camera update error", exc)


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
    except Exception as exc:
        return _error_response("Camera delete error", exc)


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
    except Exception as exc:
        return _error_response("Camera test error", exc)


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
            _safe_log_value(camera_id),
            _safe_log_value(resolved["effective_mode"]),
            _safe_log_value(
                resolved["video_source"][:40] if resolved["video_source"] else "(empty)"
            ),
        )

        dm = api_v1.detection_manager
        dm.update_configuration({"VIDEO_SOURCE": resolved["video_source"]})

        return jsonify({"status": "success", "message": "Video source updated"})
    except Exception as exc:
        return _error_response("Camera use error", exc)


# =============================================================================
# PTZ Control
# =============================================================================


@api_v1.route("/cameras/<int:camera_id>/ptz/config", methods=["GET"])
@login_required
def camera_ptz_config_get(camera_id: int):
    """Return stored auto-PTZ config for a saved camera."""
    try:
        config_data = ptz_service.get_config(camera_id)
        if config_data is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "config": config_data})
    except Exception as exc:
        return _error_response("PTZ config read error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/config", methods=["PUT"])
@login_required
def camera_ptz_config_put(camera_id: int):
    """Update stored auto-PTZ config for a saved camera."""
    try:
        data = request.get_json() or {}
        config_data = ptz_service.update_config(camera_id, data)
        if config_data is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "config": config_data})
    except Exception as exc:
        return _error_response("PTZ config update error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/presets", methods=["GET"])
@login_required
def camera_ptz_presets(camera_id: int):
    """List PTZ presets reported by the camera."""
    try:
        presets = ptz_service.list_presets(camera_id)
        return jsonify({"status": "success", "presets": presets})
    except Exception as exc:
        return _error_response("PTZ preset list error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/goto", methods=["POST"])
@login_required
def camera_ptz_goto(camera_id: int):
    """Move a camera to a preset token."""
    try:
        data = request.get_json() or {}
        preset_token = str(data.get("preset_token") or "").strip()
        if not preset_token:
            return jsonify(
                {"status": "error", "message": "preset_token is required"}
            ), 400
        ptz_service.goto_preset(camera_id, preset_token)
        try:
            dm = getattr(api_v1, "detection_manager", None)
            controller = getattr(dm, "auto_ptz_controller", None) if dm else None
            if controller:
                controller.notify_external_goto(preset_token)
        except Exception:
            logger.exception("Failed to notify Auto-PTZ controller of manual goto")
        return jsonify({"status": "success"})
    except Exception as exc:
        return _error_response("PTZ goto error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/move", methods=["POST"])
@login_required
def camera_ptz_move(camera_id: int):
    """Send a bounded continuous PTZ move command."""
    try:
        data = request.get_json() or {}
        ptz_service.move(
            camera_id,
            pan=float(data.get("pan") or 0.0),
            tilt=float(data.get("tilt") or 0.0),
            zoom=float(data.get("zoom") or 0.0),
            duration_ms=int(data.get("duration_ms") or 250),
        )
        return jsonify({"status": "success"})
    except Exception as exc:
        return _error_response("PTZ move error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/stop", methods=["POST"])
@login_required
def camera_ptz_stop(camera_id: int):
    """Stop PTZ movement."""
    try:
        ptz_service.stop(camera_id)
        return jsonify({"status": "success"})
    except Exception as exc:
        return _error_response("PTZ stop error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/presets/metadata", methods=["GET"])
@login_required
def camera_ptz_presets_metadata(camera_id: int):
    """Return PTZ preset list + Mini-Map snapshot URL."""
    try:
        config_data = ptz_service.get_config(camera_id)
        if config_data is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        show_all = request.args.get("show_all", "").lower() in ("1", "true", "yes")
        try:
            presets = ptz_service.list_presets_with_metadata(
                camera_id, show_all=show_all
            )
        except Exception as exc:
            logger.warning("PTZ presets list failed: %s", exc)
            presets = []
        snapshot_rel = (config_data.get("overview_snapshot_path") or "").strip()
        snapshot_url = (
            f"/uploads/{snapshot_rel}"
            if snapshot_rel.startswith("derivatives/")
            else ""
        )
        return jsonify(
            {
                "status": "success",
                "metadata": {
                    "overview_preset": config_data.get("overview_preset", ""),
                    "overview_snapshot_url": snapshot_url,
                    "presets": presets,
                },
            }
        )
    except Exception as exc:
        return _error_response("PTZ metadata read error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/auto/enabled", methods=["PATCH"])
@login_required
def camera_ptz_set_auto_enabled(camera_id: int):
    """Toggle the auto-PTZ enabled flag without touching other config."""
    try:
        data = request.get_json(silent=True) or {}
        if "enabled" not in data:
            return jsonify(
                {"status": "error", "message": "enabled field is required"}
            ), 400
        result = ptz_service.set_auto_enabled(camera_id, bool(data.get("enabled")))
        if result is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "config": result})
    except Exception as exc:
        return _error_response("PTZ auto-enable toggle error", exc)


@api_v1.route(
    "/cameras/<int:camera_id>/ptz/capture-overview-snapshot", methods=["POST"]
)
@login_required
def camera_ptz_capture_overview_snapshot(camera_id: int):
    """Fly to overview, fetch ONVIF snapshot, persist as Mini-Map background."""
    try:
        result = ptz_service.capture_overview_snapshot(camera_id)
        if result is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "snapshot": result})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return _error_response("PTZ snapshot capture error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/presets", methods=["POST"])
@login_required
def camera_ptz_set_preset(camera_id: int):
    """SetPreset at the current camera position (optional click metadata)."""
    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("name") or "").strip()
        if not name:
            return jsonify({"status": "error", "message": "name is required"}), 400
        result = ptz_service.set_preset_at_current_position(
            camera_id,
            name,
            preset_token=(
                str(data.get("preset_token")).strip()
                if data.get("preset_token")
                else None
            ),
            center_x_pct=(
                float(data["center_x_pct"]) if "center_x_pct" in data else None
            ),
            center_y_pct=(
                float(data["center_y_pct"]) if "center_y_pct" in data else None
            ),
            box_w_pct=(float(data["box_w_pct"]) if "box_w_pct" in data else None),
            box_h_pct=(float(data["box_h_pct"]) if "box_h_pct" in data else None),
            label=(str(data["label"]) if "label" in data else None),
        )
        if result is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "preset": result})
    except Exception as exc:
        return _error_response("PTZ set preset error", exc)


@api_v1.route("/cameras/<int:camera_id>/ptz/presets/<preset_token>", methods=["DELETE"])
@login_required
def camera_ptz_remove_preset(camera_id: int, preset_token: str):
    """RemovePreset on the camera and drop the stored metadata."""
    try:
        ok = ptz_service.remove_preset(camera_id, preset_token)
        if not ok:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success"})
    except Exception as exc:
        return _error_response("PTZ remove preset error", exc)


@api_v1.route(
    "/cameras/<int:camera_id>/ptz/presets/<preset_token>/metadata",
    methods=["PUT"],
)
@login_required
def camera_ptz_update_preset_metadata(camera_id: int, preset_token: str):
    """Update overlay-box metadata for a preset without moving the camera."""
    try:
        data = request.get_json(silent=True) or {}
        result = ptz_service.update_preset_metadata_only(
            camera_id,
            preset_token,
            center_x_pct=(
                float(data["center_x_pct"]) if "center_x_pct" in data else None
            ),
            center_y_pct=(
                float(data["center_y_pct"]) if "center_y_pct" in data else None
            ),
            box_w_pct=(float(data["box_w_pct"]) if "box_w_pct" in data else None),
            box_h_pct=(float(data["box_h_pct"]) if "box_h_pct" in data else None),
            label=(str(data["label"]) if "label" in data else None),
        )
        if result is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        return jsonify({"status": "success", "preset": result})
    except Exception as exc:
        return _error_response("PTZ metadata update error", exc)


@api_v1.route("/ptz/auto/status", methods=["GET"])
@login_required
def ptz_auto_status():
    """Return runtime auto-PTZ state from the detection manager."""
    try:
        dm = api_v1.detection_manager
        controller = getattr(dm, "auto_ptz_controller", None)
        if not controller:
            return jsonify(
                {
                    "status": "success",
                    "auto_ptz": {"enabled": False, "state": "idle"},
                }
            )
        return jsonify({"status": "success", "auto_ptz": controller.status()})
    except Exception as exc:
        return _error_response("PTZ status error", exc)


@api_v1.route("/ptz/auto/return-overview", methods=["POST"])
@login_required
def ptz_auto_return_overview():
    """Ask the runtime controller to return to the overview preset."""
    try:
        dm = api_v1.detection_manager
        controller = getattr(dm, "auto_ptz_controller", None)
        if not controller:
            return jsonify(
                {"status": "error", "message": "Auto PTZ controller unavailable"}
            ), 404
        ok = controller.return_to_overview()
        if not ok:
            return jsonify(
                {"status": "error", "message": "Overview preset is not configured"}
            ), 400
        return jsonify({"status": "success"})
    except Exception as exc:
        return _error_response("PTZ overview return error", exc)


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
    except Exception as exc:
        return _error_response("Weather API error", exc)


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
    except Exception as exc:
        return _error_response("Weather history API error", exc)


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
    except Exception as exc:
        return _error_response("Health check error", exc)


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
        except (OSError, ValueError):
            # disk_usage path missing or invalid; leave disk=None.
            pass

        # Temperature
        temp = None
        try:
            import subprocess as _sp

            result = _sp.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.replace("temp=", "").replace("'C", ""))
        except (OSError, ValueError, ImportError):
            # vcgencmd missing/timed-out; fall back to psutil sensors.
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for _name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            break
            except (AttributeError, OSError):
                # No temperature sensors exposed on this host.
                pass

        response = {"status": "success", "cpu": cpu_percent, "ram": mem.percent}
        if temp is not None:
            response["temp"] = temp
        if disk is not None:
            response["disk"] = disk

        return jsonify(response)
    except Exception as exc:
        return _error_response("Error fetching system stats", exc)


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
    except Exception as exc:
        return _error_response("System vitals error", exc)


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
            except (psutil.AccessDenied, AttributeError):
                # num_fds is POSIX-only; default of -1 already set above.
                pass

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
    except Exception as exc:
        return _error_response("System diagnostics error", exc)


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
        except OSError:
            # platform.release reads /proc; absent on non-Linux.
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
        except OSError:
            # /etc/os-release missing or unreadable.
            pass

        try:
            import shutil
            import subprocess as _sp

            if shutil.which("rpi-eeprom-update"):
                res = _sp.run(
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
        except (OSError, ImportError):
            # rpi-eeprom-update missing or denied; stays "Unknown".
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
    except Exception as exc:
        return _error_response("Error fetching system versions", exc)


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
    except Exception as exc:
        return _error_response("Error initiating shutdown", exc)


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
    except Exception as exc:
        return _error_response("Error initiating restart", exc)


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
    except Exception as exc:
        logger.error("go2rtc health API error [%s]", type(exc).__name__, exc_info=True)
        return jsonify(
            {
                "status": "error",
                "healthy": False,
                "message": "go2rtc health check failed",
            }
        ), 500


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


# OTA Update Endpoints intentionally absent.
# The /api/v1/system/updates/* routes were merged onto main ahead of
# their backing service implementation. The full implementation lives
# on a private branch (private_review_7185d29) and ships as part of
# the OTA roadmap (roadmap/2026-03-20_INFRA_ota-update-rpi_ROADMAP.md),
# which is itself blocked on USB Backup v2
# (roadmap/2026-04-29_INFRA_usb-restore-and-ota-hooks_ROADMAP.md).
# When the OTA branch is promoted, the four endpoints
# (/system/updates/{check,releases,status,install}) and a
# web/services/update_service.py module land together.


# =============================================================================
# USB Backup (write-only v1) — see docs/USB_BACKUP.md and the focus plan
# 2026-04-27_INFRA_usb-data-backup. Restore endpoints land in v2
# (2026-04-29_INFRA_usb-restore-and-ota-hooks); not in this version.
# =============================================================================


@api_v1.route("/system/backup/status", methods=["GET"])
@login_required
def system_backup_status():
    """
    Aggregate stick state + recent snapshots for the Settings card.

    Cheap (stat-only); safe to poll every 10 s.
    """
    try:
        from web.services import usb_backup_service

        return jsonify({"status": "success", **usb_backup_service.get_status()})
    except Exception as exc:
        return _error_response("Error reading USB backup status", exc)


@api_v1.route("/system/backup/list", methods=["GET"])
@login_required
def system_backup_list():
    """
    Return all snapshots (newest first), each with manifest metadata.

    Query params:
      limit (int, default 50): cap on the number of snapshots returned.
    """
    try:
        from web.services import usb_backup_service

        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, 500))
        snapshots = usb_backup_service.list_snapshots(limit=limit)
        return jsonify({"status": "success", "snapshots": snapshots})
    except Exception as exc:
        return _error_response("Error listing USB backup snapshots", exc)


@api_v1.route("/system/backup/trigger", methods=["POST"])
@login_required
def system_backup_trigger():
    """
    Spawn rpi/backup.sh --kind manual as a detached subprocess.

    JSON body (optional):
      kind (str): only 'manual' is accepted in v1. 'pre-ota' lands in v2.
    """
    try:
        from web.services import usb_backup_service

        payload = request.get_json(silent=True) or {}
        kind = payload.get("kind", "manual")
        if kind != "manual":
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"kind={kind!r} not supported in v1. Only 'manual' "
                            "is allowed; pre-ota and pre-restore land in v2."
                        ),
                    }
                ),
                400,
            )

        started, message, info = usb_backup_service.trigger_manual_backup()
        if not started:
            return (
                jsonify({"status": "error", "message": message, "info": info}),
                409,
            )
        return jsonify({"status": "success", "message": message, "info": info})
    except Exception as exc:
        return _error_response("Error triggering manual USB backup", exc)


@api_v1.route("/system/backup/<path:snapshot_name>", methods=["DELETE"])
@login_required
def system_backup_delete(snapshot_name):
    """
    Permanently delete a snapshot directory.

    Path-traversal-protected: refuses anything that escapes
    /mnt/wmb-backup/snapshots/.
    """
    try:
        from web.services import usb_backup_service

        ok, message = usb_backup_service.delete_snapshot(snapshot_name)
        if not ok:
            return jsonify({"status": "error", "message": message}), 404
        return jsonify({"status": "success", "message": message})
    except Exception as exc:
        logger.error(
            "Error deleting USB backup snapshot %r [%s]",
            _safe_log_value(snapshot_name),
            type(exc).__name__,
            exc_info=True,
        )
        return _error_response("Error deleting USB backup snapshot", exc)


@api_v1.route("/system/backup/<path:snapshot_name>/verify", methods=["POST"])
@login_required
def system_backup_verify(snapshot_name):
    """
    Re-run sha256 + sqlite integrity_check on a snapshot.

    Read-only: never mutates the snapshot. CORRUPT-marker stays as
    set by backup.sh; the operator decides what to do with the result.
    """
    try:
        from web.services import usb_backup_service

        result = usb_backup_service.verify_snapshot(snapshot_name)
        if result.get("ok") is False and result.get("error") == "Snapshot not found.":
            return jsonify({"status": "error", **result}), 404
        return jsonify({"status": "success", **result})
    except Exception as exc:
        logger.error(
            "Error verifying USB backup snapshot %r [%s]",
            _safe_log_value(snapshot_name),
            type(exc).__name__,
            exc_info=True,
        )
        return _error_response("Error verifying USB backup snapshot", exc)


# =============================================================================
# USB Stick Formatter (one-click format from Settings UI)
# =============================================================================
# DESTRUCTIVE feature. Operator picks a target USB device, types
# "FORMAT" to confirm, the Pi wipes + formats it as ext4/WMB-BACKUP.
# Backed by rpi/format_backup_stick.sh + wmb-format-backup.service
# (root, polkit-gated to watchmybirds for THIS unit only).


@api_v1.route("/system/backup/format/devices", methods=["GET"])
@login_required
def system_backup_format_devices():
    """List USB block devices the operator could format.

    Read-only: enumerates whole-disk USB sticks via lsblk + udev.
    Internal SATA/NVMe and partitions are filtered out.
    """
    try:
        from web.services import usb_format_service

        devices = usb_format_service.list_usb_block_devices()
        return jsonify(
            {
                "status": "success",
                "supported": usb_format_service.is_format_supported(),
                "devices": devices,
                # Calibration samples for the UI's remaining-time estimate.
                # Empty list on first run; gets populated by completed formats.
                "history": usb_format_service.get_format_history(limit=5),
            }
        )
    except Exception as exc:
        return _error_response("Error listing format candidates", exc)


@api_v1.route("/system/backup/format", methods=["POST"])
@login_required
def system_backup_format():
    """Trigger format on a target device.

    JSON body:
      target_device (str): e.g. "/dev/sda"  (validated against discovery list)
      confirm (str): must equal "FORMAT" -- typed by operator in modal
    """
    try:
        from web.services import usb_format_service

        payload = request.get_json(silent=True) or {}
        target = (payload.get("target_device") or "").strip()
        confirm = (payload.get("confirm") or "").strip()

        started, message = usb_format_service.trigger_format(target, confirm)
        if not started:
            return jsonify({"status": "error", "message": message}), 400
        return jsonify({"status": "success", "message": message})
    except Exception as exc:
        return _error_response("Error triggering format", exc)


@api_v1.route("/system/backup/format/status", methods=["GET"])
@login_required
def system_backup_format_status():
    """Poll the format progress.

    States: idle | starting | validating | wiping | partitioning |
            formatting | mounting | success | error
    """
    try:
        from web.services import usb_format_service

        return jsonify({"status": "success", **usb_format_service.get_format_status()})
    except Exception as exc:
        return _error_response("Error reading format status", exc)


@api_v1.route("/system/backup/format/status", methods=["DELETE"])
@login_required
def system_backup_format_status_clear():
    """Acknowledge the last format result; clears the status file."""
    try:
        from web.services import usb_format_service

        cleared = usb_format_service.clear_format_status()
        return jsonify({"status": "success" if cleared else "error"})
    except Exception as exc:
        return _error_response("Error clearing format status", exc)


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
    except Exception as exc:
        return _error_response("Error checking for updates", exc)


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
    except Exception as exc:
        return _error_response("Error fetching releases", exc)


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
    except Exception as exc:
        return _error_response("Error reading update status", exc)


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
    except Exception as exc:
        return _error_response("Error installing update", exc)


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
