"""Model-registry read service: maps the on-disk model cache into a
UI-friendly JSON payload.

It answers two questions the AI settings panel needs:

1. **What is the detector loaded right now, and what are the knobs?**
   The Flask route serializes this as the GET response.

2. **Which alternate variants are known locally?**
   That is derived from ``latest_models.json`` — specifically the
   ``pinned_models`` dict shipped with each model release.

The service is deliberately read-only. Writing (switching the active
variant) lives in :func:`utils.model_downloader.set_latest_model_id`,
invoked from the HTTP layer so the side effects stay close to the
request boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from config import get_config
from core.model_downloader_core import (
    PIN_ENV_VAR,
    PIN_ENV_VAR_PREFIX,
    _resolve_pin_for_cache_dir,
    _task_name_from_cache_dir,
)
from logging_config import get_logger

logger = get_logger(__name__)


OBJECT_DETECTION_SUBDIR = "object_detection"


@dataclass
class VariantInfo:
    """One entry in ``latest_models.json[\"pinned_models\"]`` plus liveness flags."""

    id: str
    weights_path: str
    labels_path: str
    weights_exists: bool
    labels_exists: bool
    is_active: bool
    is_hf_latest: bool
    int8_qdq_available: bool = False
    active_precision: str = "fp32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weights_path": self.weights_path,
            "labels_path": self.labels_path,
            "weights_exists": self.weights_exists,
            "labels_exists": self.labels_exists,
            "is_available_locally": self.weights_exists and self.labels_exists,
            "is_active": self.is_active,
            "is_hf_latest": self.is_hf_latest,
            "int8_qdq_available": self.int8_qdq_available,
            "active_precision": self.active_precision,
        }


def _model_dir() -> str:
    config = get_config()
    base = config.get("MODEL_BASE_PATH", "models")
    return os.path.join(base, OBJECT_DETECTION_SUBDIR)


def _read_active_metadata(model_dir: str) -> dict[str, Any]:
    """Read model_metadata.json, which the deploy pipeline generates for the active default."""
    path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return {}


def _read_latest_models(model_dir: str) -> dict[str, Any]:
    path = os.path.join(model_dir, "latest_models.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return {}


def _detect_active_source(model_dir: str) -> str:
    """Return which resolver decided the current active model id.

    - "env_pin_task"   : task-scoped env var (systemd drop-in, etc.)
    - "env_pin_generic": generic fallback env var
    - "latest_models"  : whatever ``latest_models.json["latest"]`` points at
                         (both the UI switch and the HF default land here,
                         distinguishable only by looking at which side last
                         wrote the file)
    """
    task = _task_name_from_cache_dir(model_dir)
    if os.environ.get(f"{PIN_ENV_VAR_PREFIX}_{task}", "").strip():
        return "env_pin_task"
    if os.environ.get(PIN_ENV_VAR, "").strip():
        return "env_pin_generic"
    return "latest_models"


def build_detector_registry_payload(detector: Any | None) -> dict[str, Any]:
    """Assemble the GET /api/v1/models/detector response body.

    Args:
        detector: Optional reference to the live ``ONNXDetectionModel``
            (usually ``DetectionManager.detection_service._detector``).
            When provided, the ``runtime`` block is populated with the
            model that is **actually loaded**, not just what's on disk.

    Returns:
        JSON-serializable dict.
    """
    model_dir = _model_dir()
    latest = _read_latest_models(model_dir)
    metadata = _read_active_metadata(model_dir)

    # Active id on disk = what the app will pick up on next load. That is
    # either the pin (if any) or latest_models["latest"].
    hf_latest_id: str | None = latest.get("latest") if isinstance(latest, dict) else None
    active_source = _detect_active_source(model_dir)

    pinned_models = latest.get("pinned_models") if isinstance(latest, dict) else None
    if not isinstance(pinned_models, dict):
        pinned_models = {}

    # Effective active id when the app next loads: pin (any source) wins over hf_latest.
    pin_value = _resolve_pin_for_cache_dir(model_dir)
    active_on_disk_id = pin_value or hf_latest_id

    # Live runtime id = the model currently in-memory.
    runtime_id = None
    runtime: dict[str, Any] = {}
    if detector is not None:
        runtime_id = getattr(detector, "model_id", None) or None
        runtime = {
            "model_id": runtime_id,
            "model_path": getattr(detector, "model_path", None),
            "output_format": getattr(detector, "output_format", None),
            "input_size": list(getattr(detector, "input_size", ()) or ()),
            "num_classes": len(getattr(detector, "class_names", {}) or {}),
            "class_names": list((getattr(detector, "class_names", {}) or {}).values()),
            "conf_threshold_default": getattr(detector, "conf_threshold_default", None),
            "iou_threshold_default": getattr(detector, "iou_threshold_default", None),
        }

    # Build variants list. `pinned_models` may declare alternate variants
    # shipped with the release; latest_models["latest"] (if not already
    # listed) is merged in so the UI can always show the current default
    # row.
    variant_entries = dict(pinned_models)
    if hf_latest_id and hf_latest_id not in variant_entries:
        variant_entries[hf_latest_id] = {
            "weights_path": latest.get("weights_path", ""),
            "labels_path": latest.get("labels_path", ""),
        }

    # Top-level precision hint: used as the fallback when a per-variant
    # entry doesn't carry its own ``active_precision`` (true for the
    # simplest registries that only know the active default).
    top_level_precision = (
        str(latest.get("active_precision", "fp32"))
        if isinstance(latest, dict)
        else "fp32"
    )
    if top_level_precision not in ("fp32", "int8_qdq"):
        top_level_precision = "fp32"

    variants: list[dict[str, Any]] = []
    base = get_config().get("MODEL_BASE_PATH", "models")
    for mid, payload in sorted(variant_entries.items()):
        if not isinstance(payload, dict):
            continue
        weights_rel = str(payload.get("weights_path", ""))
        labels_rel = str(payload.get("labels_path", ""))
        weights_abs = os.path.join(base, weights_rel) if weights_rel else ""
        labels_abs = os.path.join(base, labels_rel) if labels_rel else ""

        # int8-QDQ availability = primary path OR any fallback path
        # actually exists on disk. Missing entirely on disk means the
        # operator cannot toggle int8 for this variant (UI grays the chip).
        int8_candidates_rel: list[str] = []
        primary_int8 = payload.get("weights_int8_qdq_path")
        if isinstance(primary_int8, str) and primary_int8.strip():
            int8_candidates_rel.append(primary_int8.strip())
        fallbacks_int8 = payload.get("weights_int8_qdq_fallback_paths")
        if isinstance(fallbacks_int8, list):
            for entry in fallbacks_int8:
                if (
                    isinstance(entry, str)
                    and entry.strip()
                    and entry not in int8_candidates_rel
                ):
                    int8_candidates_rel.append(entry.strip())
        int8_available = any(
            os.path.exists(os.path.join(base, rel)) for rel in int8_candidates_rel
        )

        # Per-variant precision (stored at the pinned_models[<id>] level);
        # fall back to top-level for the current default.
        precision_raw = payload.get("active_precision")
        if isinstance(precision_raw, str) and precision_raw in (
            "fp32",
            "int8_qdq",
        ):
            active_precision = precision_raw
        elif mid == hf_latest_id:
            active_precision = top_level_precision
        else:
            active_precision = "fp32"

        info = VariantInfo(
            id=mid,
            weights_path=weights_rel,
            labels_path=labels_rel,
            weights_exists=bool(weights_abs) and os.path.exists(weights_abs),
            labels_exists=bool(labels_abs) and os.path.exists(labels_abs),
            is_active=(mid == (runtime_id or active_on_disk_id)),
            is_hf_latest=(mid == hf_latest_id),
            int8_qdq_available=int8_available,
            active_precision=active_precision,
        )
        variants.append(info.to_dict())

    return {
        "model_dir": model_dir,
        "active": {
            "id": runtime_id or active_on_disk_id,
            "source": active_source,
            "env_pin_value": pin_value or None,
            "hf_latest_id": hf_latest_id,
            "runtime_matches_on_disk": (runtime_id == active_on_disk_id)
            if runtime_id
            else None,
        },
        "runtime": runtime,
        "metadata": metadata,
        "variants": variants,
    }


def variant_is_known(payload: dict[str, Any], model_id: str) -> bool:
    """Return True if ``model_id`` is a locally-available variant."""
    for v in payload.get("variants", []):
        if v.get("id") == model_id and v.get("is_available_locally"):
            return True
    return False


def variant_exists_in_registry(payload: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """Return the variant entry when ``model_id`` is listed in the registry,
    regardless of local availability. This is the whitelist gate for the
    install endpoint: only ids declared under ``pinned_models`` (or the
    current HF ``latest``) can be fetched — never arbitrary strings from
    the request body.
    """
    for v in payload.get("variants", []):
        if v.get("id") == model_id:
            return v
    return None
