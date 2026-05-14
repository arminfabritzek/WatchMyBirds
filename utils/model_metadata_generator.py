"""Convert ``*_model_config.yaml`` releases into the runtime
``model_metadata.json`` consumed by the detector.

This module has **two** consumers:

1. :mod:`web.blueprints.api_v1` — the pin endpoint re-runs the
   conversion whenever the user switches the active variant, so the
   next detector reload picks up the right conf/iou thresholds.
2. ``scripts/generate_model_metadata.py`` — the CLI wrapper used at
   release time to regenerate ``model_metadata.json``.

Keeping the logic here (inside ``utils/``) avoids the runtime code
path reaching into ``scripts/``, which means the Docker image does not
need a special-case ``COPY scripts/…`` line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from logging_config import get_logger

__all__ = ["config_to_metadata", "resolve_active_yaml"]

logger = get_logger(__name__)


def _coerce_suppressed_classes(raw: Any) -> list[str]:
    """Validate a ``suppressed_classes`` YAML block.

    Expected shape: a list of class-name strings. Anything else is
    dropped with a warning. Returned list is lowercased + deduplicated
    in stable order. Used both at metadata-generation time and (via
    the metadata JSON) by the detector loader to hard-drop matching
    classes before NMS / save / CLS / scoring.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "suppressed_classes is %s, expected list; ignoring.",
            type(raw).__name__,
        )
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            logger.warning(
                "suppressed_classes entry %r is not a string; dropped.",
                item,
            )
            continue
        name = item.strip().lower()
        if not name:
            logger.warning("suppressed_classes entry is empty; dropped.")
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _coerce_min_bbox_size_px(raw: Any, default: float = 8.0) -> float:
    """Validate ``min_bbox_size_px`` YAML value (default 8 px, must be >= 0)."""
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "min_bbox_size_px=%r is not numeric; using default %s.",
            raw,
            default,
        )
        return default
    if v < 0:
        logger.warning(
            "min_bbox_size_px=%s < 0; using default %s.",
            v,
            default,
        )
        return default
    return v


def _coerce_per_class(raw: Any) -> dict[str, float]:
    """Validate a ``confidence_threshold_per_class`` YAML block.

    Expected shape: ``{class_name: float in [0.0, 1.0]}``. Anything else is
    dropped with a warning and the dict is returned without that key. A
    missing/None block returns ``{}`` so downstream ``get(...)`` lookups
    are total. Used both at metadata-generation time (this module) and
    indirectly by the detector loader, which reads the dict back out of
    ``model_metadata.json``.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "confidence_threshold_per_class is %s, expected dict; ignoring.",
            type(raw).__name__,
        )
        return {}

    out: dict[str, float] = {}
    for key, value in raw.items():
        name = str(key)
        try:
            v = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "confidence_threshold_per_class[%r] is not numeric (%r); dropped.",
                name,
                value,
            )
            continue
        if not (0.0 <= v <= 1.0):
            logger.warning(
                "confidence_threshold_per_class[%r]=%s outside [0.0, 1.0]; dropped.",
                name,
                v,
            )
            continue
        out[name] = v
    return out


def config_to_metadata(
    config: dict[str, Any], *, source_yaml_name: str
) -> dict[str, Any]:
    """Convert parsed ``model_config.yaml`` into the app's ``model_metadata.json``."""
    detection = config.get("detection") or {}
    meta = config.get("meta") or {}
    metrics = config.get("metrics_at_chosen_threshold") or {}

    arch = str(detection.get("architecture") or "")
    arch_lower = arch.lower()
    if "tiny" in arch_lower:
        variant = "tiny"
    elif "_s_" in arch_lower or arch_lower.endswith("_s"):
        variant = "s"
    elif "_n_" in arch_lower or arch_lower.endswith("_n"):
        variant = "n"
    else:
        variant = "unknown"

    input_size = detection.get("input_size") or [640, 640]
    if isinstance(input_size, list):
        input_size = [int(v) for v in input_size]

    classes_raw = meta.get("classes")
    classes: list[str] = (
        [str(c) for c in classes_raw] if isinstance(classes_raw, list) else []
    )

    metadata: dict[str, Any] = {
        "framework": "yolox",
        "variant": variant,
        "architecture": arch,
        "input_size": input_size,
        "input_format": detection.get("input_format", "BGR"),
        "input_normalize": bool(detection.get("input_normalize", False)),
        "output_format": detection.get("output_format", "yolox_raw"),
        "num_classes": int(meta.get("num_classes", 0)),
        "classes": classes,
        "inference_thresholds": {
            "confidence": float(detection.get("confidence_threshold", 0.15)),
            "iou_nms": float(detection.get("nms_iou_threshold", 0.50)),
            "confidence_per_class": _coerce_per_class(
                detection.get("confidence_threshold_per_class")
            ),
            "suppressed_classes": _coerce_suppressed_classes(
                detection.get("suppressed_classes")
            ),
            "min_bbox_size_px": _coerce_min_bbox_size_px(
                detection.get("min_bbox_size_px")
            ),
        },
        "generated_from": source_yaml_name,
    }

    if metrics:
        metadata["metrics"] = {
            k: metrics[k]
            for k in (
                "bird_recall",
                "bird_precision",
                "anim_to_bird",
                "empty_fp",
                "f1",
            )
            if k in metrics
        }

    return metadata


def resolve_active_yaml(model_dir: Path) -> tuple[Path, Path]:
    """Given a model_dir, return (yaml_path, metadata_out_path) for the
    active default variant — used by the CLI wrapper when the caller
    only wants to regenerate for whatever is currently pinned."""
    latest_path = model_dir / "latest_models.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Missing {latest_path}")
    data = json.loads(latest_path.read_text())
    latest_id = data.get("latest")
    if not latest_id:
        raise ValueError(f"{latest_path} has no 'latest' field")
    yaml_path = model_dir / f"{latest_id}_model_config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Expected config YAML not present: {yaml_path}")
    return yaml_path, model_dir / "model_metadata.json"
