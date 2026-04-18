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


__all__ = ["config_to_metadata", "resolve_active_yaml"]


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

    metadata: dict[str, Any] = {
        "framework": "yolox",
        "variant": variant,
        "architecture": arch,
        "input_size": input_size,
        "input_format": detection.get("input_format", "BGR"),
        "input_normalize": bool(detection.get("input_normalize", False)),
        "output_format": detection.get("output_format", "yolox_raw"),
        "num_classes": int(meta.get("num_classes", 0)),
        "inference_thresholds": {
            "confidence": float(detection.get("confidence_threshold", 0.15)),
            "iou_nms": float(detection.get("nms_iou_threshold", 0.50)),
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
