"""Tests for utils.model_metadata_generator.

The ``*_model_config.yaml`` released alongside each model is the source
of truth for thresholds, input format, and output format. This module
converts it into the app's ``model_metadata.json``.

The module lives in ``utils/`` (not ``scripts/``) so the Flask app can
import it at runtime without shipping the ``scripts/`` directory into
the Docker image.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.model_metadata_generator import (
    config_to_metadata,
    resolve_active_yaml,
)

# Backwards-compatible alias for the old _resolve_active_yaml helper name.
_resolve_active_yaml = resolve_active_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


TINY_CONFIG = {
    "detection": {
        "confidence_threshold": 0.15,
        "nms_iou_threshold": 0.5,
        "input_size": [640, 640],
        "input_format": "BGR",
        "input_normalize": False,
        "output_format": "yolox_raw",
        "architecture": "yolox_tiny_locator_5cls",
        "weights_file": "20260417_yolox_tiny_locator_ep120_best.onnx",
    },
    "meta": {
        "version": "20260417_yolox_tiny_locator_ep120",
        "num_classes": 5,
        "classes": ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"],
    },
    "metrics_at_chosen_threshold": {
        "conf": 0.15,
        "bird_recall": 0.993,
        "bird_precision": 0.9914,
        "anim_to_bird": 0.1429,
        "empty_fp": 0.0,
        "f1": 0.9922,
    },
}

S_CONFIG = {
    "detection": {
        "confidence_threshold": 0.30,
        "nms_iou_threshold": 0.5,
        "input_size": [640, 640],
        "input_format": "BGR",
        "input_normalize": False,
        "output_format": "yolox_raw",
        "architecture": "yolox_s_locator_5cls",
    },
    "meta": {"num_classes": 5},
    "metrics_at_chosen_threshold": {
        "bird_recall": 0.9875,
        "bird_precision": 0.9953,
        "anim_to_bird": 0.0779,
        "empty_fp": 0.0,
        "f1": 0.9914,
    },
}


# ---------------------------------------------------------------------------
# config_to_metadata
# ---------------------------------------------------------------------------


def test_tiny_config_produces_expected_metadata():
    md = config_to_metadata(TINY_CONFIG, source_yaml_name="tiny.yaml")

    assert md["framework"] == "yolox"
    assert md["variant"] == "tiny"
    assert md["architecture"] == "yolox_tiny_locator_5cls"
    assert md["input_size"] == [640, 640]
    assert md["input_format"] == "BGR"
    assert md["input_normalize"] is False
    assert md["output_format"] == "yolox_raw"
    assert md["num_classes"] == 5
    assert md["inference_thresholds"] == {"confidence": 0.15, "iou_nms": 0.5}
    assert md["generated_from"] == "tiny.yaml"
    assert md["metrics"]["bird_recall"] == 0.993
    assert md["metrics"]["anim_to_bird"] == 0.1429


def test_s_config_produces_different_threshold():
    md = config_to_metadata(S_CONFIG, source_yaml_name="s.yaml")
    assert md["variant"] == "s"
    assert md["inference_thresholds"]["confidence"] == 0.30
    # S has lower anim_to_bird than Tiny
    assert md["metrics"]["anim_to_bird"] == pytest.approx(0.0779)


def test_missing_metrics_still_valid():
    cfg = {
        "detection": {
            "confidence_threshold": 0.15,
            "nms_iou_threshold": 0.5,
            "input_size": [640, 640],
            "architecture": "yolox_tiny_locator_5cls",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="minimal.yaml")
    assert "metrics" not in md
    assert md["inference_thresholds"]["confidence"] == 0.15


def test_unknown_architecture_variant_defaults():
    cfg = {
        "detection": {
            "confidence_threshold": 0.15,
            "nms_iou_threshold": 0.5,
            "input_size": [640, 640],
            "architecture": "some_future_detector",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="future.yaml")
    assert md["variant"] == "unknown"


def test_default_thresholds_when_missing():
    """Defensive: missing threshold keys should fall back to F1-optimal defaults."""
    cfg = {
        "detection": {
            "input_size": [640, 640],
            "architecture": "yolox_tiny_locator_5cls",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="defaults.yaml")
    assert md["inference_thresholds"]["confidence"] == 0.15
    assert md["inference_thresholds"]["iou_nms"] == 0.50


# ---------------------------------------------------------------------------
# _resolve_active_yaml
# ---------------------------------------------------------------------------


def test_resolve_active_yaml_picks_latest(tmp_path):
    (tmp_path / "latest_models.json").write_text(
        json.dumps({"latest": "20260417_yolox_tiny_locator_ep120"})
    )
    (tmp_path / "20260417_yolox_tiny_locator_ep120_model_config.yaml").write_text("detection: {}\n")

    yaml_path, output_path = _resolve_active_yaml(tmp_path)
    assert yaml_path.name == "20260417_yolox_tiny_locator_ep120_model_config.yaml"
    assert output_path.name == "model_metadata.json"
    assert output_path.parent == tmp_path


def test_resolve_active_yaml_missing_latest_errors(tmp_path):
    (tmp_path / "latest_models.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="no 'latest' field"):
        _resolve_active_yaml(tmp_path)


def test_resolve_active_yaml_missing_config_errors(tmp_path):
    (tmp_path / "latest_models.json").write_text(
        json.dumps({"latest": "nonexistent_model_id"})
    )
    with pytest.raises(FileNotFoundError, match="config YAML not present"):
        _resolve_active_yaml(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end: real release YAMLs round-trip through yaml.safe_load
# ---------------------------------------------------------------------------


def test_real_release_yaml_roundtrip_tiny():
    yaml_path = REPO_ROOT / "inbox" / "release" / "object_detection" / "20260417_yolox_tiny_locator_ep120_model_config.yaml"
    if not yaml_path.is_file():
        pytest.skip(f"Release YAML not present at {yaml_path}")
    import yaml as _yaml

    config = _yaml.safe_load(yaml_path.read_text())
    md = config_to_metadata(config, source_yaml_name=yaml_path.name)
    assert md["variant"] == "tiny"
    assert md["inference_thresholds"]["confidence"] == 0.15
    assert md["num_classes"] == 5
