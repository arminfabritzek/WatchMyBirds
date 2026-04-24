"""Regression guard: the classifier metadata extractor surfaces
calibrated thresholds and input size from the canonical YAML layout.

Before 2026-04-23 the extractor only looked under ``classification.*``
or ``model.*`` for input size, and never read the decision thresholds
at all. That meant the settings page showed "31 species · 224²" for
the CLS-v2 model with no way to see how strictly each variant labels.

This test locks in that:
- Input size is read from ``detection.input_size`` (canonical)
- Confidence + genus thresholds are read from ``detection.*``
- Legacy-nested layouts (``classification.input_size``) still work
- num_classes falls back to the top-level ``classes`` list length
"""

import json
import os
import tempfile

import yaml

from web.services.model_registry_service import _build_classifier_variant_metadata


def _write_variant(model_dir: str, model_id: str, yaml_doc: dict, metrics: dict | None = None) -> None:
    with open(os.path.join(model_dir, f"{model_id}_model_config.yaml"), "w") as f:
        yaml.dump(yaml_doc, f)
    if metrics is not None:
        with open(os.path.join(model_dir, f"{model_id}_metrics.json"), "w") as f:
            json.dump(metrics, f)


class TestCanonicalYamlLayout:
    """The layout shipped from 20260423_062443 onward (Dev handoff)."""

    def test_extracts_architecture_from_detection_section(self):
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "20260424_130619",
                {
                    "detection": {
                        "architecture": "efficientnet_b2",
                        "input_size": [256, 256],
                        "confidence_threshold": 0.76,
                        "genus_fallback_threshold": 0.55,
                    },
                    "meta": {"num_classes": 33},
                },
            )
            meta = _build_classifier_variant_metadata(d, "20260424_130619")
            assert meta["architecture"] == "efficientnet_b2"

    def test_extracts_input_size_from_detection_section(self):
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "20260423_062443",
                {
                    "detection": {
                        "input_size": [288, 288],
                        "confidence_threshold": 0.88,
                        "genus_fallback_threshold": 0.55,
                    },
                    "meta": {"num_classes": 31, "trained_at": "2026-04-23"},
                    "classes": ["A", "B"] * 15 + ["C"],  # 31
                },
            )
            meta = _build_classifier_variant_metadata(d, "20260423_062443")
            assert meta["input_size"] == [288, 288]

    def test_extracts_confidence_and_genus_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "v2",
                {
                    "detection": {
                        "input_size": [288, 288],
                        "confidence_threshold": 0.88,
                        "genus_fallback_threshold": 0.55,
                    },
                    "meta": {"num_classes": 31},
                },
            )
            meta = _build_classifier_variant_metadata(d, "v2")
            assert meta["confidence_threshold"] == 0.88
            assert meta["genus_fallback_threshold"] == 0.55

    def test_legacy_calibrated_point_55_surfaces(self):
        """20250817_213043 got the app-default 0.55 stamped in so the
        App has one single code path. UI must show it just like any
        other calibrated threshold."""
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "20250817_213043",
                {
                    "detection": {
                        "input_size": [288, 288],
                        "confidence_threshold": 0.55,
                        "genus_fallback_threshold": 0.55,
                    },
                    "meta": {"num_classes": 29},
                },
            )
            meta = _build_classifier_variant_metadata(d, "20250817_213043")
            assert meta["confidence_threshold"] == 0.55


class TestBackwardCompatFallbacks:
    """Older / alternate YAML shapes must still load, just without the
    fields they don't carry."""

    def test_classification_nested_input_size_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "legacy",
                {
                    "classification": {"input_size": [224, 224]},
                    "meta": {"num_classes": 10},
                },
            )
            meta = _build_classifier_variant_metadata(d, "legacy")
            assert meta["input_size"] == [224, 224]

    def test_yaml_without_detection_section_omits_thresholds(self):
        """Legacy YAMLs without the ``detection`` section must not
        raise — they simply don't surface threshold chips."""
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "legacy",
                {
                    "model": {"input_size": [224, 224]},
                    "meta": {"num_classes": 10},
                },
            )
            meta = _build_classifier_variant_metadata(d, "legacy")
            assert "confidence_threshold" not in meta
            assert "genus_fallback_threshold" not in meta

    def test_num_classes_fallback_from_classes_list(self):
        """When ``meta.num_classes`` is absent, length of the top-level
        ``classes`` list is the source of truth (matches ONNX output
        layout)."""
        with tempfile.TemporaryDirectory() as d:
            _write_variant(
                d,
                "v3",
                {
                    "detection": {"input_size": [288, 288]},
                    "classes": [f"sp_{i}" for i in range(34)],
                },
            )
            meta = _build_classifier_variant_metadata(d, "v3")
            assert meta["num_classes"] == 34

    def test_no_yaml_at_all_returns_released_only(self):
        """No YAML, no metrics — the extractor must still run and
        return the release date inferred from the model id prefix."""
        with tempfile.TemporaryDirectory() as d:
            meta = _build_classifier_variant_metadata(d, "20250817_213043")
            assert "confidence_threshold" not in meta
            # released date comes from the id prefix
            assert meta.get("released", "").startswith("2025-08-17")


def test_metrics_json_top1_accuracy_is_still_surfaced():
    """Legacy metrics path must still work alongside the new YAML path."""
    with tempfile.TemporaryDirectory() as d:
        _write_variant(
            d,
            "v4",
            {
                "detection": {
                    "input_size": [288, 288],
                    "confidence_threshold": 0.74,
                    "genus_fallback_threshold": 0.55,
                },
                "meta": {"num_classes": 34},
            },
            metrics={"top1_accuracy": 0.94, "top5_accuracy": 0.99, "num_classes": 34},
        )
        meta = _build_classifier_variant_metadata(d, "v4")
        assert meta["top1_accuracy"] == 0.94
        assert meta["confidence_threshold"] == 0.74
        assert meta["num_classes"] == 34
