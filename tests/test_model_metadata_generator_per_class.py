"""Tests for the per-class confidence threshold pass-through in
``utils.model_metadata_generator``.

The v2-coco YAML (2026-05-13 HF release) ships
``detection.confidence_threshold_per_class`` — a dict that maps each OD
class name to its calibrated operating point (e.g. ``person: 0.30``,
``squirrel: 0.70``). The generator must propagate that block verbatim
into ``model_metadata.json`` so the detector can apply per-class
thresholds at decode time. Malformed entries are dropped with a warning,
not raised — the goal is "as-good-as-possible per-class behaviour" with
a clean fallback to the scalar threshold for any class missing or invalid.
"""

from __future__ import annotations

from utils.model_metadata_generator import (
    _coerce_min_bbox_size_px,
    _coerce_per_class,
    _coerce_suppressed_classes,
    config_to_metadata,
)

V2_COCO_CONFIG = {
    "detection": {
        "confidence_threshold": 0.30,
        "nms_iou_threshold": 0.5,
        "input_size": [640, 640],
        "input_format": "BGR",
        "input_normalize": False,
        "output_format": "yolox_raw",
        "architecture": "yolox_s_locator_v2_6cls",
        "confidence_threshold_per_class": {
            "bird": 0.30,
            "squirrel": 0.70,
            "cat": 0.70,
            "marten_mustelid": 0.45,
            "hedgehog": 0.35,
            "person": 0.30,
        },
    },
    "meta": {
        "num_classes": 6,
        "classes": ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog", "person"],
    },
}


def test_v2_coco_propagates_per_class_block():
    md = config_to_metadata(V2_COCO_CONFIG, source_yaml_name="v2_coco.yaml")
    per_class = md["inference_thresholds"]["confidence_per_class"]
    assert per_class == {
        "bird": 0.30,
        "squirrel": 0.70,
        "cat": 0.70,
        "marten_mustelid": 0.45,
        "hedgehog": 0.35,
        "person": 0.30,
    }
    assert md["num_classes"] == 6
    assert md["classes"] == [
        "bird",
        "squirrel",
        "cat",
        "marten_mustelid",
        "hedgehog",
        "person",
    ]
    # Scalar fallback stays present alongside per-class — callers without
    # per-class awareness keep working.
    assert md["inference_thresholds"]["confidence"] == 0.30


def test_5class_model_emits_empty_per_class():
    """5-class models without a per-class block produce {} (not missing).

    Downstream ``get(...)`` should always be total — the detector loader
    treats ``{}`` as "use the scalar threshold for everything".
    """
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "nms_iou_threshold": 0.5,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_5cls",
        },
        "meta": {
            "num_classes": 5,
            "classes": ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"],
        },
    }
    md = config_to_metadata(cfg, source_yaml_name="five.yaml")
    assert md["inference_thresholds"]["confidence_per_class"] == {}
    assert md["inference_thresholds"]["confidence"] == 0.30


def test_malformed_entries_are_dropped_not_raised(caplog):
    """Malformed values log a warning and are dropped; valid entries survive."""
    raw = {
        "bird": 0.30,
        "person": "not_a_number",  # not float-coercible
        "squirrel": 1.5,  # outside [0, 1]
        "cat": -0.1,  # outside [0, 1]
        "marten_mustelid": 0.45,
        "hedgehog": None,  # not float-coercible
    }
    with caplog.at_level("WARNING"):
        out = _coerce_per_class(raw)
    assert out == {"bird": 0.30, "marten_mustelid": 0.45}
    # One warning per dropped entry.
    assert sum(1 for r in caplog.records if r.levelname == "WARNING") == 4


def test_per_class_block_not_a_dict_falls_back_to_empty(caplog):
    """If someone ships ``confidence_threshold_per_class: 0.5`` (scalar) or
    a list, log once and return ``{}``."""
    with caplog.at_level("WARNING"):
        assert _coerce_per_class(0.5) == {}
        assert _coerce_per_class(["bird", "person"]) == {}
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2


def test_coerce_per_class_none_returns_empty():
    """Missing block (None) → empty dict, no log spam."""
    assert _coerce_per_class(None) == {}


def test_classes_list_preserved_when_absent_in_meta():
    """``classes`` should be ``[]`` (not missing) when meta has no list."""
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "nms_iou_threshold": 0.5,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_5cls",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="no_classes.yaml")
    assert md["classes"] == []


# ---------------------------------------------------------------------------
# Suppressed classes
# ---------------------------------------------------------------------------


def test_suppressed_classes_propagates():
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_v2_6cls",
            "suppressed_classes": ["person"],
        },
        "meta": {
            "num_classes": 6,
            "classes": [
                "bird",
                "squirrel",
                "cat",
                "marten_mustelid",
                "hedgehog",
                "person",
            ],
        },
    }
    md = config_to_metadata(cfg, source_yaml_name="v2_coco.yaml")
    assert md["inference_thresholds"]["suppressed_classes"] == ["person"]


def test_suppressed_classes_lowercases_and_dedupes():
    raw = ["Person", "person", "  PERSON  ", "dog", "Dog"]
    assert _coerce_suppressed_classes(raw) == ["person", "dog"]


def test_suppressed_classes_drops_non_strings(caplog):
    with caplog.at_level("WARNING"):
        result = _coerce_suppressed_classes(["person", 42, None, "", "dog"])
    assert result == ["person", "dog"]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 3  # 42, None, empty string


def test_suppressed_classes_not_a_list_falls_back_to_empty(caplog):
    with caplog.at_level("WARNING"):
        assert _coerce_suppressed_classes("person") == []
        assert _coerce_suppressed_classes({"person": True}) == []
    assert sum(1 for r in caplog.records if r.levelname == "WARNING") == 2


def test_suppressed_classes_none_returns_empty():
    assert _coerce_suppressed_classes(None) == []


def test_5class_model_emits_empty_suppressed_classes():
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_5cls",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="five.yaml")
    assert md["inference_thresholds"]["suppressed_classes"] == []


# ---------------------------------------------------------------------------
# Min bbox size
# ---------------------------------------------------------------------------


def test_min_bbox_size_default_is_8():
    """When the YAML doesn't set min_bbox_size_px, default to 8 px."""
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_5cls",
        },
        "meta": {"num_classes": 5},
    }
    md = config_to_metadata(cfg, source_yaml_name="default.yaml")
    assert md["inference_thresholds"]["min_bbox_size_px"] == 8.0


def test_min_bbox_size_explicit_value():
    cfg = {
        "detection": {
            "confidence_threshold": 0.30,
            "input_size": [640, 640],
            "architecture": "yolox_s_locator_v2_6cls",
            "min_bbox_size_px": 12,
        },
        "meta": {"num_classes": 6},
    }
    md = config_to_metadata(cfg, source_yaml_name="custom.yaml")
    assert md["inference_thresholds"]["min_bbox_size_px"] == 12.0


def test_min_bbox_size_negative_falls_back(caplog):
    with caplog.at_level("WARNING"):
        result = _coerce_min_bbox_size_px(-1.0)
    assert result == 8.0
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_min_bbox_size_non_numeric_falls_back(caplog):
    with caplog.at_level("WARNING"):
        result = _coerce_min_bbox_size_px("eight")
    assert result == 8.0
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_min_bbox_size_zero_disables_filter():
    """0.0 is a legitimate "disable" value, not a fallback trigger."""
    assert _coerce_min_bbox_size_px(0) == 0.0
    assert _coerce_min_bbox_size_px(0.0) == 0.0
