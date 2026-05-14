"""Tests for per-class confidence thresholds in the YOLOX postprocess.

The v2-coco model (2026-05-13) ships
``confidence_threshold_per_class`` in its ``model_config.yaml``. The
detector reads that into ``self.conf_per_class_id`` (ndarray indexed by
class id) and uses it to filter decoded boxes BEFORE NMS — so a 0.32
person prediction can survive even when squirrel's per-class threshold
is 0.70.

These tests use a fake ONNXDetectionModel (no ONNX, no Session) — they
exercise only the postprocess decode + per-class filter logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from detectors.detector import OUTPUT_FORMAT_YOLOX_RAW, ONNXDetectionModel


def _build_yolox_raw(boxes_xywh, obj_conf, cls_probs):
    """Build a raw YOLOX output tensor (1, N, 5+C) for tests."""
    assert len(boxes_xywh) == len(obj_conf) == len(cls_probs)
    n = len(boxes_xywh)
    c = len(cls_probs[0]) if n else 0
    arr = np.zeros((1, n, 5 + c), dtype=np.float32)
    for i in range(n):
        arr[0, i, 0:4] = boxes_xywh[i]
        arr[0, i, 4] = obj_conf[i]
        arr[0, i, 5 : 5 + c] = cls_probs[i]
    return arr


def _make_6class_model_with_per_class():
    """v2-coco-shaped fake model with the YAML's per-class thresholds wired up."""
    model = ONNXDetectionModel.__new__(ONNXDetectionModel)
    model.input_size = (640, 640)
    model.class_names = {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
        "5": "person",
    }
    model.conf_threshold_default = 0.30
    model.iou_threshold_default = 0.5
    model.output_format = OUTPUT_FORMAT_YOLOX_RAW

    per_class = {
        "bird": 0.30,
        "squirrel": 0.70,
        "cat": 0.70,
        "marten_mustelid": 0.45,
        "hedgehog": 0.35,
        "person": 0.30,
    }
    model.conf_per_class_name = per_class
    arr = np.array(
        [per_class[model.class_names[str(i)]] for i in range(6)], dtype=np.float32
    )
    model.conf_per_class_id = arr
    # Suppression + min-bbox defaults: off, so existing per-class tests
    # exercise only the threshold filter. Tests for suppression and
    # min-bbox live in test_detector_suppression_and_min_bbox.py.
    model.suppressed_classes = frozenset()
    model.suppressed_class_ids = frozenset()
    model.min_bbox_size_px = 0.0
    return model


def _make_5class_model_without_per_class():
    """Baseline 5-class model — byte-identical to pre-per-class behaviour."""
    model = ONNXDetectionModel.__new__(ONNXDetectionModel)
    model.input_size = (640, 640)
    model.class_names = {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }
    model.conf_threshold_default = 0.30
    model.iou_threshold_default = 0.5
    model.output_format = OUTPUT_FORMAT_YOLOX_RAW
    model.conf_per_class_name = {}
    model.conf_per_class_id = None
    model.suppressed_classes = frozenset()
    model.suppressed_class_ids = frozenset()
    model.min_bbox_size_px = 0.0
    return model


# ---------------------------------------------------------------------------
# Per-class threshold acceptance
# ---------------------------------------------------------------------------


def test_person_at_032_passes_per_class_floor_030():
    """Person prediction with score 0.32 passes its per-class floor (0.30)."""
    model = _make_6class_model_with_per_class()
    # obj=0.95, cls_prob person ~ 0.34 -> score = 0.95 * 0.34 ~= 0.32
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 200.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.01, 0.01, 0.01, 0.01, 0.34)],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert len(dets) == 1
    assert dets[0]["class"] == 5  # person
    assert dets[0]["confidence"] == pytest.approx(0.323, abs=0.01)


def test_squirrel_at_055_dropped_by_per_class_floor_070():
    """Squirrel prediction with score 0.55 fails its per-class floor (0.70).

    The pre-per-class code path would have kept it (0.55 > scalar 0.30).
    Per-class is strictly stricter than the scalar fallback for squirrel.
    """
    model = _make_6class_model_with_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.58, 0.01, 0.01, 0.01, 0.01)],  # squirrel ~0.55
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert dets == []


def test_marten_at_050_passes_per_class_floor_045():
    """Marten at 0.50 passes its 0.45 floor; same conf would be dropped by
    the old global non-bird floor of 0.80, surfacing the change's value."""
    model = _make_6class_model_with_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.01, 0.01, 0.53, 0.01, 0.01)],  # marten ~0.50
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert len(dets) == 1
    assert dets[0]["class"] == 3  # marten_mustelid


def test_marten_at_040_dropped_by_per_class_floor_045():
    """Borderline: marten at 0.40 just below its 0.45 floor — dropped."""
    model = _make_6class_model_with_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.9],
        cls_probs=[(0.01, 0.01, 0.01, 0.45, 0.01, 0.01)],  # marten ~0.40
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert dets == []


def test_mixed_frame_keeps_only_classes_passing_their_floor():
    """One frame, three boxes: bird@0.40 (keep), squirrel@0.55 (drop),
    person@0.32 (keep). Verifies per-class filter is per-prediction, not
    per-frame."""
    model = _make_6class_model_with_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[
            (100.0, 100.0, 80.0, 80.0),  # bird
            (300.0, 300.0, 80.0, 80.0),  # squirrel (will drop)
            (500.0, 300.0, 60.0, 200.0),  # person
        ],
        obj_conf=[0.9, 0.9, 0.95],
        cls_probs=[
            (0.45, 0.01, 0.01, 0.01, 0.01, 0.01),  # bird ~0.40
            (0.01, 0.61, 0.01, 0.01, 0.01, 0.01),  # squirrel ~0.55
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.34),  # person ~0.32
        ],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    classes = sorted(d["class"] for d in dets)
    assert classes == [0, 5]  # bird + person, squirrel dropped


# ---------------------------------------------------------------------------
# 5-class regression guard
# ---------------------------------------------------------------------------


def test_5class_no_per_class_block_uses_scalar_floor():
    """A 5-class model with conf_per_class_id=None must use the scalar
    conf_threshold — byte-identical to pre-per-class behaviour."""
    model = _make_5class_model_without_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[
            (200.0, 200.0, 80.0, 80.0),
            (400.0, 400.0, 80.0, 80.0),
        ],
        obj_conf=[0.9, 0.9],
        cls_probs=[
            (0.45, 0.01, 0.01, 0.01, 0.01),  # bird ~0.40 -> keep
            (0.01, 0.61, 0.01, 0.01, 0.01),  # squirrel ~0.55 -> keep (scalar 0.30)
        ],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    classes = sorted(d["class"] for d in dets)
    assert classes == [0, 1]  # both kept — proves scalar path is alive


def test_5class_low_score_still_dropped_by_scalar():
    """A 5-class model still drops 0.20 detections at scalar 0.30."""
    model = _make_5class_model_without_per_class()
    raw = _build_yolox_raw(
        boxes_xywh=[(200.0, 200.0, 80.0, 80.0)],
        obj_conf=[0.5],
        cls_probs=[(0.40, 0.01, 0.01, 0.01, 0.01)],  # bird ~0.20
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert dets == []
