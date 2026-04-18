"""Unit tests for the YOLOX detector backend (Phase 0 of the YOLOX swap).

Covers the pure helpers in detectors/detector.py:
- _normalize_class_names (dict + list formats)
- _detect_output_format (FasterRCNN vs YOLOX)
- _assert_yolox_labels_compatible (Model-Compatibility-Guard)
- ONNXDetectionModel.preprocess_image_yolox (letterbox math)
- ONNXDetectionModel.postprocess_output_yolox (raw YOLOX decode)

These tests do NOT load a real ONNX; they exercise the decoding math and the
guard logic directly so they run fast and don't depend on downloaded weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from detectors.detector import (
    OUTPUT_FORMAT_YOLOX_RAW,
    ONNXDetectionModel,
    _assert_yolox_labels_compatible,
    _detect_output_format,
    _normalize_class_names,
)


# ---------------------------------------------------------------------------
# _normalize_class_names
# ---------------------------------------------------------------------------


def test_normalize_class_names_dict_passes_through():
    raw = {"0": "Turdus_merula", "1": "Parus_major"}
    assert _normalize_class_names(raw) == {"0": "Turdus_merula", "1": "Parus_major"}


def test_normalize_class_names_list_indexed():
    raw = ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"]
    assert _normalize_class_names(raw) == {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }


def test_normalize_class_names_rejects_bad_type():
    with pytest.raises(ValueError, match="Unsupported labels.json format"):
        _normalize_class_names("not a dict or list")


# ---------------------------------------------------------------------------
# _detect_output_format
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, shape):
        self.shape = shape


class _FakeSession:
    def __init__(self, shape):
        self._outputs = [_FakeOutput(shape)]

    def get_outputs(self):
        return self._outputs


def test_detect_output_format_rejects_legacy_fasterrcnn():
    # Legacy FasterRCNN post-NMS (last-axis 6) is no longer supported.
    # The startup cleanup removes such artefacts, but we still want a
    # clear error if someone points LOCAL_PATH at one.
    session = _FakeSession([1, "num_dets", 6])
    labels = {"0": "Turdus_merula", "1": "Parus_major"}
    with pytest.raises(ValueError, match="Legacy FasterRCNN"):
        _detect_output_format(session, labels)


def test_detect_output_format_yolox_5_classes():
    # YOLOX raw: last axis == 5 + num_classes
    session = _FakeSession([1, 8400, 10])
    labels = {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }
    assert _detect_output_format(session, labels) == OUTPUT_FORMAT_YOLOX_RAW


def test_detect_output_format_rejects_unknown():
    session = _FakeSession([1, 100, 12])  # neither 6 nor 5+5=10
    labels = {"0": "bird", "1": "squirrel", "2": "cat", "3": "marten", "4": "hedgehog"}
    with pytest.raises(ValueError, match="Unrecognized ONNX output shape"):
        _detect_output_format(session, labels)


def test_detect_output_format_rejects_multi_output():
    class _MultiOut:
        def get_outputs(self):
            return [_FakeOutput([1, 8400, 10]), _FakeOutput([1, 8400, 10])]

    with pytest.raises(ValueError, match="expected 1 output tensor"):
        _detect_output_format(_MultiOut(), {"0": "bird"})


# ---------------------------------------------------------------------------
# Model-Compatibility-Guard
# ---------------------------------------------------------------------------


def test_guard_accepts_new_yolox_labels():
    labels = {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }
    # Should not raise.
    _assert_yolox_labels_compatible(labels)


def test_guard_rejects_29_species_rollback():
    # Exactly the labels from the legacy FasterRCNN labels.json.
    labels = {
        "0": "Turdus_merula",
        "1": "Fringilla_montifringilla",
        "2": "Cyanistes_caeruleus",
        "3": "Fringilla_coelebs",
    }
    with pytest.raises(ValueError, match="expected YOLOX-style locator"):
        _assert_yolox_labels_compatible(labels)


def test_guard_rejects_single_class_without_bird():
    labels = {"0": "vehicle"}
    with pytest.raises(ValueError, match="expected YOLOX-style locator"):
        _assert_yolox_labels_compatible(labels)


def test_guard_accepts_bird_plus_anything():
    # Guard only requires that 'bird' is present, not a specific count.
    labels = {"0": "bird", "1": "squirrel"}
    _assert_yolox_labels_compatible(labels)


# ---------------------------------------------------------------------------
# Model-owned thresholds regression guard (2026-04-18)
# ---------------------------------------------------------------------------


def test_detect_signature_ignores_caller_confidence():
    """Regression guard: detect() must ignore the confidence_threshold
    argument. An earlier implementation ceiling'd the model's own
    conf_threshold_default with the caller value via max(), which
    silently defeated per-variant calibration (Tiny 0.15 / S 0.30
    got lifted to the old user default 0.65 and destroyed recall).

    This test does not exercise the full inference stack; it only
    verifies the signature accepts the legacy kwarg without using it
    at runtime. A behavioural test lives in test_detector_yolox_load.
    """
    import inspect

    from detectors.detector import ONNXDetectionModel

    sig = inspect.signature(ONNXDetectionModel.detect)
    params = list(sig.parameters)
    # self + frame + confidence_threshold (optional, kept for back-compat)
    assert params[:3] == ["self", "frame", "confidence_threshold"]
    # The kwarg must be optional so DetectionService may omit it.
    assert sig.parameters["confidence_threshold"].default is None


# ---------------------------------------------------------------------------
# preprocess_image_yolox — letterbox math
# ---------------------------------------------------------------------------


def _make_fake_model():
    """Construct a bare ONNXDetectionModel WITHOUT running __init__.

    We only need a few attributes to exercise the pure preprocessing and
    postprocessing helpers; constructing the full model would require loading
    real ONNX weights which this test suite deliberately avoids.
    """
    model = ONNXDetectionModel.__new__(ONNXDetectionModel)
    model.input_size = (640, 640)
    model.class_names = {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }
    model.conf_threshold_default = 0.15
    model.iou_threshold_default = 0.5
    model.output_format = OUTPUT_FORMAT_YOLOX_RAW
    return model


def test_preprocess_yolox_square_image():
    model = _make_fake_model()
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)  # solid BGR
    chw, original, scale = model.preprocess_image_yolox(img)

    assert chw.shape == (1, 3, 640, 640)
    assert chw.dtype == np.float32
    assert scale == pytest.approx(1.0)
    # BGR preserved: channel 0 == 10, channel 1 == 20, channel 2 == 30
    assert chw[0, 0, 0, 0] == pytest.approx(10.0)
    assert chw[0, 1, 0, 0] == pytest.approx(20.0)
    assert chw[0, 2, 0, 0] == pytest.approx(30.0)
    # No /255 normalization: value is still in 0..255 range
    assert chw.max() >= 30.0


def test_preprocess_yolox_rectangular_pads_with_114():
    model = _make_fake_model()
    img = np.zeros((320, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    chw, original, scale = model.preprocess_image_yolox(img)

    # 640/320 = 2.0, 640/640 = 1.0 -> scale = min = 1.0
    assert scale == pytest.approx(1.0)
    # New size = (640, 320), padded bottom region should be 114
    assert chw[0, 0, 320, 0] == pytest.approx(114.0)
    assert chw[0, 0, 639, 0] == pytest.approx(114.0)
    # Filled region still carries source value
    assert chw[0, 0, 0, 0] == pytest.approx(50.0)


def test_preprocess_yolox_downscale_big_image():
    model = _make_fake_model()
    img = np.zeros((1280, 1280, 3), dtype=np.uint8)
    chw, original, scale = model.preprocess_image_yolox(img)

    assert scale == pytest.approx(0.5)
    assert chw.shape == (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# postprocess_output_yolox — raw decode math
# ---------------------------------------------------------------------------


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


def test_postprocess_yolox_picks_bird():
    model = _make_fake_model()
    # One centred box (cx=320, cy=320, w=100, h=100) with high obj + high bird prob
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.9, 0.01, 0.01, 0.01, 0.01)],  # bird
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.15,
        iou_threshold=0.5,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d["class"] == 0  # bird
    assert d["confidence"] > 0.8
    # Box centred at (320,320) with size 100 -> xyxy = (270, 270, 370, 370)
    assert d["x1"] == pytest.approx(270, abs=2)
    assert d["y1"] == pytest.approx(270, abs=2)
    assert d["x2"] == pytest.approx(370, abs=2)
    assert d["y2"] == pytest.approx(370, abs=2)


def test_postprocess_yolox_picks_squirrel():
    model = _make_fake_model()
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.9],
        cls_probs=[(0.01, 0.95, 0.01, 0.01, 0.01)],  # squirrel
    )
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    assert len(dets) == 1
    assert dets[0]["class"] == 1  # squirrel


def test_postprocess_yolox_filters_low_confidence():
    model = _make_fake_model()
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.1],
        cls_probs=[(0.1, 0.01, 0.01, 0.01, 0.01)],  # score ~0.01
    )
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    assert dets == []


def test_postprocess_yolox_nms_keeps_one_of_overlapping():
    model = _make_fake_model()
    # Two boxes same class, heavily overlapping
    raw = _build_yolox_raw(
        boxes_xywh=[
            (320.0, 320.0, 100.0, 100.0),
            (322.0, 322.0, 100.0, 100.0),
        ],
        obj_conf=[0.95, 0.90],
        cls_probs=[(0.9, 0.01, 0.01, 0.01, 0.01), (0.85, 0.01, 0.01, 0.01, 0.01)],
    )
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    assert len(dets) == 1  # NMS dedups to highest-score


def test_postprocess_yolox_nms_keeps_different_classes():
    model = _make_fake_model()
    # Two boxes same region but different classes -> per-class NMS keeps both
    raw = _build_yolox_raw(
        boxes_xywh=[
            (320.0, 320.0, 100.0, 100.0),
            (320.0, 320.0, 100.0, 100.0),
        ],
        obj_conf=[0.95, 0.90],
        cls_probs=[
            (0.9, 0.01, 0.01, 0.01, 0.01),  # bird
            (0.01, 0.85, 0.01, 0.01, 0.01),  # squirrel
        ],
    )
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    classes = sorted(d["class"] for d in dets)
    assert classes == [0, 1]


def test_postprocess_yolox_scale_inverse_maps_to_original():
    model = _make_fake_model()
    # Input image is 1280x1280 -> scale=0.5 (640/1280).
    # Letterbox-space box at centre (320,320, 100x100) -> original (640,640, 200x200)
    # i.e. xyxy = (540, 540, 740, 740)
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.9, 0.01, 0.01, 0.01, 0.01)],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=0.5,
        original_width=1280,
        original_height=1280,
        conf_threshold=0.15,
        iou_threshold=0.5,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d["x1"] == pytest.approx(540, abs=2)
    assert d["y1"] == pytest.approx(540, abs=2)
    assert d["x2"] == pytest.approx(740, abs=2)
    assert d["y2"] == pytest.approx(740, abs=2)


def test_postprocess_yolox_clips_to_image_bounds():
    model = _make_fake_model()
    # Box centred at (10, 10) with size 100 -> xyxy (-40, -40, 60, 60).
    # After clipping: (0, 0, 60, 60).
    raw = _build_yolox_raw(
        boxes_xywh=[(10.0, 10.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.9, 0.01, 0.01, 0.01, 0.01)],
    )
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d["x1"] == 0
    assert d["y1"] == 0
    assert d["x2"] == pytest.approx(60, abs=2)
    assert d["y2"] == pytest.approx(60, abs=2)


def test_postprocess_yolox_handles_empty_input():
    model = _make_fake_model()
    raw = np.zeros((1, 0, 10), dtype=np.float32)
    dets = model.postprocess_output_yolox(
        raw, scale=1.0, original_width=640, original_height=640,
        conf_threshold=0.15, iou_threshold=0.5,
    )
    assert dets == []
