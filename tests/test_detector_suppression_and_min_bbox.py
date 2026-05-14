"""Tests for class-suppression + min-bbox-size filter in the YOLOX postprocess.

These two filters run in this order, BEFORE the per-class threshold:

1. Class suppression — drop and audit-log detections whose class is in
   ``suppressed_class_ids`` (model-owned via YAML, with settings.yaml
   override).
2. Min bbox size — drop detections whose w or h is below
   ``min_bbox_size_px`` in 640er input-space pixels. Targets v2-coco's
   top-left tiny-box edge artifact (1.5x1.6 px at conf 0.4-0.55).

The tests use a fake ONNXDetectionModel (no ONNX, no Session) — they
exercise only the postprocess decode + filter logic.
"""

from __future__ import annotations

import json

import numpy as np

from detectors.detector import OUTPUT_FORMAT_YOLOX_RAW, ONNXDetectionModel


def _build_yolox_raw(boxes_xywh, obj_conf, cls_probs):
    assert len(boxes_xywh) == len(obj_conf) == len(cls_probs)
    n = len(boxes_xywh)
    c = len(cls_probs[0]) if n else 0
    arr = np.zeros((1, n, 5 + c), dtype=np.float32)
    for i in range(n):
        arr[0, i, 0:4] = boxes_xywh[i]
        arr[0, i, 4] = obj_conf[i]
        arr[0, i, 5 : 5 + c] = cls_probs[i]
    return arr


def _make_v2_coco_model_with_suppression(suppressed=None, min_size=8.0):
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
    # Per-class active (matches v2-coco)
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
    # Suppression + min-bbox
    suppressed = suppressed or []
    model.suppressed_classes = frozenset(s.lower() for s in suppressed)
    model.suppressed_class_ids = frozenset(
        int(idx)
        for idx, name in model.class_names.items()
        if name.lower() in model.suppressed_classes
    )
    model.min_bbox_size_px = float(min_size)
    return model


# ---------------------------------------------------------------------------
# Class suppression
# ---------------------------------------------------------------------------


def test_person_suppressed_does_not_persist(monkeypatch, tmp_path):
    """When 'person' is in suppressed_classes, a high-conf person box
    yields zero detections out of postprocess and zero NMS calls."""
    # Redirect audit log to tmp
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=0.0)
    # Person at 0.9 — would normally CONFIRM
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 120.0, 240.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.01, 0.01, 0.01, 0.01, 0.95)],
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


def test_audit_log_written_for_each_suppressed_detection(monkeypatch, tmp_path):
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=0.0)
    # Three person boxes
    raw = _build_yolox_raw(
        boxes_xywh=[
            (100.0, 100.0, 80.0, 200.0),
            (300.0, 300.0, 80.0, 200.0),
            (500.0, 100.0, 80.0, 200.0),
        ],
        obj_conf=[0.9, 0.9, 0.9],
        cls_probs=[
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.9),
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.85),
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.7),
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
    assert dets == []

    audit_path = tmp_path / "logs" / "suppressed.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        entry = json.loads(line)
        assert entry["class"] == "person"
        assert entry["class_id"] == 5
        assert entry["reason"] == "suppressed_classes"
        assert 0 < entry["od_confidence"] <= 1.0
        assert len(entry["bbox_xyxy_input"]) == 4
        assert len(entry["bbox_xyxy_orig"]) == 4
        assert entry["frame_dims"] == [640, 640]


def test_audit_log_failure_does_not_crash(monkeypatch, tmp_path):
    """Audit log failures must not propagate — detection keeps running."""
    from config import get_config

    cfg = get_config()
    # Point OUTPUT_DIR at a path that can't be created (a file)
    bad_file = tmp_path / "not_a_dir"
    bad_file.write_text("blocking file")
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(bad_file))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=0.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 80.0, 200.0)],
        obj_conf=[0.9],
        cls_probs=[(0.01, 0.01, 0.01, 0.01, 0.01, 0.9)],
    )
    # Should not raise even though audit write fails
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert dets == []  # still dropped


def test_other_classes_pass_through_when_only_person_suppressed(monkeypatch, tmp_path):
    """Bird detections must survive a 'person'-only suppression list."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=0.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 100.0, 100.0)],
        obj_conf=[0.95],
        cls_probs=[(0.9, 0.01, 0.01, 0.01, 0.01, 0.01)],  # bird
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
    assert dets[0]["class"] == 0  # bird


def test_no_suppression_when_set_empty(monkeypatch, tmp_path):
    """Empty suppressed_classes = byte-identical to pre-suppression."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=[], min_size=0.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(320.0, 320.0, 80.0, 200.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.01, 0.01, 0.01, 0.01, 0.9)],  # person
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
    assert dets[0]["class"] == 5  # person passes through


# ---------------------------------------------------------------------------
# Min bbox size
# ---------------------------------------------------------------------------


def test_tiny_edge_artifact_dropped_by_min_size(monkeypatch, tmp_path):
    """Reproduces v2-coco's 1.5x1.6 px corner detection — must be dropped."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=[], min_size=8.0)
    # Mimics the live ONNX probe output
    raw = _build_yolox_raw(
        boxes_xywh=[
            (1.5, 0.9, 1.5, 1.6),
            (0.5, 0.9, 1.4, 1.6),
            (-0.6, -0.1, 1.5, 1.6),
        ],
        obj_conf=[0.95, 0.95, 0.95],
        cls_probs=[
            (0.58, 0.01, 0.01, 0.01, 0.01, 0.01),
            (0.5, 0.01, 0.01, 0.01, 0.01, 0.01),
            (0.55, 0.01, 0.01, 0.01, 0.01, 0.01),
        ],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=0.25,
        original_width=2560,
        original_height=1920,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    assert dets == []


def test_min_size_zero_disables_filter(monkeypatch, tmp_path):
    """min_bbox_size_px=0 = filter off; tiny boxes survive."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=[], min_size=0.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(50.0, 50.0, 2.0, 2.0)],
        obj_conf=[0.95],
        cls_probs=[(0.58, 0.01, 0.01, 0.01, 0.01, 0.01)],
    )
    dets = model.postprocess_output_yolox(
        raw,
        scale=1.0,
        original_width=640,
        original_height=640,
        conf_threshold=0.30,
        iou_threshold=0.5,
    )
    # Box with w=h=2 should survive with filter off (NMS gives 1 box)
    assert len(dets) == 1


def test_legitimate_box_at_min_size_threshold_kept(monkeypatch, tmp_path):
    """A box at exactly w=h=min_bbox_size_px must be kept (>= comparison)."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=[], min_size=8.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(100.0, 100.0, 8.0, 8.0)],
        obj_conf=[0.95],
        cls_probs=[(0.6, 0.01, 0.01, 0.01, 0.01, 0.01)],
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


def test_min_size_drops_w_or_h_below(monkeypatch, tmp_path):
    """Either dimension below threshold drops the box (AND semantics: BOTH
    must be >= threshold to keep)."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=[], min_size=8.0)
    raw = _build_yolox_raw(
        boxes_xywh=[
            (100.0, 100.0, 4.0, 50.0),  # w=4 < 8 -> drop
            (200.0, 100.0, 50.0, 4.0),  # h=4 < 8 -> drop
            (300.0, 100.0, 50.0, 50.0),  # both >= 8 -> keep
        ],
        obj_conf=[0.95, 0.95, 0.95],
        cls_probs=[
            (0.6, 0.01, 0.01, 0.01, 0.01, 0.01),
            (0.6, 0.01, 0.01, 0.01, 0.01, 0.01),
            (0.6, 0.01, 0.01, 0.01, 0.01, 0.01),
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
    assert len(dets) == 1
    # The 50x50 box centred at (300, 100) -> xyxy ~ (275, 75, 325, 125)
    assert 270 < dets[0]["x1"] < 280


# ---------------------------------------------------------------------------
# Filter ordering: suppression runs BEFORE min-bbox + per-class threshold
# ---------------------------------------------------------------------------


def test_suppression_short_circuits_before_other_filters(monkeypatch, tmp_path):
    """A suppressed person at conf 0.36 (above scalar floor, below the
    per-class threshold for some classes) AND size 2x2 (would fail
    min-bbox anyway) is audit-logged exactly once. Proves suppression
    is the first filter, not redundantly stacked, and that the audit
    record survives even when downstream filters would also have
    dropped the box."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=8.0)
    # conf = 0.95 * 0.38 = 0.361 → above conf_threshold_default=0.30,
    # so the audit floor lets it through.
    raw = _build_yolox_raw(
        boxes_xywh=[(100.0, 100.0, 2.0, 2.0)],
        obj_conf=[0.95],
        cls_probs=[(0.01, 0.01, 0.01, 0.01, 0.01, 0.38)],
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
    audit_path = tmp_path / "logs" / "suppressed.jsonl"
    # Suppression still logs even if downstream filters would have dropped
    # the box too — useful for debugging
    assert audit_path.exists()
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_min_bbox_runs_after_suppression_so_unsuppressed_tiny_dropped_silently(
    monkeypatch, tmp_path
):
    """A tiny bird box (not suppressed) is dropped by min-bbox without an
    audit log line — only suppression writes audit entries."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=8.0)
    raw = _build_yolox_raw(
        boxes_xywh=[(100.0, 100.0, 1.5, 1.6)],
        obj_conf=[0.95],
        cls_probs=[(0.58, 0.01, 0.01, 0.01, 0.01, 0.01)],  # bird
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
    audit_path = tmp_path / "logs" / "suppressed.jsonl"
    assert not audit_path.exists()  # never wrote


def test_5class_legacy_path_unchanged(monkeypatch, tmp_path):
    """5-class model + no suppression + min_size=0 → identical behaviour
    to pre-suppression code path."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    # Manually build a 5-class model with no per-class, no suppression,
    # min_size=0 — full legacy
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

    raw = _build_yolox_raw(
        boxes_xywh=[(100.0, 100.0, 80.0, 80.0), (300.0, 300.0, 2.0, 2.0)],
        obj_conf=[0.9, 0.9],
        cls_probs=[
            (0.45, 0.01, 0.01, 0.01, 0.01),
            (0.4, 0.01, 0.01, 0.01, 0.01),
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
    # Both boxes survive: tiny one too, since min_size disabled
    assert len(dets) == 2


def test_audit_skips_argmax_noise_below_scalar_floor(monkeypatch, tmp_path):
    """argmax routinely picks the suppressed class with negligible
    confidence (e.g. 1e-06). Those entries are still dropped, but the
    audit JSONL must skip them — otherwise the log explodes to millions
    of lines per hour. The audit floor is the model's scalar
    conf_threshold_default (0.30 for v2-coco)."""
    from config import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg, "OUTPUT_DIR", str(tmp_path))

    model = _make_v2_coco_model_with_suppression(suppressed=["person"], min_size=0.0)
    # Two person boxes:
    #   - real:  obj_conf 0.9 * cls 0.9   = 0.81   → audit
    #   - noise: obj_conf 0.001 * cls 0.6 = 0.0006 → drop silently
    raw = _build_yolox_raw(
        boxes_xywh=[
            (320.0, 320.0, 80.0, 200.0),
            (100.0, 100.0, 80.0, 200.0),
        ],
        obj_conf=[0.9, 0.001],
        cls_probs=[
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.9),
            (0.1, 0.1, 0.1, 0.1, 0.1, 0.6),
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
    assert dets == []  # both suppressed, neither persists

    audit_path = tmp_path / "logs" / "suppressed.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().strip().splitlines()
    # Only the real (0.81) detection lands in the audit log; the 0.0006
    # noise is below conf_threshold_default=0.30 and silently dropped.
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["class"] == "person"
    assert entry["od_confidence"] >= 0.30
