"""Integration-style tests: full ONNXDetectionModel load path with a synthetic
YOLOX-shaped ONNX.

We build a tiny ONNX graph on the fly that returns a constant tensor of shape
(1, 8400, 10) — mimicking the YOLOX-S 5-class locator's output shape — and
exercise:

- LOCAL_PATH env-var override
- labels.json array-format support
- output-format sniffing selecting OUTPUT_FORMAT_YOLOX_RAW
- Model-Compatibility-Guard passing with 'bird' label present
- warm-up path (calls detect() before returning from __init__)

This gives us end-to-end coverage of Phase 0 without depending on downloaded
weights or a GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from detectors.detector import (
    LOCAL_MODEL_ENV_VAR,
    OUTPUT_FORMAT_YOLOX_RAW,
    ONNXDetectionModel,
)

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _build_synthetic_yolox_onnx(path: Path, num_classes: int = 5, num_anchors: int = 8400):
    """Create a minimal ONNX that returns a constant (1, num_anchors, 5+num_classes) tensor.

    Graph: input 'images' (1,3,640,640) -> unused; output 'output' = Constant tensor.
    A few anchors carry high scores for class 0 (bird) and class 1 (squirrel) so
    the warm-up detect() call produces non-zero decodable output.
    """
    last_dim = 5 + num_classes
    data = np.zeros((1, num_anchors, last_dim), dtype=np.float32)
    # Anchor 0: centred bird box
    data[0, 0, 0:4] = [320.0, 320.0, 100.0, 100.0]  # xywh
    data[0, 0, 4] = 0.95  # obj
    data[0, 0, 5] = 0.9   # bird prob
    # Anchor 1: different location, squirrel
    data[0, 1, 0:4] = [100.0, 100.0, 80.0, 80.0]
    data[0, 1, 4] = 0.90
    data[0, 1, 6] = 0.85  # squirrel prob
    # All other anchors remain zeros -> filtered by conf_thr

    const_tensor = numpy_helper.from_array(data, name="const_output_value")

    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 640, 640]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, num_anchors, last_dim]
    )

    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=const_tensor,
    )

    graph = helper.make_graph(
        nodes=[const_node],
        name="synthetic_yolox",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    opset_imports = [helper.make_opsetid("", 13)]
    model = helper.make_model(graph, opset_imports=opset_imports, producer_name="test")
    model.ir_version = 7  # keep compatible with older onnxruntime
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture
def synthetic_yolox_dir(tmp_path, monkeypatch):
    """Create a tmp dir with best.onnx + labels.json + model_metadata.json."""
    model_dir = tmp_path / "yolox_synthetic"
    model_dir.mkdir()

    _build_synthetic_yolox_onnx(model_dir / "best.onnx")

    labels = ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"]
    (model_dir / "labels.json").write_text(json.dumps(labels))

    metadata = {
        "framework": "yolox",
        "variant": "s",
        "input_size": [640, 640],
        "input_format": "BGR",
        "input_normalize": False,
        "output_format": "yolox_raw",
        "num_classes": 5,
        "inference_thresholds": {"confidence": 0.15, "iou_nms": 0.50},
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata))

    monkeypatch.setenv(LOCAL_MODEL_ENV_VAR, str(model_dir))
    return model_dir


def test_local_override_loads_synthetic_yolox(synthetic_yolox_dir):
    """End-to-end: LOCAL_PATH env var -> sniff -> guard passes -> warm-up runs."""
    model = ONNXDetectionModel(debug=True)

    assert model.output_format == OUTPUT_FORMAT_YOLOX_RAW
    assert model.class_names == {
        "0": "bird",
        "1": "squirrel",
        "2": "cat",
        "3": "marten_mustelid",
        "4": "hedgehog",
    }
    assert model.input_size == (640, 640)
    assert model.conf_threshold_default == pytest.approx(0.15)
    assert model.iou_threshold_default == pytest.approx(0.50)
    assert model.model_id.startswith("local:")


def test_local_override_detect_produces_bird_and_squirrel(synthetic_yolox_dir):
    """Full detect() path returns decoded boxes with correct class names."""
    model = ONNXDetectionModel(debug=True)

    # Give it a 640x640 dummy frame — the synthetic ONNX ignores input content
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = model.detect(frame, 0.15)

    class_names = sorted(d["class_name"] for d in dets)
    assert class_names == ["bird", "squirrel"]
    # Both detections should carry non-trivial confidence (obj * cls_prob)
    for d in dets:
        assert d["confidence"] > 0.5


def test_local_override_rejects_29_species_labels(tmp_path, monkeypatch):
    """Model-Compatibility-Guard rejects FasterRCNN rollback via LOCAL_PATH."""
    model_dir = tmp_path / "bad_rollback"
    model_dir.mkdir()
    _build_synthetic_yolox_onnx(model_dir / "best.onnx", num_classes=29)

    # 29 bird-species labels without a 'bird' token
    species = [f"species_{i}" for i in range(29)]
    (model_dir / "labels.json").write_text(json.dumps(species))

    monkeypatch.setenv(LOCAL_MODEL_ENV_VAR, str(model_dir))

    with pytest.raises(ValueError, match="expected YOLOX-style locator"):
        ONNXDetectionModel(debug=False)


def test_local_override_missing_onnx_raises(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "labels.json").write_text(json.dumps(["bird"]))
    monkeypatch.setenv(LOCAL_MODEL_ENV_VAR, str(empty_dir))

    with pytest.raises(FileNotFoundError, match="best.onnx"):
        ONNXDetectionModel(debug=False)


def test_local_override_missing_labels_raises(tmp_path, monkeypatch):
    model_dir = tmp_path / "no_labels"
    model_dir.mkdir()
    _build_synthetic_yolox_onnx(model_dir / "best.onnx")
    monkeypatch.setenv(LOCAL_MODEL_ENV_VAR, str(model_dir))

    with pytest.raises(FileNotFoundError, match="labels.json"):
        ONNXDetectionModel(debug=False)


def test_local_override_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(LOCAL_MODEL_ENV_VAR, str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ONNXDetectionModel(debug=False)
